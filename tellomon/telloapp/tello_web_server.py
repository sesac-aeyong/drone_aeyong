# tello_web_server.py
import traceback
import cv2
from djitellopy import Tello
import threading
import time
import numpy as np
import queue
from hailorun import HailoRun
from yolo_tools import draw_detections_on_frame
from .app_tools import connect_to_tello_wifi
from settings import settings as S


class TelloWebServer:
    def __init__(self, socketio):
        self.tello = None
        self.socketio = socketio
        self.is_streaming = False
        self.is_connected = False
        self.current_frame = None
        self.current_frame_updated = False
        #self.current_depth_map = None
        self.current_detections = []
        self.target_class = None
        self.target_identity_id = None
        self.target_bbox = None  # Store in [x1, y1, x2, y2] format
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        #self.frame_center = (480, 360)

        # 이륙 안정화 시간
        self.last_takeoff_time = None
        self.takeoff_stabilization_time = 3.0  # 이륙 후 3초간 대기

        # RC 명령 설정
        #self.use_rc_for_manual = False
        self.use_rc_for_tracking = True
        #self.rc_speed = 40
        self.tracking_rc_speed = 30
        #self.rc_command_duration = 0.4
        
        # 웹 로그 시스템
        self.log_queue = queue.Queue(maxsize=100)  # 최대 100개 로그 저장
        self.log_thread = None
        self.is_logging = True
        self.start_log_broadcaster()
        
        # 실내 고도 제한 
        self.alt_min_cm = getattr(S, "ALT_MIN_CM", 40)    # 바닥 여유 40cm
        self.alt_max_cm = getattr(S, "ALT_MAX_CM", 200)   # 천장 2.0m
        self.alt_guard_cm = getattr(S, "ALT_GUARD_CM", 15) # 근접 완충대(소프트 밴드)

        # --- Simple escape (anti-rush) params ---
        self.ESCAPE_ALT_CM    = getattr(S, "ESCAPE_ALT_CM", 200)  # 목표 회피 고도(2m)
        self.ESCAPE_HOLD_S    = getattr(S, "ESCAPE_HOLD_S", 3.0)  # 정지 시간 3초

        # --- Escape state ---
        self._escape_mode       = None   # None | 'UP' | 'HOLD' | 'DOWN'
        self._escape_origin_alt = None   # 회피 시작 시 고도(cm)
        self._escape_t0         = None   # 모드 시작 시각(time.time())

        # 추론 엔진 초기화
        self.log("INFO", "Loading inference engine...")
        try:
            self.inference_engine = HailoRun()
            self.inference_engine.load()
            self.log("SUCCESS", "✅ Inference engine loaded successfully")
        except Exception as e:
            self.log("ERROR", f"❌ Failed to load inference engine: {e}")
            traceback.print_exc()
            self.inference_engine = None


    def log(self, level, message):
        """
        로그 메시지를 터미널과 웹에 동시 전송
        
        Args:
            level: "INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"
            message: 로그 메시지
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        # 터미널 출력
        if level == "ERROR":
            print(f"[{timestamp}] ❌ {message}")
        elif level == "SUCCESS":
            print(f"[{timestamp}] ✅ {message}")
        elif level == "WARNING":
            print(f"[{timestamp}] ⚠️ {message}")
        elif level == "DEBUG":
            print(f"[{timestamp}] 🔍 {message}")
        else:
            print(f"[{timestamp}] ℹ️ {message}")
        
        # 웹으로 전송 (큐에 추가)
        try:
            if self.log_queue.full():
                self.log_queue.get()  # 오래된 로그 제거
            self.log_queue.put(log_entry)
        except:
            pass
    

    def start_log_broadcaster(self):
        """로그를 웹으로 전송하는 스레드 시작"""
        def broadcast_logs():
            while self.is_logging:
                try:
                    log_entry = self.log_queue.get(timeout=0.5)
                    self.socketio.emit('log_message', log_entry)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Log broadcast error: {e}")
        
        self.log_thread = threading.Thread(target=broadcast_logs, daemon=True)
        self.log_thread.start()


    def connect_tello(self):
        """텔로 드론 연결"""
        try:
            self.log("INFO", "🔍 Checking Tello WiFi connection...")
            if not connect_to_tello_wifi():
                self.log("ERROR", "Failed to connect to Tello WiFi")
                return False
            
            self.log("SUCCESS", "Tello WiFi connected")
            
            if self.tello:
                try:
                    self.log("INFO", "Cleaning up old connection...")
                    self.is_streaming = False
                    
                    if hasattr(self.tello, 'background_frame_read') and self.tello.background_frame_read:
                        try:
                            self.tello.background_frame_read.stop()
                        except:
                            pass
                    
                    self.tello.streamoff()
                    self.tello.end()
                    
                except Exception as e:
                    self.log("WARNING", f"Cleanup error (ignored): {e}")
                finally:
                    self.tello = None
            
            self.log("INFO", "Creating new Tello connection...")
            self.tello = Tello()
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.log("INFO", f"Connection attempt {attempt + 1}/{max_retries}...")
                    self.tello.connect()
                    break
                except Exception as e:
                    self.log("WARNING", f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise
            
            self.battery = self.tello.get_battery()
            self.log("SUCCESS", f"Tello connected. Battery: {self.battery}%")
            # self.log('INFO', f'Tello speed: {self.tello.query_speed()}')
            
            # 배터리 경고
            if self.battery < 20:
                self.log("WARNING", f"⚠️ Low battery: {self.battery}%")
            
            self.log("INFO", "Starting video stream...")
            if self.tello.stream_on:
                try:
                    self.tello.streamoff()
                    self.log('INFO', 'Waiting for video stream to end...')
                except:
                    pass
            
            self.tello.streamon()
            self.log('INFO', 'Waiting for tello video stream to start...')
            
            self.log("SUCCESS", "🎥 Stream started successfully")
            self.is_connected = True
            return True
        
        except Exception as e:
            self.log("ERROR", f"Connection error: {e}")
            traceback.print_exc()
            self.is_connected = False
            self.tello = None
            return False
    
    
# === Add/Replace inside class TelloWebServer =================================

    # ---------------------------
    # Helpers for smooth control
    # ---------------------------
    def _slew(self, prev, target, max_step):
        """한 루프당 변화량을 제한해 급격한 명령 변화를 방지"""
        delta = target - prev
        if delta > max_step:   return prev + max_step
        if delta < -max_step:  return prev - max_step
        return target

    def _ema(self, name, value, alpha):
        """self._ema_state[name]에 EMA 저장"""
        if not hasattr(self, "_ema_state"):
            self._ema_state = {}
        if name not in self._ema_state:
            self._ema_state[name] = value
        self._ema_state[name] = (1 - alpha) * self._ema_state[name] + alpha * value
        return self._ema_state[name]

    def _init_tracker_state(self):
        """트래킹 상태 변수 초기화"""
        self._cmd_lr = 0
        self._cmd_fb = 0
        self._cmd_ud = 0
        self._cmd_yaw = 0
        self._last_bbox = None
        self._last_seen_t = None
        self._lost_since_t = None
        self._ema_state = {}
        self._integral_fb = 0.0     # 거리(I) 성분 약간
        self._integral_clip = 50.0  # 바람/기체 바이어스 보정용
        # >>> ADD: occlusion strategy
        self._lost_strategy = None   # None | 'FWD' | 'SWEEP'
        self._lost_t0 = None
        self._last_ratio = None      # 마지막 관측 ratio 저장
        self._last_center_norm = None # 마지막 관측 중심 (cx/W, cy/H)

    # -------------------------------------------------------------------------
    # New tracking thread (drop-in replacement)
    # -------------------------------------------------------------------------
    def tracking_thread(self):
        """
        고급 IBVS 스타일 자동 추적 스레드 (bbox-only)
        - 목표: 매끄럽고 일관된 추종, 대각선 추종, 프레임 이탈 시 재탐색
        - 입력: self.target_bbox = [x1,y1,x2,y2] (픽셀)
        - 출력: send_rc_control(lr, fb, ud, yaw)
        """

        # ====== 제어 파라미터 (필요시 조절) ===================================
        LOOP_HZ                = 30.0
        DT                     = 1.0 / LOOP_HZ
        EMA_ALPHA_ERR          = 0.35    # 오차 EMA
        EMA_ALPHA_VEL          = 0.50    # 속도 EMA
        EMA_ALPHA_SIZE         = 0.30

        # Deadbands & 게인
        YAW_DEADBAND           = 0.06     # x정규오차 6% 이하면 yaw 안함
        LR_DEADBAND            = 0.02
        UD_DEADBAND            = 0.02
        SIZE_DEADBAND          = 0.02     # log-area 오차

        K_YAW                  = 130.0    # yaw 스케일 (deg/s 환산 없이 RC 단위)
        K_LR                   = 100.0
        K_UD                   = 110.0
        K_FB_P                 = 300.0    # 거리 P
        K_FB_I                 =  25.0    # 거리 I(소량만)

        # 속도/명령 제한
        MAX_RC                 = int(self.tracking_rc_speed)  # 기존 설정 사용
        SLEW_RC_STEP           = 18        # 루프당 최대 변화 (부드러움)
        COAST_DECAY            = 0.85      # 타겟 상실 시 점감
        COAST_MAX_TIME         = 1.0       # 최대 coast 유지 시간(s)

        # 목표 크기(프레임 면적 대비 비율)
        TARGET_RATIO           = 0.28      # 28% 정도 화면 차지하도록
        
        # SAFETY 파라미터
        NEAR_RATIO             = 0.35   # 이 이상은 근거리: 전진 금지(또는 후퇴만 허용)
        PUSHBACK_RATIO         = 0.45   # 강한 근접: 전진 금지 + 약한 후퇴
        HARD_STOP_RATIO        = 0.50   # 절대 상한: 즉시 후퇴(면적 50% 초과 금지)
        SAFE_FB_FWD_CAP        = 20     # 전진 soft cap
        SAFE_FB_BWD_CAP        = 20     # 후퇴 soft cap(|-|)

        # 급접근(급확대) 감지: ratio의 시간 미분 임계값(초당 면적비 변화)
        EMA_ALPHA_RATIO        = 0.40
        RAPID_ENLARGE_WARN     = 0.25   # 이 이상(+/s)이면 전진 금지
        RAPID_ENLARGE_PANIC    = 0.35   # 이 이상(+/s)이면 소폭 후퇴

        # 전진 가속도 제한: 루프당 전진 목표 증가량 제한(추가 안전)
        FB_ACCEL_STEP_FWD      = 8      # +방향(전진) 증분 제한
        FB_ACCEL_STEP_BWD      = 12     # -방향(후퇴) 증분 제한
        
        # 프레임 경계 근접 힌트
        EDGE_FRAC              = 0.06      # 가장자리 6%를 '위험영역'으로 판단
        EDGE_BOOST             = 0.6       # 경계 근접 시 해당 축 추가 가중

        # 재탐색(Search) 파라미터
        SEARCH_YAW_SPEED       = 40        # 분실 시 회전 기본속도
        SEARCH_UD_SPEED        = 28
        SEARCH_FB_SPEED        = 0         # 분실 시 전후는 보수적으로 0

        # 안전/기타
        MIN_BATT               = 10        # 10% 이하면 즉시 정지
        STABILIZE_AFTER_TAKEOFF= self.takeoff_stabilization_time

        # >>> ADD: Occlusion-forward strategy params
        OCCLUDED_GRACE_S       = 3.0   # 3초 이상 끊기면 '가림'으로 가정
        OCC_FWD_MAX_S          = 2.5   # 전진 시도 최대 시간
        OCC_FWD_SPEED          = min(MAX_RC, 18)  # 전진 속도 캡
        OCC_CENTER_BAND        = 0.25  # 마지막 중심이 화면 중앙 ±25% 안이면 '프레임 내 가림'으로 추정
        RATIO_GOAL_OCCLUDED    = 0.30  # 재관측 시 이 이상이면 충분히 붙었다고 판단
        SWEEP_HALF_PERIOD_S    = 1.2   # 좌/우 반주기(초)로 지그재그 회전 탐색
        
        # ======================================================================
        self.log("INFO", "🎯 IBVS tracking thread started")
        self._init_tracker_state()

        # 이륙 안정화 대기 (버퍼)
        if self.last_takeoff_time is not None:
            while True:
                dt_take = time.time() - self.last_takeoff_time
                if dt_take >= STABILIZE_AFTER_TAKEOFF:
                    break
                self.tello.send_rc_control(0, 0, 0, 0)
                self.log("INFO", f"⏳ Stabilizing... {STABILIZE_AFTER_TAKEOFF - dt_take:.1f}s")
                time.sleep(0.1)
            self.last_takeoff_time = None
            self.log("SUCCESS", "✅ Stabilization complete - starting IBVS tracking")

        while self.is_tracking:
            loop_start = time.time()
            try:
                # ===== 안전 가드 =====
                try:
                    if self.tello and isinstance(self.battery, (int, float)) and self.battery <= MIN_BATT:
                        self.log("WARNING", "🔋 Critically low battery - halting RC")
                        self.tello.send_rc_control(0, 0, 0, 0)
                        time.sleep(0.5)
                        continue
                except Exception:
                    pass

                bbox = None
                with self.lock:
                    bbox = self.target_bbox
                    frm = self.current_frame

                if frm is None:
                    time.sleep(DT)
                    continue

                h, w = frm.shape[:2]
                cx, cy = w * 0.5, h * 0.5

                if bbox is not None:
                    # ------------------ 타겟 관측 유효 -------------------------
                    x1, y1, x2, y2 = bbox
                    bx = (x1 + x2) * 0.5
                    by = (y1 + y2) * 0.5
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    area = bw * bh
                    ratio = area / float(w * h)

                    # 정규 오차(화면 대비)
                    ex = (bx - cx) / w      # -0.5 ~ 0.5
                    ey = (by - cy) / h
                    # log-area 오차: TARGET_RATIO를 기준으로 곱배 변화에 민감
                    e_size_raw = np.log(max(1e-6, ratio) / max(1e-6, TARGET_RATIO))

                    # EMA로 노이즈 완화
                    ex_f   = self._ema("ex",   ex,   EMA_ALPHA_ERR)
                    ey_f   = self._ema("ey",   ey,   EMA_ALPHA_ERR)
                    es_f   = self._ema("esz",  e_size_raw, EMA_ALPHA_SIZE)

                    # 속도 추정(프레임 속 좌표 변화율)
                    if self._last_bbox is not None and self._last_seen_t is not None:
                        dtv = max(1e-3, loop_start - self._last_seen_t)
                        last_cx = (self._last_bbox[0] + self._last_bbox[2]) * 0.5
                        last_cy = (self._last_bbox[1] + self._last_bbox[3]) * 0.5
                        vx = ((bx - last_cx) / w) / dtv
                        vy = ((by - last_cy) / h) / dtv
                    else:
                        vx = vy = 0.0
                    vx_f = self._ema("vx", vx, EMA_ALPHA_VEL)
                    vy_f = self._ema("vy", vy, EMA_ALPHA_VEL)

                    # 프레임 가장자리 근접 가중 (이탈 방지용)
                    edge_x = 0.0
                    edge_y = 0.0
                    if bx < w * EDGE_FRAC:        edge_x = -EDGE_BOOST
                    elif bx > w * (1 - EDGE_FRAC):edge_x =  EDGE_BOOST
                    if by < h * EDGE_FRAC:        edge_y = -EDGE_BOOST
                    elif by > h * (1 - EDGE_FRAC):edge_y =  EDGE_BOOST

                    # ----------- 제어 로직 (IBVS) -----------------------------
                    # yaw: 좌우 큰 오차일수록 yaw 우선 -> 잔여 오차는 LR로 병행
                    if abs(ex_f) > YAW_DEADBAND:
                        yaw_cmd = int(np.clip(K_YAW * ex_f, -MAX_RC, MAX_RC))
                        lr_cmd  = 0
                    else:
                        yaw_cmd = 0
                        lr_cmd  = int(np.clip(K_LR * (ex_f + 0.35 * vx_f + edge_x), -MAX_RC, MAX_RC))
                        if abs(lr_cmd) < int(MAX_RC * 0.1):
                            lr_cmd = 0

                    # ud: 세로 오차 + 경계 근접 + 약간의 속도 선행
                    if abs(ey_f) > UD_DEADBAND:
                        ud_cmd = int(np.clip(-K_UD * (ey_f + 0.25 * vy_f + edge_y), -MAX_RC, MAX_RC))
                    else:
                        ud_cmd = 0

                    # ----- ALTITUDE LIMITS (ceiling/floor clamp) -----
                    h_cm = None
                    try:
                        # TOF는 간헐적으로 -1/0이 나올 수 있으니 유효성 검사
                        if isinstance(self.height, (int, float)) and self.height > 0:
                            h_cm = float(self.height)
                    except:
                        h_cm = None

                    if h_cm is not None:
                        # 1) 절대 천장: alt_max_cm 초과 시 무조건 하강 방향(양의 ud는 금지)
                        if h_cm >= self.alt_max_cm:
                            if ud_cmd > 0: ud_cmd = 0
                            ud_cmd = min(ud_cmd, -10)  # 살짝이라도 내려오게
                            self.log("WARNING", f"[ALT] HARD_CEILING h={h_cm:.0f}cm → ud={ud_cmd}")

                        # 2) 천장 근접 소프트 밴드: 더 올라가지 못하게(상승 금지)
                        elif h_cm >= self.alt_max_cm - self.alt_guard_cm:
                            if ud_cmd > 0: ud_cmd = 0  # 상승 차단
                            # 필요시 천장 근접시 FB도 살짝 캡: 대각상향 추세 억제
                            # self._cmd_fb = min(self._cmd_fb, SAFE_FB_FWD_CAP // 2)

                        # 3) 절대 바닥: alt_min_cm 이하이면 반드시 상승 방향(음의 ud 금지)
                        if h_cm <= self.alt_min_cm:
                            if ud_cmd < 0: ud_cmd = 0
                            ud_cmd = max(ud_cmd, +10)
                            self.log("WARNING", f"[ALT] HARD_FLOOR h={h_cm:.0f}cm → ud={ud_cmd}")

                        # 4) 바닥 근접 소프트 밴드: 더 내려가지 못하게(하강 금지)
                        elif h_cm <= self.alt_min_cm + self.alt_guard_cm:
                            if ud_cmd < 0: ud_cmd = 0  # 하강 차단
                    else:
                        # TOF 불가 시 보수적으로: 과한 상승/하강 제한(실내 안전)
                        ud_cmd = int(np.clip(ud_cmd, -10, +10))

                    # fb: 거리(log-area) P + I. (n배 멀어지면 n배 전진 느낌)
                    if abs(es_f) > SIZE_DEADBAND:
                        self._integral_fb += es_f * DT * K_FB_I
                        self._integral_fb = float(np.clip(self._integral_fb, -self._integral_clip, self._integral_clip))
                        fb_raw = K_FB_P * es_f + self._integral_fb
                        fb_cmd = int(np.clip(fb_raw, -MAX_RC, MAX_RC))
                        if fb_cmd < 0:
                            fb_cmd = int(0.4 * fb_cmd)  # 후퇴는 보수적으로
                    else:
                        self._integral_fb *= 0.98
                        fb_cmd = 0

                    # ---------- 🔒 SAFETY: ratio 변화율(급확대·급축소) 계산 ----------
                    ratio_f = self._ema("ratio", ratio, EMA_ALPHA_RATIO)
                    if not hasattr(self, "_ratio_prev"):
                        self._ratio_prev = ratio_f
                        self._ratio_prev_t = loop_start
                    dt_ratio = max(1e-3, loop_start - getattr(self, "_ratio_prev_t", loop_start))
                    dratio_dt = (ratio_f - getattr(self, "_ratio_prev", ratio_f)) / dt_ratio
                    self._ratio_prev = ratio_f
                    self._ratio_prev_t = loop_start

                    # ---------- 🔒 SAFETY: 근접·상한·급접근 보호 ----------
                    # >>> ADD: Simple escape trigger (rush or too close)
                    should_escape = (ratio_f >= PUSHBACK_RATIO) or (dratio_dt >= RAPID_ENLARGE_PANIC)
                    if (self._escape_mode is None) and should_escape:
                        # 캡된 목표고도 계산(천장 보호)
                        ceiling_soft = self.alt_max_cm - self.alt_guard_cm
                        target_alt = min(self.ESCAPE_ALT_CM, ceiling_soft)

                        # 현재 고도 스냅샷
                        origin = None
                        try:
                            if isinstance(self.height, (int, float)) and self.height > 0:
                                origin = float(self.height)
                        except:
                            pass

                        self._escape_origin_alt = origin
                        self._escape_target_alt = target_alt
                        self._escape_mode = 'UP'
                        self._escape_t0 = time.time()
                        self.log("WARNING", f"[ESCAPE] Triggered → UP to ~{target_alt:.0f} cm (origin={origin})")
                    
                    # 1) 절대 상한: 화면 50% 초과 금지 → 즉시 후퇴
                    if ratio_f >= HARD_STOP_RATIO:
                        self.log("WARNING", f"[SAFETY] HARD_STOP ratio={ratio_f:.2f} fb -> {fb_cmd}")
                        if fb_cmd > 0: fb_cmd = 0
                        fb_cmd = min(fb_cmd, -15)   # 강제 살짝 후퇴
                        # 근접 시 yaw/lr 우선(전진 금지)
                        # (yaw/lr 제한은 아래 근거리 규칙에서 처리)

                    # 2) 강한 근접: 45% 이상 → 전진 금지 + 가벼운 후퇴
                    elif ratio_f >= PUSHBACK_RATIO:
                        self.log("WARNING", f"[SAFETY] PUSHBACK ratio={ratio_f:.2f} fb -> {fb_cmd}")
                        if fb_cmd > 0: fb_cmd = 0
                        fb_cmd = min(fb_cmd, -10)

                    # 3) 근거리 일반: 35% 이상 → 전진 금지(0) 또는 후퇴만 허용
                    elif ratio_f >= NEAR_RATIO:
                        if fb_cmd > 0: fb_cmd = 0  # 근거리에서는 전진 금지(충돌 방지)

                    # 4) 급접근 보호: ratio가 빠르게 커짐(양수) → 전진 차단/후퇴
                    if dratio_dt >= RAPID_ENLARGE_PANIC:
                        self.log("WARNING", f"[SAFETY] RAPID_ENLARGE_PANIC dr/dt={dratio_dt:.2f} fb -> {fb_cmd}")
                        # 매우 빠르게 가까워짐 → 즉시 약간 후퇴
                        fb_cmd = min(fb_cmd, -12)
                    elif dratio_dt >= RAPID_ENLARGE_WARN:
                        # 빠르게 가까워짐 → 전진 금지
                        if fb_cmd > 0: fb_cmd = 0

                    # 5) 전진/후퇴 소프트 캡
                    if fb_cmd > 0:
                        fb_cmd = min(fb_cmd, SAFE_FB_FWD_CAP)
                    else:
                        fb_cmd = max(fb_cmd, -SAFE_FB_BWD_CAP)

                    # 6) 전진/후퇴 가속도(증분) 제한: 이전 명령 대비 증분 제한
                    #    (slew에 앞서 fb만 한 번 더 보수적으로 제한)
                    fb_target = fb_cmd
                    if fb_target > self._cmd_fb:
                        # 전진 쪽으로 증가
                        fb_cmd = min(self._cmd_fb + FB_ACCEL_STEP_FWD, fb_target)
                    else:
                        # 후퇴 쪽으로 증가
                        fb_cmd = max(self._cmd_fb - FB_ACCEL_STEP_BWD, fb_target)

                    # ---------- 대각선 추종: yaw와 lr를 상보적으로 병합 ----------
                    # 큰 ex면 yaw에, 작은 ex면 lr에 더 배분했으므로 그 상태 유지 (근거리에서는 fb가 0 또는 음수라 yaw/lr 중심으로 대각 추종)

                    # >>> ADD: Simple ESCAPE state machine (UP -> HOLD -> DOWN)
                    if self._escape_mode is not None:
                        esc_lr, esc_fb, esc_ud, esc_yaw = 0, 0, 0, 0  # 정면 고정, 수직만 사용

                        # 현재 고도 읽기
                        h_cm = None
                        try:
                            if isinstance(self.height, (int, float)) and self.height > 0:
                                h_cm = float(self.height)
                        except:
                            pass

                        mode = self._escape_mode
                        now  = time.time()

                        if mode == 'UP':
                            # 목표 고도 근처까지 상승
                            target = self._escape_target_alt
                            if (h_cm is not None) and (h_cm < target - self.alt_guard_cm):
                                esc_ud = +min(20, int(self.tracking_rc_speed))  # 부드럽게 상승
                            else:
                                # 고도 도달 → HOLD로 전환
                                self._escape_mode = 'HOLD'
                                self._escape_t0   = now
                                self.log("INFO", f"[ESCAPE] Reached ~{h_cm} cm → HOLD {self.ESCAPE_HOLD_S}s")

                        elif mode == 'HOLD':
                            # 3초 정지
                            if (now - self._escape_t0) >= self.ESCAPE_HOLD_S:
                                self._escape_mode = 'DOWN'
                                self._escape_t0   = now
                                self.log("INFO", "[ESCAPE] HOLD done → DOWN")
                            # esc_* 모두 0 (정지 유지)

                        elif mode == 'DOWN':
                            # 원고도(있으면) 또는 안전 최소고도까지 하강
                            fallback = self.alt_min_cm + max(40, self.alt_guard_cm)  # 너무 낮게 붙지 않도록
                            target_down = self._escape_origin_alt if (self._escape_origin_alt is not None) else fallback

                            if (h_cm is not None) and (h_cm > target_down + self.alt_guard_cm):
                                esc_ud = -min(18, int(self.tracking_rc_speed))  # 부드럽게 하강
                            else:
                                # 회피 종료
                                self._escape_mode = None
                                self._escape_origin_alt = None
                                self._escape_target_alt = None
                                self._escape_t0 = None
                                self.log("SUCCESS", f"[ESCAPE] Down complete (~{h_cm} cm) → RESUME tracking")

                        # 명령 적용(회피가 활성화된 동안에는 IBVS를 덮어씀)
                        self._cmd_lr  = int(self._slew(self._cmd_lr,  esc_lr,  SLEW_RC_STEP))
                        self._cmd_fb  = int(self._slew(self._cmd_fb,  esc_fb,  SLEW_RC_STEP))
                        self._cmd_ud  = int(self._slew(self._cmd_ud,  esc_ud,  SLEW_RC_STEP))
                        self._cmd_yaw = int(self._slew(self._cmd_yaw, esc_yaw, SLEW_RC_STEP))
                        if self.use_rc_for_tracking:
                            self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)
                        # 회피 루틴이 이 루프의 RC를 소비했으니, 아래 IBVS 일반 경로는 건너뜀
                        # (이 줄이 중요)
                        continue

                    # ---------- Slew-rate limit + 적용 ----------
                    self._cmd_yaw = int(self._slew(self._cmd_yaw, yaw_cmd, SLEW_RC_STEP))
                    self._cmd_lr  = int(self._slew(self._cmd_lr,  lr_cmd,  SLEW_RC_STEP))
                    self._cmd_ud  = int(self._slew(self._cmd_ud,  ud_cmd,  SLEW_RC_STEP))
                    self._cmd_fb  = int(self._slew(self._cmd_fb,  fb_cmd,  SLEW_RC_STEP))

                    # 전송
                    if self.use_rc_for_tracking:
                        self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)

                    # 기록
                    self._last_bbox = bbox
                    self._last_seen_t = loop_start
                    self._lost_since_t = None

                    # >>> ADD: keep last hints for occlusion-strategy
                    self._last_ratio = ratio_f
                    self._last_center_norm = (bx / w, by / h)

                    # >>> ADD: if we were in occlusion strategy and 이제 충분히 가까워졌다면 전략 해제
                    if self._lost_strategy is not None and self._last_ratio is not None:
                        if self._last_ratio >= RATIO_GOAL_OCCLUDED:
                            self._lost_strategy = None
                            self._lost_t0 = None
                            self.log("INFO", "[OCC] Reacquired with sufficient size → resume normal IBVS")

                else:
                    # ------------------ 타겟 분실/가림 -------------------------
                    now = loop_start
                    last_seen_ago = 1e9 if self._last_seen_t is None else (now - self._last_seen_t)


                    # >>> ADD: '프레임 내 가림'으로 보이면 먼저 전진해서 30%까지 붙고, 그 후 회전 탐색
                    # 판단 기준:
                    #  - 3초 이상 미관측 (OCCLUDED_GRACE_S)
                    #  - 마지막 중심이 화면 중앙부 (±OCC_CENTER_BAND) 안이었다면 '프레임 내 가림'으로 가정
                    #  - 배터리/고도 안전은 기존 가드 + 전/후 캡 사용
                    in_center_band = False
                    if self._last_center_norm is not None:
                        lx, ly = self._last_center_norm
                        in_center_band = (abs(lx - 0.5) <= OCC_CENTER_BAND) and (abs(ly - 0.5) <= OCC_CENTER_BAND)

                    # 마지막 ratio 힌트가 있고, 이미 충분히 컸던 상황이라면 FWD 생략(근접 돌진 방지)
                    last_small_enough = (self._last_ratio is None) or (self._last_ratio < RATIO_GOAL_OCCLUDED)
                    if (last_seen_ago >= OCCLUDED_GRACE_S) and in_center_band and last_small_enough:
                        # 전략 진입 결정
                        if self._lost_strategy is None:
                            self._lost_strategy = 'FWD'
                            self._lost_t0 = now
                            self.log("WARNING", "[OCC] Likely occlusion (not out-of-frame) → FWD-to-30% then SWEEP")

                        # --- FWD 단계: 일정 시간 전진해서 시야 확보 ---
                        if self._lost_strategy == 'FWD':
                            # 전진만 수행, yaw/ud는 0 (충돌 방지 위해 전진은 제한)
                            yaw_cmd = 0
                            lr_cmd  = 0
                            ud_cmd  = 0
                            fb_cmd  = min(SAFE_FB_FWD_CAP, OCC_FWD_SPEED)

                            # Slew 적용
                            self._cmd_yaw = int(self._slew(self._cmd_yaw, yaw_cmd, SLEW_RC_STEP))
                            self._cmd_lr  = int(self._slew(self._cmd_lr,  lr_cmd,  SLEW_RC_STEP))
                            self._cmd_ud  = int(self._slew(self._cmd_ud,  ud_cmd,  SLEW_RC_STEP))
                            self._cmd_fb  = int(self._slew(self._cmd_fb,  fb_cmd,  SLEW_RC_STEP))

                            if self.use_rc_for_tracking:
                                self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)

                            # 전진 시간 종료 → SWEEP 전환
                            if (now - self._lost_t0) >= OCC_FWD_MAX_S:
                                self._lost_strategy = 'SWEEP'
                                self._lost_t0 = now
                                self.log("INFO", "[OCC] FWD stage done → SWEEP rotate-search")
                            # 이 루프는 소비되었으므로 아래 일반 로직은 건너뜀
                            time.sleep(DT)
                            continue

                        # --- SWEEP 단계: 좌/우 교대 회전으로 가림면 가장자리를 찾아줌 ---
                        elif self._lost_strategy == 'SWEEP':
                            # 반주기마다 부호를 바꿈: ... ← → ← → ...
                            phase = int((now - self._lost_t0) / SWEEP_HALF_PERIOD_S)
                            yaw_dir = -1 if (phase % 2 == 0) else 1
                            yaw_cmd = int(np.clip(yaw_dir * SEARCH_YAW_SPEED, -MAX_RC, MAX_RC))
                            ud_cmd  = 0
                            lr_cmd  = 0
                            fb_cmd  = 0  # 회전 중심 탐색

                            self._cmd_yaw = int(self._slew(self._cmd_yaw, yaw_cmd, SLEW_RC_STEP))
                            self._cmd_ud  = int(self._slew(self._cmd_ud,  ud_cmd,  SLEW_RC_STEP))
                            self._cmd_lr  = int(self._slew(self._cmd_lr,  lr_cmd,  SLEW_RC_STEP))
                            self._cmd_fb  = int(self._slew(self._cmd_fb,  fb_cmd,  SLEW_RC_STEP))

                            if self.use_rc_for_tracking:
                                self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)

                            # SWEEP은 타임아웃 없이 지속 (재관측 되면 위에서 전략 자동 해제)
                            time.sleep(DT)
                            continue

                    # ===== 기존 기본 동작(프레임 밖/일반 분실) =====

                    # 1) 직후(<= COAST_MAX_TIME)는 마지막 명령을 점감(coast)
                    if last_seen_ago <= COAST_MAX_TIME:
                        self._cmd_lr  = int(self._cmd_lr  * COAST_DECAY)
                        self._cmd_fb  = int(self._cmd_fb  * COAST_DECAY)
                        self._cmd_ud  = int(self._cmd_ud  * COAST_DECAY)
                        self._cmd_yaw = int(self._cmd_yaw * COAST_DECAY)
                    else:
                        # 2) 재탐색: 마지막 관측 에러/속도 부호를 이용해 회전/상하 스캔
                        ex_f = self._ema_state.get("ex", 0.0)
                        ey_f = self._ema_state.get("ey", 0.0)
                        vx_f = self._ema_state.get("vx", 0.0)
                        vy_f = self._ema_state.get("vy", 0.0)

                        yaw_cmd = int(np.clip(np.sign(ex_f if abs(ex_f) > YAW_DEADBAND else vx_f) * SEARCH_YAW_SPEED,
                                              -MAX_RC, MAX_RC))
                        ud_cmd  = int(np.clip(-np.sign(ey_f if abs(ey_f) > UD_DEADBAND else vy_f) * SEARCH_UD_SPEED,
                                              -MAX_RC, MAX_RC))
                        lr_cmd  = 0
                        fb_cmd  = SEARCH_FB_SPEED

                        self._cmd_yaw = int(self._slew(self._cmd_yaw, yaw_cmd, SLEW_RC_STEP))
                        self._cmd_ud  = int(self._slew(self._cmd_ud,  ud_cmd,  SLEW_RC_STEP))
                        self._cmd_lr  = int(self._slew(self._cmd_lr,  lr_cmd,  SLEW_RC_STEP))
                        self._cmd_fb  = int(self._slew(self._cmd_fb,  fb_cmd,  SLEW_RC_STEP))

                    if self.use_rc_for_tracking:
                        self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)

                    if self._lost_since_t is None:
                        self._lost_since_t = now
                        self.log("WARNING", "⚠️ Target lost - entering search mode")

                # 루프 타이밍 정렬
                elapsed = time.time() - loop_start
                sleep_t = max(0.0, DT - elapsed)
                time.sleep(sleep_t)

            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except:
                    pass
                time.sleep(0.2)

        # 종료 시 안전 정지
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except:
            pass
        self.log("INFO", "🛑 IBVS tracking thread stopped")

    # start_tracking에서 트래킹 상태 초기화 훅 추가 (선택)
    def _spawn_tracking_thread(self):
        if getattr(self, "_tracking_thread", None) and self._tracking_thread.is_alive():
            return
        # 새 루프마다 제어 상태 리셋(EMA, I성분, 이전 명령값 등)
        self._init_tracker_state()
        t = threading.Thread(target=self.tracking_thread, daemon=True)
        t.start()
        self._tracking_thread = t


    def video_stream_thread(self):
        """비디오 스트리밍 스레드"""
        print("📹 Starting video stream thread...")
        
        try:
            time.sleep(3)
            frame_reader = self.tello.get_frame_read()
            print("✅ Frame reader initialized")
        except Exception as e:
            print(f"❌ Failed to initialize frame reader: {e}")
            traceback.print_exc()
            self.is_streaming = False
            self.socketio.emit('stream_error', {
                'message': 'Failed to start video stream. Please reconnect.'
            })
            return
            
        error_count = 0
        max_errors = 10
        
        while self.is_streaming:
            try:
                frame = frame_reader.frame
                
                if frame is None:
                    error_count += 1
                    if error_count >= max_errors:
                        print("⚠️ Too many frame errors")
                        self.is_streaming = False
                        self.socketio.emit('stream_error', {
                            'message': 'Video stream lost. Please reconnect.'
                        })
                        break                    
                    continue
            
                error_count = 0
                
                # 추론 실행
                detections, depth_map, *_ = self.inference_engine.run(frame)
                
                with self.lock:
                    self.current_detections = detections
                    
                    if self.is_tracking:
                        # 1) 도둑 모드 후보 찾기: thief_dist <= gate 인 것 중 최솟값
                        best = None
                        for det in detections:
                            get = det.get if isinstance(det, dict) else (lambda k, d=None: getattr(det, k, d))
                            td = get("thief_dist")
                            tg = get("thief_cos_dist")
                            if td is None or tg is None:
                                continue
                            if td <= tg:
                                if (best is None) or (td < best.get("thief_dist", 1e9)):
                                    best = det

                        if best is not None:
                            # 매칭 통과: 이 bbox만 추적 대상으로
                            self.target_bbox  = best["bbox"] if isinstance(best, dict) else best.bbox
                            self.target_class = (best.get("class", "person") if isinstance(best, dict)
                                                else getattr(best, "cls", "person"))
                        else:
                            # 매칭 실패: 타겟 상실 처리
                            if self.target_bbox is not None:
                                self.log("WARNING", f"⚠️ Thief not found under gate; holding position")
                            self.target_bbox = None
                
                # 감지 결과 그리기
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_with_detections = draw_detections_on_frame(frame, detections)
                
                # 프레임 중심 십자선 표시
                h, w = frame_with_detections.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.line(frame_with_detections, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
                cv2.line(frame_with_detections, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)
                cv2.circle(frame_with_detections, (cx, cy), 5, (255, 255, 255), -1)
                
                # 배터리 및 높이 정보 업데이트
                try:
                    old_battery = self.battery
                    self.battery = self.tello.get_battery()
                    self.height = self.tello.get_distance_tof()
                    
                    # 배터리 경고
                    if self.battery < 15 and old_battery >= 15:
                        self.log("WARNING", f"⚠️ Critical battery: {self.battery}% - Land soon!")
                    elif self.battery < 25 and old_battery >= 25:
                        self.log("WARNING", f"⚠️ Low battery: {self.battery}%")
                except:
                    pass
                
                # 프레임 저장
                with self.lock:
                    self.current_frame = frame_with_detections
                    self.current_frame_updated = True
                
                # 감지 정보를 클라이언트에 전송
                self.socketio.emit('detections_update', {
                    'detections': detections,
                    'battery': self.battery,
                    'height': self.height,
                    'is_tracking': self.is_tracking,
                    'target_identity_id': self.target_identity_id,
                    'target_class': self.target_class
                })
                
                
                # time.sleep(0.033)
                
            except Exception as e:
                print(f"Stream error: {e}")
                traceback.print_exc()
                error_count += 1
                if error_count >= max_errors:
                    print("❌ Stream failed completely")
                    self.is_streaming = False
                    break
                time.sleep(0.1)
                
        print("📹 Video stream thread ended")
    

    def start_streaming(self):
        """스트리밍 시작"""
        if not self.is_streaming and self.is_connected:
            self.is_streaming = True
            thread = threading.Thread(target=self.video_stream_thread)
            thread.daemon = True
            thread.start()
            return True
        return False
    

    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False

    def start_tracking(self):
        """자동 추적 시작 (identity 우선, 실패 시 bbox 폴백)"""
        if self.is_tracking:
            self.log("WARNING", "Already tracking. Ignoring start request.")
            self._emit_tracking_status(True, target_identity_id=self.target_identity_id)
            return True

        iid = None if self.target_identity_id is None else int(self.target_identity_id)
        bbox = self.target_bbox

        # 1) identity 우선
        if iid is not None:
            if self.inference_engine.enter_thief_mode(iid):
                self.is_tracking = True
                self._spawn_tracking_thread()
                self._emit_tracking_status(True, target_identity_id=iid)
                self.log("SUCCESS", f"🎯 Started tracking: ID {iid} ({self.target_class})")
                return True
            self.log("WARNING", f"enter_thief_mode failed for ID {iid}; trying bbox fallback...")

        # 2) bbox 폴백
        if bbox is not None and self.inference_engine.lock_by_bbox(bbox):
            self.is_tracking = True
            self._spawn_tracking_thread()
            self._emit_tracking_status(True, target_identity_id=None)
            self.log("SUCCESS", "🎯 Started tracking by bbox-lock (ID pending)")
            return True

        # 3) 실패
        self._emit_tracking_status(False, message="lock_by_identity and bbox fallback both failed")
        self.log("ERROR", "Failed to start tracking")
        return False

    def _emit_tracking_status(self, is_on, target_identity_id=None, message=None):
        """프론트로 추적 상태 송신 (routes.py에서 socketio.emit 쓰는 콜백을 주입해도 됨)"""
        try:
            if hasattr(self, "socketio"):
                self.socketio.emit('tracking_status', {
                    'is_tracking': bool(is_on),
                    'target_identity_id': target_identity_id,
                    'class': getattr(self, 'target_class', None),
                    'message': message,
                })
        except Exception:
            pass
    

    def get_current_frame_jpeg(self):
        """현재 프레임을 JPEG로 반환 (BGR 그대로 인코딩)"""
        frame = None
        with self.lock:
            if self.current_frame is not None and self.current_frame_updated:
                frame = self.current_frame  # copy() 불필요: 바로 imencode 하고 끝
                self.current_frame_updated = False
        if frame is None:
            return None

        try:
            # >>> 색 변환 금지! (OpenCV는 BGR 그대로 JPEG 인코딩)
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                with self.lock:
                    self.current_frame_updated = True
                return None
            return buf.tobytes()
        except Exception as e:
            with self.lock:
                self.current_frame_updated = True
            self.log("ERROR", f"JPEG encode failed: {e}")
            return None
    
    
    def execute_command(self, command):
        """드론 명령 실행"""
        if not self.is_connected or not self.tello:
            return {'success': False, 'message': 'Not connected to Tello'}
        
        try:
            if command == 'takeoff':
                self.log("INFO", "🚁 Taking off...")
                self.tello.takeoff()
                self.last_takeoff_time = time.time()  # 이륙 시간 기록
                self.log("SUCCESS", f"Takeoff successful - stabilizing for {self.takeoff_stabilization_time}s")
                
                # 안정화 후 바닥 근접이면 alt_min까지 끌어올림
                applied_raise = False
                try:
                    h = self.tello.get_distance_tof()
                    if isinstance(h, (int, float)) and h > 0 and h < self.alt_min_cm:
                        delta = int(self.alt_min_cm - h)
                        self.log("INFO", f"[ALT] raise to min: {h:.0f}→{self.alt_min_cm}cm (+{delta}cm)")
                        self.tello.move_up(max(20, min(60, delta)))  # 20~60cm 사이로 안전 상승
                        applied_raise = True
                except Exception:
                    pass
                
                if not applied_raise:
                    self.tello.move_up(20)
                time.sleep(self.takeoff_stabilization_time)
                return {'success': True, 'message': 'Takeoff successful'}
                
            elif command == 'land':
                self.log("INFO", "🛬 Landing...")
                self.tello.land()
                self.last_takeoff_time = None  # 착륙 시 초기화
                time.sleep(2)
                self.log("SUCCESS", "Landing successful")
                return {'success': True, 'message': 'Landing successful'}
                
            elif command == 'emergency':
                self.log("WARNING", "🚨 Emergency stop!")
                self.tello.emergency()
                self.last_takeoff_time = None  # 비상 정지 시 초기화
                return {'success': True, 'message': 'Emergency stop'}
            
            elif command == 'up':
                self.tello.move_up(30)
            elif command == 'down':
                self.tello.move_down(30)
            elif command == 'left':
                self.tello.move_left(30)
            elif command == 'right':
                self.tello.move_right(30)
            elif command == 'forward':
                self.tello.move_forward(30)
            elif command == 'back':
                self.tello.move_back(30)
            elif command == 'cw':
                self.tello.rotate_clockwise(30)
            elif command == 'ccw':
                self.tello.rotate_counter_clockwise(30)
            else:
                return {'success': False, 'message': f'Unknown command: {command}'}
            
            time.sleep(1.0)
            self.log("DEBUG", f"Command {command} completed")
            return {'success': True, 'message': f'Command {command} executed'}
        
        except Exception as e:
            self.log("ERROR", f"Command execution error: {e}")
            return {'success': False, 'message': str(e)}
    

    def cleanup(self):
        """리소스 정리"""
        self.is_logging = False
        if self.inference_engine:
            self.inference_engine.close()


    def stop_tracking(self):
        """자동 추적 중지 → 일반 모드로 복귀"""
        if not self.is_tracking:
            self._emit_tracking_status(False, message="Already stopped")
            return

        # 도둑 모드 해제(있으면)
        try:
            if hasattr(self.inference_engine, "exit_thief_mode"):
                self.inference_engine.exit_thief_mode()
        except Exception as e:
            self.log("WARNING", f"exit_thief_mode error: {e}")

        # 상태 초기화
        self.is_tracking = False
        self.target_identity_id = None
        self.target_bbox = None
        self.target_class = None

        # 드론 정지 (안전)
        try:
            if self.tello:
                self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass

        # 프론트 알림
        self._emit_tracking_status(False, message="Back to normal mode")
        self.log("INFO", "Stopped tracking and returned to normal mode.")
