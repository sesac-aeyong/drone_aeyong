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
        self.current_detections = []
        self.target_class = None
        self.target_identity_id = None
        self.target_bbox = None  # Store in [x1, y1, x2, y2] format
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()

        # 이륙 안정화 시간
        self.last_takeoff_time = None
        self.takeoff_stabilization_time = 3.0  # 이륙 후 3초간 대기

        # RC 명령 설정
        self.use_rc_for_tracking = True
        self.tracking_rc_speed = 30
        
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
        
        # --- undistort state ---
        self._ud_size = None  # (w, h)
        self._ud_map1 = None
        self._ud_map2 = None
        self._crop_roi = None  # (x1, y1, x2, y2)
        self._ud_initialized = False
        self.show_calib_debug = getattr(S, "SHOW_CALIB_DEBUG", True)

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

    # === performance constants (class-level) =================================
    # 제어 루프/필터 상수: 매 프레임 재할당 방지
    LOOP_HZ                = 30.0
    DT                     = 1.0 / LOOP_HZ
    EMA_ALPHA_ERR          = 0.35
    EMA_ALPHA_VEL          = 0.50
    EMA_ALPHA_SIZE         = 0.30
    EMA_ALPHA_RATIO        = 0.40

    YAW_DEADBAND           = 0.06
    LR_DEADBAND            = 0.02
    UD_DEADBAND            = 0.02
    SIZE_DEADBAND          = 0.02

    K_YAW                  = 130.0
    K_LR                   = 100.0
    K_UD                   = 110.0
    K_FB_P                 = 250.0
    K_FB_I                 = 25.0

    SLEW_RC_STEP           = 18
    COAST_DECAY            = 0.85
    COAST_MAX_TIME         = 1.0

    TARGET_RATIO           = 0.40
    NEAR_RATIO             = 0.35
    PUSHBACK_RATIO         = 0.45
    HARD_STOP_RATIO        = 0.50
    SAFE_FB_FWD_CAP        = 20
    SAFE_FB_BWD_CAP        = 20
    FB_ACCEL_STEP_FWD      = 8
    FB_ACCEL_STEP_BWD      = 12

    EDGE_FRAC              = 0.06
    EDGE_BOOST             = 0.6

    SEARCH_YAW_SPEED       = 40
    SEARCH_UD_SPEED        = 28
    SEARCH_FB_SPEED        = 0

    OCCLUDED_GRACE_S       = 3.0
    OCC_FWD_MAX_S          = 2.5
    OCC_CENTER_BAND        = 0.25
    RATIO_GOAL_OCCLUDED    = 0.30
    SWEEP_HALF_PERIOD_S    = 1.2

    MIN_BATT               = 10
    
    RAPID_ENLARGE_WARN     = 0.25   # ratio가 초당 이 값 이상 증가하면 전진 금지
    RAPID_ENLARGE_PANIC    = 0.35   # 이 값 이상이면 즉시 소폭 후퇴
    
    UNDISTORT_ALPHA = getattr(S, "UNDISTORT_ALPHA", 1.0)  # 1.0=FOV 최대(테두리 O), 0.0=자동 크롭(테두리 X)
    DEPTH_USE_CROP  = getattr(S, "DEPTH_USE_CROP", True)  # depth에는 crop 적용
    DEPTH_IN_SIZE   = getattr(S, "DEPTH_IN_SIZE", (384, 256))  # (W,H) scdepth 등 입력

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
        s = self._ema_state
        prev = s.get(name, value)
        s[name] = (1 - alpha) * prev + alpha * value
        return s[name]

    def _get_altitude_cm(self):
        # 지상 튜닝 모드면 가상 고도
        if self.ground_tune_mode and not self._airborne:
            return float(self.virtual_height_cm)
        try:
            h = self.tello.get_distance_tof()
            if isinstance(h, (int, float)) and 0 < h < 1000:
                return float(h)
        except:
            pass
        return None

    def _enforce_altitude_limits(self, ud_cmd):
        h_cm = self._get_altitude_cm()
        if h_cm is None:
            return int(np.clip(ud_cmd, -10, +10))
        # ceiling hard
        if h_cm >= self.alt_max_cm:
            return min(0, -10)
        # ceiling soft
        if h_cm >= self.alt_max_cm - self.alt_guard_cm and ud_cmd > 0:
            ud_cmd = 0
        # floor hard
        if h_cm <= self.alt_min_cm:
            return max(ud_cmd, +10)
        # floor soft
        if h_cm <= self.alt_min_cm + self.alt_guard_cm and ud_cmd < 0:
            ud_cmd = 0
        return ud_cmd

    def _apply_slew_and_send(self, yaw_cmd, lr_cmd, ud_cmd, fb_cmd):
        self._cmd_yaw = int(self._slew(self._cmd_yaw, yaw_cmd, self.SLEW_RC_STEP))
        self._cmd_lr  = int(self._slew(self._cmd_lr,  lr_cmd,  self.SLEW_RC_STEP))
        self._cmd_ud  = int(self._slew(self._cmd_ud,  ud_cmd,  self.SLEW_RC_STEP))
        self._cmd_fb  = int(self._slew(self._cmd_fb,  fb_cmd,  self.SLEW_RC_STEP))
        if self.use_rc_for_tracking:
            self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)

    def _select_best_thief_detection(self, detections):
        # thief_dist <= thief_cos_dist (gate) 이면서 최소값
        best = None
        best_td = 1e9
        for d in detections:
            get = d.get if isinstance(d, dict) else (lambda k, default=None: getattr(d, k, default))
            td = get("thief_dist"); gate = get("thief_cos_dist")
            if td is None or gate is None or td > gate: 
                continue
            if td < best_td:
                best = d; best_td = td
        return best

    def _throttle(self, name, interval_s):
        now = time.time()
        tmap = getattr(self, "_throttle_map", None)
        if tmap is None:
            self._throttle_map = {}
            tmap = self._throttle_map
        last = tmap.get(name, 0.0)
        if now - last >= interval_s:
            tmap[name] = now
            return True
        return False

    def _init_tracker_state(self):
        """트래킹 상태 변수 초기화"""
        self._cmd_lr = self._cmd_fb = self._cmd_ud = self._cmd_yaw = 0
        self._last_bbox = None
        self._last_seen_t = None
        self._lost_since_t = None
        self._integral_fb = 0.0
        self._integral_clip = 50.0
        self._lost_strategy = None
        self._lost_t0 = None
        self._last_ratio = None
        self._last_center_norm = None
        self._ema_state = {}
        self.ground_tune_mode = getattr(S, "GROUND_TUNE_MODE", False)
        self.virtual_height_cm = getattr(S, "VIRTUAL_HEIGHT_CM", 80)
        self._airborne = False
        
    def _init_undistort_if_needed(self, frame_shape_hw):
        """첫 유효 프레임 크기(H,W)로 remap 맵과 5% 크롭 ROI를 1회 준비"""
        if self._ud_initialized:
            return
        h, w = frame_shape_hw
        if w < 640 or h < 360:
            return
        img_size = (w, h)

        # ⚙️ alpha를 설정 가능하게: 표시/탐지는 alpha=1.0(=FOV 보존), depth는 validROI 사용
        newK, validROI = cv2.getOptimalNewCameraMatrix(
            S.CAMERA_MATRIX, S.DIST_COEFFS, img_size,
            self.UNDISTORT_ALPHA, img_size, centerPrincipalPoint=True
        )
        self._ud_map1, self._ud_map2 = cv2.initUndistortRectifyMap(
            S.CAMERA_MATRIX, S.DIST_COEFFS, None, newK, img_size, cv2.CV_32FC1
        )

        # 표시/탐지용 기본 ROI는 전체 —> depth 전용 ROI로 validROI 따로 보관
        self._crop_roi = (0, 0, w, h)
        # validROI는 (x, y, w, h) 형태
        self._valid_roi_xywh = validROI  # depth 전용으로 사용 예정

        self._ud_initialized = True
        self._ud_size = (w, h)
        self.log("INFO", f"[CALIB] UD ready → proc {w}x{h}, validROI={validROI}, alpha={self.UNDISTORT_ALPHA}")

    def _undistort_and_crop(self, bgr_frame):
        """미리 계산된 맵으로 빠르게 보정 (크롭 없음, 검은 테두리 허용)"""
        undist = cv2.remap(
            bgr_frame, self._ud_map1, self._ud_map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,  # 검은 테두리
            borderValue=(0, 0, 0)
        )
        # 전체 ROI (0,0,w,h)라 slicing 영향 없음
        return undist

    def _make_depth_input(self, undistorted_bgr):
        """
        depth 전용 입력 프레임 생성:
        - DEPTH_USE_CROP=True면 validROI로 잘라 검은 테두리 제거
        - DEPTH_IN_SIZE로 리사이즈 (W,H)
        """
        src = undistorted_bgr
        if self.DEPTH_USE_CROP and hasattr(self, "_valid_roi_xywh") and self._valid_roi_xywh is not None:
            x, y, w, h = self._valid_roi_xywh
            # validROI가 너무 작거나 비정상일 때 안전장치
            if w > 0 and h > 0 and (x+w) <= src.shape[1] and (y+h) <= src.shape[0]:
                src = src[y:y+h, x:x+w]

        W, H = self.DEPTH_IN_SIZE
        if (src.shape[1], src.shape[0]) != (W, H):
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
        return src

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
        self.log("INFO", "🎯 IBVS tracking thread started")
        self._init_tracker_state()

        # 이륙 안정화
        if self.last_takeoff_time is not None:
            while time.time() - self.last_takeoff_time < self.takeoff_stabilization_time:
                self.tello.send_rc_control(0,0,0,0); time.sleep(0.1)
            self.last_takeoff_time = None
            self.log("SUCCESS", "✅ Stabilization complete - starting IBVS tracking")

        while self.is_tracking:
            t0 = time.time()
            try:
                # 배터리 가드(스로틀 無: 프레임 임계)
                if self.tello and isinstance(self.battery, (int,float)) and self.battery <= self.MIN_BATT:
                    self.tello.send_rc_control(0,0,0,0); time.sleep(0.2); continue

                with self.lock:
                    bbox = self.target_bbox
                    frm  = self.current_frame

                if frm is None:
                    time.sleep(self.DT); continue

                H,W = frm.shape[:2]; cx, cy = 0.5*W, 0.5*H

                if bbox is not None:
                    # --- 관측 유효 ---
                    x1,y1,x2,y2 = bbox
                    bx,by = 0.5*(x1+x2), 0.5*(y1+y2)
                    bw,bh = max(1,x2-x1), max(1,y2-y1)
                    ratio  = (bw*bh)/(W*H)

                    ex = (bx - cx)/W
                    ey = (by - cy)/H
                    es = np.log(max(1e-6,self.TARGET_RATIO)/max(1e-6,ratio))

                    ex_f = self._ema("ex", ex, self.EMA_ALPHA_ERR)
                    ey_f = self._ema("ey", ey, self.EMA_ALPHA_ERR)
                    es_f = self._ema("esz", es, self.EMA_ALPHA_SIZE)

                    if self._last_bbox is not None and self._last_seen_t is not None:
                        dtv = max(1e-3, t0 - self._last_seen_t)
                        last_cx = 0.5*(self._last_bbox[0]+self._last_bbox[2])
                        last_cy = 0.5*(self._last_bbox[1]+self._last_bbox[3])
                        vx = ((bx-last_cx)/W)/dtv; vy = ((by-last_cy)/H)/dtv
                    else:
                        vx=vy=0.0
                    vx_f = self._ema("vx", vx, self.EMA_ALPHA_VEL)
                    vy_f = self._ema("vy", vy, self.EMA_ALPHA_VEL)

                    edge_x = (-self.EDGE_BOOST if bx < W*self.EDGE_FRAC else
                            self.EDGE_BOOST  if bx > W*(1-self.EDGE_FRAC) else 0.0)
                    edge_y = (-self.EDGE_BOOST if by < H*self.EDGE_FRAC else
                            self.EDGE_BOOST  if by > H*(1-self.EDGE_FRAC) else 0.0)

                    # yaw/lr
                    if abs(ex_f) > self.YAW_DEADBAND:
                        yaw_cmd = int(np.clip(self.K_YAW*ex_f, -self.tracking_rc_speed, self.tracking_rc_speed)); lr_cmd=0
                    else:
                        yaw_cmd = 0
                        lr_cmd  = int(np.clip(self.K_LR*(ex_f + 0.35*vx_f + edge_x), -self.tracking_rc_speed, self.tracking_rc_speed))
                        if abs(lr_cmd) < int(self.tracking_rc_speed*0.1): lr_cmd = 0

                    # ud (alt limit 적용은 나중에)
                    ud_cmd = int(np.clip(-self.K_UD*(ey_f + 0.25*vy_f + edge_y), -self.tracking_rc_speed, self.tracking_rc_speed)) \
                            if abs(ey_f) > self.UD_DEADBAND else 0

                    # fb (PI)
                    if abs(es_f) > self.SIZE_DEADBAND:
                        self._integral_fb = float(np.clip(self._integral_fb + es_f*self.DT*self.K_FB_I,
                                                        -self._integral_clip, self._integral_clip))
                        fb_cmd = int(np.clip(self.K_FB_P*es_f + self._integral_fb,
                                            -self.tracking_rc_speed, self.tracking_rc_speed))
                        if fb_cmd < 0: fb_cmd = int(0.4*fb_cmd)
                    else:
                        self._integral_fb *= 0.98
                        fb_cmd = 0

                    # ratio dynamics
                    ratio_f = self._ema("ratio", ratio, self.EMA_ALPHA_RATIO)
                    if not hasattr(self, "_ratio_prev"):
                        self._ratio_prev, self._ratio_prev_t = ratio_f, t0
                    dt_r = max(1e-3, t0 - getattr(self, "_ratio_prev_t", t0))
                    drdt = (ratio_f - getattr(self, "_ratio_prev", ratio_f))/dt_r
                    self._ratio_prev, self._ratio_prev_t = ratio_f, t0

                    # ESCAPE trigger
                    if (self._escape_mode is None) and (ratio_f >= self.PUSHBACK_RATIO or drdt >= self.RAPID_ENLARGE_PANIC):
                        ceiling_soft = self.alt_max_cm - self.alt_guard_cm
                        self._escape_target_alt = min(self.ESCAPE_ALT_CM, ceiling_soft)
                        try:
                            self._escape_origin_alt = float(self.height) if self.height and self.height > 0 else None
                        except:
                            self._escape_origin_alt = None
                        self._escape_mode = 'UP'; self._escape_t0 = time.time()
                        self.log("WARNING", f"[ESCAPE] Triggered → UP to ~{self._escape_target_alt:.0f} cm")

                    # 근접/속증가 보호 + caps
                    if ratio_f >= self.HARD_STOP_RATIO:
                        fb_cmd = min(0, -15)
                    elif ratio_f >= self.PUSHBACK_RATIO:
                        fb_cmd = min(0, -10)
                    elif ratio_f >= self.NEAR_RATIO and fb_cmd > 0:
                        fb_cmd = 0
                    if drdt >= self.RAPID_ENLARGE_PANIC:
                        fb_cmd = min(fb_cmd, -12)
                    elif drdt >= self.RAPID_ENLARGE_WARN and fb_cmd > 0:
                        fb_cmd = 0

                    fb_cmd = min(fb_cmd, self.SAFE_FB_FWD_CAP) if fb_cmd > 0 else max(fb_cmd, -self.SAFE_FB_BWD_CAP)
                    # accel limit
                    fb_cmd = (min(self._cmd_fb + self.FB_ACCEL_STEP_FWD, fb_cmd) if fb_cmd > self._cmd_fb
                            else max(self._cmd_fb - self.FB_ACCEL_STEP_BWD, fb_cmd))

                    # ALT limits (여기서만 한번)
                    ud_cmd = self._enforce_altitude_limits(ud_cmd)

                    # ESCAPE state machine (활성시 IBVS 덮어씀)
                    if self._escape_mode is not None:
                        h_cm = None
                        try:
                            if isinstance(self.height,(int,float)) and self.height>0: h_cm=float(self.height)
                        except: pass
                        esc_lr=esc_fb=esc_yaw=0; esc_ud=0
                        mode=self._escape_mode; now=time.time()
                        if mode=='UP':
                            target=self._escape_target_alt
                            if (h_cm is not None) and (h_cm < target - self.alt_guard_cm):
                                esc_ud=+min(20, int(self.tracking_rc_speed))
                            else:
                                self._escape_mode='HOLD'; self._escape_t0=now
                        elif mode=='HOLD':
                            if (now - self._escape_t0) >= self.ESCAPE_HOLD_S:
                                self._escape_mode='DOWN'; self._escape_t0=now
                        elif mode=='DOWN':
                            fallback = self.alt_min_cm + max(40, self.alt_guard_cm)
                            target_down = self._escape_origin_alt if self._escape_origin_alt is not None else fallback
                            if (h_cm is not None) and (h_cm > target_down + self.alt_guard_cm):
                                esc_ud = -min(18, int(self.tracking_rc_speed))
                            else:
                                self._escape_mode = None
                                self._escape_origin_alt = self._escape_target_alt = self._escape_t0 = None
                        self._apply_slew_and_send(esc_yaw, esc_lr, esc_ud, esc_fb)
                        time.sleep(max(0.0, self.DT - (time.time()-t0))); continue

                    # 정상 IBVS 적용
                    self._apply_slew_and_send(yaw_cmd, lr_cmd, ud_cmd, fb_cmd)

                    # 기록
                    self._last_bbox = bbox; self._last_seen_t = t0; self._lost_since_t = None
                    self._last_ratio = ratio_f; self._last_center_norm = (bx/W, by/H)
                    if self._lost_strategy is not None and self._last_ratio is not None and self._last_ratio >= self.RATIO_GOAL_OCCLUDED:
                        self._lost_strategy = None; self._lost_t0 = None

                else:
                    # --- 분실/가림 ---
                    now = t0
                    last_seen_ago = 1e9 if self._last_seen_t is None else (now - self._last_seen_t)

                    in_center = False
                    if self._last_center_norm is not None:
                        lx,ly = self._last_center_norm
                        in_center = (abs(lx-0.5)<=self.OCC_CENTER_BAND) and (abs(ly-0.5)<=self.OCC_CENTER_BAND)
                    last_small = (self._last_ratio is None) or (self._last_ratio < self.RATIO_GOAL_OCCLUDED)

                    if (last_seen_ago >= self.OCCLUDED_GRACE_S) and in_center and last_small:
                        if self._lost_strategy is None:
                            self._lost_strategy='FWD'; self._lost_t0=now
                            self.log("WARNING", "[OCC] Occlusion → FWD then SWEEP")
                        if self._lost_strategy=='FWD':
                            self._apply_slew_and_send(0,0,0, min(self.SAFE_FB_FWD_CAP, 18))
                            if (now - self._lost_t0) >= self.OCC_FWD_MAX_S:
                                self._lost_strategy='SWEEP'; self._lost_t0=now
                            time.sleep(max(0.0, self.DT - (time.time()-t0))); continue
                        elif self._lost_strategy=='SWEEP':
                            phase = int((now - self._lost_t0)/self.SWEEP_HALF_PERIOD_S)
                            yaw_dir = -1 if (phase % 2 == 0) else 1
                            self._apply_slew_and_send(int(np.clip(yaw_dir*self.SEARCH_YAW_SPEED,-self.tracking_rc_speed,self.tracking_rc_speed)),
                                                    0,0,0)
                            time.sleep(max(0.0, self.DT - (time.time()-t0))); continue

                    if last_seen_ago <= self.COAST_MAX_TIME:
                        self._cmd_lr  = int(self._cmd_lr  * self.COAST_DECAY)
                        self._cmd_fb  = int(self._cmd_fb  * self.COAST_DECAY)
                        self._cmd_ud  = int(self._cmd_ud  * self.COAST_DECAY)
                        self._cmd_yaw = int(self._cmd_yaw * self.COAST_DECAY)
                        if self.use_rc_for_tracking:
                            self.tello.send_rc_control(self._cmd_lr, self._cmd_fb, self._cmd_ud, self._cmd_yaw)
                    else:
                        ex_f = self._ema_state.get("ex", 0.0)
                        ey_f = self._ema_state.get("ey", 0.0)
                        vx_f = self._ema_state.get("vx", 0.0)
                        vy_f = self._ema_state.get("vy", 0.0)
                        yaw_cmd = int(np.clip(np.sign(ex_f if abs(ex_f) > self.YAW_DEADBAND else vx_f)*self.SEARCH_YAW_SPEED,
                                            -self.tracking_rc_speed, self.tracking_rc_speed))
                        ud_cmd  = int(np.clip(-np.sign(ey_f if abs(ey_f) > self.UD_DEADBAND else vy_f)*self.SEARCH_UD_SPEED,
                                            -self.tracking_rc_speed, self.tracking_rc_speed))
                        self._apply_slew_and_send(yaw_cmd, 0, ud_cmd, self.SEARCH_FB_SPEED)

                    if self._lost_since_t is None:
                        self._lost_since_t = now
                        self.log("WARNING", "⚠️ Target lost - entering search mode")

                # 주기 정렬
                time.sleep(max(0.0, self.DT - (time.time()-t0)))

            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                try: self.tello.send_rc_control(0,0,0,0)
                except: pass
                time.sleep(0.2)

        # 종료 안전정지
        try: self.tello.send_rc_control(0,0,0,0)
        except: pass
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
                        self.is_streaming = False
                        self.socketio.emit('stream_error', {'message': 'Video stream lost. Please reconnect.'})
                        break
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                if self._throttle("raw_size_log", 1.0):
                    self.log("DEBUG", f"[RAW] {w}x{h} (UD init:{self._ud_initialized}, ud_size:{self._ud_size})")
                if (not self._ud_initialized) or (self._ud_size != (w, h)):
                    # 사이즈가 바뀌었거나 아직 초기화 안됨 → 다시 준비
                    self._ud_initialized = False
                    self._init_undistort_if_needed((h, w))
                # 이후에만 undistort 적용
                if self._ud_initialized:
                    frame = self._undistort_and_crop(frame)

                frame = self._make_depth_input(frame)
                
                # 추론 (BGR 입력 그대로)
                detections, depth_map, *_ = self.inference_engine.run(frame)

                # 트래킹 타겟 갱신 (잠금일관성)
                with self.lock:
                    self.current_detections = detections
                    if self.is_tracking:
                        best = self._select_best_thief_detection(detections)
                        if best is not None:
                            self.target_bbox  = best["bbox"] if isinstance(best, dict) else best.bbox
                            self.target_class = (best.get("class", "person") if isinstance(best, dict)
                                                else getattr(best, "cls", "person"))
                        else:
                            if self.target_bbox is not None:
                                self.log("WARNING", "⚠️ Thief not found under gate; holding position")
                            self.target_bbox = None

                # 오버레이는 비용이 크므로 한 번만 변환 → 표시 경로만 RGB
                disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.show_calib_debug:
                    h, w = disp.shape[:2]
                    color = (0,255,0) if self._ud_initialized else (0,255,255)
                    cv2.rectangle(disp, (0,0), (w-1,h-1), color, 1)
                    cv2.putText(disp, f"{'UD ONLY'} {w}x{h}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    for px,py in [(3,3),(w-4,3),(3,h-4),(w-4,h-4)]:
                        cv2.circle(disp, (px,py), 2, color, -1)

                disp = draw_detections_on_frame(disp, detections)

                # 십자선
                h, w = disp.shape[:2]
                cx, cy = w//2, h//2
                cv2.line(disp, (cx-30, cy), (cx+30, cy), (255,255,255), 2)
                cv2.line(disp, (cx, cy-30), (cx, cy+30), (255,255,255), 2)
                cv2.circle(disp, (cx, cy), 5, (255,255,255), -1)

                # 배터리/높이 5Hz 스로틀
                if self._throttle("poll_state", 0.2):
                    try:
                        old_batt = self.battery
                        self.battery = self.tello.get_battery()
                        if self.ground_tune_mode and not self._airborne:
                            self.height = self.virtual_height_cm
                        else:
                            self.height = self.tello.get_distance_tof()
                        if self.battery < 15 <= old_batt:
                            self.log("WARNING", f"⚠️ Critical battery: {self.battery}% - Land soon!")
                        elif self.battery < 25 <= old_batt:
                            self.log("WARNING", f"⚠️ Low battery: {self.battery}%")
                    except:
                        pass

                # 프레임 저장 (표시 프레임만 저장 → 송출/트래커가 공유)
                with self.lock:
                    self.current_frame = disp
                    self.current_frame_updated = True

                # UI 업데이트 10Hz 스로틀 (소켓 부하 절감)
                if self._throttle("emit_ui", 0.1):
                    self.socketio.emit('detections_update', {
                        'detections': detections,
                        'battery': self.battery,
                        'height': self.height,
                        'is_tracking': self.is_tracking,
                        'target_identity_id': self.target_identity_id,
                        'target_class': self.target_class
                    })

            except Exception as e:
                traceback.print_exc()
                error_count += 1
                if error_count >= max_errors:
                    self.is_streaming = False
                    break
                time.sleep(0.05)
                
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
                self._airborne = True
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
                self._airborne = False
                self.log("INFO", "🛬 Landing...")
                self.tello.land()
                self.last_takeoff_time = None  # 착륙 시 초기화
                time.sleep(2)
                self.log("SUCCESS", "Landing successful")
                return {'success': True, 'message': 'Landing successful'}
                
            elif command == 'emergency':
                self._airborne = False
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
