# tello_web_server.py
import traceback
import cv2
from djitellopy import Tello
import threading
import time
from datetime import datetime
import queue
from hailorun import HailoRun
from yolo_tools import draw_detections_on_frame
from .app_tools import connect_to_tello_wifi

# 제어/유틸 전부 모듈에서 가져오기
from .control_fusion import (
    ControlFusion, clip_bbox_to_frame,
    compute_flow_from_spine_strip, update_pose_stats,
    spine_depth_mode_and_brake, SearchManager, SearchParams,
    want_depth, want_pose, want_flow, feature_alpha, prepare_pose_for_rc,
    depth_to_vis, overlay_flow_arrow, overlay_pose_points_min9
)


class TelloWebServer:
    def __init__(self, socketio):
        self.tello = None
        self.socketio = socketio
        self.is_streaming = False
        self.is_connected = False
        self.current_frame = None
        self.current_frame_updated = False
        self.current_depth_map = None
        self.current_detections = []
        self.target_class = None
        self.target_identity_id = None
        self.target_bbox = None  # [x1,y1,x2,y2]
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()

        # 이륙 안정화 시간
        self.last_takeoff_time = None
        self.takeoff_stabilization_time = 3.0

        # RC 설정
        self.use_rc_for_tracking = True
        self.tracking_rc_speed = 30

        # 웹 로그
        self.log_queue = queue.Queue(maxsize=100)
        self.log_thread = None
        self.is_logging = True
        self.start_log_broadcaster()

        # 추론 엔진
        self.log("INFO", "Loading inference engine...")
        try:
            self.inference_engine = HailoRun(feature_state_getter=self.get_feature_state) # 프론트 토글 상태 제공 콜백 연결
            self.inference_engine.load()
            self.log("SUCCESS", "✅ Inference engine loaded successfully")
        except Exception as e:
            self.log("ERROR", f"❌ Failed to load inference engine: {e}")
            traceback.print_exc()
            self.inference_engine = None

        # 토글
        self.USE_POSE       = True
        self.USE_FLOW       = True
        self.USE_DEPTH_VIEW = True
        self.USE_OBS_BRAKE  = True

        # 포즈 상태
        self.pose_quality = 0.0
        self.pose_should_ref = None; self.pose_should_ema = None
        self.pose_spine_ref  = None; self.pose_spine_ema  = None

        # 플로우 상태
        self.prev_gray = None
        self.last_flow_vec = (0.0, 0.0)   # (vx,vy) px/frame

        # 장애물 브레이크/표시
        self._obstacle_brake = False
        self._last_depth_mode_spine = None

        # 포즈 중심/스케일(EMA) 공유 상태
        self._hip_mid = None
        self._sh_mid = None
        self._pelvis_ema = None

        # 제어 융합기 & 유실 검색 매니저
        self.fuser = ControlFusion(tracking_rc_speed=self.tracking_rc_speed)
        self.search = SearchManager(params=SearchParams())
        
        # 웹 토글(라우트에서 set_features로 갱신) - OFF이면 연산/오버레이/제어 반영 모두 중지
        self.features = {'depth': False, 'pose':  False, 'flow':  False, 
                         'alpha': 0.5,}
        self._miss_cnt = 0
        self._miss_hold = 3  # 연속 3프레임 미스 시에만 해제
        
    # 프론트 토글 getter (스레드 안전)
    def get_feature_state(self):
        with self.lock:
            return dict(self.features)

    # ──────────────────────────────────────────────────────────────────────
    # 로깅
    # ──────────────────────────────────────────────────────────────────────
    def log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {'timestamp': timestamp,'level': level,'message': message}
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
        try:
            if self.log_queue.full():
                try: self.log_queue.get_nowait()
                except queue.Empty: pass
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            pass

    def start_log_broadcaster(self):
        def broadcast_logs():
            while self.is_logging:
                try:
                    log_entry = self.log_queue.get(timeout=0.5)
                    self.socketio.emit('log_message', log_entry)
                except queue.Empty:
                    continue
        self.log_thread = threading.Thread(target=broadcast_logs, daemon=True)
        self.log_thread.start()

    # ──────────────────────────────────────────────────────────────────────
    # 연결
    # ──────────────────────────────────────────────────────────────────────
    def connect_tello(self):
        self.log("INFO", "🔍 Checking Tello WiFi connection...")
        if not connect_to_tello_wifi():
            self.log("ERROR", "Failed to connect to Tello WiFi")
            return False

        self.log("SUCCESS", "Tello WiFi connected")

        # 이전 연결 정리(에러는 조용히 무시)
        if self.tello:
            self.log("INFO", "Cleaning up old connection...")
            self.is_streaming = False
            try:
                fr = getattr(self.tello, 'background_frame_read', None)
                if fr: fr.stop()
                self.tello.streamoff()
                self.tello.end()
            except Exception:
                pass
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
                    self.is_connected = False
                    self.tello = None
                    return False

        self.battery = self.tello.get_battery()
        self.log("SUCCESS", f"Tello connected. Battery: {self.battery}%")

        if self.battery < 20:
            self.log("WARNING", f"⚠️ Low battery: {self.battery}%")

        self.log("INFO", "Starting video stream...")
        try: self.tello.streamoff()
        except Exception: pass
        self.tello.streamon()

        self.log("SUCCESS", "🎥 Stream started successfully")
        self.is_connected = True
        return True

    # ──────────────────────────────────────────────────────────────────────
    # 추적 쓰레드(제어만 담당)
    # ──────────────────────────────────────────────────────────────────────
    def tracking_thread(self):
        target_lost_time = None
        target_lost_warning_sent = False
        self.log("INFO", "🎯 RC tracking (bbox + pose + flow) started")

        while self.is_tracking:
            try:
                # 이륙 안정화
                if self.last_takeoff_time is not None:
                    elapsed = time.time() - self.last_takeoff_time
                    if elapsed < self.takeoff_stabilization_time:
                        remain = self.takeoff_stabilization_time - elapsed
                        if int(remain * 10) % 10 == 0:
                            self.log("INFO", f"⏳ Stabilizing... {remain:.1f}s")
                        time.sleep(0.1)
                        continue
                    else:
                        self.log("SUCCESS", "✅ Stabilization complete - starting tracking")
                        self.last_takeoff_time = None

                with self.lock:
                    feat = dict(self.features)
                use_depth = want_depth(feat)
                use_pose  = want_pose(feat)
                use_flow  = want_flow(feat)
                # 타깃 있음 → 제어
                if self.target_bbox and self.current_frame is not None:
                    # 재획득 시 검색리셋
                    if target_lost_time is not None:
                        self.log("SUCCESS", "🎯 Target re-acquired!")
                        target_lost_time = None
                        target_lost_warning_sent = False
                        self.search.reset()

                    # ▶ 포즈센터/골반폭: video_thread에서 계산한 최신값 사용
                    with self.lock:
                        hip_mid = self._hip_mid
                        sh_mid  = self._sh_mid
                        pelvis_ema = self._pelvis_ema

                    # ▶ 포즈: ref/ema가 있을 때만 보조 반영
                    pose_dict = prepare_pose_for_rc(
                        self.USE_POSE and use_pose,
                        self.pose_quality,
                        self.pose_should_ref, self.pose_should_ema,
                        self.pose_spine_ref,  self.pose_spine_ema,
                        hip_mid=hip_mid, shoulder_mid=sh_mid, pelvis_ema=pelvis_ema
                    )
                    
                    # ▶ 깊이 브레이크
                    with self.lock:
                        obstacle_brake = (
                            self._obstacle_brake
                            if (self.USE_OBS_BRAKE and use_depth) else False
                        )

                    lr, fb, ud, yaw = self.fuser.compute_rc(
                        self.current_frame.shape, self.target_bbox,
                        pose_dict=pose_dict,
                        flow_vec=(self.last_flow_vec if (self.USE_FLOW and use_flow) else None),
                        size_target_range=(0.20, 0.30),
                        obstacle_brake=obstacle_brake,
                        desired_spine_pct=0.28,       # 화면 높이 대비 28%
                        desired_pelvis_pct=0.18,      # 화면 너비 대비 18%
                        pose_center_mode="hip-shoulder-mid",
                    )
                    self.tello.send_rc_control(lr, fb, ud, yaw)

                # 타깃 없음 → 검색
                else:
                    now = time.time()
                    if target_lost_time is None:
                        target_lost_time = now
                        self.tello.send_rc_control(0, 0, 0, 0)
                        # flow 방향 기반 검색 시작
                        self.search.start(self.last_flow_vec[0] if self.last_flow_vec else 0.0, now)

                    # 경고
                    if not target_lost_warning_sent and (now - target_lost_time) > 3:
                        self.log("WARNING", f"⚠️ Target lost for 3 seconds (ID: {self.target_identity_id})")
                        target_lost_warning_sent = True

                    # 검색 커맨드
                    yaw_cmd = self.search.command(now)
                    if yaw_cmd is not None:
                        self.tello.send_rc_control(0, 0, 0, int(yaw_cmd))
                    else:
                        # 짧은 정지
                        self.tello.send_rc_control(0, 0, 0, 0)

                time.sleep(0.05)  # 20Hz

            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                if self.use_rc_for_tracking:
                    self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)

        # 종료 시 정지
        self.tello.send_rc_control(0, 0, 0, 0)
        self.log("INFO", "🛑 Tracking stopped - drone halted")
        self.log("INFO", "🎯 Tracking thread stopped")

    # ──────────────────────────────────────────────────────────────────────
    # 비디오/추론 쓰레드(상태 업데이트만 담당)
    # ──────────────────────────────────────────────────────────────────────
    def video_stream_thread(self):
        print("📹 Starting video stream thread...")
        time.sleep(3)
        frame_reader = self.tello.get_frame_read()
        print("✅ Frame reader initialized")

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
                        self.socketio.emit('stream_error', {'message': 'Video stream lost. Please reconnect.'})
                        break
                    continue
                error_count = 0

                # ── 토글 스냅샷
                with self.lock:
                    feat = dict(self.features)
                use_depth = want_depth(feat)
                use_pose  = want_pose(feat)
                use_flow  = want_flow(feat)

                # ── 추론 (Thief 모드 전제: 0개 또는 1개)
                detections, depth_map, extra = self.inference_engine.run(frame)
                alpha_from_engine = extra[0].get("alpha") if (isinstance(extra, list) and extra and isinstance(extra[0], dict)) else None

                # if self.is_tracking:
                #     self.log("DEBUG", f"[THIEF] detections_len={len(detections)} (expected 0 or 1)")

                det = (detections[0] if (self.is_tracking and len(detections) == 1) else None)

                # ── 타깃 bbox/클래스/포즈 스냅샷
                target_bbox = None
                target_class = None
                pose_obj = None
                if det is not None:
                    if isinstance(det, dict):
                        target_bbox = det.get("bbox")
                        target_class = det.get("class", "person")
                        if use_pose:  # 🔸 pose 토글 ON일 때만 읽음
                            pose_obj = det.get("pose", None)
                    else:
                        target_bbox = getattr(det, "bbox", None)
                        target_class = getattr(det, "cls", "person")
                        if use_pose:  # 🔸 pose 토글 ON일 때만 읽음
                            pose_obj = getattr(det, "pose", None)

                    if target_bbox is not None:
                        h, w = frame.shape[:2]
                        target_bbox = clip_bbox_to_frame(target_bbox, w, h)

                # ── 포즈 중심(힙/어깨) + 골반폭(EMA) 계산/저장
                if self.USE_POSE and use_pose and (pose_obj is not None):
                    def _ok(kp, thr=60):
                        return kp and len(kp) >= 3 and int(kp[2]) >= thr
                    hip_mid = None; sh_mid = None; pelvis_len = None
                    try:
                        lhip = pose_obj[11]; rhip = pose_obj[12]
                        lsh  = pose_obj[5];  rsh  = pose_obj[6]
                        if _ok(lhip) and _ok(rhip):
                            hip_mid = (
                                0.5*(float(lhip[0])+float(rhip[0])),
                                0.5*(float(lhip[1])+float(rhip[1]))
                            )
                            dx = float(lhip[0]) - float(rhip[0])
                            dy = float(lhip[1]) - float(rhip[1])
                            pelvis_len = (dx*dx + dy*dy) ** 0.5
                        if _ok(lsh) and _ok(rsh):
                            sh_mid = (
                                0.5*(float(lsh[0])+float(rsh[0])),
                                0.5*(float(lsh[1])+float(rsh[1]))
                            )
                    except:
                        pass
                    if pelvis_len is not None:
                        prev = self._pelvis_ema if (self._pelvis_ema is not None) else pelvis_len
                        pelvis_ema = 0.75*prev + 0.25*pelvis_len
                    else:
                        pelvis_ema = self._pelvis_ema
                    with self.lock:
                        self._hip_mid = hip_mid
                        self._sh_mid = sh_mid
                        self._pelvis_ema = pelvis_ema

                # ── 상태 저장(최소 락)
                with self.lock:
                    self.current_detections = detections
                    if target_bbox is not None:
                        self._miss_cnt = 0
                        self.target_bbox = target_bbox
                        self.target_class = target_class
                    else:
                        if self.is_tracking:
                            self._miss_cnt += 1
                            if self._miss_cnt >= self._miss_hold:
                                if self.target_bbox is not None:
                                    self.log("WARNING", "⚠️ Thief not found; holding position")
                                self.target_bbox = None
                        else:
                            self.target_bbox = None

                # ── Pose (토글 ON + bbox 존재 시에만)
                if self.USE_POSE and use_pose and (self.target_bbox is not None):
                    try:
                        (self.pose_quality,
                        self.pose_should_ref, self.pose_should_ema,
                        self.pose_spine_ref,  self.pose_spine_ema) = update_pose_stats(
                            pose_obj, self.pose_quality,
                            self.pose_should_ref, self.pose_should_ema,
                            self.pose_spine_ref,  self.pose_spine_ema,
                            alpha=0.25
                        )
                    except Exception as e:
                        self.log("DEBUG", f"[POSE] update error: {e}")
                        traceback.print_exc()

                # ── Optical Flow (토글 ON + bbox 존재 시에만)
                if self.USE_FLOW and use_flow and (self.target_bbox is not None):
                    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.prev_gray is not None:
                        fv = compute_flow_from_spine_strip(self.prev_gray, gray_now, tuple(self.target_bbox), frame.shape)
                        if fv is not None:
                            with self.lock:
                                self.last_flow_vec = fv
                    with self.lock:
                        self.prev_gray = gray_now
                else:
                    with self.lock:
                        self.prev_gray = None
                        self.last_flow_vec = (0.0, 0.0)

                # ── Depth (토글 ON + bbox + depth_map 존재 시에만)
                depth_mode_local = None
                obstacle_brake_local = False
                if self.USE_DEPTH_VIEW and use_depth and (self.target_bbox is not None) and (depth_map is not None):
                    depth_mode_local, obstacle_brake_local = spine_depth_mode_and_brake(
                        depth_map, tuple(self.target_bbox), use_brake=self.USE_OBS_BRAKE
                    )

                with self.lock:
                    if self.USE_DEPTH_VIEW and use_depth and (self.target_bbox is not None) and (depth_map is not None):
                        self._last_depth_mode_spine = depth_mode_local
                        self._obstacle_brake = obstacle_brake_local
                        self.current_depth_map = depth_map
                    else:
                        self._last_depth_mode_spine = None
                        self._obstacle_brake = False
                        self.current_depth_map = None

                # ── 오버레이
                alpha = alpha_from_engine if isinstance(alpha_from_engine, (int, float)) else feature_alpha(feat, 0.5)
                frame_with_detections = draw_detections_on_frame(frame, detections)

                if use_depth and depth_map is not None and alpha > 0:
                    depth_vis_bgr = depth_to_vis(depth_map)
                    frame_with_detections = cv2.addWeighted(frame_with_detections, 1.0 - alpha, depth_vis_bgr, alpha, 0)

                if use_pose and (self.target_bbox is not None):
                    frame_with_detections = overlay_pose_points_min9(
                        frame_with_detections,
                        self.pose_should_ema, self.pose_spine_ema, alpha,
                        pose_kpts=(pose_obj if (self.target_bbox is not None) else None),
                        conf_thresh=60,
                        draw_indices=False,     # 필요하면 True
                        draw_midpoints=True
                    )

                if use_flow and (self.target_bbox is not None) and (self.last_flow_vec is not None):
                    frame_with_detections = overlay_flow_arrow(
                        frame_with_detections, self.target_bbox, self.last_flow_vec, alpha
                    )

                # ── 중앙 십자 & 텍스트
                h, w = frame_with_detections.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.line(frame_with_detections, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 2)
                cv2.line(frame_with_detections, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 2)
                cv2.circle(frame_with_detections, (cx, cy), 5, (255, 255, 255), -1)

                try:
                    txt = []
                    if use_depth and self._last_depth_mode_spine is not None:
                        txt.append(f"spine-depth(mode): {self._last_depth_mode_spine:.3f}")
                    if self.USE_FLOW and use_flow and self.last_flow_vec is not None:
                        vx, vy = self.last_flow_vec; txt.append(f"flow(vx,vy): ({vx:.1f},{vy:.1f})")
                    if self.USE_POSE and use_pose:
                        txt.append(f"poseQ: {self.pose_quality:.2f}")
                    if use_depth and getattr(self, "_obstacle_brake", False):
                        txt.append("BRAKE")
                    if txt:
                        cv2.putText(frame_with_detections, " | ".join(txt), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                # ── 배터리/고도
                try:
                    old_battery = self.battery
                    self.battery = self.tello.get_battery()
                    self.height = self.tello.get_distance_tof()
                    if self.battery < 15 and old_battery >= 15:
                        self.log("WARNING", f"⚠️ Critical battery: {self.battery}% - Land soon!")
                    elif self.battery < 25 and old_battery >= 25:
                        self.log("WARNING", f"⚠️ Low battery: {self.battery}%")
                except:
                    pass

                # ── 프레임 저장
                with self.lock:
                    self.current_frame = frame_with_detections
                    self.current_frame_updated = True

                # ── UI 업데이트
                self.socketio.emit('detections_update', {
                    'detections': detections,
                    'battery': self.battery,
                    'height': self.height,
                    'is_tracking': self.is_tracking,
                    'target_identity_id': self.target_identity_id,
                    'target_class': self.target_class,
                    'pose_quality': self.pose_quality,
                    'flow_vec': self.last_flow_vec,
                    'spine_depth_mode': getattr(self, "_last_depth_mode_spine", None),
                    'brake': getattr(self, "_obstacle_brake", False),
                })

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

    # ──────────────────────────────────────────────────────────────────────
    # 스트리밍/트래킹 온오프
    # ──────────────────────────────────────────────────────────────────────
    def start_streaming(self):
        if not self.is_streaming and self.is_connected:
            self.is_streaming = True
            thread = threading.Thread(target=self.video_stream_thread, daemon=True)
            thread.start()
            return True
        return False

    def stop_streaming(self):
        self.is_streaming = False

    def start_tracking(self):
        if self.is_tracking:
            self.log("WARNING", "Already tracking. Ignoring start request.")
            return True
        iid = None if self.target_identity_id is None else int(self.target_identity_id)

        if iid is not None and iid > 0:
            if self.inference_engine.enter_thief_mode(iid):
                self._miss_cnt = 0        # ⬅️ 시작 시 리셋
                self.is_tracking = True
                self._spawn_tracking_thread()
                self.log("SUCCESS", f"🎯 Started tracking: ID {iid} ({self.target_class})")
                return True
            self.log("WARNING", f"enter_thief_mode failed for ID {iid}.")

        self.log("ERROR", "Failed to start tracking")
        return False

    def _spawn_tracking_thread(self):
        if getattr(self, "_tracking_thread", None) and self._tracking_thread.is_alive():
            return
        t = threading.Thread(target=self.tracking_thread, daemon=True)
        t.start()
        self._tracking_thread = t

    def _emit_tracking_status(self, is_on, target_identity_id=None, message=None):
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

    # ──────────────────────────────────────────────────────────────────────
    # 프레임/명령/정리
    # ──────────────────────────────────────────────────────────────────────
    def get_current_frame_jpeg(self):
        frame = None
        with self.lock:
            if self.current_frame is not None and self.current_frame_updated:
                frame = self.current_frame
                self.current_frame_updated = False
        if frame is None:
            return None
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            with self.lock:
                self.current_frame_updated = True
            return None
        return buf.tobytes()

    def execute_command(self, command):
        if not self.is_connected or not self.tello:
            return {'success': False, 'message': 'Not connected to Tello'}

        manual_commands = ['up', 'down', 'left', 'right', 'forward', 'back', 'cw', 'ccw']
        if self.is_tracking and command in manual_commands:
            return {'success': False, 'message': 'Manual control disabled during tracking. Stop tracking first.'}

        try:
            if command == 'takeoff':
                self.log("INFO", "🚁 Taking off...")
                self.tello.takeoff()
                self.last_takeoff_time = time.time()
                self.log("SUCCESS", f"Takeoff successful - stabilizing for {self.takeoff_stabilization_time}s")
                time.sleep(self.takeoff_stabilization_time)
                return {'success': True, 'message': 'Takeoff successful'}

            elif command == 'land':
                self.log("INFO", "🛬 Landing...")
                self.tello.land()
                self.last_takeoff_time = None
                time.sleep(2)
                self.log("SUCCESS", "Landing successful")
                return {'success': True, 'message': 'Landing successful'}

            elif command == 'emergency':
                self.log("WARNING", "🚨 Emergency stop!")
                self.tello.emergency()
                self.last_takeoff_time = None
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
        self.is_logging = False
        if self.inference_engine:
            self.inference_engine.close()

    def stop_tracking(self):
        if not self.is_tracking:
            self._emit_tracking_status(False, message="Already stopped")
            return
        try:
            if hasattr(self.inference_engine, "exit_thief_mode"):
                self.inference_engine.exit_thief_mode()
                self._miss_cnt = 0                # ⬅️ 종료 시 리셋
        except Exception as e:
            self.log("WARNING", f"exit_thief_mode error: {e}")

        self.is_tracking = False
        self.target_identity_id = None
        self.target_bbox = None
        self.target_class = None
        # 보조 상태 리셋
        self.prev_gray = None
        self.last_flow_vec = (0.0, 0.0)
        self._obstacle_brake = False
        self._last_depth_mode_spine = None

        if self.tello:
            self.tello.send_rc_control(0, 0, 0, 0)

        self.log("INFO", "Stopped tracking and returned to normal mode.")