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
        self.current_depth_map = None
        self.current_detections = []
        self.target_class = None
        self.target_identity_id = None
        self.target_bbox = None  # Store in [x1, y1, x2, y2] format
        self.target_body_bbox = None
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        self.frame_center = (480, 360)
        self.target_lost_time = None
        self.yaw_started = False

        self.target_bbox = None
        self.last_seen_cx = None        # 마지막 타겟 x, 좌우 사라짐 판단
        self.last_seen_cy = None        # 마지막 타겟 y, 아래로 사라짐 판단

        self.cmd_fb = 0   # 전후
        self.cmd_lr = 0   # 좌우 (사용 안함)
        self.cmd_ud = 0   # 상하 (사용 안함)
        self.cmd_yaw = 0  # 회전

        # 이륙 안정화 시간
        self.last_takeoff_time = None
        self.takeoff_stabilization_time = 3.0  # 이륙 후 3초간 대기

        # RC 명령 설정
        self.use_rc_for_manual = False
        self.use_rc_for_tracking = True
        self.rc_speed = 40
        self.tracking_rc_speed = 30
        self.rc_command_duration = 0.4
        
        # 웹 로그 시스템
        self.log_queue = queue.Queue(maxsize=100)  # 최대 100개 로그 저장
        self.log_thread = None
        self.is_logging = True
        self.start_log_broadcaster()
        
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
                try:
                    self.log_queue.get_nowait()  # 오래된 로그 제거
                except queue.Empty:
                    pass
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
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
    
    
    def get_body_bbox(self, pose, visible_parts):
        """
        visible_parts에 True로 표시된 torso 부위만 사용하여 body bbox 계산.
        confidence 재확인 필요 없음 (이미 visible_parts에서 필터링됨)
        """

        if pose is None or visible_parts is None:
            return None

        # 안전성: visible_parts가 dict인지 보장
        if not isinstance(visible_parts, dict):
            return None

        xs, ys = [], []

        # head keypoints
        if visible_parts.get("head", False):
            for i in [0, 1, 2, 3, 4]:
                if i >= len(pose):
                    continue
                x, y, _ = pose[i]
                xs.append(x)
                ys.append(y)

        # shoulder keypoints
        if visible_parts.get("shoulder", False):
            for i in [5, 6]:
                if i >= len(pose):
                    continue
                x, y, _ = pose[i]
                xs.append(x)
                ys.append(y)

        # hip keypoints
        if visible_parts.get("hip", False):
            for i in [11, 12]:
                if i >= len(pose):
                    continue
                x, y, _ = pose[i]
                xs.append(x)
                ys.append(y)

        if len(xs) < 2 or len(ys) < 2:
            return None

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # 폭이 지나치게 좁을 때 보정
        h = y2 - y1
        w = x2 - x1

        if w < h * 0.3:
            cx = (x1 + x2) / 2
            expand = h * 0.15
            x1 = cx - expand
            x2 = cx + expand

        return [int(x1), int(y1), int(x2), int(y2)]


    def get_target_detection(self, detections):
        """
        기존 thief_dist 기반 타겟 선정 로직을 그대로 사용해서,
        target_det 반환하는 함수로 분리
        """
        target_det = None

        if not detections:
            return None

        for det in detections:
            get = det.get if isinstance(det, dict) else (lambda k, d=None: getattr(det, k, d))

            td = get("thief_dist")
            tg = get("thief_cos_dist")

            if td is None or tg is None:
                continue
            
            if td <= tg:
                if target_det is None:
                    target_det = det
                else:
                    get_curr = target_det.get if isinstance(target_det, dict) else (lambda k, d=None: getattr(target_det, k, d))
                    curr_td = get_curr("thief_dist", 9999999)
                    if td < curr_td:
                        target_det = det

        return target_det
        
    def get_target_area_by_pose(self, visible_parts):
        """
        visible_parts (dict) 기반으로 목표 면적 반환.
        NOTE: 이 함수는 detections/pose를 직접 참조하지 않음.
        """
        # 기본값
        BASE = 26460

        # 안전성: visible_parts 없으면 default
        if visible_parts is None or not isinstance(visible_parts, dict):
            return None

        if not any(visible_parts.values()):
            return None

        # 튜닝 테이블 (head, shoulder, hip)
        TARGET_AREA = {
            (True, True, True):  int(BASE * 1.0),  # head + shoulder + hip
            (True, True, False): int(BASE * 0.3),
            (True, False, False): int(BASE * 0.06),

            (False, True, True): int(BASE * 0.7),
            (False, True, False): None,

            (False, False, True): None,  # hip only
        }

        key = (visible_parts.get("head", False),
            visible_parts.get("shoulder", False),
            visible_parts.get("hip", False))

        return TARGET_AREA.get(key, None)
    
    def get_visible_torso_parts(self, pose, th=25):
        """
        Torso 관련된 keypoints만 confidence 기반으로 visible 여부 반환
        항상 dict 반환: {"head":bool, "shoulder":bool, "hip":bool}
        """
        # 기본 False 딕셔너리
        visible = {"head": False, "shoulder": False, "hip": False}

        if pose is None:
            return visible

        # 안전: pose 길이 확인 (pose는 list/iterable of [x,y,c])
        L = len(pose)

        # Head 영역 (0~4)
        head_idxs = [i for i in [0, 1, 2, 3, 4] if i < L]
        if head_idxs and any(pose[i][2] > th for i in head_idxs):
            visible["head"] = True

        # Shoulder (5,6)
        sh_idxs = [i for i in [5, 6] if i < L]
        if sh_idxs and any(pose[i][2] > th for i in sh_idxs):
            visible["shoulder"] = True

        # Hip (11,12)
        hip_idxs = [i for i in [11, 12] if i < L]
        if hip_idxs and any(pose[i][2] > th for i in hip_idxs):
            visible["hip"] = True

        return visible


    def tracking_thread(self):
        """
        [Torso Pose 기반] 사람 몸통 크기를 기준으로 거리 유지 & Yaw 정렬
        - 팔/다리 동작에 영향받지 않음
        """
        self.log("INFO", "🚀 Tracking Started (Torso-Pose Area Mode)")

        Kp_yaw_normal = 0.6
        Kp_yaw_fast = 1.2
        Kp_area = 0.005   # 전진 게인(면적 오차 기반)

        while self.is_tracking:
            if self.tello is None:
                self.log("WARNING", "🛑 Tello instance is None. Stopping tracking thread.")
                break

            try:
                with self.lock:
                    detections = self.current_detections
                    frame = self.current_frame

                if detections is None or frame is None:
                    if self.tello:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.1)
                    continue

                if not detections:
                    # 타겟을 마지막으로 본 중앙 위치 기반으로 사라진 방향 판단
                    lost_direction_x = None
                    lost_direction_y = None

                    if self.last_seen_cx is not None:
                        norm = self.last_seen_cx
                        if norm < 0.25:
                            lost_direction_x = "left"
                        elif norm > 0.75:
                            lost_direction_x = "right"
                        else:
                            lost_direction_x = "center"   # 중앙에서 사라짐 = 장애물 뒤?
                    else:
                        lost_direction_x = "unknown"

                    if self.last_seen_cy is not None:
                        if self.last_seen_cy > 0.75:
                            lost_direction_y = "down"
                        else:
                            lost_direction_y = "center"

                    if self.target_lost_time is None:
                        # 타겟이 처음 사라진 시점 저장
                        self.target_lost_time = time.time()
                        self.tello.send_rc_control(0, self.cmd_fb//2, 0, self.cmd_yaw//2)
                    else:
                        # 타겟 사라진지 1초 지나면 회전 시작
                        if time.time() - self.target_lost_time > 1:
                            if lost_direction_y == "down":
                                if not hasattr(self, 'descend_start_height'):
                                    self.log('INFO', f'saving original height: {self.tello.get_distance_tof()}')
                                    setattr(self, 'descend_start_height', self.tello.get_distance_tof())
                                
                                self.log("INFO", "Target lost DOWNWARD → descending to find target")
                                
                                if self.tello.get_distance_tof() > 80:
                                    self.tello.send_rc_control(0, 0, -20, 0)  # 천천히 하강
                                else:
                                    # 하강한 고도에서 Hover
                                    self.tello.send_rc_control(0, 0, 0, 0)

                                # 하강 직후 잠깐 기다리면서 탐색
                                time.sleep(0.5)
                                continue
                                
                            if lost_direction_x in ("left", "right"):
                                # --- 회전을 처음 시작할 때 yaw 초기화 ---
                                if not hasattr(self, "yaw_started") or not self.yaw_started:
                                    self.yaw_started = True
                                    self.yaw_accumulated = 0  # 몇 도 회전했는지 누적
                                    self.prev_yaw = self.tello.get_yaw()  # 시작 yaw 저장
                                    self.spin_direction = 1 if self.cmd_yaw >= 0 else -1
                                    self.log("INFO", f"Start 360 spin, dir={self.spin_direction}")

                                # --- 현재 yaw 읽기 ---
                                curr_yaw = self.tello.get_yaw()

                                # --- yaw 변화량 계산 (wrap-around 처리) ---
                                delta = curr_yaw - self.prev_yaw
                                if delta > 180:
                                    delta -= 360
                                elif delta < -180:
                                    delta += 360

                                # 회전 방향에 맞는 yaw만 누적
                                self.yaw_accumulated += delta
                                self.prev_yaw = curr_yaw

                                # --- 회전 명령 보내기 ---
                                self.tello.send_rc_control(0, 0, 0, 60 * self.spin_direction)

                                # --- 360도 회전 완료 체크 ---
                                if abs(self.yaw_accumulated) >= 360:
                                    self.log("INFO", "360 spin complete. Landing now...")

                                    # RC 중지
                                    self.tello.send_rc_control(0, 0, 0, 0)

                                    # 상태 리셋
                                    self.yaw_started = False
                                    self.target_lost_time = None

                                    # 착륙
                                    self.tello.land()

                                    time.sleep(0.5)
                                    continue
                            elif lost_direction_x == "center":
                                self.log("INFO", "Target lost in CENTER → Hover & wait")
                                # 제자리에서 정지
                                self.tello.send_rc_control(0, 0, 0, 0)

                                # 필요하면 천천히 위로 올라가서 시야 확보도 가능:
                                # self.tello.send_rc_control(0, 0, 20, 0)

                                # 회전 상태 초기화
                                self.yaw_started = False

                            else:
                                # 방향 모르면 기본 hover
                                self.tello.send_rc_control(0, 0, 0, 0)

                    time.sleep(0.1)
                    continue

                elif self.target_lost_time is not None:
                    # 타겟 다시 찾으면 리셋
                    self.target_lost_time = None
                    self.yaw_started = False


                # ---------------------------
                # 1) 타겟 탐색
                # ---------------------------
                target_det = self.get_target_detection(detections)

                if target_det is None:
                    self.log("info", "There is no target")
                    if self.tello:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.1)
                    continue

                # ---------------------------
                # 2) Pose로 torso bbox 생성
                # ---------------------------
                pose = target_det.get("pose") if isinstance(target_det, dict) else getattr(target_det, "pose", None)
                visible_parts = self.get_visible_torso_parts(pose)
                body_bbox = self.get_body_bbox(pose, visible_parts)
                self.target_body_bbox = body_bbox

                if body_bbox is None:
                    self.log("info", "Torso BBox not available, waiting...")
                    if self.tello:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.1)
                    continue

                x1, y1, x2, y2 = body_bbox
                h, w = frame.shape[:2]

                # ---------------------------
                # A. Yaw 제어 (중앙 정렬)
                # ---------------------------
                target_cx = (x1 + x2) / 2
                err_x = (target_cx - w/2) / w
                
                # 마지막 본 위치 저장
                self.last_seen_cx = target_cx / w
                self.last_seen_cy = ((y1 + y2) / 2) / h

                if abs(err_x) > 0.15:
                    self.cmd_yaw = int(err_x * 100 * Kp_yaw_fast * 2)
                else:
                    self.cmd_yaw = int(err_x * 100 * Kp_yaw_normal * 2)

                # ---------------------------
                # B. Forward 제어 (면적 유지)
                # ---------------------------
                target_area = self.get_target_area_by_pose(visible_parts)
                if target_area is None: target_area = None

                current_area = (x2 - x1) * (y2 - y1)

                if target_area is None:
                    continue
                elif current_area < target_area:
                    diff = target_area - current_area
                    self.cmd_fb = int(diff * Kp_area)
                    self.cmd_fb = min(self.cmd_fb, 60)
                else:
                    self.cmd_fb = 0

                # ---------------------------
                # C. fb, yaw clipping
                # ---------------------------
                self.cmd_fb = int(np.clip(self.cmd_fb, 0, 100))
                self.cmd_yaw = int(np.clip(self.cmd_yaw, -100, 100))

                # ---------------------------
                # D. UD 제어 (BBOX 외곽선)
                # ---------------------------
                pad = 15
                need_u = y1 < pad
                need_d = y2 >= h - pad
                if need_u and not need_d:
                    if self.tello.get_distance_tof() < 200: # max height around 200CM
                        self.cmd_ud = 20
                elif not need_u and need_d:
                    if self.tello.get_distance_tof() > 40: # min height around 40CM
                        self.cmd_ud = -20
                else:
                    self.cmd_ud = 0

                if self.tello:
                    self.tello.send_rc_control(0, self.cmd_fb, self.cmd_ud, self.cmd_yaw)

                time.sleep(0.1)

            except Exception as e:
                self.log("ERROR", f"Tracking Error: {e}")
                traceback.print_exc()
                try:
                    if self.tello:
                        self.tello.send_rc_control(0, 0, 0, 0)
                except:
                    pass
                time.sleep(1)


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
        # skip = False
        fps = S.video_fps
        dt = 1 / fps
        next_t = time.time()


        while self.is_streaming:
            try:
                now = time.time()

                if now < next_t:
                    time.sleep(0.003)  
                    continue

                next_t += dt

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

                # 디버깅용 출력 (Torso 기반)
                if self.target_body_bbox is not None and best is not None:
                    x1, y1, x2, y2 = map(int, self.target_body_bbox)

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    torso_area = (x2 - x1) * (y2 - y1)
                    visible_parts = self.get_visible_torso_parts(best.get("pose", []))
                    goal_area = self.get_target_area_by_pose(visible_parts)
                    # visible_parts가 None일 가능성 방지
                    if not visible_parts:
                        visible_parts = []

                    cv2.putText(frame_with_detections, 
                                f"TORSO cx: {cx:.1f}, cy: {cy:.1f}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (50, 255, 255), 4)

                    if goal_area is not None:

                        diff_area = goal_area - torso_area

                        # visible_parts를 문자열로 합침
                        parts_str = ",".join(visible_parts) if isinstance(visible_parts, list) else str(visible_parts)

                        cv2.putText(
                            frame_with_detections, 
                            f"goal_area: {goal_area}, torso_area: {torso_area}, diff: {diff_area}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (50, 255, 255), 4
                        )

                        # --- 추가된 부분: 어떤 부위 기준인지 표시 ---
                        cv2.putText(
                            frame_with_detections,
                            f"visible: {parts_str}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (50, 255, 255), 4
                        )

                # 명령 출력
                cv2.putText(
                    frame_with_detections,
                    f"CMD: FB={self.cmd_fb} YAW={self.cmd_yaw} | LR={self.cmd_lr} UD={self.cmd_ud}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 255, 255), 4
                )

                
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


    def _spawn_tracking_thread(self):
        """트래킹 스레드 안전 생성 (중복 방지)"""
        if getattr(self, "_tracking_thread", None) and self._tracking_thread.is_alive():
            return
        t = threading.Thread(target=self.tracking_thread, daemon=True)
        t.start()
        self._tracking_thread = t

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
        
        manual_commands = ['up', 'down', 'left', 'right', 'forward', 'back', 'cw', 'ccw']
        if self.is_tracking and command in manual_commands:
            return {'success': False, 'message': 'Manual control disabled during tracking. Stop tracking first.'}

        try:
            if command == 'takeoff':
                self.log("INFO", "🚁 Taking off...")
                self.tello.takeoff()
                self.last_takeoff_time = time.time()  # 이륙 시간 기록
                self.log("SUCCESS", f"Takeoff successful - stabilizing for {self.takeoff_stabilization_time}s")
                
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
