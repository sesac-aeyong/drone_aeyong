# tello_web_server.py
import math
import os
import traceback
import cv2
import patches
from djitellopy import Tello
import threading
import time
import numpy as np
import queue
import os
from datetime import datetime
from hailorun import HailoRun
from yolo_tools import draw_detections_on_frame
from .app_tools import *
from settings import settings as S

class TelloOpticalFlow:
    def __init__(self, tello_instance, drone_color='blue'):
        self.tello = tello_instance
        self.is_optical_flow_running = True
        self.lock = ... # (기존 lock 객체 등 필요)
        
        # ==========================================
        # 1. 드론 선택 및 캘리브레이션 데이터 로드
        # ==========================================
        if drone_color == 'blue':
            # 파란 텔로 (Tello 1)
            self.mtx = np.array([[921.73863554, 0.0, 484.9439379],
                                 [0.0, 921.06520894, 355.5162763],
                                 [0.0, 0.0, 1.0]])
            self.dist = np.array([0.01635261, -0.19644455, -0.00021575, 0.00116993, 0.56532416])
            self.new_camera_mtx = np.array([[922.83538766, 0.0, 485.70362399],
                                            [0.0, 920.62929167, 355.42781255],
                                            [0.0, 0.0, 1.0]])
            print(">>> [INFO] Loaded Calibration Data: BLUE Tello")
            
        elif drone_color == 'white':
            # 흰색 텔로 (Tello 2)
            self.mtx = np.array([[918.20765312, 0.0, 481.1811003],
                                 [0.0, 918.144143, 351.56850948],
                                 [0.0, 0.0, 1.0]])
            self.dist = np.array([0.01513358, -0.32789949, -0.00590646, -0.00200168, 0.96440547])
            self.new_camera_mtx = np.array([[917.04620423, 0.0, 479.64715048],
                                            [0.0, 914.76700761, 348.73281015],
                                            [0.0, 0.0, 1.0]])
            print(">>> [INFO] Loaded Calibration Data: WHITE Tello")

        # 정확한 초점거리 (fx) 추출 (Undistort 후의 newCameraMatrix 사용)
        self.fx = self.new_camera_mtx[0, 0] 
        
        # Optical Flow 파라미터
        self.of_lk_params = dict(winSize=(15, 15), maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.of_max_corners = 100
        self.of_quality = 0.3
        self.of_min_dist = 7

class TelloWebServer:
    def __init__(self, socketio):
        self.tello = None
        self.socketio = socketio
        self.is_streaming = False
        self.is_connected = False
        self.current_frame = None
        self.current_depth_map = None            # float32 depth (m) visualized by depth_feed
        self.current_detections = []
        self.target_class = None
        self.target_track_id = None
        self.target_bbox = None  # Store in [x1, y1, x2, y2] format
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        self.frame_center = (480, 360)
        self.target_depth = None

        # RC 명령 설정
        self.use_rc_for_manual = False
        self.use_rc_for_tracking = True
        self.rc_speed = 40
        self.tracking_rc_speed = 25
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
            import traceback
            traceback.print_exc()
            self.inference_engine = None

        # --- Optical Flow 관련 상태/파라미터 ---
        # focal (픽셀) - 기본값은 네가 줬던 fx 값 사용
        self.focal_px = 922.837110
        self.of_max_corners = 300
        self.of_quality = 0.01
        self.of_min_dist = 7
        self.of_lk_params = dict(winSize=(21, 21), maxLevel=3,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.of_bg_percentile = 30          # 하위 퍼센타일을 background 후보로
        self.is_optical_flow_running = False
        self.optical_flow_thread_obj = None
        self.of_last_stats = {}
        self.of_blur_kernel = 51            # sparse->dense 시 Gaussian blur kernel (홀수)
        # 최소 전진 속도 (m/s) 안정화
        self.min_forward_speed = 0.02

        # 녹화 관련 변수 추가
        self.is_recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_dir = "recordings"
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir)
            self.log("INFO", f"📁 Recording directory created: {self.recording_dir}")

        # # 스크린샷 저장 디렉토리 생성
        # self.screenshot_dir = "screenshots"
        # if not os.path.exists(self.screenshot_dir):
        #     os.makedirs(self.screenshot_dir)
        #     self.log("INFO", f"📁 Screenshot directory created: {self.screenshot_dir}")

    # ----------------------
    # 로깅
    # ----------------------
    def log(self, level, message):
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
                self.log_queue.get()
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

    # ----------------------
    # Tello 연결 / 스트리밍
    # ----------------------
    def connect_tello(self):
        """텔로 드론 연결"""
        try:
            self.log("INFO", "🔍 Checking Tello WiFi connection...")
            if not connect_to_tello_wifi():
                self.log("ERROR", "Failed to connect to Tello WiFi")
                return False

            self.log("SUCCESS", "Tello WiFi connected")
            time.sleep(2)

            # 기존 연결 정리
            if self.tello:
                try:
                    self.log("INFO", "Cleaning up old connection...")
                    self.is_streaming = False
                    time.sleep(1)
                    if hasattr(self.tello, 'background_frame_read') and self.tello.background_frame_read:
                        try:
                            self.tello.background_frame_read.stop()
                        except:
                            pass
                    self.tello.streamoff()
                    time.sleep(1)
                    self.tello.end()
                except Exception as e:
                    self.log("WARNING", f"Cleanup error (ignored): {e}")
                finally:
                    self.tello = None
                    time.sleep(3)

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

            # 배터리 경고
            if self.battery < 20:
                self.log("WARNING", f"⚠️ Low battery: {self.battery}%")

            self.log("INFO", "Starting video stream...")
            try:
                self.tello.streamoff()
                time.sleep(2)
            except:
                pass

            self.tello.streamon()
            time.sleep(3)

            self.log("SUCCESS", "🎥 Stream started successfully")
            self.is_connected = True
            return True

        except Exception as e:
            self.log("ERROR", f"Connection error: {e}")
            self.is_connected = False
            self.tello = None
            return False

    # ----------------------
    # 추론 (기존)
    # ----------------------
    def process_frame_with_inference(self, frame):
        """추론 엔진으로 객체 감지 및 깊이 추정"""
        if self.inference_engine is None:
            print("❌ Inference engine is None!")
            return [], None

        try:
            detections, depth_map, _ = self.inference_engine.run(frame)
            return detections, depth_map
        except Exception as e:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
            return [], None

    # ----------------------
    # 자동 추적 (기존)
    # ----------------------
    def tracking_thread(self):
        """자동 추적 스레드"""
        last_command_time = time.time()
        command_interval = 1.0
        target_lost_time = None
        target_lost_warning_sent = False
        depth_threshold = 0.20
        prev_depth = None

        self.log("INFO", "🎯 Tracking thread started (safe mode: 1s interval)")

        while self.is_tracking:
            try:
                if self.target_bbox and self.current_frame is not None:
                    current_time = time.time()

                    # 타겟 재발견 시 경고 리셋
                    if target_lost_time is not None:
                        self.log("SUCCESS", "🎯 Target re-acquired!")
                        target_lost_time = None
                        target_lost_warning_sent = False

                    if current_time - last_command_time >= command_interval:
                        # 제어 명령 계산
                        h, w = self.current_frame.shape[:2]
                        center_x = w // 2
                        center_y = h // 2

                        # target_bbox is in [x1, y1, x2, y2] format
                        x1, y1, x2, y2 = self.target_bbox
                        target_center_x = (x1 + x2) // 2
                        target_center_y = (y1 + y2) // 2

                        # 오차 계산
                        error_x = target_center_x - center_x
                        error_y = target_center_y - center_y
                        if prev_depth is not None:
                            error_d = self.target_depth - prev_depth
                        else:
                            error_d = None

                        # depth 계산
                        prev_depth = self.target_depth

                        # 타겟 크기
                        target_width = x2 - x1
                        target_height = y2 - y1
                        target_area = target_width * target_height
                        frame_area = w * h
                        target_ratio = target_area / frame_area

                        # 임계값
                        threshold_x = w * 0.1
                        threshold_y = h * 0.1
                        threshold_size_min = 0.06
                        threshold_size_max = 0.20

                        action = None

                        # 우선순위 기반 제어
                        # 1. 좌우 정렬 (Yaw)
                        if abs(error_x) > threshold_x:
                            if self.use_rc_for_tracking:
                                yaw_speed = int(np.clip(error_x * 0.06, -self.tracking_rc_speed, self.tracking_rc_speed))
                                self.tello.send_rc_control(0, 0, 0, yaw_speed)
                                time.sleep(self.rc_command_duration)
                                self.tello.send_rc_control(0, 0, 0, 0)
                                action = f"RC yaw={yaw_speed}"
                            else:
                                angle = 15
                                if error_x > 0:
                                    self.tello.rotate_clockwise(angle)
                                    action = f"CW {angle}°"
                                else:
                                    self.tello.rotate_counter_clockwise(angle)
                                    action = f"CCW {angle}°"

                        # 3. 상하 정렬
                        elif abs(error_y) > threshold_y:
                            if self.use_rc_for_tracking:
                                ud_speed = int(np.clip(-error_y * 0.06, -self.tracking_rc_speed, self.tracking_rc_speed))
                                self.tello.send_rc_control(0, 0, ud_speed, 0)
                                time.sleep(self.rc_command_duration)
                                self.tello.send_rc_control(0, 0, 0, 0)
                                action = f"RC ud={ud_speed}"
                            else:
                                if error_y > 0:
                                    self.tello.move_down(20)
                                    action = "Down 20cm"
                                else:
                                    self.tello.move_up(20)
                                    action = "Up 20cm"
                        else:
                            action = "Centered ✅"

                        if action:
                            self.log("DEBUG", f"🎯 {action} | Error: x={error_x:.0f}, y={error_y:.0f} | Size: {target_ratio:.3f}")
                            action = None

                        elif error_d and abs(error_d) > depth_threshold:
                            # 사람이 너무 멀다 → 앞으로 이동해야 함
                            if error_d > 0:
                                if self.use_rc_for_tracking:
                                    self.tello.send_rc_control(0, self.tracking_rc_speed, 0, 0)
                                    time.sleep(self.rc_command_duration)
                                    self.tello.send_rc_control(0, 0, 0, 0)
                                    action = f"RC forward (error_d={error_d:.2f})"
                                else:
                                    self.tello.move_forward(20)
                                    action = "Forward 20cm"

                        else:
                            action = "Distance OK (within threshold)"

                        if action:
                            self.log("DEBUG", f"🎯 {action} | Error: depth={error_d:.0f} | Size: {target_ratio:.3f}")
                            action = None

                        last_command_time = current_time
                        time.sleep(0.5)

                else:
                    # 타겟을 잃어버림
                    if target_lost_time is None:
                        target_lost_time = time.time()

                    # 3초 이상 타겟을 못 찾으면 경고
                    if not target_lost_warning_sent and (time.time() - target_lost_time) > 3:
                        self.log("WARNING", f"⚠️ Target lost for 3 seconds (ID: {self.target_track_id})")
                        target_lost_warning_sent = True

                time.sleep(0.2)

            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                if self.use_rc_for_tracking:
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    except:
                        pass
                time.sleep(1)

        if self.use_rc_for_tracking:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.log("INFO", "🛑 Tracking stopped - drone halted")
            except:
                pass

        self.log("INFO", "🎯 Tracking thread stopped")

    # ----------------------
    # Video stream (기존)
    # ----------------------
    def video_stream_thread(self):
        """비디오 스트리밍 스레드"""
        print("📹 Starting video stream thread...")

        try:
            self.frame_reader = self.tello.get_frame_read()
            print("✅ Frame reader initialized")
        except Exception as e:
            print(f"❌ Failed to initialize frame reader: {e}")
            self.is_streaming = False
            self.socketio.emit('stream_error', {
                'message': 'Failed to start video stream. Please reconnect.'
            })
            return

        error_count = 0
        max_errors = 10

        while self.is_streaming:
            try:
                frame = self.frame_reader.frame

                if frame is not None:
                    error_count = 0

                    # BGR → RGB 변환 (inference expects RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 추론 실행
                    detections, depth_map = self.process_frame_with_inference(frame_rgb)

                    with self.lock:
                        self.current_detections = detections
                        # inference에서 나온 depth_map 크기와 맞추려면 주의
                        try:
                            if depth_map is not None:
                                self.current_depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
                        except Exception:
                            pass

                        # 타겟 추적중이면 해당 객체 찾기
                        if self.is_tracking and self.target_track_id is not None:
                            target_found = False
                            for det in detections:
                                if det['track_id'] == self.target_track_id:
                                    # bbox 업데이트 (detections는 dict 포맷 가정)
                                    self.target_bbox = det['bbox']
                                    self.target_class = det['class']
                                    target_found = True
                                    break

                            if not target_found:
                                self.log("WARNING", f"⚠️ Target ID {self.target_track_id} lost from view")
                            else:
                                x1, y1, x2, y2 = map(int, self.target_bbox)

                                # depth_map에서 bbox 부분만 crop
                                try:
                                    bbox_depth_map = self.current_depth_map[y1:y2, x1:x2]
                                except Exception:
                                    bbox_depth_map = np.array([])

                                if bbox_depth_map.size > 0:
                                    # 중앙값이 가장 안정적
                                    target_depth = float(np.median(bbox_depth_map))

                                    # 신뢰도(옵션)
                                    depth_conf = float(np.var(bbox_depth_map))

                                    # 저장 (다른 쓰레드나 controller가 쓰게)
                                    self.target_depth = target_depth
                                    self.target_depth_conf = depth_conf

                                    self.log("INFO", f"🎯 Target depth: {target_depth:.3f}, conf: {depth_conf:.5f}")
                                else:
                                    self.log("WARNING", "Target depth crop invalid")

                    # 감지 결과 그리기 (inference 엔진의 helper 사용)
                    frame_with_detections = frame_rgb.copy()
                    try:
                        frame_with_detections = self.inference_engine.draw_detections_on_frame(
                            frame_with_detections,
                            detections,
                            target_track_id=self.target_track_id if self.is_tracking else None
                        )
                    except Exception:
                        pass

                    # 프레임 중심 십자선 표시
                    h, w = frame_with_detections.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    cv2.line(frame_with_detections, (center_x - 30, center_y), (center_x + 30, center_y), (255, 255, 255), 2)
                    cv2.line(frame_with_detections, (center_x, center_y - 30), (center_x, center_y + 30), (255, 255, 255), 2)
                    cv2.circle(frame_with_detections, (center_x, center_y), 5, (255, 255, 255), -1)

                    # 배터리 및 높이 정보 업데이트
                    try:
                        old_battery = self.battery
                        self.battery = self.tello.get_battery()
                        self.height = self.tello.get_height()

                        # 배터리 경고
                        if self.battery < 15 and old_battery >= 15:
                            self.log("WARNING", f"⚠️ Critical battery: {self.battery}% - Land soon!")
                        elif self.battery < 25 and old_battery >= 25:
                            self.log("WARNING", f"⚠️ Low battery: {self.battery}%")
                    except:
                        pass

                    # 프레임 저장 (convert back to BGR for web)
                    # with self.lock:
                    #     try:
                    #         frame_bgr = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
                    #     except Exception:
                    #         frame_bgr = frame.copy()
                    #     self.current_frame = frame_bgr
                    
                    self.write_frame_to_video()

                    # 감지 정보를 클라이언트에 전송
                    self.socketio.emit('detections_update', {
                        'detections': [det.to_dict() for det in detections],
                        'battery': self.battery,
                        'height': self.height,
                        'is_tracking': self.is_tracking,
                        'target_track_id': self.target_track_id,
                        'target_class': self.target_class
                    })

                else:
                    error_count += 1
                    if error_count >= max_errors:
                        print("⚠️ Too many frame errors")
                        self.is_streaming = False
                        self.socketio.emit('stream_error', {
                            'message': 'Video stream lost. Please reconnect.'
                        })
                        break

                time.sleep(0.033)

            except Exception as e:
                print(f"Stream error: {e}")
                error_count += 1
                if error_count >= max_errors:
                    print("❌ Stream failed completely")
                    self.is_streaming = False
                    break
                time.sleep(0.1)

        print("📹 Video stream thread ended")

    def start_streaming(self):
        """스트리밍 시작 + Optical Flow 자동 시작"""
        if not self.is_streaming and self.is_connected:
            self.is_streaming = True

            # frame_reader 싱글톤화
            if not hasattr(self, "frame_reader") or self.frame_reader is None:
                try:
                    self.frame_reader = self.tello.get_frame_read()
                    self.log("INFO", "✅ Frame reader initialized")
                except Exception as e:
                    self.log("ERROR", f"Failed to initialize frame reader: {e}")
                    self.is_streaming = False
                    self.socketio.emit('stream_error', {'message': 'Failed to start video stream.'})
                    return False

            # 비디오 스트리밍 쓰레드 시작
            threading.Thread(target=self.video_stream_thread, daemon=True).start()

            # Optical Flow도 자동 시작
            self.start_optical_flow()

            return True
        return False

    def stop_streaming(self):
        """스트리밍 중지"""
        self.is_streaming = False

    # ----------------------
    # 스크린샷 캡처
    # ----------------------

    # def save_screenshot(self):
    #     """현재 프레임을 스크린샷으로 저장"""
    #     try:
    #         with self.lock:
    #             if self.current_frame is None:
    #                 return {'success': False, 'message': 'No frame available'}
                
    #             frame = self.current_frame.copy()
            
    #         # 타임스탬프로 파일명 생성
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"tello_capture_{timestamp}.jpg"
    #         filepath = os.path.join(self.screenshot_dir, filename)
            
    #         # 이미지 저장
    #         cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
    #         return {
    #             'success': True, 
    #             'message': f'Screenshot saved: {filename}',
    #             'filename': filename,
    #             'filepath': filepath
    #         }
            
    #     except Exception as e:
    #         self.log("ERROR", f"Screenshot error: {e}")
    #         return {'success': False, 'message': str(e)}

    # ----------------------
    # 녹화 기능
    # ----------------------
    
    def start_recording(self):
        """녹화 시작"""
        try:
            if self.is_recording:
                return {'success': False, 'message': 'Already recording'}
            
            with self.lock:
                if self.current_frame is None:
                    return {'success': False, 'message': 'No frame available'}
                
                # 비디오 파라미터 설정
                frame_height, frame_width = self.current_frame.shape[:2]
                fps = 20  # FPS 설정 (조정 가능)
            
            # 타임스탬프로 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_filename = f"tello_recording_{timestamp}.mp4"
            filepath = os.path.join(self.recording_dir, self.recording_filename)
            
            # VideoWriter 초기화 (XVID 코덱 사용)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                filepath, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            
            if not self.video_writer.isOpened():
                self.video_writer = None
                return {'success': False, 'message': 'Failed to initialize video writer'}
            
            self.is_recording = True
            self.log("SUCCESS", f"🎥 Recording started: {self.recording_filename}")
            
            return {
                'success': True,
                'message': f'Recording started: {self.recording_filename}',
                'filename': self.recording_filename
            }
            
        except Exception as e:
            self.log("ERROR", f"Recording start error: {e}")
            self.is_recording = False
            self.video_writer = None
            return {'success': False, 'message': str(e)}

    def stop_recording(self):
        """녹화 중지"""
        try:
            if not self.is_recording:
                return {'success': False, 'message': 'Not recording'}
            
            self.is_recording = False
            
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            filename = self.recording_filename
            self.recording_filename = None
            
            self.log("SUCCESS", f"🎬 Recording stopped: {filename}")
            
            return {
                'success': True,
                'message': f'Recording saved: {filename}',
                'filename': filename
            }
            
        except Exception as e:
            self.log("ERROR", f"Recording stop error: {e}")
            self.is_recording = False
            self.video_writer = None
            return {'success': False, 'message': str(e)}
    
    def write_frame_to_video(self):
        """현재 프레임을 비디오에 기록"""
        if self.is_recording and self.video_writer is not None:
            try:
                with self.lock:
                    if self.current_frame is not None:
                        # BGR 형식으로 프레임 기록
                        self.video_writer.write(self.current_frame)
            except Exception as e:
                self.log("ERROR", f"Frame write error: {e}")

    # ----------------------
    # Optical Flow Depth 관련 함수들 (추가)
    # ----------------------

    def stop_optical_flow(self):
        if not self.is_optical_flow_running:
            return False
        self.is_optical_flow_running = False
        if self.optical_flow_thread_obj:
            self.optical_flow_thread_obj.join(timeout=1.0)
        self.log("INFO", "Optical flow depth thread stopped")
        return True
    

    # def optical_flow_thread(self):
    #     """Optical Flow 기반 절대 거리 계산 (Grid Sampling + 에러 방지)"""
        
    #     # =========================================================
    #     # 1. 텔로 캘리브레이션 데이터
    #     # =========================================================
    #     # [옵션 A] 파란 텔로 (Blue Tello)
    #     mtx = np.array([[921.73863554, 0.0, 484.9439379],
    #                     [0.0, 921.06520894, 355.5162763],
    #                     [0.0, 0.0, 1.0]])
    #     dist = np.array([0.01635261, -0.19644455, -0.00021575, 0.00116993, 0.56532416])
    #     new_camera_mtx = np.array([[922.83538766, 0.0, 485.70362399],
    #                             [0.0, 920.62929167, 355.42781255],
    #                             [0.0, 0.0, 1.0]])

    #     fx = new_camera_mtx[0, 0]
        
    #     # [설정] 격자 간격 (픽셀 단위)
    #     GRID_STEP = 40 
    #     # =========================================================

    #     try:
    #         frame_reader = self.tello.get_frame_read()
    #     except Exception as e:
    #         self.log("ERROR", f"OpticalFlow: failed to get frame_read: {e}")
    #         self.is_optical_flow_running = False
    #         return

    #     prev_gray = None
    #     prev_pts = None
    #     prev_time = time.time()

    #     while self.is_optical_flow_running:
    #         # 1. 프레임 획득
    #         raw_frame = frame_reader.frame
    #         if raw_frame is None:
    #             time.sleep(0.01)
    #             continue

    #         # 2. 왜곡 보정 (Undistort)
    #         frame = cv2.undistort(raw_frame, mtx, dist, None, new_camera_mtx)
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #         # =========================================================
    #         # [에러 방지 1] 해상도 변경 감지
    #         # 이전 프레임과 현재 프레임 크기가 다르면 리셋합니다.
    #         # =========================================================
    #         if prev_gray is not None and prev_gray.shape != gray.shape:
    #             print(f"[Warning] Frame size changed: {prev_gray.shape} -> {gray.shape}. Resetting Flow.")
    #             prev_gray = None
    #             prev_pts = None

    #         # 3. 특징점(격자) 생성
    #         # 초기 상태이거나, 점이 너무 적게 남았거나, 리셋된 경우
    #         if prev_gray is None or prev_pts is None or len(prev_pts) < 50:
    #             h, w = gray.shape
                
    #             # np.mgrid를 사용하여 격자 좌표 생성
    #             grid_y, grid_x = np.mgrid[GRID_STEP//2:h:GRID_STEP, GRID_STEP//2:w:GRID_STEP].astype(np.float32)
                
    #             # (N, 1, 2) 형태로 변환
    #             prev_pts = np.expand_dims(np.stack((grid_x.flatten(), grid_y.flatten()), axis=1), axis=1)
                
    #             prev_gray = gray
    #             prev_time = time.time()
    #             time.sleep(0.01)
    #             continue

    #         # 4. Optical Flow 계산 (Lucas-Kanade)
    #         # =========================================================
    #         # [에러 방지 2] try-except로 감싸서 크래시 방지
    #         # =========================================================
    #         try:
    #             next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
    #                 prev_gray, gray, prev_pts, None, **self.of_lk_params
    #             )
    #         except cv2.error as e:
    #             # OpenCV 에러 발생 시(크기 불일치 등) 리셋하고 넘어감
    #             print(f"[Error] calcOpticalFlowPyrLK failed: {e}")
    #             prev_gray = None
    #             prev_pts = None
    #             continue

    #         # 추적 성공한 점들만 유지
    #         good_prev = prev_pts[status == 1]
    #         good_next = next_pts[status == 1]

    #         # 추적 점이 너무 적으면 다음 루프에서 재생성하도록 유도
    #         if len(good_prev) < 10:
    #             prev_gray = None
    #             continue

    #         vis = frame.copy()

    #         # =========================================================
    #         # 거리 계산 로직
    #         # =========================================================
    #         current_time = time.time()
    #         dt = current_time - prev_time
    #         prev_time = current_time

    #         vx_cm = self.tello.get_speed_x()
    #         vy_cm = self.tello.get_speed_y()
    #         vz_cm = self.tello.get_speed_z()
    #         tx = vx_cm / 100.0  # m/s
    #         dx = tx * dt        # 이동 거리 (m)

    #         draw_count = 0
    #         valid_points_count = 0

    #         for p0, p1 in zip(good_prev, good_next):
    #             x0, y0 = p0.ravel()
    #             x1, y1 = p1.ravel()
                
    #             u = (x1 - x0)
                
    #             # 필터링: 움직임이 너무 작거나(노이즈), 드론이 멈춰있으면 스킵
    #             if abs(u) < 0.1 or abs(dx) < 0.001:
    #                 cv2.circle(vis, (int(x1), int(y1)), 2, (0, 150, 0), -1)
    #                 continue

    #             try:
    #                 Z = (dx * fx) / u
    #             except ZeroDivisionError:
    #                 continue

    #             # 유효 거리 필터링 (0 ~ 10m)
    #             if Z <= 0 or Z > 10.0:
    #                 cv2.circle(vis, (int(x1), int(y1)), 2, (0, 0, 255), -1) 
    #                 continue

    #             valid_points_count += 1
                
    #             # 시각화
    #             cv2.circle(vis, (int(x1), int(y1)), 3, (0, 255, 255), -1)
    #             cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                
    #             # 텍스트 가독성 조절 (2번에 1번 출력)
    #             if draw_count % 2 == 0:
    #                 cv2.putText(vis, f"{Z:.1f}m", (int(x1) - 10, int(y1) - 5),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 255), 1)
    #             draw_count += 1

    #         # 정보 표시
    #         info_text = f"Grid Mode | Speed vx: {vx_cm} cm/s | Speed vy: {vy_cm} cm/s | Speed vz: {vz_cm} cm/s | Valid: {valid_points_count}"
    #         cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #         with self.lock:
    #             self.current_frame = vis

    #         # 다음 프레임 준비
    #         prev_gray = gray.copy()
    #         prev_pts = good_next.reshape(-1, 1, 2)

    #         time.sleep(0.01)

    def optical_flow_thread(self):
        """Optical Flow 기반 절대 거리 계산 (goodFeaturesToTrack 사용)"""
        
        # =========================================================
        # 1. 텔로 캘리브레이션 데이터
        # =========================================================
        # [옵션 A] 파란 텔로 (Blue Tello) - 주석 처리됨
        # mtx = np.array([[921.73863554, 0.0, 484.9439379],
        #                 [0.0, 921.06520894, 355.5162763],
        #                 [0.0, 0.0, 1.0]])
        # dist = np.array([0.01635261, -0.19644455, -0.00021575, 0.00116993, 0.56532416])
        # new_camera_mtx = np.array([[922.83538766, 0.0, 485.70362399],
        #                            [0.0, 920.62929167, 355.42781255],
        #                            [0.0, 0.0, 1.0]])

        # [옵션 B] 흰색 텔로 (White Tello) - 현재 적용 중!
        # 보내주신 원본 Intrinsic (fx, fy, cx, cy) 적용
        mtx = np.array([[918.21, 0.0, 481.18],
                        [0.0, 918.14, 351.57],
                        [0.0, 0.0, 1.0]])
        
        # 보내주신 Distortion Coefficients
        dist = np.array([0.01513, -0.32790, -0.005906, -0.002002, 0.96441])
        
        # 보내주신 New Camera Matrix (Undistort 후)
        new_camera_mtx = np.array([[917.04620423, 0.0, 479.64715048],
                                [0.0, 914.76700761, 348.73281015],
                                [0.0, 0.0, 1.0]])

        fx = new_camera_mtx[0, 0]  # ~917.04
        fy = new_camera_mtx[1, 1]  # ~914.76
        cx = new_camera_mtx[0, 2]  # 약 479.64
        cy = new_camera_mtx[1, 2]  # 약 348.73
        
        # [설정] goodFeaturesToTrack 파라미터
        MAX_CORNERS = 200        # 추출할 최대 특징점 개수
        MIN_CORNERS = 100        # 재추출 임계값
        QUALITY_LEVEL = 0.01     # 코너 품질 (0.01 ~ 0.1)
        MIN_DISTANCE = 25        # 특징점 간 최소 거리 (픽셀)
        # =========================================================

        try:
            frame_reader = self.tello.get_frame_read()
        except Exception as e:
            self.log("ERROR", f"OpticalFlow: failed to get frame_read: {e}")
            self.is_optical_flow_running = False
            return

        prev_gray = None
        prev_pts = None
        prev_time = time.time()

        while self.is_optical_flow_running:
            # 1. 프레임 획득
            raw_frame = frame_reader.frame
            if raw_frame is None:
                time.sleep(0.01)
                continue

            # 2. 왜곡 보정 (Undistort)
            frame = cv2.undistort(raw_frame, mtx, dist, None, new_camera_mtx)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # =========================================================
            # [에러 방지 1] 해상도 변경 감지
            # 이전 프레임과 현재 프레임 크기가 다르면 리셋합니다.
            # =========================================================
            if prev_gray is not None and prev_gray.shape != gray.shape:
                print(f"[Warning] Frame size changed: {prev_gray.shape} -> {gray.shape}. Resetting Flow.")
                prev_gray = None
                prev_pts = None

            # 3. 특징점 추출 (goodFeaturesToTrack 사용)
            # 초기 상태이거나, 점이 MIN_CORNERS 미만으로 떨어지면 재추출
            if prev_gray is None or prev_pts is None or len(prev_pts) < MIN_CORNERS:
                # goodFeaturesToTrack으로 특징점 추출
                prev_pts = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=MAX_CORNERS,
                    qualityLevel=QUALITY_LEVEL,
                    minDistance=MIN_DISTANCE,
                    blockSize=7
                )
                
                if prev_pts is None or len(prev_pts) == 0:
                    # 특징점을 찾지 못한 경우
                    with self.lock:
                        self.current_frame = frame
                    time.sleep(0.01)
                    continue
                
                self.log("DEBUG", f"🔍 Extracted {len(prev_pts)} feature points")
                prev_gray = gray
                prev_time = time.time()
                with self.lock:
                    self.current_frame = frame
                time.sleep(0.01)
                continue

            # 4. Optical Flow 계산 (Lucas-Kanade)
            # =========================================================
            # [에러 방지 2] try-except로 감싸서 크래시 방지
            # =========================================================
            try:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **self.of_lk_params
                )
            except cv2.error as e:
                # OpenCV 에러 발생 시(크기 불일치 등) 리셋하고 넘어감
                print(f"[Error] calcOpticalFlowPyrLK failed: {e}")
                prev_gray = None
                prev_pts = None
                continue

            # 추적 성공한 점들만 유지
            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]

            # 추적 점이 너무 적으면 다음 루프에서 재생성하도록 유도
            if len(good_prev) < MIN_CORNERS:
                self.log("DEBUG", f"⚠️ Feature points dropped to {len(good_prev)}, re-extracting...")
                prev_gray = None
                prev_pts = None
                continue

            vis = frame.copy()

            # =========================================================
            # 거리 계산 로직
            # =========================================================
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            vx_cm = self.tello.get_speed_x() * 10
            vy_cm = self.tello.get_speed_y() * 10
            # vz_cm = self.tello.get_speed_z() * 10
            yaw = self.tello.get_yaw()

            # yaw를 라디안으로 변환
            yaw_rad = np.deg2rad(yaw)
            vx_forward = vx_cm * np.cos(yaw_rad) + vy_cm * np.sin(yaw_rad)  # 드론 앞방향 속도
            vy_lateral = -vx_cm * np.sin(yaw_rad) + vy_cm * np.cos(yaw_rad) # 드론 좌우방향 속도

            tx = vx_forward / 100.0  # m/s
            dx = tx * dt        # 이동 거리 (m)

            draw_count = 0
            valid_points_count = 0
            depth_measurements = []  # 유효한 거리 측정값 저장

            for p0, p1 in zip(good_prev, good_next):
                x0, y0 = p0.ravel()
                x1, y1 = p1.ravel()
                
                # u = (x1 - x0)
                
                # # 필터링: 움직임이 너무 작거나(노이즈), 드론이 멈춰있으면 스킵
                # if abs(u) < 0.1 or abs(dx) < 0.001:
                #     cv2.circle(vis, (int(x1), int(y1)), 2, (0, 150, 0), -1)
                #     continue

                # try:
                #     Z = (dx * fx) / u
                # except ZeroDivisionError:
                #     continue

                # 노이즈 필터링: 이동 거리가 너무 짧으면(호버링 등) 계산 스킵
                if abs(dx) < 0.001:
                    continue

                # -----------------------------------------------------
                # [Step 1] 정밀 각도 계산 (fx, fy 반영)
                # -----------------------------------------------------
                # a: 이전 프레임 점의 각도
                norm_x0 = (x0 - cx) / fx
                norm_y0 = (y0 - cy) / fy
                tan_a = math.sqrt(norm_x0**2 + norm_y0**2)
                
                # b: 현재 프레임 점의 각도
                norm_x1 = (x1 - cx) / fx
                norm_y1 = (y1 - cy) / fy
                tan_b = math.sqrt(norm_x1**2 + norm_y1**2)

                # 중심 부근 노이즈 필터링
                if tan_a < 0.01:
                    continue

                angle_a = math.atan(tan_a)
                angle_b = math.atan(tan_b)
                
                # -----------------------------------------------------
                # [Step 2] 공식 적용: Z = dx * sin(a) / sin(b - a)
                # -----------------------------------------------------
                # delta_angle (b - a)는 시차(Parallax)를 의미합니다.
                delta_angle = angle_b - angle_a
                
                # 각도 변화가 너무 작으면(무한대 거리 or 정지) 스킵
                if abs(delta_angle) < 1e-5:
                    continue

                try:
                    Z = dx * math.sin(angle_a) / math.sin(delta_angle)
                except ZeroDivisionError:
                    continue

                # 유효 거리 필터링 (0 ~ 10m)
                if Z <= 0 or Z > 10.0:
                    cv2.circle(vis, (int(x1), int(y1)), 2, (0, 0, 255), -1) 
                    continue

                valid_points_count += 1
                depth_measurements.append(Z)
                
                # 시각화
                cv2.circle(vis, (int(x1), int(y1)), 3, (0, 255, 255), -1)
                cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                
                # 텍스트 가독성 조절 (3번에 1번 출력)
                if draw_count % 3 == 0:
                    cv2.putText(vis, f"{Z:.1f}m", (int(x1) - 10, int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 255), 1)
                draw_count += 1

            # 평균/중간값 거리 계산
            avg_depth = np.mean(depth_measurements) if depth_measurements else 0.0
            median_depth = np.median(depth_measurements) if depth_measurements else 0.0

            # 정보 표시
            print(self.tello.get_current_state())
            info_text = f"Features: {len(good_next)} | Speed v_forward: {vx_forward/100:.2f} m/s | Speed v_lateral: {vy_lateral/100:.2f} m/s | Yaw: {yaw} degree | Valid: {valid_points_count}"
            cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # cv2.putText(vis, f"query speed: {self.tello.query_speed()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # 평균 거리 표시
            if depth_measurements:
                depth_text = f"Avg Depth: {avg_depth:.2f}m | Median: {median_depth:.2f}m"
                cv2.putText(vis, depth_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with self.lock:
                self.current_frame = vis

            # 다음 프레임 준비
            prev_gray = gray.copy()
            prev_pts = good_next.reshape(-1, 1, 2)

            time.sleep(0.01)


    def start_optical_flow(self):
        """Optical Flow depth 추정 스레드 시작"""
        if not self.is_connected:
            self.log("ERROR", "Cannot start optical flow: Not connected")
            return False
        if self.is_optical_flow_running:
            self.log("WARNING", "Optical flow already running")
            return False

        self.is_optical_flow_running = True
        threading.Thread(target=self.optical_flow_thread, daemon=True).start()
        self.log("INFO", "Optical flow depth thread started")
        return True


    # ----------------------
    # Depth feed (MJPEG) - 컬러맵으로 제공
    # ----------------------
    def get_depth_colormap_jpeg(self):
        with self.lock:
            if self.current_depth_map is None:
                return None
            depth = self.current_depth_map.copy()

        # 0 값(없는 지점)은 min으로 처리해서 시각화 왜곡 방지
        mask = depth > 0
        if not np.any(mask):
            return None

        # normalize to 0-255 for colormap
        depth_nonzero = depth.copy()
        # clip extremes
        vmin = np.percentile(depth_nonzero[mask], 5)
        vmax = np.percentile(depth_nonzero[mask], 95)
        if vmin == vmax:
            vmax = vmin + 1e-3
        norm = np.zeros_like(depth_nonzero, dtype=np.uint8)
        clip = np.clip(depth_nonzero, vmin, vmax)
        norm = ((clip - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

        # fill zeros with 0 (black)
        norm[~mask] = 0

        colormap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # return JPEG bytes
        _, buffer = cv2.imencode('.jpg', colormap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes()

    # ----------------------
    # 명령 실행 (기존)
    # ----------------------
    def get_current_frame_jpeg(self):
        """현재 프레임을 JPEG로 반환"""
        with self.lock:
            if self.current_frame is not None:
                _, buffer = cv2.imencode('.jpg', self.current_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()
        return None

    def execute_command(self, command):
        """드론 명령 실행"""
        if not self.is_connected or not self.tello:
            return {'success': False, 'message': 'Not connected to Tello'}

        try:
            if command == 'takeoff':
                self.log("INFO", "🚁 Taking off...")
                self.tello.takeoff()
                time.sleep(3)
                self.log("SUCCESS", "Takeoff successful")
                return {'success': True, 'message': 'Takeoff successful'}

            elif command == 'land':
                self.log("INFO", "🛬 Landing...")
                self.tello.land()
                time.sleep(2)
                self.log("SUCCESS", "Landing successful")
                return {'success': True, 'message': 'Landing successful'}

            elif command == 'emergency':
                self.log("WARNING", "🚨 Emergency stop!")
                self.tello.emergency()
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
                # self.tello.move_forward(30)
                # self.tello.go_xyz_speed(30, 0, 0, 100)
                self.tello.send_rc_control(left_right_velocity=0, forward_backward_velocity=100, up_down_velocity=0, yaw_velocity=0)
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
        # Stop optical flow if running
        # 녹화 중이면 중지
        if self.is_recording:
            try:
                self.stop_recording()
            except:
                pass
        try:
            self.stop_optical_flow()
        except:
            pass
        # Stop streaming
        try:
            self.stop_streaming()
        except:
            pass
        if self.inference_engine:
            try:
                self.inference_engine.close()
            except:
                pass
