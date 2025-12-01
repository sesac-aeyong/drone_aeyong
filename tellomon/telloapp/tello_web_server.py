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
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        self.frame_center = (480, 360)

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
    
    
    def tracking_thread(self):
        """자동 추적 스레드"""
        target_lost_time = None
        target_lost_warning_sent = False
        
        # 제어 게인 (단순 비례 제어)
        gain_yaw = 0.80      # 회전 게인
        gain_lr = 0.80       # 좌우 이동 게인
        gain_ud = 0.40       # 상하 이동 게인
        gain_fb = 200         # 전후 이동 게인
        
        # 임계값
        yaw_threshold = 0.20    # 20% 이상 오차면 회전
        lr_threshold = 0.05     # 8% 이상 오차면 좌우 이동
        ud_threshold = 0.05     # 8% 이상 오차면 상하 이동
        size_threshold = 0.025  # 크기 오차 임계값

        self.log("INFO", "🎯 Simple RC tracking started")
        
        while self.is_tracking:
            try:
                # 이륙 후 안정화 시간 체크
                if self.last_takeoff_time is not None:
                    time_since_takeoff = time.time() - self.last_takeoff_time
                    if time_since_takeoff < self.takeoff_stabilization_time:
                        remaining = self.takeoff_stabilization_time - time_since_takeoff
                        if int(remaining * 10) % 10 == 0:  # 0.1초마다 로그
                            self.log("INFO", f"⏳ Stabilizing... {remaining:.1f}s remaining")
                        time.sleep(0.1)
                        continue
                    else:
                        # 안정화 완료
                        if self.last_takeoff_time is not None:
                            self.log("SUCCESS", "✅ Stabilization complete - starting tracking")
                            self.last_takeoff_time = None  # 한 번만 로그 출력

                if self.target_bbox and self.current_frame is not None:
                    # 타겟 재발견 시 경고 리셋
                    if target_lost_time is not None:
                        self.log("SUCCESS", "🎯 Target re-acquired!")
                        target_lost_time = None
                        target_lost_warning_sent = False
                    
                    # 제어 명령 계산
                    h, w = self.current_frame.shape[:2]
                    center_x = w // 2
                    center_y = h // 2
                    
                    # target_bbox is in [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = self.target_bbox
                    target_center_x = (x1 + x2) // 2
                    target_center_y = (y1 + y2) // 2
                    
                    # 오차 계산 (정규화)
                    error_x = (target_center_x - center_x) / w  # -0.5 ~ 0.5
                    error_y = (target_center_y - center_y) / h  # -0.5 ~ 0.5
                    
                    # 타겟 크기
                    target_width = x2 - x1
                    target_height = y2 - y1
                    target_area = target_width * target_height
                    frame_area = w * h
                    target_ratio = target_area / frame_area
                    
                    # 목표 크기
                    target_size_ideal = 0.3
                    error_size = target_size_ideal - target_ratio
                    
                    # === 간단한 비례 제어 ===
                    
                    # 1. 좌우 제어: 큰 오차는 회전, 작은 오차는 평행이동
                    if abs(error_x) > yaw_threshold:
                        # 회전
                        yaw_speed = int(np.clip(error_x * gain_yaw * 100, -self.tracking_rc_speed, self.tracking_rc_speed))
                        lr_speed = 0
                    elif abs(error_x) > lr_threshold:
                        # 좌우 이동
                        yaw_speed = 0
                        lr_speed = int(np.clip(error_x * gain_lr * 100, -self.tracking_rc_speed, self.tracking_rc_speed))
                    else:
                        # 중앙 정렬됨
                        yaw_speed = 0
                        lr_speed = 0
                    
                    # 2. 상하 제어
                    if abs(error_y) > ud_threshold:
                        ud_speed = int(np.clip(-error_y * gain_ud * 100, -self.tracking_rc_speed, self.tracking_rc_speed))
                    else:
                        ud_speed = 0
                    
                    # 3. 전후 제어
                    if abs(error_size) > size_threshold:
                        fb_speed = int(np.clip(error_size * gain_fb, 0, self.tracking_rc_speed))
                    else:
                        fb_speed = 0
                    
                    # RC 명령 전송
                    self.tello.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)
                    
                    # 로그 출력
                    # if yaw_speed != 0 or lr_speed != 0 or ud_speed != 0 or fb_speed != 0:
                        # action = f"RC[lr={lr_speed:+3d}, fb={fb_speed:+3d}, ud={ud_speed:+3d}, yaw={yaw_speed:+3d}]"
                        # self.log("DEBUG", 
                            # f"🎯 {action} | Err[x={error_x:+.3f}, y={error_y:+.3f}, s={error_size:+.3f}] | Size={target_ratio:.3f}")
                
                else:
                    # 타겟을 잃어버림
                    if target_lost_time is None:
                        target_lost_time = time.time()
                        self.tello.send_rc_control(0, 0, 0, 0)
                    
                    # 3초 이상 타겟을 못 찾으면 경고
                    if not target_lost_warning_sent and (time.time() - target_lost_time) > 3:
                        self.log("WARNING", f"⚠️ Target lost for 3 seconds (ID: {self.target_identity_id})")
                        target_lost_warning_sent = True
                
                time.sleep(0.05)  # 20Hz 제어 루프
                
            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                if self.use_rc_for_tracking:
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    except:
                        pass
                time.sleep(0.5)
        
        # 추적 종료 시 정지
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.log("INFO", "🛑 Tracking stopped - drone halted")
        except:
            pass
        
        self.log("INFO", "🎯 Tracking thread stopped")


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