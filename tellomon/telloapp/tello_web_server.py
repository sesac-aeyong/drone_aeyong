# tello_web_server.py
import traceback
import cv2
import patches
from djitellopy import Tello
import threading
import time
import numpy as np
import queue
from hailorun import HailoRun
from yolo_tools import draw_detections_on_frame
from .app_tools import *


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
            time.sleep(1)
            
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
            
            self.log("INFO", "Creating new Tello connection...")
            self.tello = Tello()
            ### Tello parameters
            # these don't work on tello v1.3
            # self.tello.set_video_fps(Tello.FPS_15)
            # self.tello.set_video_resolution(Tello.RESOLUTION_480P)
            
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
            if self.tello.stream_on:
                try:
                    self.tello.streamoff()
                    self.log('INFO', 'Waiting for video stream to end...')
                    while self.tello.stream_on:
                        time.sleep(0.1)
                except:
                    pass
            
            self.tello.streamon()
            self.log('INFO', 'Waiting for tello video stream to start...')
            while not self.tello.stream_on:
                time.sleep(0.1)
            # time.sleep(10)
            
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
                        
                        # # 2. 거리 조정
                        # elif target_ratio < threshold_size_min:
                        #     if self.use_rc_for_tracking:
                        #         self.tello.send_rc_control(0, self.tracking_rc_speed, 0, 0)
                        #         time.sleep(self.rc_command_duration)
                        #         self.tello.send_rc_control(0, 0, 0, 0)
                        #         action = f"RC forward={self.tracking_rc_speed}"
                        #     else:
                        #         self.tello.move_forward(20)
                        #         action = "Forward 20cm"
                        
                        # elif target_ratio > threshold_size_max:
                        #     if self.use_rc_for_tracking:
                        #         self.tello.send_rc_control(0, -self.tracking_rc_speed, 0, 0)
                        #         time.sleep(self.rc_command_duration)
                        #         self.tello.send_rc_control(0, 0, 0, 0)
                        #         action = f"RC back={self.tracking_rc_speed}"
                        #     else:
                        #         self.tello.move_back(20)
                        #         action = "Back 20cm"
                        
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

                            # # 사람이 너무 가깝다 → 뒤로 이동
                            # else: 
                            #     if self.use_rc_for_tracking:
                            #         self.tello.send_rc_control(0, -self.tracking_rc_speed, 0, 0)
                            #         time.sleep(self.rc_command_duration)
                            #         self.tello.send_rc_control(0, 0, 0, 0)
                            #         action = f"RC back (error_d={error_d:.2f})"
                            #     else:
                            #         self.tello.move_back(20)
                            #         action = "Back 20cm"

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

    def video_stream_thread(self):
        """비디오 스트리밍 스레드"""
        print("📹 Starting video stream thread...")
        
        try:
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
                
                # BGR → RGB 변환
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 추론 실행
                detections, depth_map, *_ = self.inference_engine.run(frame)
                
                with self.lock:
                    self.current_detections = detections
                    self.current_depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0])) 
                    dep_vis = cv2.normalize(self.current_depth_map, None, 0, 255).astype(np.uint8)
                    dep_vis = cv2.applyColorMap(dep_vis, cv2.COLORMAP_JET)
                    cv2.imshow('dep', dep_vis)
                    cv2.waitKey(1)
                        
                    # don't resize the depth array, resize points instead?
                    
                    # 타겟 추적중이면 해당 객체 찾기
                    if self.is_tracking and self.target_track_id is not None:
                        target_found = False
                        for det in detections:
                            if det.track_id == self.target_track_id:
                                self.target_bbox = det.bbox
                                self.target_class = det.cls
                                target_found = True
                                break
                        
                        if not target_found:
                            self.log("WARNING", f"⚠️ Target ID {self.target_track_id} lost from view")
                        else:
                            x1, y1, x2, y2 = self.target_bbox

                            # depth_map에서 bbox 부분만 crop
                            bbox_depth_map = self.current_depth_map[y1:y2, x1:x2]

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
                
                # 감지 결과 그리기
                frame_with_detections = draw_detections_on_frame(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    detections,
                    target_track_id=self.target_track_id if self.is_tracking else None
                )
                
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
                
                # 프레임 저장
                with self.lock:
                    self.current_frame = frame_with_detections
                    self.current_frame_updated = True
                
                # 감지 정보를 클라이언트에 전송
                self.socketio.emit('detections_update', {
                    'detections': [det.to_dict() for det in detections],
                    'battery': self.battery,
                    'height': self.height,
                    'is_tracking': self.is_tracking,
                    'target_track_id': self.target_track_id,
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
        """자동 추적 시작"""
        if not self.is_tracking and self.target_track_id is not None:
            # ThiefTracker 활성화
            success = self.inference_engine.enter_thief_mode(self.target_track_id)
            if not success:
                self.log("ERROR", f"Failed to enter thief mode for ID {self.target_track_id}")
                return False

            self.is_tracking = True
            thread = threading.Thread(target=self.tracking_thread)
            thread.daemon = True
            thread.start()
            self.log("SUCCESS", f"🎯 Started tracking: ID {self.target_track_id} ({self.target_class})")
            return True
        return False
    

    def stop_tracking(self):
        """자동 추적 중지"""
        if not self.is_tracking:
            return

        self.is_tracking = False
        self.target_bbox = None
        self.log("INFO", "⏹️ Stopped tracking")

        # ThiefTracker 모드 종료
        self.inference_engine.exit_thief_mode()
    

    def get_current_frame_jpeg(self):
        """현재 프레임을 JPEG로 반환"""
        with self.lock:
            if self.current_frame is not None and self.current_frame_updated:
                _, buffer = cv2.imencode('.jpg', self.current_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.current_frame_updated = False
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
