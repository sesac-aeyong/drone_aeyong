# tello_web_server.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from djitellopy import Tello
import threading
import time
import numpy as np
import socket
import signal
import sys
import subprocess
import queue
from hailorun import HailoRun

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tello_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def list_wifi_networks():
    result = subprocess.run(["nmcli", "-t", "-f", "SSID", "dev", "wifi"], capture_output=True, text=True)
    ssids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ssids

def disconnect_wifi():
    subprocess.run(['nmcli', 'dev', 'disconnect', 'wlan0'])

def connect_to_wifi(ssid, password=None):
    cmd = ["nmcli", "dev", "wifi", "connect", ssid]
    if password:
        cmd.extend(["password", password])
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        return False
    return True

def get_current_ssid():
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.strip().split("\n"):
            active, ssid = line.split(":")
            if active == "yes":
                return ssid
        return None
    except Exception as e:
        print("Error:", e)
        return None
    
def connect_to_tello_wifi():
    """Tello WiFi에 자동으로 연결"""
    ssid = get_current_ssid()
    print('Current SSID:', ssid)
    if ssid and ssid.startswith('TELLO-'):
        return True
    
    print('Looking for Tello WiFi...')
    for attempt in range(10):
        networks = set(list_wifi_networks())
        for ssid in networks:
            if ssid.startswith('TELLO-'):
                print(f'Connecting to {ssid}...')
                if connect_to_wifi(ssid):
                    print(f'✅ Connected to {ssid}')
                    return True
                else:
                    print(f'❌ Failed to connect to {ssid}')
        print(f'Retry {attempt + 1}/10...')
        time.sleep(5)
    return False

def get_local_ip():
    """현재 사용중인 IP 주소 가져오기"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.10.1", 8889))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unknown"

class TelloWebServer:
    def __init__(self):
        self.tello = None
        self.is_streaming = False
        self.is_connected = False
        self.current_frame = None
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
        # self.target_depth = None

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
                    socketio.emit('log_message', log_entry)
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
            time.sleep(2)
            
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
                    target_size_ideal = 0.2
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
                        self.log("WARNING", f"⚠️ Target lost for 3 seconds (ID: {self.target_track_id})")
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
            frame_reader = self.tello.get_frame_read()
            print("✅ Frame reader initialized")
        except Exception as e:
            print(f"❌ Failed to initialize frame reader: {e}")
            self.is_streaming = False
            socketio.emit('stream_error', {
                'message': 'Failed to start video stream. Please reconnect.'
            })
            return
            
        error_count = 0
        max_errors = 10
        
        while self.is_streaming:
            try:
                frame = frame_reader.frame
                
                if frame is not None:
                    error_count = 0
                    
                    # BGR → RGB 변환
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 추론 실행
                    detections, depth_map = self.process_frame_with_inference(frame)
                    
                    with self.lock:
                        self.current_detections = detections
                        # self.current_depth_map = cv2.resize(depth_map, frame.shape[:2])
                        
                        # 타겟 추적중이면 해당 객체 찾기
                        if self.is_tracking and self.target_track_id is not None:
                            target_found = False
                            for det in detections:
                                if det['track_id'] == self.target_track_id:
                                    # bbox 업데이트 (detections는 [x1, y1, x2, y2] format)
                                    self.target_bbox = det['bbox']
                                    self.target_class = det['class']
                                    target_found = True
                                    break
                            
                            if not target_found:
                                self.log("WARNING", f"⚠️ Target ID {self.target_track_id} lost from view")
                            # else:
                            #     x1, y1, x2, y2 = map(int, self.target_bbox)

                            #     # depth_map에서 bbox 부분만 crop
                            #     bbox_depth_map = self.current_depth_map[y1:y2, x1:x2]

                            #     if bbox_depth_map.size > 0:
                            #         # 중앙값이 가장 안정적
                            #         target_depth = float(np.median(bbox_depth_map))

                            #         # 신뢰도(옵션)
                            #         depth_conf = float(np.var(bbox_depth_map))

                            #         # 저장 (다른 쓰레드나 controller가 쓰게)
                            #         self.target_depth = target_depth
                            #         self.target_depth_conf = depth_conf

                            #         self.log("INFO", f"🎯 Target depth: {target_depth:.3f}, conf: {depth_conf:.5f}")
                            #     else:
                            #         self.log("WARNING", "Target depth crop invalid")
                    
                    # 감지 결과 그리기
                    frame_with_detections = self.inference_engine.draw_detections_on_frame(
                        frame.copy(), 
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
                    
                    # 감지 정보를 클라이언트에 전송
                    socketio.emit('detections_update', {
                        'detections': detections,
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
                        socketio.emit('stream_error', {
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
                self.last_takeoff_time = time.time()  # 이륙 시간 기록
                time.sleep(3)
                self.log("SUCCESS", f"Takeoff successful - stabilizing for {self.takeoff_stabilization_time}s")
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

# 전역 인스턴스
tello_server = TelloWebServer()

# Flask 라우트
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = tello_server.get_current_frame_jpeg()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO 이벤트
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('connect_tello')
def handle_connect_tello():
    success = tello_server.connect_tello()
    if success:
        tello_server.start_streaming()
        emit('tello_status', {
            'connected': True, 
            'battery': tello_server.battery
        })
    else:
        emit('tello_status', {'connected': False})

@socketio.on('reconnect_tello')
def handle_reconnect_tello():
    print("🔄 Reconnecting to Tello...")
    tello_server.stop_tracking()
    tello_server.stop_streaming()
    time.sleep(1)
    
    print("🔌 Disconnecting WiFi...")
    disconnect_wifi()
    time.sleep(2)
    
    success = tello_server.connect_tello()
    if success:
        tello_server.start_streaming()
        emit('tello_status', {
            'connected': True, 
            'battery': tello_server.battery
        })
    else:
        emit('tello_status', {'connected': False})

@socketio.on('send_command')
def handle_command(data):
    command = data.get('command')
    result = tello_server.execute_command(command)
    emit('command_response', result)

@socketio.on('set_target')
def handle_set_target(data):
    """타겟 설정 (bbox는 [x1, y1, x2, y2] format)"""
    target_track_id = data.get('track_id')
    target_class = data.get('class')
    target_bbox = data.get('bbox')  # [x1, y1, x2, y2]
    
    tello_server.target_track_id = target_track_id
    tello_server.target_class = target_class
    tello_server.target_bbox = target_bbox
    
    tello_server.log("INFO", f"🎯 Target set to: ID {target_track_id} ({target_class}), bbox: {target_bbox}")
    emit('target_response', {
        'track_id': target_track_id,
        'class': target_class,
        'bbox': target_bbox
    })

@socketio.on('start_tracking')
def handle_start_tracking():
    if tello_server.target_track_id is not None:
        success = tello_server.start_tracking()
        emit('tracking_status', {
            'is_tracking': success,
            'target_track_id': tello_server.target_track_id,
            'target_class': tello_server.target_class
        })
    else:
        emit('tracking_status', {
            'is_tracking': False,
            'message': 'No target selected'
        })

@socketio.on('stop_tracking')
def handle_stop_tracking():
    tello_server.stop_tracking()
    emit('tracking_status', {'is_tracking': False})

def cleanup_and_exit():
    """완전한 정리 후 종료"""
    print("\n🛑 Cleaning up...")
    
    global tello_server
    
    try:
        if tello_server.is_tracking:
            tello_server.stop_tracking()
            time.sleep(0.5)
    except:
        pass
    
    try:
        if tello_server.is_streaming:
            tello_server.stop_streaming()
            time.sleep(0.5)
    except:
        pass
    
    try:
        tello_server.cleanup()
    except:
        pass
    
    try:
        if tello_server.tello:
            if hasattr(tello_server.tello, 'background_frame_read'):
                if tello_server.tello.background_frame_read:
                    try:
                        tello_server.tello.background_frame_read.stop()
                        print("✅ Background frame read stopped")
                    except:
                        pass
            
            try:
                tello_server.tello.streamoff()
                time.sleep(1)
                print("✅ Stream off")
            except:
                pass
            
            try:
                tello_server.tello.end()
                print("✅ Tello connection ended")
            except:
                pass
    except:
        pass
    
    try:
        print("🔧 Killing processes on UDP port 11111...")
        subprocess.run(['fuser', '-k', '11111/udp'], 
                      stderr=subprocess.DEVNULL, 
                      stdout=subprocess.DEVNULL,
                      timeout=2)
        time.sleep(1)
        print("✅ UDP port released")
    except:
        pass
    
    print("✅ Cleanup complete")

def signal_handler(sig, frame):
    cleanup_and_exit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print("🔧 Cleaning up UDP port 11111...")
    try:
        subprocess.run(['fuser', '-k', '11111/udp'], 
                      stderr=subprocess.DEVNULL, 
                      stdout=subprocess.DEVNULL,
                      timeout=2)
        time.sleep(1)
        print("✅ Port cleaned")
    except:
        print("⚠️ Could not clean port (may not be in use)")

    if '--auto-connect' in sys.argv or '-a' in sys.argv:
        print("\n🔍 Auto-connecting to Tello WiFi...")
        disconnect_wifi()
        time.sleep(1)
        if connect_to_tello_wifi():
            print("✅ Auto-connected to Tello WiFi")
        else:
            print("⚠️ Auto-connect failed")
        time.sleep(2)

    local_ip = get_local_ip()
    print("\n" + "="*50)
    print(f"🚁 Tello Web Server Started!")
    print("="*50 + "\n")

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        cleanup_and_exit()