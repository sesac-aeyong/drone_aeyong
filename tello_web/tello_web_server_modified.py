# tello_web_server_modified.py
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
from djitellopy import Tello
import threading
import time
import base64
import json
import numpy as np
import socket
import signal
import sys
import subprocess
import os

# object_detection 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
object_detection_dir = os.path.join(parent_dir, 'object_detection')
if object_detection_dir not in sys.path:
    sys.path.insert(0, object_detection_dir)

# YOLO 대신 첫 번째 코드의 추론 엔진 사용
from tello_inference import TelloInference

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
    if ssid and ssid.startswith('TELLO-'):  # 이미 Tello에 연결됨
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
        self.current_depth = None
        self.current_detections = []
        self.target_class = None
        self.target_track_id = None  # YOLO 클래스 대신 track_id 사용
        self.target_bbox = None
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        self.frame_center = (480, 360)
        
        # 추론 엔진 초기화 (첫 번째 코드의 추론 시스템 사용)
        print("Loading inference engine...")
        try:
            self.inference_engine = TelloInference()
            print("✅ Inference engine loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load inference engine: {e}")
            self.inference_engine = None
        

    def connect_tello(self):
        """텔로 드론 연결 (에러 복구 기능 추가)"""
        try:
            # WiFi 자동 연결 시도
            print("🔍 Checking Tello WiFi connection...")
            if not connect_to_tello_wifi():
                print('❌ Failed to connect to Tello WiFi')
                return False
            
            print("✅ Tello WiFi connected")
            time.sleep(2)  # WiFi 연결 안정화 대기
            
            # 기존 연결 완전히 정리
            if self.tello:
                try:
                    print("Cleaning up old connection...")
                    self.is_streaming = False
                    time.sleep(1)
                    
                    # ⭐ background_frame_read 명시적으로 정리
                    if hasattr(self.tello, 'background_frame_read') and self.tello.background_frame_read:
                        try:
                            self.tello.background_frame_read.stop()
                        except:
                            pass
                    
                    self.tello.streamoff()
                    time.sleep(1)  # 스트림 완전히 닫히길 대기
                    self.tello.end()
                    
                except Exception as e:
                    print(f"Cleanup error (ignored): {e}")
                finally:
                    self.tello = None
                    time.sleep(3)  # 더 길게 대기
            
            # 새로운 연결 생성
            print("Creating new Tello connection...")
            self.tello = Tello()
            
            # 연결 시도 (재시도 로직)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Connection attempt {attempt + 1}/{max_retries}...")
                    self.tello.connect()
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise
            
            # 배터리 확인
            self.battery = self.tello.get_battery()
            print(f"✅ Tello connected. Battery: {self.battery}%")
            
            # 스트림 시작
            print("Starting video stream...")
            try:
                self.tello.streamoff()
                time.sleep(2)  # 더 길게 대기
            except:
                pass
            
            self.tello.streamon()
            time.sleep(3)  # 더 길게 대기
            
            print("🎥 Stream started successfully")
            self.is_connected = True
            return True
        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.is_connected = False
            self.tello = None
            return False
    
    def process_frame_with_inference(self, frame):
        """
        첫 번째 코드의 추론 엔진으로 객체 감지 및 깊이 추정
        
        Args:
            frame: RGB 이미지
        
        Returns:
            detections: 감지된 객체 리스트
            depth_map: 깊이 맵
        """
        if self.inference_engine is None:
            return [], None
        
        try:
            detections, depth_map = self.inference_engine.process_frame(frame)
            return detections, depth_map
        except Exception as e:
            print(f"Inference error: {e}")
            return [], None
    
    def draw_detections(self, frame, detections, depth_map=None):
        """
        프레임에 감지 결과 그리기
        
        Args:
            frame: RGB 이미지
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            label = f"{det['class']} (ID:{track_id}): {det['confidence']:.2f}"
            
            # 타겟이면 빨간색, 아니면 초록색 - track_id만으로 비교
            is_target = (track_id == self.target_track_id)
            
            # RGB 색상 (OpenCV는 BGR이지만 우리는 RGB 프레임 사용)
            color = (255, 0, 0) if is_target else (0, 255, 0)  # RGB: Red or Green
            thickness = 3 if is_target else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # 라벨 배경
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 라벨 텍스트
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 타겟이면 중심점 표시
            if is_target:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (center_x, center_y), 12, (0, 0, 255), 2)
        
        # 프레임 중심 십자선 표시
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 십자선
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (255, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (255, 255, 255), 2)
        
        # 중심점
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        
        return frame
    
    def calculate_control_commands(self, target_bbox, frame_shape):
        """타겟 위치에 따라 드론 제어 명령 계산"""
        if not target_bbox:
            return None
        
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # 타겟의 중심점
        x1, y1, x2, y2 = target_bbox
        target_center_x = (x1 + x2) // 2
        target_center_y = (y1 + y2) // 2
        
        # 타겟의 크기 (거리 추정용)
        target_width = x2 - x1
        target_height = y2 - y1
        target_area = target_width * target_height
        frame_area = w * h
        target_ratio = target_area / frame_area
        
        # 오차 계산
        error_x = target_center_x - center_x
        error_y = target_center_y - center_y
        
        # 임계값 설정 (픽셀)
        threshold_x = w * 0.15  # 프레임 너비의 15%
        threshold_y = h * 0.15  # 프레임 높이의 15%
        threshold_size_min = 0.03  # 너무 작으면 전진
        threshold_size_max = 0.30  # 너무 크면 후진
        
        commands = []
        
        # 좌우 제어 (Yaw 회전) - 우선순위 1
        if abs(error_x) > threshold_x:
            if error_x > 0:
                commands.append('cw')  # 오른쪽으로 회전
            else:
                commands.append('ccw')  # 왼쪽으로 회전
        
        # 전후 제어 (크기 기반) - 우선순위 2
        elif target_ratio < threshold_size_min:
            commands.append('forward')  # 전진
        elif target_ratio > threshold_size_max:
            commands.append('back')  # 후진
        
        # 상하 제어 - 우선순위 3
        elif abs(error_y) > threshold_y:
            if error_y > 0:
                commands.append('down')  # 아래로
            else:
                commands.append('up')  # 위로
        
        return commands if commands else None
    
    def tracking_thread(self):
        """자동 추적 스레드"""
        last_command_time = time.time()
        command_interval = 1.5  # 1.5초마다 명령 실행 (더 안정적)
        
        while self.is_tracking:
            try:
                if self.target_bbox and self.current_frame is not None:
                    current_time = time.time()
                    
                    # 일정 시간마다 제어 명령 실행
                    if current_time - last_command_time >= command_interval:
                        commands = self.calculate_control_commands(
                            self.target_bbox, 
                            self.current_frame.shape
                        )
                        
                        if commands:
                            # 첫 번째 명령만 실행 (우선순위 기반)
                            command = commands[0]
                            print(f"🎯 Tracking command: {command}")
                            result = self.execute_command(command)
                            
                            if result['success']:
                                last_command_time = current_time
                            else:
                                print(f"Command failed: {result['message']}")
                        else:
                            print("🎯 Target centered - no adjustment needed")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Tracking error: {e}")
                time.sleep(0.5)
    
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
                print(frame.shape[:])
                
                if frame is not None:
                    error_count = 0  # 정상 프레임 수신시 에러 카운트 리셋
                    
                    # 첫 번째 코드와 정확히 동일한 처리
                    # Tello 원본: 400x300 → 640x480으로 리사이즈
                    frame = cv2.resize(frame, (640, 480))
                    
                    # BGR → RGB 변환
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 첫 번째 프레임에서만 크기 출력
                    # if not hasattr(self, '_frame_size_logged'):
                    #     print(f"📐 Frame size after resize: {frame.shape} (H={frame.shape[0]}, W={frame.shape[1]})")
                    #     self._frame_size_logged = True
                    
                    # 추론 실행 (bbox는 640x480 기준으로 반환됨)
                    detections, depth_map = self.process_frame_with_inference(frame)
                    
                    # 첫 몇 개 프레임에서 detection 정보 출력
                    if not hasattr(self, '_detection_logged_count'):
                        self._detection_logged_count = 0
                    if self._detection_logged_count < 3 and len(detections) > 0:
                        det = detections[0]
                        print(f"🔍 Sample detection:")
                        print(f"    Class: {det['class']}")
                        print(f"    BBox: {det['bbox']} (x: 0-640, y: 0-480)")
                        print(f"    Confidence: {det['confidence']:.2f}")
                        self._detection_logged_count += 1
                    
                    with self.lock:
                        self.current_detections = detections
                        self.current_depth = depth_map
                        
                        # 타겟 추적중이면 해당 객체 찾기 (track_id로만 찾음)
                        if self.is_tracking and self.target_track_id is not None:
                            target_found = False
                            for det in detections:
                                if det['track_id'] == self.target_track_id:
                                    # bbox 업데이트 (실제 객체 위치로)
                                    self.target_bbox = det['bbox']
                                    self.target_class = det['class']
                                    target_found = True
                                    break
                            
                            if not target_found:
                                print(f"⚠️ Target ID {self.target_track_id} lost from view")
                                # 타겟을 잃어버려도 bbox는 유지 (마지막 위치 기억)
                    
                    # 감지 결과 그리기 (640x480 프레임에 그림)
                    frame_with_detections = self.draw_detections(frame.copy(), detections, depth_map)
                    
                    # 배터리 및 높이 정보 업데이트
                    try:
                        self.battery = self.tello.get_battery()
                        self.height = self.tello.get_height()
                    except:
                        pass
                    
                    # 프레임 저장 (detection이 그려진 프레임, RGB)
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
                        print("⚠️ Too many frame errors, attempting reconnection...")
                        self.is_streaming = False
                        socketio.emit('stream_error', {
                            'message': 'Video stream lost. Please reconnect.'
                        })
                        break
                
                time.sleep(0.033)  # ~30 FPS
                
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
            self.is_tracking = True
            thread = threading.Thread(target=self.tracking_thread)
            thread.daemon = True
            thread.start()
            print(f"🎯 Started tracking: ID {self.target_track_id} ({self.target_class})")
            return True
        return False
    
    def stop_tracking(self):
        """자동 추적 중지"""
        self.is_tracking = False
        self.target_bbox = None
        print("⏹️ Stopped tracking")
    
    def get_current_frame_jpeg(self):
        """현재 프레임을 JPEG로 반환 (RGB → JPEG)"""
        with self.lock:
            if self.current_frame is not None:
                # current_frame은 RGB 형식
                # OpenCV의 imencode는 BGR을 기대하지만, RGB를 넣으면
                # BGR로 "생각"하고 인코딩함 → 웹 브라우저가 RGB로 해석 → 정상 출력
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
                self.tello.takeoff()
            elif command == 'land':
                self.tello.land()
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
            elif command == 'emergency':
                self.tello.emergency()
            else:
                return {'success': False, 'message': f'Unknown command: {command}'}
            
            return {'success': True, 'message': f'Command {command} executed'}
        
        except Exception as e:
            print(f"Command execution error: {e}")
            return {'success': False, 'message': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        if self.inference_engine:
            self.inference_engine.cleanup()

# 전역 인스턴스
tello_server = TelloWebServer()

# Flask 라우트
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    def generate():
        while True:
            frame = tello_server.get_current_frame_jpeg()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# SocketIO 이벤트
@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    print('Client disconnected')

@socketio.on('connect_tello')
def handle_connect_tello():
    """텔로 연결"""
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
    """텔로 재연결"""
    print("🔄 Reconnecting to Tello...")
    tello_server.stop_tracking()
    tello_server.stop_streaming()
    time.sleep(1)
    
    # WiFi 연결도 다시 시도
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
    """드론 명령 처리"""
    command = data.get('command')
    result = tello_server.execute_command(command)
    emit('command_response', result)

@socketio.on('set_target')
def handle_set_target(data):
    """타겟 설정 (track_id 기반)"""
    target_track_id = data.get('track_id')
    target_class = data.get('class')
    target_bbox = data.get('bbox')
    
    tello_server.target_track_id = target_track_id
    tello_server.target_class = target_class
    tello_server.target_bbox = target_bbox
    
    print(f"🎯 Target set to: ID {target_track_id} ({target_class}), bbox: {target_bbox}")
    emit('target_response', {
        'track_id': target_track_id,
        'class': target_class,
        'bbox': target_bbox
    })

@socketio.on('start_tracking')
def handle_start_tracking():
    """자동 추적 시작"""
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
    """자동 추적 중지"""
    tello_server.stop_tracking()
    emit('tracking_status', {'is_tracking': False})

def cleanup_and_exit():
    """완전한 정리 후 종료"""
    print("\n🛑 Cleaning up...")
    
    global tello_server
    
    # 추적 중지
    try:
        if tello_server.is_tracking:
            tello_server.stop_tracking()
            time.sleep(0.5)
    except:
        pass
    
    # 스트리밍 중지
    try:
        if tello_server.is_streaming:
            tello_server.stop_streaming()
            time.sleep(1)
    except:
        pass
    
    # 추론 엔진 정리
    try:
        tello_server.cleanup()
    except:
        pass
    
    # 텔로 연결 종료
    try:
        if tello_server.tello:
            # BackgroundFrameRead 정리
            if hasattr(tello_server.tello, 'background_frame_read'):
                if tello_server.tello.background_frame_read:
                    try:
                        tello_server.tello.background_frame_read.stop()
                        print("✅ Background frame read stopped")
                    except:
                        pass
            
            # 스트림 끄기
            try:
                tello_server.tello.streamoff()
                time.sleep(1)
                print("✅ Stream off")
            except:
                pass
            
            # 연결 종료
            try:
                tello_server.tello.end()
                print("✅ Tello connection ended")
            except:
                pass
    except:
        pass
    
    # UDP 포트 강제 해제
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
    """Ctrl+C 처리"""
    cleanup_and_exit()
    sys.exit(0)

# 시그널 핸들러 등록
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

    # 시작 시 Tello WiFi 자동 연결 시도 (선택사항)
    import sys
    if '--auto-connect' in sys.argv or '-a' in sys.argv:
        print("\n🔍 Auto-connecting to Tello WiFi...")
        disconnect_wifi()
        time.sleep(1)
        if connect_to_tello_wifi():
            print("✅ Auto-connected to Tello WiFi")
        else:
            print("⚠️ Auto-connect failed, but you can connect manually from the web interface")
        time.sleep(2)

    local_ip = get_local_ip()
    print("\n" + "="*50)
    print(f"🚁 Tello Web Server Started!")
    print(f"📱 Access from phone: http://{local_ip}:5000")
    print(f"🌐 Or use: http://raspberrypi.local:5000")
    print("\n💡 Tip: Use --auto-connect or -a flag to auto-connect to Tello WiFi on startup")
    print("="*50 + "\n")

    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        cleanup_and_exit()