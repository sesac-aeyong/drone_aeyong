# tello_web_server_new.py
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
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tello_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
        self.current_detections = []
        self.target_class = None
        self.target_bbox = None
        self.is_tracking = False
        self.battery = 0
        self.height = 0
        self.lock = threading.Lock()
        self.frame_center = (480, 360)
        
        # YOLO 모델 로드
        print("Loading YOLO model...")
        try:
            # YOLOv8n (nano) 모델 사용 - 가장 가벼움
            self.yolo_model = YOLO('yolov8n.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            print("Downloading YOLOv8n model...")
            self.yolo_model = YOLO('yolov8n.pt')  # 자동으로 다운로드됨
        
        # COCO 클래스 이름
        self.class_names = self.yolo_model.names
        

    def connect_tello(self):
        """텔로 드론 연결 (에러 복구 기능 추가)"""
        try:
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
    
    def process_frame_with_yolo(self, frame):
        """
        YOLO로 객체 감지
        """
        try:
            # YOLO 추론 실행 (verbose=False로 출력 최소화)
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            
            detections = []
            
            # 결과 파싱
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 신뢰도
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 클래스
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            return detections
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return []
    
    def process_frame_with_hailo(self, frame):
        """
        Hailo NPU로 추론 실행
        현재는 YOLO를 사용하지만, 나중에 Hailo 코드로 교체 가능
        """
        # YOLO로 추론
        return self.process_frame_with_yolo(frame)
    
    def draw_detections(self, frame, detections):
        """프레임에 감지 결과 그리기"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            # 타겟이면 빨간색, 아니면 초록색
            is_target = (det['class'] == self.target_class and 
                        self.target_bbox and 
                        det['bbox'] == self.target_bbox)
            
            color = (255, 0, 0) if is_target else (0, 255, 0)
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
                
                if frame is not None:
                    error_count = 0  # 정상 프레임 수신시 에러 카운트 리셋
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # YOLO로 객체 인식
                    detections = self.process_frame_with_hailo(frame)
                    
                    with self.lock:
                        self.current_detections = detections
                        
                        # 타겟 추적중이면 해당 객체 찾기
                        if self.is_tracking and self.target_class:
                            target_found = False
                            for det in detections:
                                if det['class'] == self.target_class:
                                    # 가장 가까운 객체 선택 (여러 개 있을 경우)
                                    self.target_bbox = det['bbox']
                                    target_found = True
                                    break
                            
                            if not target_found:
                                print(f"⚠️ Target '{self.target_class}' lost from view")
                    
                    # 감지 결과 그리기
                    frame = self.draw_detections(frame, detections)
                    
                    # 배터리 및 높이 정보 업데이트
                    try:
                        self.battery = self.tello.get_battery()
                        self.height = self.tello.get_height()
                    except:
                        pass
                    
                    # 프레임 저장
                    with self.lock:
                        self.current_frame = frame.copy()
                    
                    # 감지 정보를 클라이언트에 전송
                    socketio.emit('detections_update', {
                        'detections': detections,
                        'battery': self.battery,
                        'height': self.height,
                        'is_tracking': self.is_tracking,
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
        if not self.is_tracking and self.target_class:
            self.is_tracking = True
            thread = threading.Thread(target=self.tracking_thread)
            thread.daemon = True
            thread.start()
            print(f"🎯 Started tracking: {self.target_class}")
            return True
        return False
    
    def stop_tracking(self):
        """자동 추적 중지"""
        self.is_tracking = False
        self.target_bbox = None
        print("⏹️ Stopped tracking")
    
    def get_current_frame_jpeg(self):
        """현재 프레임을 JPEG로 반환"""
        with self.lock:
            if self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame_rgb, 
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
    """타겟 클래스 설정"""
    target_class = data.get('target_class')
    target_bbox = data.get('bbox')
    
    tello_server.target_class = target_class
    tello_server.target_bbox = target_bbox
    
    print(f"🎯 Target set to: {target_class}, bbox: {target_bbox}")
    emit('target_response', {'target': target_class, 'bbox': target_bbox})

@socketio.on('start_tracking')
def handle_start_tracking():
    """자동 추적 시작"""
    if tello_server.target_class:
        success = tello_server.start_tracking()
        emit('tracking_status', {
            'is_tracking': success,
            'target': tello_server.target_class
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

    local_ip = get_local_ip()
    print("\n" + "="*50)
    print(f"🚁 Tello Web Server Started!")
    print(f"📱 Access from phone: http://{local_ip}:5000")
    print(f"🌐 Or use: http://raspberrypi.local:5000")
    print("="*50 + "\n")

    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        cleanup_and_exit()