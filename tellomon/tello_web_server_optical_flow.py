# tello_web_server.py (통합 버전: 기존 기능 + Optical Flow depth 추정)
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
from djitellopy import Tello
import threading
import time
import numpy as np
import math
import socket
import signal
import sys
import subprocess
import queue
from hailorun import HailoRun
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tello_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ----------------------
# 유틸 함수들
# ----------------------
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

# ----------------------
# 메인 클래스
# ----------------------
class TelloWebServer:
    def __init__(self):
        self.tello = None
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

        # 스크린샷 저장 디렉토리 생성
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            self.log("INFO", f"📁 Screenshot directory created: {self.screenshot_dir}")

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
                    socketio.emit('log_message', log_entry)
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
            socketio.emit('stream_error', {
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
                    with self.lock:
                        try:
                            frame_bgr = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
                        except Exception:
                            frame_bgr = frame.copy()
                        self.current_frame = frame_bgr

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
                    socketio.emit('stream_error', {'message': 'Failed to start video stream.'})
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

    def save_screenshot(self):
        """현재 프레임을 스크린샷으로 저장"""
        try:
            with self.lock:
                if self.current_frame is None:
                    return {'success': False, 'message': 'No frame available'}
                
                frame = self.current_frame.copy()
            
            # 타임스탬프로 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tello_capture_{timestamp}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            # 이미지 저장
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return {
                'success': True, 
                'message': f'Screenshot saved: {filename}',
                'filename': filename,
                'filepath': filepath
            }
            
        except Exception as e:
            self.log("ERROR", f"Screenshot error: {e}")
            return {'success': False, 'message': str(e)}

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

    def optical_flow_thread(self):
        """Optical Flow 기반 절대 거리 계산 (드론 속도 사용)"""
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
            frame = frame_reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 특징점 부족 → 다시 추출
            if prev_gray is None or prev_pts is None or len(prev_pts) < 40:
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.of_max_corners,
                    qualityLevel=self.of_quality,
                    minDistance=self.of_min_dist
                )
                time.sleep(0.01)
                continue

            # Optical Flow 계산
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **self.of_lk_params
            )

            good_prev = prev_pts[status == 1]
            good_next = next_pts[status == 1]

            # 화면 렌더링용
            vis = frame.copy()

            # =============================
            # 절대 거리 계산에 필요한 값
            # =============================
            # 1) 드론 전진 속도 (cm/s → m/s)
            vx = self.tello.get_speed_x() / 100.0
            print(f"Speed -> X:   {vx} (m/s)")

            # 드론이 정지 상태면 거리 계산 의미 없음
            if abs(vx) < 0.05:     # 5cm/s 이하
                vx = 0.0

            # 2) 프레임 간 시간 Δt
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # 3) 전체 optical flow 평균 계산
            flow_u = good_next[:, 0] - good_prev[:, 0]
            u_mean = np.median(flow_u)

            # ============ 절대 깊이 계산 공식 ===============
            # Z = f * v * dt / u
            if abs(u_mean) < 0.1 or vx == 0:
                Z = None
            else:
                Z = (self.focal_px * vx * dt) / u_mean

            # =============================
            # 시각화 (가독성 중요)
            # =============================

            # 특징점 전체는 표시하지 말고 → 소수만 찍기
            sample_idx = np.linspace(0, len(good_prev) - 1, 20).astype(int)
            sampled_prev = good_prev[sample_idx]
            sampled_next = good_next[sample_idx]

            for p0, p1 in zip(sampled_prev, sampled_next):
                x0, y0 = p0.ravel()
                x1, y1 = p1.ravel()
                cv2.circle(vis, (int(x1), int(y1)), 3, (0, 255, 0), -1)
                cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

            # 화면 상단에 전체 거리만 표시
            if Z is not None:
                cv2.putText(vis, f"Depth (Absolute) ~ {Z:.2f} m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Depth: -- (No motion)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            # 프레임 업데이트 → 웹 전송됨
            with self.lock:
                self.current_frame = vis

            # 다음 프레임 준비
            prev_gray = gray.copy()
            prev_pts = good_next.reshape(-1, 1, 2)

            time.sleep(0.005)

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
                self.tello.move_forward(30)
                # self.tello.send_rc_control(left_right_velocity=0, forward_backward_velocity=10, up_down_velocity=0, yaw_velocity=0)
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

# 전역 인스턴스
tello_server = TelloWebServer()

# ----------------------
# Flask 라우트
# ----------------------
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

@app.route('/depth_feed')
def depth_feed():
    """Depth 컬러맵을 MJPEG로 스트리밍"""
    def generate():
        while True:
            frame = tello_server.get_depth_colormap_jpeg()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # depth 아직 준비 안됨 → 검은 빈 프레임
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------
# SocketIO 이벤트
# ----------------------
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

@socketio.on('start_optical_flow')
def handle_start_optical_flow():
    """SocketIO로 optical flow 시작 요청"""
    ok = tello_server.start_optical_flow()
    emit('optical_flow_status', {'running': ok})

@socketio.on('stop_optical_flow')
def handle_stop_optical_flow():
    ok = tello_server.stop_optical_flow()
    emit('optical_flow_status', {'running': not ok})

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

@socketio.on('capture_screenshot')
def handle_capture_screenshot():
    """스크린샷 캡처 요청"""
    result = tello_server.save_screenshot()
    emit('screenshot_response', result)

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

# ----------------------
# 종료/정리
# ----------------------
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
