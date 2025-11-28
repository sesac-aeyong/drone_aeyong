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
from tracking_controller import DroneTrackingController, TrackingConfig

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
        self.tracking_rc_speed = 50
        self.rc_command_duration = 0.4
        
        # 웹 로그 시스템
        self.log_queue = queue.Queue(maxsize=100)  # 최대 100개 로그 저장
        self.log_thread = None
        self.is_logging = True
        self.start_log_broadcaster()

        self.tracking_controller = DroneTrackingController(TrackingConfig())
        # Optical flow 데이터 저장
        self.current_ego_velocity = None
        self.ego_velocity_history = []
        self.max_ego_history = 15
        
        # 거리 유지 목표
        self.forward_only = True
        self.target_distance = 3.0  # 3m 유지
        self.min_safe_distance = 1.8 # 최소 안전 거리
        self.max_track_distance = 6.0   # 최대 추적 거리
        self.depth_history = []
        self.max_depth_history = 5
  
        self.depth_scale = 0.85
        self.depth_scale_history = []
        self.max_scale_history = 20
        self.frame_count = 0
        self.depth_diagnostic_interval = 40  # 20Hz * 2s
        self.last_depth_values = []
        self._last_control_log_time = time.time()

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

    def _log_depth_diagnostic(self, raw_depth, smoothed_depth):
        """
        Depth 필터링 진단 로깅
        
        Args:
            raw_depth: 필터링 전 depth 값
            smoothed_depth: 필터링 후 depth 값
        """
        import numpy as np
        
        # 최근 값 저장 (최대 20개)
        if len(self.last_depth_values) >= 20:
            self.last_depth_values.pop(0)
        
        self.last_depth_values.append({
            'raw': raw_depth,
            'smoothed': smoothed_depth,
            'time': time.time()
        })
        
        # 5개 이상 모았을 때만 로그
        if len(self.last_depth_values) < 5:
            return
        
        # 통계 계산
        raw_values = [v['raw'] for v in self.last_depth_values]
        smoothed_values = [v['smoothed'] for v in self.last_depth_values]
        
        raw_mean = np.mean(raw_values)
        raw_std = np.std(raw_values)
        raw_min = np.min(raw_values)
        raw_max = np.max(raw_values)
        
        smoothed_mean = np.mean(smoothed_values)
        smoothed_std = np.std(smoothed_values)
        
        # # 간단한 로그 (매번)
        # self.log("DEBUG",
        #     f"[DEPTH] Raw={raw_mean:.2f}±{raw_std:.3f}m "
        #     f"({raw_min:.2f}~{raw_max:.2f}) | "
        #     f"Smoothed={smoothed_mean:.2f}±{smoothed_std:.3f}m")
        
        # 상세 로그 (2초마다)
        if self.frame_count % 60 == 0:
            recent_raw = [f"{v['raw']:.2f}" for v in self.last_depth_values[-5:]]
            recent_smooth = [f"{v['smoothed']:.2f}" for v in self.last_depth_values[-5:]]
            
            self.log("DEBUG",
                f"[DEPTH DETAIL] Raw: {recent_raw} | "
                f"Smoothed: {recent_smooth}")
            
            # 필터링 효율 계산
            if raw_std > 0:
                reduction_ratio = (raw_std - smoothed_std) / raw_std * 100
                self.log("DEBUG",
                    f"[FILTER] Std reduction: {reduction_ratio:.1f}% "
                    f"({raw_std:.3f}m → {smoothed_std:.3f}m)")
            


    def _log_control_state(self, depth, state_str, rc_cmd):
        """
        제어 상태 로깅
        
        Args:
            depth: 현재 depth 값
            state_str: 상태 문자열
            rc_cmd: RC 명령 dict
        """
        # 1초마다 로그
        current_time = time.time()
        if current_time - self._last_control_log_time < 1.0:
            return
        
        self.log("DEBUG",
            f"[CONTROL] Depth={depth:.2f}m | State={state_str} | "
            f"RC[LR={rc_cmd['left_right']:+3d}, "
            f"FB={rc_cmd['forward_backward']:+3d}, "
            f"UD={rc_cmd['up_down']:+3d}, "
            f"YAW={rc_cmd['yaw']:+3d}]")
        
        self._last_control_log_time = current_time

    def update_adaptive_depth_scale(self, ego_velocity, depth_history):
        """
        Optical flow + Depth로 스케일 자동 조정
        
        필요한 데이터:
        - ego_velocity: 현재 프레임의 드론 이동
        - depth_history: 최근 깊이 히스토리
        """

        if ego_velocity is None or depth_history is None:
            return self.depth_scale

        # 데이터 충분 체크
        if len(depth_history) < 2:
            return self.depth_scale

        if isinstance(ego_velocity, (tuple, list)):
            if len(ego_velocity) >= 2:
                # 속도의 크기(magnitude) 계산
                # ego_velocity = (vx, vy)
                ego_vel = (ego_velocity[0]**2 + ego_velocity[1]**2) ** 0.5
            else:
                return self.depth_scale
        else:
            ego_vel = abs(ego_velocity)

        # 드론이 거의 안 움직임
        if abs(ego_vel) < 0.01:  # 1cm 미만
            return self.depth_scale
        
        # 깊이 변화 계산
        delta_depth = depth_history[-1] - depth_history[-2]
        
        # 깊이가 거의 안 변함
        if abs(delta_depth) < 0.001:  # 1mm 미만
            return self.depth_scale
        
        # 스케일 계산
        # 예상 깊이 변화 = -드론 이동 / 현재 스케일
        expected_delta_depth = -ego_vel / self.depth_scale
        
        # 오류율 계산
        error_ratio = delta_depth / expected_delta_depth
        
        # EMA 평활화 (천천히 적응)
        alpha = 0.05
        new_scale = (alpha * self.depth_scale / error_ratio +
                     (1 - alpha) * self.depth_scale)
        
        # 타당성 검사
        if 0.5 < new_scale < 2.0:
            # 범위 내 - 업데이트
            self.depth_scale_history.append(new_scale)
            
            if len(self.depth_scale_history) > self.max_scale_history:
                self.depth_scale_history.pop(0)
            
            # 로그 출력
            print(f"[SCALE] {self.depth_scale:.3f} -> {new_scale:.3f}, "
                  f"avg={np.median(self.depth_scale_history):.3f}")
            
            self.depth_scale = new_scale
        
        return self.depth_scale

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
    
    
    def tracking_thread(self):
        """자동 추적 스레드"""
        target_lost_time = None
        target_lost_warning_sent = False
        
        self.log("INFO", "🎯 PID-based tracking started")
        
        while self.is_tracking:
            try:
                # 이륙 후 안정화 시간 체크
                if self.last_takeoff_time is not None:
                    time_since_takeoff = time.time() - self.last_takeoff_time
                    if time_since_takeoff < self.takeoff_stabilization_time:
                        remaining = self.takeoff_stabilization_time - time_since_takeoff
                        if int(remaining * 10) % 10 == 0:
                            self.log("INFO", f"⏳ Stabilizing... {remaining:.1f}s")
                        time.sleep(0.1)
                        continue
                    else:
                        if self.last_takeoff_time is not None:
                            self.log("SUCCESS", "✅ Stabilization complete")
                            self.last_takeoff_time = None
                
                # 타겟 존재 여부 확인
                if not (self.target_bbox and self.current_frame is not None):
                    # 타겟 상실
                    if target_lost_time is None:
                        target_lost_time = time.time()
                    
                    self.tello.send_rc_control(0, 0, 0, 0)
                    
                    if not target_lost_warning_sent and (time.time() - target_lost_time) > 3:
                        self.log("WARNING", f"⚠️ Target lost for 3s (ID: {self.target_identity_id})")
                        target_lost_warning_sent = True
                    
                    time.sleep(0.05)
                    continue
                
                # 타겟 재발견 처리
                if target_lost_time is not None:
                    self.log("SUCCESS", "🎯 Target re-acquired!")
                    target_lost_time = None
                    target_lost_warning_sent = False
                
                # ===== 제어 명령 계산 =====
                
                h, w = self.current_frame.shape[:2]
                frame_center = (w // 2, h // 2)
                
                # Depth 정보 추출
                if self.current_depth_map is None:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.05)
                    continue
                
                target_depth = self.inference_engine.extract_target_depth(
                    self.current_depth_map,
                    self.target_bbox
                )
                
                if target_depth is None:
                    self.tello.send_rc_control(0, 0, 0, 0)
                    time.sleep(0.05)
                    continue
                
                # Optical flow ego-velocity (있으면 사용)
                ego_velocity = None
                if self.current_ego_velocity is not None:
                    ego_velocity = self.current_ego_velocity
                
                # 🆕 개선된 제어 명령 생성
                control_cmd = self.tracking_controller.compute_control_command(
                    frame=self.current_frame,
                    bbox=self.target_bbox,
                    depth=target_depth,
                    ego_velocity=ego_velocity,
                    frame_center=frame_center
                )
                
                # RC 명령 전송
                lr_speed = control_cmd['left_right']
                fb_speed = control_cmd['forward_backward']
                ud_speed = control_cmd['up_down']
                yaw_speed = control_cmd['yaw']
                
                self.tello.send_rc_control(lr_speed, fb_speed, ud_speed, yaw_speed)
                
                # 🔴 Depth 진단 로깅 (필터링 전후 비교)
                smoothed_depth = self.tracking_controller.depth_filter.smoothed_value
                if smoothed_depth is None:
                    smoothed_depth = target_depth
                
                self._log_depth_diagnostic(target_depth, smoothed_depth)
                
                # 🔴 제어 상태 로깅
                self._log_control_state(smoothed_depth, control_cmd['state'], {
                    'left_right': lr_speed,
                    'forward_backward': fb_speed,
                    'up_down': ud_speed,
                    'yaw': yaw_speed
                })
                time.sleep(0.05)  # 20Hz 제어 루프
            
            except Exception as e:
                self.log("ERROR", f"Tracking error: {e}")
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except:
                    pass
                time.sleep(0.5)
        
        # 추적 종료 시 정지
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.log("INFO", "🛑 Tracking stopped")
        except:
            pass
        
        self.log("INFO", "🎯 Tracking thread stopped")

    def _log_ego_speed_stats(self, frame_count):
        """기존 ego_velocity_history를 이용해 로그 출력 (이상치 필터링)"""
        if not self.ego_velocity_history:
            return
        
        #  드론 최대 속도 임계값
        MAX_DRONE_SPEED = 2.0  # m/s
        
        ego_speeds = []
        filtered_ego_speeds = []  #  필터링된 속도
        
        for vel in self.ego_velocity_history:
            if vel is not None:
                vx, vy = vel
                speed = np.sqrt(vx**2 + vy**2)
                ego_speeds.append(speed)
                
                #  이상치 제거
                if speed <= MAX_DRONE_SPEED:
                    filtered_ego_speeds.append(speed)
        
        if not ego_speeds:
            return
        
        # 원본 통계
        current = ego_speeds[-1]
        avg_raw = np.mean(ego_speeds)
        
        # 필터링된 통계
        if filtered_ego_speeds:
            avg_filtered = np.mean(filtered_ego_speeds)
            std_filtered = np.std(filtered_ego_speeds)
        else:
            avg_filtered = 0
            std_filtered = 0
        
        # 안정성 판정
        if avg_filtered < 0.2:
            stability = "Excellent"
        elif avg_filtered < 0.4:
            stability = "Good"
        elif avg_filtered < 0.6:
            stability = "Fair"
        else:
            stability = "Poor"
        
        print(f"\n{'='*60}")
        print(f" Ego-speed Stats [Frame {frame_count}]")
        print(f"{'='*60}")
        
        print(f"\n Filtered Data (< {MAX_DRONE_SPEED} m/s):")
        print(f"  Average:    {avg_filtered:.3f} m/s ✓")
        print(f"  Std Dev:    {std_filtered:.3f} m/s")
        print(f"  Samples:    {len(filtered_ego_speeds)}")
        print(f"  Stability:  {stability}")
                
        print(f"{'='*60}\n")


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
        frame_count = 0

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
                
                # 추론 실행
                detections, depth_map, _, optical_flow_data = self.inference_engine.run(frame)
                self.current_depth_map = depth_map

                # self.update_adaptive_depth_scale(self.current_ego_velocity, self.depth_history)
                # self.inference_engine.depth_scale = self.depth_scale

                with self.lock:
                    self.current_detections = detections
                    self.current_depth_map = cv2.resize(depth_map, frame.shape[1::-1])
                        
                    # Ego-velocity 저장
                    if optical_flow_data['has_flow'] and optical_flow_data['ego_velocity'] is not None:
                        self.current_ego_velocity = optical_flow_data['ego_velocity']
                        self.ego_velocity_history.append(self.current_ego_velocity)
                        if len(self.ego_velocity_history) > self.max_ego_history:
                            self.ego_velocity_history.pop(0)
                    
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
                            # Depth 계산
                            target_depth = self.inference_engine.extract_target_depth(
                                self.current_depth_map,
                                self.target_bbox
                            )

                            if target_depth is not None:
                                self.depth_history.append(target_depth)
                                if len(self.depth_history) > self.max_depth_history:
                                    self.depth_history.pop(0)
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
                
                self.frame_count += 1
                # if frame_count % 30 == 0:
                #     self._log_ego_speed_stats(frame_count)
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
