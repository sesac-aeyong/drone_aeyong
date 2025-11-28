# tracking_controller.py
"""
개선된 드론 추적 제어 모듈
- PID 기반 3축 독립 제어
- Depth 안정화
- Optical flow 예측 통합
- 하이스테리시스 기반 상태 관리
"""

import numpy as np
from collections import deque
import time
from dataclasses import dataclass

@dataclass
class TrackingConfig:
    """추적 제어 설정"""
    # ===== 거리 제어 =====
    target_distance: float = 2.5  # 2.5m 유지
    distance_tolerance: float = 0.2  # ±20cm 허용
    min_safe_distance: float = 1.8
    max_track_distance: float = 8.0
    
    # ===== PID 파라미터 =====
    # 거리 제어 (전후)
    pid_distance_kp: float = 20.0  # 매우 강함 (depth는 느린 반응)
    pid_distance_ki: float = 0.5   # 누적 오차 보정
    pid_distance_kd: float = 10.0   # 미분으로 진동 감소
    
    # 수평 위치 제어 (좌우)
    pid_horizontal_kp: float = 0.5
    pid_horizontal_ki: float = 0.02
    pid_horizontal_kd: float = 0.10
    
    # 수직 위치 제어 (상하)
    pid_vertical_kp: float = 0.3
    pid_vertical_ki: float = 0.04
    pid_vertical_kd: float = 0.05
    
    # ===== 임계값 =====
    yaw_threshold: float = 0.12  # 10% 이상 오차면 회전
    lr_threshold: float = 0.20   # 15% 이상 오차면 좌우
    ud_threshold: float = 0.20   # 20% 이상 오차면 상하
    depth_threshold: float = 0.2  # ±20cm 이내면 유지
    
    # ===== 평활화 =====
    depth_history_size: int = 8  # Depth history 길이
    smoothing_alpha: float = 0.3  # EMA 평활화 계수 (0.3 = 70% 이전 값)
    
    # ===== 속도 제한 =====
    max_rc_speed: int = 50  # RC 명령 최대값
    tracking_rc_speed: int = 40
    
    # ===== 상태 관리 =====
    forward_only: bool = True  # 뒤로 안 가기
    use_ego_motion: bool = True  # Optical flow 예측 활용


class PIDController:
    """PID 컨트롤러"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, 
                 max_integral: float = 1.0, dt: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.dt = dt
        
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.derivative_error = 0.0
    
    def update(self, error: float) -> float:
        """
        PID 제어 출력 계산
        
        Args:
            error: 현재 오차
        
        Returns:
            제어 출력값
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (Anti-windup)
        self.integral_error += error * self.dt
        self.integral_error = np.clip(
            self.integral_error, 
            -self.max_integral, 
            self.max_integral
        )
        i_term = self.ki * self.integral_error
        
        # Derivative term
        self.derivative_error = (error - self.previous_error) / self.dt
        d_term = self.kd * self.derivative_error
        
        # Total output
        output = p_term + i_term + d_term
        
        self.previous_error = error
        
        return output
    
    def reset(self):
        """제어기 상태 초기화"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.derivative_error = 0.0


class DepthFilter:
    """Depth 값 안정화 필터"""
    
    def __init__(self, history_size: int = 8, alpha: float = 0.3):
        self.history = deque(maxlen=history_size)
        self.alpha = alpha  # EMA 평활화 계수
        self.smoothed_value = None
        self.last_raw_value = None  

    def update(self, depth_value: float) -> float:
        """
        Depth 값을 필터링하고 반환
        
        Args:
            depth_value: 현재 프레임의 depth
        
        Returns:
            평활화된 depth 값
        """
        # 유효 범위 필터링
        if depth_value < 0.5 or depth_value > 8.0:
            return self.smoothed_value if self.smoothed_value else depth_value
        
        self.history.append(depth_value)
        
        if len(self.history) < 3:
            self.smoothed_value = depth_value
            self.last_raw_value = depth_value
            return depth_value

        # 1. 극단값 제거 (IQR 방식)
        if len(self.history) >= 3:
            median = np.median(list(self.history))
            q1 = np.percentile(list(self.history), 25)
            q3 = np.percentile(list(self.history), 75)
            iqr = q3 - q1
            
            # IQR 기반 필터
            valid_values = [d for d in self.history 
                          if abs(d - median) < 1.2 * iqr]
        else:
            valid_values = list(self.history)

        if not valid_values:
            return self.smoothed_value if self.smoothed_value else depth_value
        
        # 2. EMA (Exponential Moving Average) 평활화
        raw_filtered = np.median(valid_values)
        
        if self.smoothed_value is None:
            self.smoothed_value = raw_filtered
        else:
            # 새 값에 더 가중치를 주려면 alpha를 크게
            self.smoothed_value = (self.alpha * raw_filtered + 
                                  (1 - self.alpha) * self.smoothed_value)
        
        return self.smoothed_value


class DroneTrackingController:
    """드론 추적 제어기 (PID + 예측 기반)"""
    
    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()
        
        # PID 컨트롤러 초기화
        self.pid_distance = PIDController(
            kp=self.config.pid_distance_kp,
            ki=self.config.pid_distance_ki,
            kd=self.config.pid_distance_kd,
            max_integral=2.0,
            dt=0.05  # 20Hz 제어
        )
        
        self.pid_horizontal = PIDController(
            kp=self.config.pid_horizontal_kp,
            ki=self.config.pid_horizontal_ki,
            kd=self.config.pid_horizontal_kd,
            max_integral=0.5,
            dt=0.05
        )
        
        self.pid_vertical = PIDController(
            kp=self.config.pid_vertical_kp,
            ki=self.config.pid_vertical_ki,
            kd=self.config.pid_vertical_kd,
            max_integral=0.5,
            dt=0.05
        )
        
        # Depth 필터
        self.depth_filter = DepthFilter(
            history_size=self.config.depth_history_size,
            alpha=self.config.smoothing_alpha
        )
        
        # 상태 추적
        self.last_valid_depth = None
        self.last_ego_velocity = None
        self.state_history = deque(maxlen=10)
        
        # 디버깅 로깅
        self.last_log_time = time.time()
        self.log_interval = 1.0  # 1초마다 로그
    
    def compute_control_command(self, 
                               frame: np.ndarray,
                               bbox: tuple,
                               depth: float,
                               ego_velocity: tuple = None,
                               frame_center: tuple = (480, 360)) -> dict:
        """
        현재 상태로부터 드론 제어 명령 생성
        
        Args:
            frame: 현재 프레임 (H, W, C)
            bbox: 타겟 바운딩박스 [x1, y1, x2, y2]
            depth: 타겟까지의 거리 (미터)
            ego_velocity: Optical flow로부터 추정한 드론 ego velocity [vx, vy]
            frame_center: 프레임 중심 좌표 (cx, cy)
        
        Returns:
            RC 제어 명령 {
                'left_right': int,
                'forward_backward': int,
                'up_down': int,
                'yaw': int,
                'state': str,
                'diagnostics': dict
            }
        """
        
        h, w = frame.shape[:2]
        cx, cy = frame_center
        
        # 타겟 정보 추출
        x1, y1, x2, y2 = bbox
        target_cx = (x1 + x2) / 2.0
        target_cy = (y1 + y2) / 2.0
        target_width = x2 - x1
        target_height = y2 - y1
        
        # ===== 1단계: Depth 필터링 =====
        smoothed_depth = self.depth_filter.update(depth)
        if smoothed_depth is not None:
            self.last_valid_depth = smoothed_depth
        
        # ===== 2단계: 위치 오차 계산 =====
        
        # 수평 오차 (정규화: -0.5 ~ 0.5)
        horizontal_error = (target_cx - cx) / w
        
        # 수직 오차 (정규화: -0.5 ~ 0.5)
        vertical_error = (target_cy - cy) / h
        
        # ===== 3단계: 거리 제어 (Depth 기반) =====
        
        if self.last_valid_depth is None:
            fb_speed = 0
            distance_state = "NO_DEPTH"
        elif self.last_valid_depth < self.config.min_safe_distance:
            fb_speed = 0
            distance_state = "TOO_CLOSE"
        elif self.last_valid_depth > self.config.max_track_distance:
            fb_speed = 0
            distance_state = "TOO_FAR"
        else:
            # 정상 범위 - PID 제어
            distance_error = self.last_valid_depth - self.config.target_distance
            
            # Ego-motion 예측 추가 (선택적)
            if self.config.use_ego_motion and ego_velocity is not None:
                vx, vy = ego_velocity  # m/s
                # 드론이 앞으로 가고 있으면 그만큼 보정
                distance_error -= vx * 0.1  # 100ms 미래 예측
            
            if abs(distance_error) <= self.config.distance_tolerance:
                fb_speed = 0
                distance_state = "HOLD"
            else:
                # PID 계산
                pid_output = self.pid_distance.update(distance_error)
                
                # Forward-only 모드
                if self.config.forward_only:
                    if pid_output > 0:
                        # 앞으로 추적 가능
                        fb_speed = int(np.clip(
                            pid_output,
                            0,
                            self.config.tracking_rc_speed
                        ))
                        distance_state = "FORWARD"
                    else:
                        # 뒤로는 안 됨 - 정지
                        fb_speed = 0
                        distance_state = "STOP_BACKWARD"
                else:
                    # 양방향 가능
                    fb_speed = int(np.clip(
                        pid_output,
                        -self.config.tracking_rc_speed,
                        self.config.tracking_rc_speed
                    ))
                    distance_state = "FORWARD" if fb_speed > 0 else "BACKWARD"
        
        # ===== 4단계: 수평 제어 (좌우) =====
        
        if abs(horizontal_error) > self.config.yaw_threshold:
            # 큰 오차 - 회전
            yaw_speed = int(np.clip(
                self.pid_horizontal.update(horizontal_error) * 100,
                -self.config.tracking_rc_speed,
                self.config.tracking_rc_speed
            ))
            lr_speed = 0
            horizontal_state = "YAW"
        elif abs(horizontal_error) > self.config.lr_threshold:
            # 작은 오차 - 평행이동
            yaw_speed = 0
            lr_speed = int(np.clip(
                self.pid_horizontal.update(horizontal_error) * 100,
                -self.config.tracking_rc_speed,
                self.config.tracking_rc_speed
            ))
            horizontal_state = "STRAFE"
        else:
            # 중앙 정렬
            yaw_speed = 0
            lr_speed = 0
            self.pid_horizontal.reset()
            horizontal_state = "CENTER"
        
        # ===== 5단계: 수직 제어 (상하) =====
    
        # 1. 거리가 가까우면 하강 완전 금지
        is_too_close_for_down = (self.last_valid_depth is not None and 
                                self.last_valid_depth < 1.8)
        
        if abs(vertical_error) > self.config.ud_threshold:
            # PID 계산
            ud_output = self.pid_vertical.update(vertical_error) * 100
            
            # 2. 하강 속도 제한 (핵심!)
            if ud_output < 0:  # 하강 명령일 때
                if is_too_close_for_down:
                    # 가까우면 완전 금지
                    ud_speed = 0
                    vertical_state = "LOCKED_LOW"
                else:
                    # 멀어도 하강은 최대 -15로 제한!
                    ud_speed = int(max(ud_output, -15))
                    vertical_state = "ADJUST_LIMITED"
            else:
                # 상승은 자유롭게
                ud_speed = int(np.clip(
                    ud_output,
                    -self.config.tracking_rc_speed,
                    self.config.tracking_rc_speed
                ))
                vertical_state = "ADJUST"
        else:
            ud_speed = 0
            self.pid_vertical.reset()
            vertical_state = "LEVEL"
        
        # ===== 6단계: 상태 저장 =====
        
        state_snapshot = {
            'timestamp': time.time(),
            'distance': self.last_valid_depth,
            'horizontal_error': horizontal_error,
            'vertical_error': vertical_error,
            'distance_state': distance_state,
            'horizontal_state': horizontal_state,
            'vertical_state': vertical_state,
        }
        self.state_history.append(state_snapshot)
        
        # ===== 최종 RC 명령 =====
        
        return {
            'left_right': lr_speed,
            'forward_backward': fb_speed,
            'up_down': ud_speed,
            'yaw': yaw_speed,
            'state': f"{distance_state}|{horizontal_state}|{vertical_state}",
            'diagnostics': {
                'depth': self.last_valid_depth,
                'horizontal_error': horizontal_error,
                'vertical_error': vertical_error,
                'ego_velocity': ego_velocity,
            }
        }
    
    def get_state_string(self) -> str:
        """현재 상태를 문자열로 반환"""
        if not self.state_history:
            return "INIT"
        
        latest = self.state_history[-1]
        return (f"D:{latest['distance']:.2f}m | "
                f"H:{latest['horizontal_error']:+.3f} | "
                f"V:{latest['vertical_error']:+.3f} | "
                f"State:{latest['distance_state']}/{latest['horizontal_state']}")
    
    def reset(self):
        """제어기 초기화"""
        self.pid_distance.reset()
        self.pid_horizontal.reset()
        self.pid_vertical.reset()
        self.depth_filter = DepthFilter(
            history_size=self.config.depth_history_size,
            alpha=self.config.smoothing_alpha
        )
        self.last_valid_depth = None
        self.state_history.clear()
