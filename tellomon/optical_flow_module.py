# optical_flow_module.py
import numpy as np
import cv2
from typing import Tuple, Optional

class OpticalFlowEgoMotion:
    """배경 Optical Flow로부터 드론의 Ego-motion 추정"""
    
    def __init__(self, use_calibration=True):
        """
        DJI Tello 카메라 파라미터 초기화
        """
        if use_calibration:
            # 당신의 Calibration 결과 (Undistorted)
            self.fx = 911.77
            self.fy = 909.18
            self.cx = 473.64
            self.cy = 345.08
            
            source = "Calibration (Undistorted)"
        else:
            # FOV 기반 추정 (백업용)
            w, h = 960, 720
            fov_rad = np.radians(110)
            self.fx = (w / 2) / np.tan(fov_rad / 2)
            self.fy = self.fx
            self.cx = w / 2
            self.cy = h / 2
            
            source = "FOV-based estimation"
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray = None
        self.prev_points = None
        
        print(f"✅ Optical Flow initialized with {source}")
        print(f"   fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
    
    def compute_sparse_optical_flow(
        self, 
        curr_frame: np.ndarray, 
        background_mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Sparse optical flow 계산 (Lucas-Kanade)"""
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
        
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = curr_gray
            mask_for_corners = background_mask.astype(np.uint8)
            self.prev_points = cv2.goodFeaturesToTrack(
                curr_gray,
                mask=mask_for_corners,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7
            )
            return None
        
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_gray = curr_gray
            self.prev_points = None
            return None
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        if next_points is None:
            self.prev_gray = curr_gray
            self.prev_points = None
            return None
        
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        self.prev_gray = curr_gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return good_new, good_old
    
    def estimate_ego_velocity(
        self,
        good_new: np.ndarray,
        good_old: np.ndarray,
        depth_map: np.ndarray,
        dt: float = 0.033  # 30fps 기준
    ) -> Optional[Tuple[float, float]]:
        """배경 optical flow로부터 드론 ego-velocity 추정"""
        if good_new is None or good_old is None or len(good_new) < 10:
            return None
        
        # ===== 수정: depth_map 차원 확인 및 처리 =====
        if len(depth_map.shape) == 3:
            # 3차원인 경우 (h, w, c) → (h, w)로 변환
            if depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]
            else:
                # 여러 채널이 있으면 평균 사용
                depth_map = np.mean(depth_map, axis=2)
        
        h, w = depth_map.shape  # 이제 안전하게 2차원
        
        flow_vectors = good_new - good_old
        
        depths = []
        valid_flows = []
        
        for i in range(len(good_new)):
            x, y = int(good_new[i, 0]), int(good_new[i, 1])
            
            if 0 <= x < w and 0 <= y < h:
                z = depth_map[y, x]
                if 0.1 < z < 10.0:
                    depths.append(z)
                    valid_flows.append(flow_vectors[i])
        
        if len(depths) < 10:
            return None
        
        depths = np.array(depths)
        valid_flows = np.array(valid_flows)
        
        flow_x = valid_flows[:, 0]
        flow_y = valid_flows[:, 1]
        
        # ego_velocity = -optical_flow * depth / (focal_length * dt)
        ego_vx_samples = -flow_x * depths / (self.fx * dt)
        ego_vy_samples = -flow_y * depths / (self.fy * dt)
        
        # IQR 기반 이상치 제거
        q1_x, q3_x = np.percentile(ego_vx_samples, [25, 75])
        iqr_x = q3_x - q1_x
        mask_x = (ego_vx_samples >= q1_x - 1.5*iqr_x) & (ego_vx_samples <= q3_x + 1.5*iqr_x)
        
        q1_y, q3_y = np.percentile(ego_vy_samples, [25, 75])
        iqr_y = q3_y - q1_y
        mask_y = (ego_vy_samples >= q1_y - 1.5*iqr_y) & (ego_vy_samples <= q3_y + 1.5*iqr_y)
        
        inlier_mask = mask_x & mask_y
        
        if np.sum(inlier_mask) < 5:
            ego_vx = np.median(ego_vx_samples)
            ego_vy = np.median(ego_vy_samples)
        else:
            ego_vx = np.median(ego_vx_samples[inlier_mask])
            ego_vy = np.median(ego_vy_samples[inlier_mask])
        
        return float(ego_vx), float(ego_vy)
    
    def reset(self):
        """상태 초기화"""
        self.prev_gray = None
        self.prev_points = None
