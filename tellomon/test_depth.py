import numpy as np
import math
import cv2

class DepthToAbsolute:
    def __init__(self):
        # Camera intrinsics
        self.fx = 922.837110
        self.fy = 922.837110
        self.cx = 480.0
        self.cy = 360.0

        self.width = 960
        self.height = 720

        # Distortion coefficients (네가 제공한 값)
        self.dist = np.array([-0.137695, 0.330662, -0.010519, 0.005196, 0.0], dtype=np.float64)

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0,     0,      1    ]
        ], dtype=np.float64)

    # ----------------------------------------------------------
    # 1) Drone attitude
    # ----------------------------------------------------------
    def get_drone_attitude(self, tello):
        roll  = math.radians(tello.get_roll())
        pitch = math.radians(tello.get_pitch())
        yaw   = math.radians(tello.get_yaw())
        return roll, pitch, yaw

    # ----------------------------------------------------------
    # 2) Rotation matrix
    # ----------------------------------------------------------
    def rotation_matrix(self, roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])

        Ry = np.array([
            [ math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    # ----------------------------------------------------------
    # 3) Pixel → normalized camera ray (undistortion included)
    # ----------------------------------------------------------
    def pixel_to_ray(self, u, v):
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.dist, P=None)
        x, y = und[0, 0]
        ray = np.array([x, y, 1.0])
        return ray / np.linalg.norm(ray)

    # ----------------------------------------------------------
    # 4) Ray-world-ground intersection
    # ----------------------------------------------------------
    def ray_ground_intersection(self, ray_world, drone_height):
        vz = ray_world[2]
        if abs(vz) < 1e-9:
            return None

        t = drone_height / vz
        if t <= 0:
            return None

        point = t * ray_world
        dist = np.linalg.norm(point)
        return dist

    # ----------------------------------------------------------
    # 5) Scale = real_ground_distance / relative_ground_depth
    # ----------------------------------------------------------
    def compute_scale(self, tello, depth_map, ground_pixel):
        u, v = ground_pixel

        # 1) 상대 거리
        d_rel_ground = depth_map[int(v), int(u)]
        if d_rel_ground <= 0:
            return None

        # 2) ray
        ray_cam = self.pixel_to_ray(u, v)
        roll, pitch, yaw = self.get_drone_attitude(tello)
        R = self.rotation_matrix(roll, pitch, yaw)
        ray_world = R @ ray_cam

        # 3) 드론 높이
        H = tello.get_distance_tof() / 100.0  # meters

        # 4) 실제 바닥까지 절대거리 계산
        D_ground = self.ray_ground_intersection(ray_world, H)
        if D_ground is None:
            return None

        # 5) 스케일 계산
        scale = D_ground / d_rel_ground
        return scale

    # ----------------------------------------------------------
    # 6) Target bbox → absolute distance (scale 적용)
    # ----------------------------------------------------------
    def compute_absolute_distance(self, depth_map, bbox, scale):
        x1, y1, x2, y2 = map(int, bbox)
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        d_rel_target = float(np.median(crop))
        D_target = scale * d_rel_target
        return D_target


converter = DepthToAbsolute()

# 1) 바닥 픽셀을 지정 (예: 바닥 세그멘테이션 등으로 찾은 좌표)
ground_pixel = (450, 650)

# 2) scale 계산
scale = converter.compute_scale(tello, depth_map, ground_pixel)
if scale is None:
    print("Failed to compute scale")
    exit()

print("Scale factor =", scale)

# 3) 도둑 bbox
bbox = (x1, y1, x2, y2)

# 4) 절대거리 계산
absolute_distance = converter.compute_absolute_distance(depth_map, bbox, scale)
print("Absolute distance to target =", absolute_distance, "meters")
