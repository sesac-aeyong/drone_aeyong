# control_fusion.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


# ───────────────────────────────────────────────────────────────────────────────
# Core: bbox + pose + optical-flow 융합 RC 제어기
# ───────────────────────────────────────────────────────────────────────────────
class ControlFusion:
    """
    bbox 중심/면적 기반 기본 P제어에
    - pose 스케일(어깨/척추) 로그 오차 보조
    - optical flow (vx, vy) 소량 피드포워드
    를 섞어 RC(lr, fb, ud, yaw)를 산출한다.
    """

    def __init__(
        self,
        tracking_rc_speed: int = 30,
        yaw_gain: float = 0.80,
        lr_gain: float = 0.80,
        ud_gain: float = 0.40,
        fb_gain: float = 200.0,
        yaw_thresh: float = 0.20,
        lr_thresh: float = 0.05,
        ud_thresh: float = 0.05,
        size_deadband: float = 0.025,
        pose_gain: float = 140.0,
        pose_qmin: float = 0.25,
        flow_ff_gain: float = 0.20,
    ):
        self.vmax = int(tracking_rc_speed)
        self.g_yaw, self.g_lr, self.g_ud, self.g_fb = yaw_gain, lr_gain, ud_gain, fb_gain
        self.t_yaw, self.t_lr, self.t_ud = yaw_thresh, lr_thresh, ud_thresh
        self.t_size = size_deadband
        self.pose_gain = pose_gain
        self.pose_qmin = pose_qmin
        self.flow_ff_gain = flow_ff_gain  # yaw/lr/ud에 곱해줄 소량 피드포워드 계수

    @staticmethod
    def _area_ratio(bb, w, h):
        x1, y1, x2, y2 = bb
        a = max(0, x2 - x1) * max(0, y2 - y1)
        return a / max(1, w * h)

    @staticmethod
    def _center_error(bb, w, h):
        x1, y1, x2, y2 = bb
        cx, cy = w // 2, h // 2
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        return ((tx - cx) / max(1, w), (ty - cy) / max(1, h))  # -0.5~0.5

    @staticmethod
    def _clip(v, vmax):
        return int(np.clip(v, -vmax, vmax))

    def compute_rc(
        self,
        frame_shape,
        bbox,
        pose_dict: Optional[Dict[str, Any]] = None,  # {'shoulder':{'ref':v,'ema':v}, 'spine':{...}, 'quality':0~1}
        flow_vec: Optional[Tuple[float, float]] = None,  # (vx, vy) in px/frame
        size_target_range: Tuple[float, float] = (0.20, 0.30),
        obstacle_brake: bool = False,
    ):
        """
        returns (lr, fb, ud, yaw)
        - size_target_range: (low, high) → 중앙값을 목표로 deadband 적용
        - obstacle_brake: True면 fb=0 강제(가까운 장애물 감지 시)
        """
        h, w = frame_shape[:2]
        err_x, err_y = self._center_error(bbox, w, h)
        ratio = self._area_ratio(bbox, w, h)
        target = 0.5 * (size_target_range[0] + size_target_range[1])
        err_size = target - ratio

        # yaw / lr
        if abs(err_x) > self.t_yaw:
            yaw = self._clip(err_x * self.g_yaw * 100, self.vmax)
            lr = 0
        elif abs(err_x) > self.t_lr:
            yaw = 0
            lr = self._clip(err_x * self.g_lr * 100, self.vmax)
        else:
            yaw, lr = 0, 0

        # ud
        if abs(err_y) > self.t_ud:
            ud = self._clip(-err_y * self.g_ud * 100, self.vmax)
        else:
            ud = 0

        # fb: 면적 + (옵션) pose 스케일
        fb_from_size = 0
        if abs(err_size) > self.t_size:
            fb_from_size = self._clip(err_size * self.g_fb, self.vmax)

        fb_from_pose = 0
        if pose_dict is not None and float(pose_dict.get("quality", 1.0)) >= self.pose_qmin:
            terms = []
            for k in ("shoulder", "spine"):
                d = pose_dict.get(k)
                if not d:
                    continue
                ref, ema = d.get("ref"), d.get("ema")
                if ref and ema and ref > 1e-6 and ema > 1e-6:
                    # 현재가 커지면(가까워짐) log(ema/ref)>0 → 뒤로
                    terms.append(np.log(ema / ref) * self.pose_gain)
            if terms:
                fb_from_pose = self._clip(np.median(terms), self.vmax)

        # depth를 쓰지 않겠다는 요구조건 → fb는 size+pose로만 구성
        # 가중치는 0.6(size) / 0.4(pose)로 시작(튜닝 지점)
        fb = self._clip(0.6 * fb_from_size + 0.4 * fb_from_pose, self.vmax)

        # flow 피드포워드(소량): 화면 이동을 약간 선반영
        if flow_vec is not None:
            vx, vy = flow_vec
            yaw += self._clip(self.flow_ff_gain * vx, self.vmax)
            lr += self._clip(self.flow_ff_gain * vx, self.vmax)   # 좌우엔 동일 상수로 시작(튜닝 지점)
            ud += self._clip(-self.flow_ff_gain * vy, self.vmax)

        if obstacle_brake:
            fb = 0

        return int(lr), int(fb), int(ud), int(yaw)


# ───────────────────────────────────────────────────────────────────────────────
# Tracking 유틸 (선택 로직 / bbox 클리핑)
# ───────────────────────────────────────────────────────────────────────────────
def clip_bbox_to_frame(bb, w, h):
    if bb is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bb]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return [x1, y1, x2, y2] if (x2 > x1 and y2 > y1) else None


def select_thief_candidate(detections):
    """
    thief_dist <= thief_cos_dist 를 만족하는 후보 중
    td가 최소인 것 하나만 반환.
    """
    best, best_td = None, 1e9
    if not detections:
        return None

    for det in detections:
        get = det.get if isinstance(det, dict) else (lambda k, d=None: getattr(det, k, d))
        td = get("thief_dist")
        gate = get("thief_cos_dist")

        if td is None or gate is None:
            continue
        if td <= gate and td < best_td:
            best, best_td = det, td

    return best


# ───────────────────────────────────────────────────────────────────────────────
# Optical Flow (척추 스트립 기반)
# ───────────────────────────────────────────────────────────────────────────────
def compute_flow_from_spine_strip(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
) -> Optional[Tuple[float, float]]:
    """
    bbox 중앙 '척추 스트립'에서만 LK flow를 계산해 (vx,vy) 반환.
    팔/경계 노이즈 억제용. 중앙 20% 폭, 5x7 격자, 20~80% 절사평균.
    """
    if prev_gray is None or gray is None or bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    cx = (x1 + x2) // 2
    half = max(2, (x2 - x1) // 10)  # ~20%
    xs1, xs2 = max(0, cx - half), min(w - 1, cx + half)

    xs = np.linspace(xs1 + 3, xs2 - 3, 5, dtype=np.float32)
    ys = np.linspace(max(0, y1 + 6), min(h - 1, y2 - 6), 7, dtype=np.float32)
    if xs.size == 0 or ys.size == 0:
        return None

    pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if p1 is None or st is None:
        return None
    good = st.squeeze() == 1
    if not np.any(good):
        return None

    dx = p1[good, :, 0] - pts[good, :, 0]
    dy = p1[good, :, 1] - pts[good, :, 1]

    def robust_mean(v):
        v = v.flatten()
        p20, p80 = np.percentile(v, [20, 80])
        v = v[(v >= p20) & (v <= p80)]
        return float(np.mean(v)) if v.size > 0 else float(np.median(v))

    return (robust_mean(dx), robust_mean(dy))


# ───────────────────────────────────────────────────────────────────────────────
# Pose 스케일 상태 업데이트(어깨/척추)
# ───────────────────────────────────────────────────────────────────────────────
def update_pose_stats(
    pose: Optional[Dict[str, Any]],
    quality: float,
    should_ref: Optional[float],
    should_ema: Optional[float],
    spine_ref: Optional[float],
    spine_ema: Optional[float],
    alpha: float = 0.25,
):
    """
    포즈 품질/척추/어깨 레퍼런스 및 EMA를 업데이트하여 반환.
    pose 예시: {'shoulder': px, 'spine': px, 'quality': 0~1}
    """
    if pose is None:
        return quality, should_ref, should_ema, spine_ref, spine_ema

    q = float(pose.get("quality", 0.0) or 0.0)
    quality = q

    sh = pose.get("shoulder")
    sp = pose.get("spine")

    if should_ref is None and sh:
        should_ref = float(sh)
        should_ema = float(sh)
    if spine_ref is None and sp:
        spine_ref = float(sp)
        spine_ema = float(sp)

    if sh:
        should_ema = (1 - alpha) * (should_ema if should_ema is not None else sh) + alpha * float(sh)
    if sp:
        spine_ema = (1 - alpha) * (spine_ema if spine_ema is not None else sp) + alpha * float(sp)

    return quality, should_ref, should_ema, spine_ref, spine_ema


# ───────────────────────────────────────────────────────────────────────────────
# Depth(상대) 기반 표시 & 브레이크 판단
# ───────────────────────────────────────────────────────────────────────────────
def spine_depth_mode_and_brake(
    depth_map: Optional[np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]],
    use_brake: bool = True,
) -> Tuple[Optional[float], bool]:
    """
    척추 주변(중앙 1/3폭) depth 최빈값(mode)과 '중앙 전방 근접 브레이크' 플래그를 계산.
    depth는 상대값이므로 임계는 경험적으로 조정.
    """
    if depth_map is None or bbox is None:
        return None, False
    x1, y1, x2, y2 = bbox
    H, W = depth_map.shape[:2]

    # 척추 주변 모드(depth)
    cx = (x1 + x2) // 2
    w3 = max(2, (x2 - x1) // 6)
    xs1, xs2 = max(0, cx - w3), min(W - 1, cx + w3)
    roi = depth_map[max(0, y1) : min(H - 1, y2), xs1:xs2]
    mode = None
    if roi.size > 0:
        hist, bin_edges = np.histogram(roi.flatten(), bins=32)
        idx = int(hist.argmax())
        mode = float(0.5 * (bin_edges[idx] + bin_edges[idx + 1]))

    # 중앙 전방 근접 브레이크
    brake = False
    if use_brake:
        strip = depth_map[:, W // 2 - W // 16 : W // 2 + W // 16]  # 화면 중앙 1/8 폭
        if strip.size > 0:
            dmin = float(np.percentile(strip, 5))  # 아주 가까운 물체
            if dmin < 0.15:  # 튠 포인트
                brake = True

    return mode, brake


# ───────────────────────────────────────────────────────────────────────────────
# 타깃 유실 시 재획득 전략 (flow 기반 회전/좌우 스윕)
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class SearchParams:
    base_yaw: int = 22          # flow 기반 검색 yaw 속도
    sweep_yaw: int = 18         # 좌↔우 스윕 yaw 속도
    flow_thresh: float = 0.3    # flow vx 임계
    flow_spin_time: float = 1.2 # flow 방향으로 도는 시간
    sweep_time: float = 0.8     # 좌/우 한 사이클 시간


class SearchManager:
    """
    타겟 유실 시 '마지막 flow 방향'으로 짧게 도는 검색.
    flow 없으면 좌↔우 스윕. 재획득 시 reset() 호출 필요.
    """

    def __init__(self, params: SearchParams = SearchParams()):
        self.p = params
        self.active = False
        self.until = 0.0
        self.yaw = 0
        self._sweep_dir = 1

    def start(self, flow_vx: float, now: float):
        if abs(flow_vx) > self.p.flow_thresh:
            self.yaw = self.p.base_yaw if flow_vx > 0 else -self.p.base_yaw
            self.until = now + self.p.flow_spin_time
        else:
            self._sweep_dir *= -1
            self.yaw = self._sweep_dir * self.p.sweep_yaw
            self.until = now + self.p.sweep_time
        self.active = True

    def command(self, now: float) -> Optional[int]:
        """활성 상태면 현재 yaw 명령을, 끝났으면 None."""
        if not self.active:
            return None
        if now < self.until:
            return self.yaw
        # 끝
        self.active = False
        return None

    def reset(self):
        self.active = False
        self.until = 0.0
        self.yaw = 0


__all__ = [
    "ControlFusion",
    "clip_bbox_to_frame",
    "select_thief_candidate",
    "compute_flow_from_spine_strip",
    "update_pose_stats",
    "spine_depth_mode_and_brake",
    "SearchParams",
    "SearchManager",
]
