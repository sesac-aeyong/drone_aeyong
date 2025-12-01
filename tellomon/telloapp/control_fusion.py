# control_fusion.py
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Dict, Any, List
import numpy as np
import cv2

# ───────────────────────────────────────────────────────────────────────────────
# Feature gates (ON/OFF + alpha)
# ───────────────────────────────────────────────────────────────────────────────
def want_depth(features: Dict[str, Any]) -> bool:
    return bool(features.get('depth', False))

def want_pose(features: Dict[str, Any]) -> bool:
    return bool(features.get('pose', False))

def want_flow(features: Dict[str, Any]) -> bool:
    return bool(features.get('flow', False))

def feature_alpha(features: Dict[str, Any], default: float = 0.5) -> float:
    try:
        a = float(features.get('alpha', default))
    except Exception:
        a = default
    return 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

# ───────────────────────────────────────────────────────────────────────────────
# Pose → RC 보조 입력 준비
# ───────────────────────────────────────────────────────────────────────────────
def prepare_pose_for_rc(
    use_pose: bool,
    pose_quality: float,
    should_ref: Optional[float],
    should_ema: Optional[float],
    spine_ref: Optional[float],
    spine_ema: Optional[float],
) -> Optional[Dict[str, Any]]:
    if not use_pose:
        return None
    has_any = any(v is not None for v in (should_ref, should_ema, spine_ref, spine_ema))
    if not has_any:
        return None
    return {
        'quality': float(pose_quality),
        'shoulder': {'ref': should_ref, 'ema': should_ema} if should_ref is not None or should_ema is not None else None,
        'spine':    {'ref': spine_ref,  'ema': spine_ema}  if spine_ref  is not None or spine_ema  is not None else None,
    }

# ───────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ───────────────────────────────────────────────────────────────────────────────
def depth_to_vis(depth_map: np.ndarray) -> np.ndarray:
    """2D depth → 3ch 컬러맵(BGR). 이미 3채널이면 그대로 반환."""
    if depth_map is None:
        return None
    if depth_map.ndim == 2:
        dm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(dm, cv2.COLORMAP_TURBO)  # BGR
    return depth_map  # already 3ch

def overlay_blend(base_bgr: np.ndarray, layer_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    if layer_bgr is None or alpha <= 0.0:
        return base_bgr
    if base_bgr.shape != layer_bgr.shape:
        layer_bgr = cv2.resize(layer_bgr, (base_bgr.shape[1], base_bgr.shape[0]))
    return cv2.addWeighted(base_bgr, 1.0 - alpha, layer_bgr, alpha, 0)

# 9 keypoints (COCO indices)
_FACE      = [0, 1, 2, 3, 4]     # nose, L eye, R eye, L ear, R ear
_SHOULDER  = [5, 6]              # L/R shoulders
_HIP       = [11, 12]            # L/R hips
_KP9       = _FACE + _SHOULDER + _HIP
# minimal limbs only among those 9
_LIMBS_9 = [
    (0, 1), (0, 2),     # nose→eyes
    (1, 3), (2, 4),     # eyes→ears
    (5, 6),             # shoulder line
    (11, 12),           # hip line
    (5, 11), (6, 12),   # shoulders→hips
    (0, 5), (0, 6),     # nose→shoulders
]
def overlay_pose_points_min9(
    img,
    should_ema=None, spine_ema=None, alpha: float = 0.0,
    pose_kpts=None,            # [[x,y,conf_uint8], ...] length >= 13 (COCO)
    conf_thresh: int = 60,
    draw_indices: bool = False,
    draw_midpoints: bool = True
):
    """
    얼굴(5) + 어깨(2) + 힙(2) = 9점만 표시하고, 해당 점들만 연결합니다.
    confidence는 0~255 정수 가정, conf_thresh 이상만 그립니다.
    """
    if pose_kpts is None or len(pose_kpts) < 17:
        return img

    h, w = img.shape[:2]

    def _pt_ok(i):
        if i >= len(pose_kpts): return False
        kp = pose_kpts[i]
        if not kp or len(kp) < 3: return False
        try:
            return int(kp[2]) > conf_thresh
        except:
            return False

    def _pt_xy(i):
        kp = pose_kpts[i]
        x = max(0, min(w - 1, int(kp[0])))
        y = max(0, min(h - 1, int(kp[1])))
        return (x, y)

    # points
    for i in _KP9:
        if _pt_ok(i):
            x, y = _pt_xy(i)
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            if draw_indices:
                cv2.putText(img, str(i), (x+2, y+2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)

    # limbs
    for a, b in _LIMBS_9:
        if _pt_ok(a) and _pt_ok(b):
            cv2.line(img, _pt_xy(a), _pt_xy(b), (255, 0, 0), 2)

    # midpoints (어깨/힙 중심은 제어 시 유용)
    if draw_midpoints:
        try:
            if _pt_ok(5) and _pt_ok(6):
                sx = (_pt_xy(5)[0] + _pt_xy(6)[0]) // 2
                sy = (_pt_xy(5)[1] + _pt_xy(6)[1]) // 2
                cv2.circle(img, (sx, sy), 4, (255, 0, 0), -1, cv2.LINE_AA)
            if _pt_ok(11) and _pt_ok(12):
                hx = (_pt_xy(11)[0] + _pt_xy(12)[0]) // 2
                hy = (_pt_xy(11)[1] + _pt_xy(12)[1]) // 2
                cv2.circle(img, (hx, hy), 4, (0, 0, 255), -1, cv2.LINE_AA)
        except:
            pass

    return img

def overlay_flow_arrow(base_bgr: np.ndarray,
                       bbox: Optional[Tuple[int,int,int,int]],
                       flow_vec: Optional[Tuple[float,float]],
                       alpha: float = 0.5) -> np.ndarray:
    if bbox is None or flow_vec is None:
        return base_bgr
    (x1,y1,x2,y2) = map(int, bbox)
    cx, cy = (x1 + x2)//2, (y1 + y2)//2
    vx, vy = flow_vec
    layer = np.zeros_like(base_bgr)
    tip = (int(cx + vx), int(cy + vy))
    try:
        cv2.arrowedLine(layer, (cx, cy), tip, (0, 200, 255), 3, tipLength=0.35)
        return overlay_blend(base_bgr, layer, alpha)
    except Exception:
        return base_bgr

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
        fb = self._clip(0.6 * fb_from_size + 0.4 * fb_from_pose, self.vmax)

        # flow 피드포워드(소량): 화면 이동을 약간 선반영
        if flow_vec is not None:
            vx, vy = flow_vec
            yaw += self._clip(self.flow_ff_gain * vx, self.vmax)
            lr  += self._clip(self.flow_ff_gain * vx, self.vmax)
            ud  += self._clip(-self.flow_ff_gain * vy, self.vmax)

        if obstacle_brake:
            fb = 0

        return int(lr), int(fb), int(ud), int(yaw)

# ───────────────────────────────────────────────────────────────────────────────
# Tracking 유틸 (선택 로직 / bbox 클리핑 / iid filter)
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


# ───────────────────────────────────────────────────────────────────────────────
# Optical Flow (척추 스트립 기반)
# ───────────────────────────────────────────────────────────────────────────────
def compute_flow_from_spine_strip(prev_gray: np.ndarray,
                                  gray: np.ndarray,
                                  bbox: Tuple[int, int, int, int],
                                  frame_shape: Tuple[int, int, int]) -> Optional[Tuple[float, float]]:
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

# COCO 17 keypoint indices
KP = {"LShoulder": 5, "RShoulder": 6, "LHip": 11, "RHip": 12}

def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _mid(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))

def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = int(0.95 * (len(vs) - 1))
    return float(vs[idx])

def _estimate_conf_scale_uint8(confs: List[float],
                               default_target: float = 200.0,
                               upper_bound: float = 255.0) -> float:
    """
    uint8(0~255) 가정. 관측값의 p95와 경험적 상한(≈200)을 결합해 스케일 결정.
    scale = clamp(max(default_target, p95), 1.0, 255.0)
    """
    if not confs:
        return default_target
    p95 = _p95(confs)
    scale = max(default_target, p95)
    scale = min(scale, upper_bound)
    return max(1.0, scale)

def _pose_list_to_metrics(pose_list: Sequence[Sequence[float]],
                          conf_thr_norm: float = 0.05) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Returns:
      quality: 정규화된 c의 평균 (없으면 0.0)
      shoulder: 좌/우 어깨 거리 (없으면 None)
      spine: 어깨중점 ↔ 엉덩이중점 거리 (없으면 None)
    """
    # 1) 스케일 추정 (uint8 0~255, 경험상 200 근처)
    raw_confs: List[float] = []
    for kp in pose_list:
        if kp and len(kp) >= 3 and kp[2] is not None:
            raw_confs.append(float(kp[2]))
    scale = _estimate_conf_scale_uint8(raw_confs, default_target=200.0, upper_bound=255.0)

    # 2) 정규화된 confidence 수집 (0~1, 상한 1.0)
    confs_norm: List[float] = []
    for kp in pose_list:
        if kp and len(kp) >= 3 and kp[2] is not None:
            c_norm = min(float(kp[2]) / scale, 1.0)
            confs_norm.append(c_norm)
    quality = float(sum(confs_norm) / len(confs_norm)) if confs_norm else 0.0

    # 3) 좌표 가져오기 (정규화 임계 적용)
    def _get_kp(idx: int) -> Optional[Tuple[float, float, float]]:
        try:
            x, y, c_raw = pose_list[idx]
            if c_raw is None:
                return None
            c_norm = min(float(c_raw) / scale, 1.0)
            if c_norm >= conf_thr_norm:
                return float(x), float(y), c_norm
        except Exception:
            pass
        return None

    lsh = _get_kp(KP["LShoulder"])
    rsh = _get_kp(KP["RShoulder"])
    lhip = _get_kp(KP["LHip"])
    rhip = _get_kp(KP["RHip"])

    shoulder = None
    spine = None
    shoulder_mid = None

    if lsh and rsh:
        shoulder = _dist((lsh[0], lsh[1]), (rsh[0], rsh[1]))
        shoulder_mid = _mid((lsh[0], lsh[1]), (rsh[0], rsh[1]))

    if shoulder_mid and lhip and rhip:
        hip_mid = _mid((lhip[0], lhip[1]), (rhip[0], rhip[1]))
        spine = _dist(shoulder_mid, hip_mid)

    return quality, shoulder, spine

def update_pose_stats(pose: Optional[Any],
                      quality: float,
                      should_ref: Optional[float],
                      should_ema: Optional[float],
                      spine_ref: Optional[float],
                      spine_ema: Optional[float],
                      alpha: float = 0.25):
    """
    pose: dict(기존) 또는 [[x,y,c_uint8], ...] (COCO 17)
    uint8 confidence는 내부에서 0~1 정규화하여 사용.
    """
    if pose is None:
        return quality, should_ref, should_ema, spine_ref, spine_ema

    if isinstance(pose, dict):
        q = float(pose.get("quality", 0.0) or 0.0)
        sh = pose.get("shoulder"); sh = float(sh) if sh is not None else None
        sp = pose.get("spine");    sp = float(sp) if sp is not None else None
    else:
        try:
            q, sh, sp = _pose_list_to_metrics(pose, conf_thr_norm=0.05)
        except Exception:
            q, sh, sp = 0.0, None, None

    quality = float(q)

    if should_ref is None and sh is not None:
        should_ref = float(sh); should_ema = float(sh)
    if spine_ref is None and sp is not None:
        spine_ref = float(sp);   spine_ema = float(sp)

    if sh is not None:
        base = should_ema if should_ema is not None else float(sh)
        should_ema = (1.0 - alpha) * base + alpha * float(sh)
    if sp is not None:
        base = spine_ema if spine_ema is not None else float(sp)
        spine_ema = (1.0 - alpha) * base + alpha * float(sp)

    return quality, should_ref, should_ema, spine_ref, spine_ema


# def update_pose_stats(pose: Optional[Dict[str, Any]],
#                       quality: float,
#                       should_ref: Optional[float],
#                       should_ema: Optional[float],
#                       spine_ref: Optional[float],
#                       spine_ema: Optional[float],
#                       alpha: float = 0.25):
#     if pose is None:
#         return quality, should_ref, should_ema, spine_ref, spine_ema
#     q = float(pose.get("quality", 0.0) or 0.0)
#     quality = q
#     sh = pose.get("shoulder")
#     sp = pose.get("spine")
#     if should_ref is None and sh:
#         should_ref = float(sh); should_ema = float(sh)
#     if spine_ref is None and sp:
#         spine_ref = float(sp); spine_ema = float(sp)
#     if sh:
#         should_ema = (1 - alpha) * (should_ema if should_ema is not None else sh) + alpha * float(sh)
#     if sp:
#         spine_ema = (1 - alpha) * (spine_ema if spine_ema is not None else sp) + alpha * float(sp)
#     return quality, should_ref, should_ema, spine_ref, spine_ema

# ───────────────────────────────────────────────────────────────────────────────
# Depth(상대) 기반 표시 & 브레이크 판단
# ───────────────────────────────────────────────────────────────────────────────
def spine_depth_mode_and_brake(depth_map: Optional[np.ndarray],
                               bbox: Optional[Tuple[int, int, int, int]],
                               use_brake: bool = True) -> Tuple[Optional[float], bool]:
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
    base_yaw: int = 22
    sweep_yaw: int = 18
    flow_thresh: float = 0.3
    flow_spin_time: float = 1.2
    sweep_time: float = 0.8

class SearchManager:
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
        if not self.active:
            return None
        if now < self.until:
            return self.yaw
        self.active = False
        return None
    def reset(self):
        self.active = False
        self.until = 0.0
        self.yaw = 0