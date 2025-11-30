# control_fusion.py
import numpy as np

class ControlFusion:
    """
    bbox 중심/면적 기반 기본 P제어에
    - pose 스케일(어깨/척추) 로그 오차 보조
    - optical flow (vx, vy) 소량 피드포워드
    를 섞어 RC(lr, fb, ud, yaw)를 산출한다.
    """
    def __init__(self,
                 tracking_rc_speed=30,
                 yaw_gain=0.80, lr_gain=0.80, ud_gain=0.40, fb_gain=200,
                 yaw_thresh=0.20, lr_thresh=0.05, ud_thresh=0.05, size_deadband=0.025,
                 pose_gain=140.0, pose_qmin=0.25, flow_ff_gain=0.20):
        self.vmax = int(tracking_rc_speed)
        self.g_yaw, self.g_lr, self.g_ud, self.g_fb = yaw_gain, lr_gain, ud_gain, fb_gain
        self.t_yaw, self.t_lr, self.t_ud = yaw_thresh, lr_thresh, ud_thresh
        self.t_size = size_deadband
        self.pose_gain = pose_gain
        self.pose_qmin = pose_qmin
        self.flow_ff_gain = flow_ff_gain  # yaw/lr에 곱해줄 소량 피드포워드 계수

    @staticmethod
    def _area_ratio(bb, w, h):
        x1,y1,x2,y2 = bb
        a = max(0, x2-x1) * max(0, y2-y1)
        return a / max(1, w*h)

    @staticmethod
    def _center_error(bb, w, h):
        x1,y1,x2,y2 = bb
        cx, cy = w//2, h//2
        tx, ty = (x1+x2)//2, (y1+y2)//2
        return ( (tx-cx)/max(1,w), (ty-cy)/max(1,h) )  # -0.5~0.5

    @staticmethod
    def _clip(v, vmax):
        return int(np.clip(v, -vmax, vmax))

    def compute_rc(self, frame_shape, bbox,
                   pose_dict=None,    # {'shoulder':{'ref':v,'ema':v}, 'spine':{...}, 'quality':0~1}
                   flow_vec=None,     # (vx, vy) in px/frame
                   size_target_range=(0.40, 0.50),
                   obstacle_brake=False):
        """
        returns (lr, fb, ud, yaw)
        - size_target_range: (low, high) → 중앙 0.45을 목표로 deadband 적용
        - obstacle_brake: True면 fb=0 강제(가까운 장애물 감지 시)
        """
        h, w = frame_shape[:2]
        err_x, err_y = self._center_error(bbox, w, h)
        ratio = self._area_ratio(bbox, w, h)
        target = 0.5*(size_target_range[0] + size_target_range[1])  # 기본 0.45
        err_size = target - ratio

        # yaw / lr
        if abs(err_x) > self.t_yaw:
            yaw = self._clip(err_x * self.g_yaw * 100, self.vmax)
            lr  = 0
        elif abs(err_x) > self.t_lr:
            yaw = 0
            lr  = self._clip(err_x * self.g_lr * 100, self.vmax)
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
        if pose_dict is not None and float(pose_dict.get('quality',1.0)) >= self.pose_qmin:
            terms = []
            for k in ('shoulder','spine'):
                d = pose_dict.get(k)
                if not d: 
                    continue
                ref, ema = d.get('ref'), d.get('ema')
                if ref and ema and ref>1e-6 and ema>1e-6:
                    # 현재가 커지면(가까워짐) log(ema/ref)>0 → 뒤로
                    terms.append(np.log(ema/ref) * self.pose_gain)
            if terms:
                fb_from_pose = self._clip(np.median(terms), self.vmax)

        # depth를 쓰지 않겠다는 요구사항이므로 fb는 size+pose로만 구성
        # 가중치는 대충 0.6(size) / 0.4(pose)로 시작 → 추후 튠
        fb = self._clip(0.6*fb_from_size + 0.4*fb_from_pose, self.vmax)

        # flow 피드포워드(소량): err_x 부호와 상관없이 화면 이동을 약간 선반영
        if flow_vec is not None:
            vx, vy = flow_vec
            yaw += self._clip(self.flow_ff_gain * vx, self.vmax)
            lr  += self._clip(self.flow_ff_gain * vx, self.vmax)  # 좌우엔 동일 상수로 시작(튜닝 지점)
            ud  += self._clip(-self.flow_ff_gain * vy, self.vmax)

        if obstacle_brake:
            fb = 0

        return int(lr), int(fb), int(ud), int(yaw)

def clip_bbox_to_frame(bb, w, h):
    if bb is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bb]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
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
        td   = get("thief_dist")
        gate = get("thief_cos_dist")

        if td is None or gate is None:
            continue
        if td <= gate and td < best_td:
            best, best_td = det, td

    return best