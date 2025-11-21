import numpy as np

def compute_yaw_rc(box, frame_width):
    """
    box: [x1, y1, x2, y2]
    frame_width: 영상 가로(px)
    return: yaw_rc (-100 ~ 100)
    """
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cx0 = frame_width / 2.0

    # 화면 중심 대비 수평 오차
    ex = cx - cx0
    if frame_width <= 0:
        return 0

    # -1 ~ 1로 정규화
    ex_norm = ex / (frame_width / 2.0)

    # 데드존: 너무 작으면 회전하지 않음 (떨림 방지)
    if abs(ex_norm) < 0.05:
        return 0

    # 비례 제어 (Kp 계수는 상황 보면서 조정)
    Kp = 0.5  # 0.3~0.7 사이에서 튜닝해보세요
    yaw = Kp * ex_norm * 100.0  # 최대 약 ±50 정도까지 나오게

    # Tello 범위 -100~100, 너무 크지 않게 ±60 정도로 제한
    yaw = int(np.clip(yaw, -60, 60))
    return yaw
