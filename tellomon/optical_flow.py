import cv2
import numpy as np
import time
from djitellopy import Tello

# ---- 카메라 파라미터 ----
focal = 500  # 픽셀 단위 초점거리

# ---- Tello 연결 ----
tello = Tello()
tello.connect()

print("Tello Connected")
print("Battery:", tello.get_battery())

# 스트리밍 시작
tello.streamoff()
time.sleep(1)
tello.streamon()
time.sleep(1)

frame_reader = tello.get_frame_read()


# ---- 첫 프레임 ----
prev_frame = frame_reader.frame
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 특징점 추출
prev_pts = cv2.goodFeaturesToTrack(
    prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=7
)

while True:
    frame = frame_reader.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- optical flow ----
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None
    )

    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    flow = good_next - good_prev
    mag = np.linalg.norm(flow, axis=1)

    # -----------------------
    # 배경 선택 (하위 30% mag)
    # -----------------------
    if len(mag) > 0:
        threshold = np.percentile(mag, 30)
        bg_mask = mag < threshold
    else:
        bg_mask = np.zeros_like(mag, dtype=bool)

    bg_prev = good_prev[bg_mask]
    bg_mag = mag[bg_mask]

    # -----------------------
    # 드론 속도 가져오기
    # -----------------------
    v_forward = tello.get_speed_y() / 100.0  # cm/s → m/s

    # 전진 속도가 거의 0이면 깊이 측정 불가
    if abs(v_forward) < 0.02:
        v_forward = 0.02  # 최소값 강제 (진동방지)

    # -----------------------
    # 절대 깊이 계산
    # Z = f * V / flow
    # -----------------------
    eps = 1e-6
    depth_est = (focal * abs(v_forward)) / (bg_mag + eps)

    # -----------------------
    # 깊이 표시
    # -----------------------
    for (x, y), Z in zip(bg_prev, depth_est):
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"{Z:.1f}m", (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 속도 디버그 표시
    cv2.putText(frame, f"Drone Speed Forward: {v_forward:.2f} m/s",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Depth Estimation", frame)

    # ---- prepare next loop ----
    prev_gray = gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)

    if cv2.waitKey(1) & 0xFF == 27:
        break

tello.streamoff()
cv2.destroyAllWindows()
