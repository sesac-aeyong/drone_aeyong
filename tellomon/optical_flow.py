import cv2
import numpy as np

# ---- 카메라 파라미터 ----
focal = 500   # px
drone_v = 0.0 # m/s  (예시)

cap = cv2.VideoCapture(0)

# 첫 프레임
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 특징점 추출
prev_pts = cv2.goodFeaturesToTrack(
    prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=7
)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None
    )

    # 점 필터링
    good_prev = prev_pts[status==1]
    good_next = next_pts[status==1]

    # movement vectors
    flow = good_next - good_prev
    mag = np.linalg.norm(flow, axis=1)  # magnitude

    # -----------------------
    # 배경 선택 (방법 A: 낮은 mag percentile 사용)
    # -----------------------
    threshold = np.percentile(mag, 30)  # 하위 30%를 배경으로
    bg_mask = mag < threshold

    bg_prev = good_prev[bg_mask]
    bg_next = good_next[bg_mask]
    bg_flow_mag = mag[bg_mask]

    # -----------------------
    # 절대 깊이 계산
    # Z = f * V / flow
    # -----------------------
    eps = 1e-6
    depth_est = (focal * drone_v) / (bg_flow_mag + eps)

    # 배경 점 + 깊이 시각화
    for (x, y), Z in zip(bg_prev, depth_est):
        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(frame, f"{Z:.1f}", (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.imshow("depth", frame)

    # 다음 루프 준비
    prev_gray = gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
