import cv2
import numpy as np
import time

# 카메라 인트린식 (픽셀 단위 초점 거리)
fx = 922.837110

# 드론 속도 입력 (m/s)
v_drone = 1.0  # 예시

# Optical Flow Farneback 화살표 시각화
def draw_flow_arrows(img, flow, step=16):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    fx_map = flow[::step, ::step, 0]
    fy_map = flow[::step, ::step, 1]
    
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y, x, dx, dy in zip(ys.flatten(), xs.flatten(), fx_map.flatten(), fy_map.flatten()):
        cv2.arrowedLine(vis, (x, y), (int(x + dx*2), int(y + dy*2)),
                        (0, 255, 0), 1, tipLength=0.3)
    return vis

# metric depth → color map 시각화
def depth_to_colormap(depth_values):
    depth_clipped = np.clip(depth_values, 0, 50)   # 너무 큰 값은 클리핑
    norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
    norm_uint8 = norm.astype(np.uint8)
    colormap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
    return colormap


# 특정 지점 거리 샘플 출력용
def print_sample_depths(Z):
    h, w = Z.shape
    sample_points = [
        (h//2, w//2),           # 중앙
        (h//4, w//4),           # 좌상단
        (h//4, 3*w//4),         # 우상단
        (3*h//4, w//4),         # 좌하단
        (3*h//4, 3*w//4),       # 우하단
        (h//2, w//4),           # 중앙 좌
        (h//2, 3*w//4)          # 중앙 우
    ]

    print("🟦 Depth Samples (meters):")
    for (y, x) in sample_points:
        d = Z[y, x]
        print(f"  Point ({y},{x}) = {d:.3f} m")
    print("----------------------------------------")


def visualize_optical_flow(video_path=0):
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("❌ 첫 번째 프레임을 가져올 수 없습니다.")
        return

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    print("▶ Optical Flow Farneback 계산 시작... ESC 를 누르면 종료")

    while True:
        start = time.time()
        ret, frame2 = cap.read()
        if not ret:
            print("❌ 프레임 없음, 종료.")
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # magnitude 계산
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # metric depth 계산  Z = f * v / mag
        Z_values = fx * v_drone / (mag + 1e-6)

        # 전체 depth color map
        depth_colormap = depth_to_colormap(Z_values)

        # 화살표 시각화
        arrow_vis = draw_flow_arrows(gray, flow)

        # 로그 출력
        mean_Z = np.mean(Z_values)
        median_Z = np.median(Z_values)
        fps = 1.0 / (time.time() - start)

        print(f"[Frame Stats] mean_depth={mean_Z:.2f} m, median_depth={median_Z:.2f} m, FPS={fps:.2f}")
        print_sample_depths(Z_values)

        # 화면 출력
        cv2.imshow("Input", frame2)
        cv2.imshow("Optical Flow Arrows", arrow_vis)
        cv2.imshow("Metric Depth Color Map", depth_colormap)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        prev_gray = gray.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_optical_flow(0)  # 웹캠
