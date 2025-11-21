# main_thief.py
"""
Thief mode:

1. 시작할 때:
   - longterm gallery(np.load)에서 thief_id 갤러리만 꺼냄
   - ThiefTracker(thief_embs=...) 초기화

2. 매 프레임:
   - YOLO → now_dets (person만)
   - crop → OVReID → now_embs
   - ThiefTracker.update(now_dets, now_embs)
   - 반환된 트랙(논리상 0 또는 1개)에 대해 cos_dist(thief_dist)가 충분히 작으면
     화면에 빨간 박스로 표시 / 드론 제어에 사용
"""

import cv2, argparse
import numpy as np
from utils.config import ULTRA_MODEL, DETECTOR_ONNX, DETECTOR_NMS_JSON, PERSON_CLASS_ID
from utils.draw import draw_focus
from tracker_thief import ThiefTracker
from utils.reid_repVGG_ov import OVReID
from utils.tello_control import compute_yaw_rc

from utils.gallery_io import save_gallery, load_gallery
GALLERY_PATH = "cache/longterm_gallery.npy"
THIEF_PATH = "cache/thief_gallery.npy"

# ------------------------------
# 영상 입력
# ------------------------------
def open_source(src):
    if src == "tello":
        from djitellopy import Tello
        t = Tello()
        print("[TELLO] Connecting...")
        t.connect()
        try:
            bat = t.get_battery()
            print(f"[TELLO] Battery: {bat}%")
        except Exception as e:
            print(f"[TELLO] battery read error: {e}")

        t.streamon()
        print("[TELLO] streamon()")
        frame_reader = t.get_frame_read()
        return "tello", t, frame_reader

    try:
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(src)
    return "cv", cap, None


# ------------------------------
# 안전 crop (now_bbox → crop)
# ------------------------------
def crop_safe(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w-1, x2)
    y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1: 
        return img[0:1, 0:1]
    return img[y1:y2, x1:x2]


# ------------------------------
# CLI argument parsing
# ------------------------------
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="int:webcam/tello:tello_udp/str:local_file")
    ap.add_argument("--display", action="store_true", help="imshow")
    ap.add_argument("--det-backend", choices=["ultra", "ov"], default="ov")
    ap.add_argument("--device", default="GPU", help="OpenVINO device: CPU/GPU")
    ap.add_argument("--onnx", default=None, help="YOLO raw-head ONNX")
    ap.add_argument("--nms-json", default=None, help="optional NMS JSON")
    ap.add_argument("--thief-id", type=int, required=True, help="LongTerm identity id to track as thief")
    return ap.parse_args()


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse()

    # ------------------------
    # Detector 선택
    # ------------------------
    if args.det_backend == "ultra":
        print("[Detector] Using Ultralytics:", ULTRA_MODEL)
        from utils.detector_yolo_ultra import UltraYoloDetector
        detector = UltraYoloDetector(model=ULTRA_MODEL)
    else:
        print(f"[Detector] Using OV ONNX: {args.onnx or DETECTOR_ONNX}")
        from utils.detector_yolo_ov import OVYoloDetector
        detector = OVYoloDetector(
            onnx_path=args.onnx or DETECTOR_ONNX,
            nms_json=args.nms_json or DETECTOR_NMS_JSON,
            device=args.device
        )

    # ReID embedder
    reid = OVReID(device=args.device)

    # ------------------------
    # LongTerm 갤러리에서 도둑 갤러리만 로드
    # ------------------------
    # gallery: {id: {"gal_embs": [emb1, emb2, ...]}, ...}
    gallery = load_gallery(GALLERY_PATH)
    if len(gallery) == 0:
        print("[THIEF] ERROR: longterm gallery is empty. Run main_xpu (search mode) first.")
        return

    thief_id = args.thief_id
    if thief_id not in gallery:
        print(f"[THIEF] ERROR: identity_id {thief_id} not found in gallery. keys={list(gallery.keys())}")
        return

    thief_info = gallery[thief_id]        # {"gal_embs": [...]}
    thief_embs = thief_info["gal_embs"]   # 실제 임베딩 리스트만 꺼내기

    print(f"[THIEF] Using gallery for identity_id={thief_id}, "
          f"K={len(thief_embs) if hasattr(thief_embs, '__len__') else '1'}")
    
    thief_tracker = ThiefTracker(thief_embs=thief_embs)
    print("[THIEF] ThiefTracker initialized.")

    mode, cap, frame_reader = open_source(args.source)
    if mode == "cv":
        if not cap.isOpened():
            print("ERROR: cannot open", args.source)
            return

    # ------------------------
    # Main loop
    # ------------------------
    frame_idx = 0 # 프레임 카운터
    
    while True:
        if mode == "tello":
            frame = frame_reader.frame
            if frame is None:
                # 아직 첫 프레임 안 온 경우도 있어서 한 번 더 기다리기
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))
        else:
            ok, frame = cap.read()
            if not ok:
                break
        frame_idx += 1

        # ---------- 0) 이 프레임에서 YOLO/ReID를 돌릴지 결정 ----------
        run_det = (frame_idx % 3 == 0) # 3프레임마다 추론

        if run_det:

            # ========== 1) YOLO → now_dets ==========
            # dets shape: [N, 6] = [x1,y1,x2,y2,score,cls]
            dets = detector.infer(frame)

            # ========== 2) crop → OVReID → now_emb ==========
            now_dets = []     # [x1,y1,x2,y2,score]
            now_embs = []     # ReID emb
            for x1, y1, x2, y2, conf, cls in dets:
                if int(cls) != PERSON_CLASS_ID:
                    continue

                crop = crop_safe(frame, (x1, y1, x2, y2))
                now_emb = reid.embed(crop)

                now_dets.append([x1, y1, x2, y2, conf])
                now_embs.append(now_emb)

            # numpy로 맞추기
            now_dets = np.asarray(now_dets, dtype=np.float32)

        else:
            # 🔥 이 프레임은 관측 없이 Kalman만 돌리기
            now_dets = np.zeros((0, 5), dtype=np.float32)
            now_embs = []

        # ========== 3) ThiefTracker.update(now_dets, now_embs) ==========
        # 내부에서 TrackState.predict/update + 도둑 갤러리 기반 매칭 처리
        tracks = thief_tracker.update(now_dets, now_embs)

        # ========== 3.5) 도둑 트랙 하나 선택 (가장 thief_dist 작은 것) ==========
        best_track = None
        best_dist = 1e9
        for t in tracks:
            d = getattr(t, "thief_dist", 1.0)
            if d > thief_tracker.thief_cos_dist:
                continue
            if d < best_dist:
                best_dist = d
                best_track = t
       
       
        # ========== 3.6) yaw_rc 계산 (웹캠/텔로 공통) ==========
        yaw_rc = 0
        if best_track is not None:
            box = best_track.last_bbox_tlbr  # [x1, y1, x2, y2]
            frame_h, frame_w = frame.shape[:2]
            yaw_rc = compute_yaw_rc(box, frame_w)

            # 디버깅용 로그
            print(f"[CTRL] frame={frame_idx} yaw_rc={yaw_rc}")
        else:
            print(f"[CTRL] frame={frame_idx} no thief detected → yaw_rc=0")
            
        if mode == "tello":
            cap.send_rc_control(0, 0, 0, yaw_rc)

                
        '''# ========== 3.6) Tello yaw 제어 ==========
        if mode == "tello":
            if best_track is not None:
                box = best_track.last_bbox_tlbr  # [x1, y1, x2, y2]
                # frame은 tello 모드에서 640x480으로 resize 했으므로 그대로 사용 가능
                frame_h, frame_w = frame.shape[:2]
                yaw_rc = compute_yaw_rc(box, frame_w)

                # 수평 회전만 제어, 나머지는 호버 상태 유지
                # (추후 거리 제어, 상하 이동 등 추가 가능)
                cap.send_rc_control(0, 0, 0, yaw_rc)
            else:
                # 도둑을 못 보고 있으면 회전 멈춤
                cap.send_rc_control(0, 0, 0, 0)'''
                     
        # ========== 4) 화면 표시 ==========
        if args.display:
            vis = frame.copy()

            # 화면 중앙 십자선 그려주면 튜닝에 도움 됨
            h, w = vis.shape[:2]
            cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
            cv2.line(vis, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

            if best_track is not None:
                box = best_track.last_bbox_tlbr
                tid = thief_id
                draw_focus(vis, box, tid)
            
            # 왼(+)/오른쪽(-) 회전 방향 숫자 표시
            cv2.putText(
                vis, f"yaw_rc={yaw_rc}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

            cv2.imshow(f"Thief Mode (id={thief_id})", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if mode == "tello":
        try: cap.streamoff()
        except: pass
        try: cap.end()
        except: pass
    else:
        cap.release()
    cv2.destroyAllWindows()

    # 종료할 때: 현재 도둑 갤러리 저장
    thief_gallery = {thief_id: {"gal_embs": thief_tracker.thief_embs}}
    save_gallery(THIEF_PATH, thief_gallery)

if __name__ == "__main__":
    main()
