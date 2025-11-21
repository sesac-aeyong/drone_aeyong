# main_xpu.py
"""
Pipeline (time axis naming):

t = now frame
t-1 = last frame

1. YOLO → now_dets   (각각 now_bbox_tlbr, now_score, now_cls)
2. crop → OVReID → now_emb(t)
3. LongTermBoTSORT.update(now_dets, now_embs) 호출
   내부 동작:
     - BoTSORT.update()
         Track.predict() → pred_bbox_tlbr
         last_bbox_tlbr & last_emb 업데이트
     - LongTerm: last_emb ↔ gal_emb 비교 → identity_id 부여
4. 화면에는 track.identity_visible (갤러리 충분할 때만 숫자) 를 그리면 됨.
"""

import cv2, argparse
import numpy as np
from utils.config import ULTRA_MODEL, DETECTOR_ONNX, DETECTOR_NMS_JSON, PERSON_CLASS_ID
from utils.draw import draw_track
from tracker_botsort import BoTSORT, LongTermBoTSORT
from utils.reid_repVGG_ov import OVReID

from utils.gallery_io import save_gallery, load_gallery
GALLERY_PATH = "cache/longterm_gallery.npy" 

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

    # Tracker: BoTSORT + LongTerm
    base_tracker = BoTSORT()
    # ★ 갤러리 3장 이상부터 화면에 숫자 ID 노출
    tracker = LongTermBoTSORT(base_tracker, gallery_min_for_display=3)

    # 🔹 시작할 때: 갤러리 파일이 있으면 불러오기
    gallery = load_gallery(GALLERY_PATH)
    if len(gallery) > 0:
        tracker.gallery = gallery
        tracker.next_identity = max(gallery.keys()) + 1
        print("[LT-GAL] start AGAIN with saved gallery")
    else:
        tracker.gallery = {}
        tracker.next_identity = 1
        print("[LT-GAL] NEW start with empty gallery")

    mode, cap, frame_reader = open_source(args.source)
    if mode == "cv":
        if not cap.isOpened():
            print("ERROR: cannot open", args.source)
            return

    # ------------------------
    # Main loop
    # ------------------------
    frame_idx = 0 #💖 프레임 카운터
    
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
            now_dets = []     # [now_bbox_tlbr, now_score]
            now_embs = []     # now_emb
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
            
        # ========== 3) LongTermBoTSORT.update(now_dets, now_embs) ==========
        tracks = tracker.update(now_dets, now_embs)

        # ========== 4) 화면 표시 ==========
        if args.display:
            vis = frame.copy()
            for t in tracks:
                # BoTSORT는 Track.last_bbox_tlbr 로 위치를 유지함
                box = t.last_bbox_tlbr

                # ★ 표시용 ID: identity_visible (갤러리 충분할 때만 숫자), 아니면 "??"
                #   - identity_visible 속성이 없으면 (옛 tracker) → identity_id 또는 track_id로 fallback
                if hasattr(t, "identity_visible"):
                    visible_id = t.identity_visible  # None이면 draw_track에서 "??" 처리
                else:
                    visible_id = getattr(t, "identity_id", t.track_id)

                draw_track(vis, box, visible_id)

            cv2.imshow("XPU ReID Tracker (LongTerm + BoTSORT)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ------------------------
    # Clean up
    # ------------------------
    if mode == "tello":
        try: cap.streamoff()
        except: pass
        try: cap.end()
        except: pass
    else:
        cap.release()
    cv2.destroyAllWindows()

    # 🔹 종료할 때: 현재 갤러리 저장
    save_gallery(GALLERY_PATH, tracker.gallery)

if __name__ == "__main__":
    main()
