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
4. 화면에 track.identity_id를 그리면 됨.
"""

import cv2, argparse
from utils.config import ULTRA_MODEL, DETECTOR_ONNX, DETECTOR_NMS_JSON, PERSON_CLASS_ID, TELLO_UDP
from utils.draw import draw_track
from tracker_botsort import BoTSORT, LongTermBoTSORT
from utils.reid_repVGG_ov import OVReID

from utils.gallery_io import save_gallery, load_gallery
GALLERY_PATH = "cache/longterm_gallery.npy" 

# ------------------------------
# 영상 입력
# ------------------------------
def open_source(src):
    if src == "tello": src = TELLO_UDP
    try:
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(src)
    return cap


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
    tracker = LongTermBoTSORT(base_tracker)

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

    cap = open_source(args.source)
    if not cap.isOpened():
        print("ERROR: cannot open", args.source)
        return

    # ------------------------
    # Main loop
    # ------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

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
        import numpy as np
        now_dets = np.asarray(now_dets, dtype=np.float32)

        # ========== 3) LongTermBoTSORT.update(now_dets, now_embs) ==========
        # 내부에서 BoTSORT.update → Track.predict/predict/update → Track.last_*, pred_* 처리
        # long-term identity까지 완성된 Track 리스트 반환
        tracks = tracker.update(now_dets, now_embs)

        # ========== 4) 화면 표시 ==========
        if args.display:
            vis = frame.copy()
            for t in tracks:
                # BoTSORT는 Track.last_bbox_tlbr 로 위치를 유지함
                box = t.last_bbox_tlbr

                # 화면 표시 ID: identity_id 우선, 없으면 track_id
                tid = getattr(t, "identity_id", t.track_id)

                draw_track(vis, box, tid)

            cv2.imshow("XPU ReID Tracker (LongTerm + BoTSORT)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # 🔹 종료할 때: 현재 갤러리 저장
    save_gallery(GALLERY_PATH, tracker.gallery)

if __name__ == "__main__":
    main()
