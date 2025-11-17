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
from utils.config import ULTRA_MODEL, DETECTOR_ONNX, DETECTOR_NMS_JSON, PERSON_CLASS_ID, TELLO_UDP
from utils.draw import draw_focus       #💖
from tracker_thief import ThiefTracker  #💖
from utils.reid_repVGG_ov import OVReID

from utils.gallery_io import save_gallery, load_gallery
GALLERY_PATH = "cache/longterm_gallery.npy"
THIEF_PATH = "cache/thief_gallery.npy"  #💖

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
    ap.add_argument("--thief-id", type=int, required=True, help="LongTerm identity id to track as thief")  #💖
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
    # LongTerm 갤러리에서 도둑 갤러리만 로드 #💖
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
    #💖

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

        # ========== 3) ThiefTracker.update(now_dets, now_embs) ==========
        # 내부에서 TrackState.predict/update + 도둑 갤러리 기반 매칭 처리
        tracks = thief_tracker.update(now_dets, now_embs)

        # ========== 4) 화면 표시 ==========
        if args.display:
            vis = frame.copy()
            for t in tracks:
                # cos_dist 기준으로 필터
                if getattr(t, "thief_dist", 1.0) > thief_tracker.thief_cos_dist: 
                    continue
                
                # 위치 유지
                box = t.last_bbox_tlbr

                # 화면 표시 ID: thief_id만 있음
                tid = thief_id

                draw_focus(vis, box, tid)

            cv2.imshow(f"Thief Mode (id={thief_id})", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # 🔹 종료할 때: 현재 도둑 갤러리 저장  #💖
    thief_gallery = {thief_id: {"gal_embs": thief_tracker.thief_embs}}
    save_gallery(THIEF_PATH, thief_gallery)

if __name__ == "__main__":
    main()
