# main_xpu.py
"""
1. YOLO가 사람 박스 뽑음 → 각 박스로 crop
2. `OVReID`로 그 crop에서 임베딩 추출
3. `boxes, embs`를 **`LongTermBoTSORT.update()`** 에 넣음
4. 이 안에서 먼저 `BoTSORT.update()`가 호출되고(헝가리안 + IoU + ReID cost),  
   그 결과 트랙들에 대해 `identity_id`를 다시 붙여서 반환
5. 화면에는 `track_id` 대신 `identity_id`를 그려줌
"""


import cv2, argparse
from config import ULTRA_MODEL, DETECTOR_ONNX, DETECTOR_NMS_JSON, PERSON_CLASS_ID, TELLO_UDP
from utils.draw import draw_track
from tracker_botsort import BoTSORT, LongTermBoTSORT #💖
from reid_repVGG_ov import OVReID

def open_source(src):
    if src == "tello": src = TELLO_UDP
    try:
        cam_index = int(src)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(src)
    return cap

def crop_safe(img, box):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = map(int, box)
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
    if x2<=x1 or y2<=y1: return img[0:1,0:1]
    return img[y1:y2, x1:x2]

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--det-backend", choices=["ultra","ov"], default="ov")
    ap.add_argument("--device", default="GPU", help="OpenVINO device for OV backends: CPU or GPU")
    ap.add_argument("--onnx", default=None, help="Path to YOLO ONNX (raw head [1,84,8400])")
    ap.add_argument("--nms-json", default=None, help="Optional NMS config JSON (conf_thres / iou_thres)")
    return ap.parse_args()

def main():
    args = parse()

    # Detector backend
    if args.det_backend == "ultra":
        print("[Detector] Forcing Ultralytics backend:", ULTRA_MODEL)
        from detector_yolo_ultra import UltraYoloDetector
        detector = UltraYoloDetector(model=ULTRA_MODEL)
    else:
        from detector_yolo_ov import OVYoloDetector
        print(f"[Detector] Using OV ONNX from CLI: {args.onnx or DETECTOR_ONNX}")
        detector = OVYoloDetector(
            onnx_path=args.onnx or DETECTOR_ONNX,
            nms_json=args.nms_json or DETECTOR_NMS_JSON,
            device=args.device)
            
    reid = OVReID(device=args.device)
    #tracker = BoTSORT() #💖
    base_tracker = BoTSORT()
    tracker = LongTermBoTSORT(base_tracker)

    cap = open_source(args.source)
    if not cap.isOpened():
        print("ERROR: cannot open", args.source); return

    while True:
        ok, frame = cap.read()
        if not ok: break

        dets = detector.infer(frame)  # [x1,y1,x2,y2,conf,cls]
        boxes, embs = [], [] 
        for x1,y1,x2,y2,conf,cls in dets:
            if int(cls) != PERSON_CLASS_ID:
                continue
            crop = crop_safe(frame, (x1,y1,x2,y2))
            emb = reid.embed(crop)
            boxes.append([x1,y1,x2,y2,conf]); embs.append(emb)
        tracks = tracker.update(boxes, embs) 

        if args.display:
            vis = frame.copy()
            for t in tracks: #💖
                box = t.tlbr ; tid = getattr(t, "identity_id", t.track_id) #💖
                draw_track(vis, box, tid)
            cv2.imshow("XPU OpenVINO ReID Tracker", vis)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
    
"""
python main_xpu.py --display
"""