# Detector using Ultralytics (PyTorch). Easiest for correctness checks.
from ultralytics import YOLO

class UltraYoloDetector:
    def __init__(self, model="yolov8n.pt", imgsz=640, conf=0.35):
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.conf = conf

    def infer(self, frame_bgr):
        # returns list of [x1,y1,x2,y2,score,cls]
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        if res.boxes is None: return out
        for b in res.boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            cls  = int(b.cls[0].item())
            out.append([x1,y1,x2,y2,conf,cls])
        return out