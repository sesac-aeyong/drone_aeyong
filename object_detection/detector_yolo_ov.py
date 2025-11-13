# detector_yolo_ov.py — ONNX + (optional) NMS config JSON, raw head [1,84,8400] 지원
import os, json
import numpy as np, cv2
from openvino.runtime import Core
from config import DET_IMG_SIZE, DET_CONF_TH, DET_NMS_TH, PERSON_CLASS_ID

# YOLOv8 strides (640 기준 80x80, 40x40, 20x20 → 8400)
STRIDES = [8, 16, 32]

def _nms_xyxy(boxes, scores, iou_th=0.45):
    if len(boxes) == 0: return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1: break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.clip(xx2-xx1,0,None) * np.clip(yy2-yy1,0,None)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        idxs = rest[iou < iou_th]
    return keep

def _make_grids(imgsz=DET_IMG_SIZE, strides=STRIDES):
    grids, strides_all = [], []
    for s in strides:
        ny, nx = imgsz // s, imgsz // s
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='xy')
        grid = np.stack((xv, yv), axis=0).reshape(2, -1)  # [2, nx*ny]
        grids.append(grid.astype(np.float32))
        strides_all.append(np.full((1, nx*ny), s, dtype=np.float32))
    grid = np.concatenate(grids, axis=1)          # [2, 8400]
    stride = np.concatenate(strides_all, axis=1)  # [1, 8400]
    return grid, stride

GRID, STRIDE = _make_grids()

class OVYoloDetector:
    """
    OpenVINO로 ONNX를 직접 로드하여 YOLOv8 raw head([1,84,8400])를 디코딩 + NMS.
    - onnx_path: 필수 (XML/BIN 불필요)
    - nms_json: 선택 (conf_thres / iou_thres 키 지원)
    """
    def __init__(self, onnx_path, device="CPU", nms_json=None):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        ie = Core()
        self.model = ie.read_model(onnx_path)
        self.compiled = ie.compile_model(self.model, device)
        self.input = self.compiled.inputs[0]
        self.output = self.compiled.outputs[0]

        # 임계값 설정 (config 기본값 → JSON 덮어쓰기)
        self.conf_th = float(DET_CONF_TH)
        self.nms_th  = float(DET_NMS_TH)
        if nms_json and os.path.exists(nms_json):
            try:
                cfg = json.load(open(nms_json, "r"))
                if "conf_thres" in cfg: self.conf_th = float(cfg["conf_thres"])
                if "iou_thres"  in cfg: self.nms_th  = float(cfg["iou_thres"])
            except Exception:
                pass  # JSON 파싱 실패 시 config 기본값 사용

    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    def _decode_raw(self, outCxN, orig_w, orig_h):
        """
        YOLOv8: outCxN = [84, 8400]
        이미 grid-offset된 절대좌표로 출력됨.
        """
        C, N = outCxN.shape
        has_obj = (C == 85)
        cls_offset = 5 if has_obj else 4

        # 1) box (cx, cy, w, h)는 이미 scale된 절대 좌표 (640 기준)
        box_cxcywh = outCxN[0:4, :].T  # [N,4]
        boxes = np.zeros_like(box_cxcywh)
        boxes[:, 0] = box_cxcywh[:, 0] - box_cxcywh[:, 2] / 2  # x1
        boxes[:, 1] = box_cxcywh[:, 1] - box_cxcywh[:, 3] / 2  # y1
        boxes[:, 2] = box_cxcywh[:, 0] + box_cxcywh[:, 2] / 2  # x2
        boxes[:, 3] = box_cxcywh[:, 1] + box_cxcywh[:, 3] / 2  # y2

        # 2) confidence = objectness * max(class_probs)
        if has_obj:
            obj = self._sigmoid(outCxN[4, :])
            cls_probs = self._sigmoid(outCxN[cls_offset:, :])
            conf_person = obj * cls_probs[PERSON_CLASS_ID, :]
            conf = conf_person
            cls_ids = np.full_like(conf_person, PERSON_CLASS_ID)
        else:
            cls_probs = self._sigmoid(outCxN[cls_offset:, :])
            conf = cls_probs[PERSON_CLASS_ID, :]
            cls_ids = np.full_like(conf, PERSON_CLASS_ID)

        # 3) 필터링 + TopK
        m = conf >= self.conf_th
        if not np.any(m):
            return []
        boxes, conf, cls_ids = boxes[m], conf[m], cls_ids[m]
        K = min(300, len(conf))
        topk = np.argsort(conf)[::-1][:K]
        boxes, conf, cls_ids = boxes[topk], conf[topk], cls_ids[topk]

        # 4) 원본 스케일 복원
        sx, sy = orig_w / DET_IMG_SIZE, orig_h / DET_IMG_SIZE
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        # 5) NMS
        keep = _nms_xyxy(boxes, conf, self.nms_th)
        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].tolist()
            results.append([x1, y1, x2, y2, float(conf[i]), int(cls_ids[i])])
        return results

    def infer(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]

        # 전처리: resize 640, BGR→RGB, [0,1], NCHW
        img = cv2.resize(frame_bgr, (DET_IMG_SIZE, DET_IMG_SIZE))
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        inp = rgb.transpose(2, 0, 1)[None]  # [1,3,640,640]

        res = self.compiled([inp])[self.output]  # 보통 [1,84,8400]
        arr = res.squeeze(0)                     # [84,8400]

        # 안전장치: 만약 IR(NMS 포함) 형태라면 [N,6]일 수도 있음
        if arr.ndim == 2 and arr.shape[1] >= 6 and arr.shape[0] != 84:
            dets = arr
            sx, sy = w0 / DET_IMG_SIZE, h0 / DET_IMG_SIZE
            boxes = dets[:, :4].copy()
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            conf = dets[:, 4]; cls = dets[:, 5].astype(int)
            m = (conf >= self.conf_th) & (cls == PERSON_CLASS_ID)
            boxes, conf, cls = boxes[m], conf[m], cls[m]
            if len(boxes) == 0: return []
            keep = _nms_xyxy(boxes, conf, self.nms_th)  # 중복 방지용
            out = []
            for i in keep:
                x1, y1, x2, y2 = boxes[i].tolist()
                out.append([x1, y1, x2, y2, float(conf[i]), int(cls[i])])
            return out

        # 기본: YOLOv8 raw head 경로
        if arr.ndim == 2 and arr.shape[0] == 84:
            return self._decode_raw(arr, w0, h0)

        # 알 수 없는 형식
        return []


"""
python main_xpu.py \
  --source 0 --display \
  --det-backend ov --device GPU \
  --onnx models/yolov8m/yolov8m.onnx \
  --nms-json models/yolov8m/yolov8m_nms_config.json
"""