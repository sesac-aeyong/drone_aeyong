# XPU/OpenVINO variant config

# --- Detector ---
# Option A: Ultralytics PyTorch (no path needed) - default
ULTRA_MODEL = "yolov8n.pt"   # you can switch to yolov8s.pt etc.

# Option B: OpenVINO IR (faster on Intel GPU once exported)
DETECTOR_IR_XML = "models/yolov8n_openvino_model/yolov8n.xml"  # change if you export
DETECTOR_IR_BIN = "models/yolov8n_openvino_model/yolov8n.bin"
DETECTOR_ONNX = "models/yolov8m/yolov8m.onnx"
DETECTOR_NMS_JSON = "models/yolov8m/yolov8m_nms_config.json"
DET_IMG_SIZE = 640
DET_CONF_TH  = 0.7
DET_NMS_TH   = 0.5
PERSON_CLASS_ID = 0

# --- ReID (OpenVINO ONNX) ---
REID_ONNX = "models/repvgg_a0_person_reid_512.onnx" #"models/osnet_x1_0.onnx"

# --- Tracker ---
IOU_WEIGHT   = 0.5
REID_WEIGHT  = 0.5
EMB_MATCH_TH = 0.6
TRACK_BUFFER = 30

# --- Sources ---
TELLO_UDP = "udp://0.0.0.0:11111?fifo_size=1000000&overrun_nonfatal=1"