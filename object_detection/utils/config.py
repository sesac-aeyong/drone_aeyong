# XPU/OpenVINO variant config

# --- Detector ---
# Option A: Ultralytics PyTorch (no path needed) - default
ULTRA_MODEL = "yolov8n.pt"   # you can switch to yolov8s.pt etc.

# Option B: OpenVINO IR (faster on Intel GPU once exported)
DETECTOR_IR_XML = "/home/intel/tello/aeyong/object_detection/models/yolov8n_openvino_model/yolov8n.xml"  # change if you export
DETECTOR_IR_BIN = "/home/intel/tello/aeyong/object_detection/models/yolov8n_openvino_model/yolov8n.bin"
DETECTOR_ONNX = "/home/intel/tello/aeyong/object_detection/models/yolov8m/yolov8m.onnx"
DETECTOR_NMS_JSON = "/home/intel/tello/aeyong/object_detection/models/yolov8m/yolov8m_nms_config.json"
DET_IMG_SIZE = 640
DET_CONF_TH  = 0.7
DET_NMS_TH   = 0.5
PERSON_CLASS_ID = 0

# --- ReID (OpenVINO ONNX) ---
REID_ONNX = "/home/intel/tello/aeyong/object_detection/models/repvgg_a0_person_reid_512.onnx" #"models/osnet_x1_0.onnx"