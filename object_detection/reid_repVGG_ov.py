# ReID using OpenVINO (reads ONNX).
import numpy as np, cv2
from openvino.runtime import Core
from config import REID_ONNX

class OVReID:
    def __init__(self, onnx_path=REID_ONNX, device="GPU"):
        print("[ReID] loading ONNX:", onnx_path, "on device:", device)
        ie = Core()
        self.model = ie.read_model(onnx_path)
        self.compiled = ie.compile_model(self.model, device)
        self.input = self.compiled.inputs[0]
        self.output = self.compiled.outputs[0]

        # Hailo yaml에서 가져온 전처리 값 (RGB)
        self.mean = np.array([111.296, 104.263, 104.396], dtype=np.float32)
        self.std  = np.array([64.193, 62.547, 60.324], dtype=np.float32)
        
    def embed(self, crop_bgr):
        img = cv2.resize(crop_bgr, (128,256))
        rgb = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        rgb = (rgb - self.mean) / self.std        # 채널별 정규화 적용
        
        tensor = rgb.transpose(2,0,1)[None].astype(np.float32)  # NCHW (1, 3, 256, 128)
        emb = self.compiled([tensor])[self.output] 
        emb = emb.astype(np.float32).squeeze()
        n = np.linalg.norm(emb) + 1e-6                          # L2 정규화 (코사인 거리용)
        return emb / n