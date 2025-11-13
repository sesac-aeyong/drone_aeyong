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

    def embed(self, crop_bgr):
        img = cv2.resize(crop_bgr, (128,256))
        rgb = img[:,:,::-1]
        tensor = rgb.transpose(2,0,1)[None].astype(np.float32)  # normalize if needed
        emb = self.compiled([tensor])[self.output]
        emb = emb.astype(np.float32).squeeze()
        n = np.linalg.norm(emb) + 1e-6
        return emb / n