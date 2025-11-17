import numpy as np
from openvino.runtime import Core

MODEL = "/home/intel/tello/tello_reid_tracker_xpu/models/yolov8m/yolov8m.onnx"   # 또는 .xml (IR)

ie = Core()
m = ie.read_model(MODEL)
execm = ie.compile_model(m, "GPU")  # GPU 사용 시 "GPU"

print("== Inputs ==")
for i in execm.inputs:
    print(i.get_any_name(), i.partial_shape, i.get_element_type())

print("\n== Outputs ==")
for o in execm.outputs:
    print(o.get_any_name(), o.partial_shape, o.get_element_type())

# 더미 추론으로 실제 shape 확인
inp = execm.inputs[0]
ps = inp.partial_shape
N = int(ps[0]) if ps[0].is_static else 1
C = int(ps[1]) if ps[1].is_static else 3
H = int(ps[2]) if ps[2].is_static else 640
W = int(ps[3]) if ps[3].is_static else 640
dummy = np.zeros((N, C, H, W), dtype=np.float32)

res = execm([dummy])
print("\n== Runtime Shapes ==")
for o in execm.outputs:
    arr = res[o]
    print(o.get_any_name(), arr.shape, arr.dtype)


"""
출력결과

== Inputs ==
images [1,3,640,640] <Type: 'float32'>

== Outputs ==
output0 [1,84,8400] <Type: 'float32'>
"""