# file: ov_devices_probe.py
from openvino import Core
ie = Core()
print("Devices:", ie.available_devices)
for d in ie.available_devices:
    print(d, ie.get_property(d, "FULL_DEVICE_NAME"))


"""
출력 결과
Devices: ['CPU', 'GPU.0', 'GPU.1']
CPU Intel(R) Core(TM) Ultra 5 225
GPU.0 Intel(R) Graphics (iGPU)
GPU.1 Intel(R) Arc(TM) B580 Graphics (dGPU)

"""