import time
import cv2
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)

# --- Callback function ---
def callback_fn(bindings_list, **kwargs):
    print("Inference completed!")
    for i, bindings in enumerate(bindings_list):
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(
                    bindings.output(name).get_buffer(), axis=0
                )
                for name in bindings._output_names
            }
        depth_scaled = ((result / result.max()) * 255).astype(np.uint8)
        result = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        result = cv2.resize(result, (810, 1080))
        cv2.imwrite('depth.png', result)
    # print(bindings_list)
    # for i, binding in enumerate(bindings_list):
    #     # binding is a Bindings object
    #     # Get outputs via configured output layer names
    #     for output_info in binding.output_layers_info:
    #         name = output_info.name
    #         buf = binding.output(name)  # get numpy array
    #         print(f"Output {name} shape: {buf.shape}, dtype: {buf.dtype}")


# --- Load image ---
image_path = "bus.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to RGB if needed
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if image.shape[2] == 4:
    image = image[:, :, :3]

# --- Preprocess if required by the model ---
# Example: resize to model input size
batch_size = 1
model = HailoInfer('fast_depth.hef', batch_size=batch_size)
input_infos = model.hef.get_input_vstream_infos()
# print(input_infos[0].name, input_infos[0].shape, input_infos[0].format.type)
input_shape = model.get_input_shape()  
print('wtf', input_shape)
image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))  # HWC -> W,H
# Convert to float32 and transpose to CHW if model expects it
# image_preprocessed = image_resized.astype(np.float32).transpose(2, 0, 1)
image_preprocessed = np.ascontiguousarray(image_resized)


# --- Run inference ---
model.run([image_preprocessed], callback_fn)

while True:
    time.sleep(0.1)
