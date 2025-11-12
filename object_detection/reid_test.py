from functools import partial
from pathlib import Path
import queue
import time
import cv2
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer
# from common.tracker.byte_tracker import BYTETracker
from common.tracker.bot_sort import BoTSORT
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from object_detection_post_process import draw_detections, extract_detections, inference_result_handler, find_best_matching_detection_index, id_to_color, draw_detection
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, visualize, FrameRateTracker



# # --- Callback function ---
# def callback_fn(bindings_list, **kwargs):
#     print("Inference completed!")
#     for i, bindings in enumerate(bindings_list):
#         if len(bindings._output_names) == 1:
#             result = bindings.output().get_buffer()
#         else:
#             result = {
#                 name: np.expand_dims(
#                     bindings.output(name).get_buffer(), axis=0
#                 )
#                 for name in bindings._output_names
#             }
        # print(result)
        # output_queue.put(result)
        # depth_scaled = ((result / result.max()) * 255).astype(np.uint8)
        # result = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        # result = cv2.resize(result, (810, 1080))
        # cv2.imwrite('reid.png', result)
    # print(bindings_list)
    # for i, binding in enumerate(bindings_list):
    #     # binding is a Bindings object
    #     # Get outputs via configured output layer names
    #     for output_info in binding.output_layers_info:
    #         name = output_info.name
    #         buf = binding.output(name)  # get numpy array
    #         print(f"Output {name} shape: {buf.shape}, dtype: {buf.dtype}")
# --- Persistent ReID wrapper for BoTSORT ---
class LongTermBoTSORT:
    """
    Wrapper around BoTSORT that keeps embeddings in memory for long-term re-identification.
    """
    def __init__(self, bot_sort_tracker, embedding_threshold=0.45):
        self.tracker = bot_sort_tracker
        self.embedding_threshold = embedding_threshold
        self.persistent_memory = {}  # track_id -> embedding

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two embeddings"""
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            return 1.0
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return 1.0 - np.dot(a_norm, b_norm)

    def update(self, detections: np.ndarray, embeddings: list):
        """
        Args:
            detections: np.ndarray of [N,5] -> [x1,y1,x2,y2,score]
            embeddings: list of N normalized embeddings
        Returns:
            List of tracks with persistent IDs
        """
        online_targets = self.tracker.update(detections, embeddings)

        # Save embeddings for active tracks
        for track in online_targets:
            feat = track.get_feature()
            if feat is not None:
                self.persistent_memory[track.track_id] = feat
                
        # Match new detections to previous IDs if distance < threshold
        for i, emb in enumerate(embeddings):
            box = detections[i][:4]
            best_id = None
            best_dist = self.embedding_threshold
            for mem_id, mem_emb in self.persistent_memory.items():
                dist = self.cosine_distance(emb, mem_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_id = mem_id

            if best_id is not None:
                # reassign track_id if box matches memory
                for track in online_targets:
                    if np.allclose(track.tlbr, box):
                        track.track_id = best_id

        return online_targets

# --- Wrap your BoTSORT tracker ---

cap = cv2.VideoCapture(0)
config = load_json_file('config.json')
labels = get_labels(str(Path(__file__).parent.parent / "common" / "coco.txt"))
tracker_config = config.get("visualization_params", {}).get("tracker", {})
# tracker = BYTETracker(SimpleNamespace(**tracker_config))
tracker = BoTSORT(max_age=60, min_hits=3, use_reid=True, iou_threshold=0.3)
tracker = LongTermBoTSORT(tracker)  # replaces previous tracker reference
yolo_model = HailoInfer('yolov8n.hef', batch_size=1)
# emb_model = HailoInfer('repvgg_a0_person_reid_512.hef', batch_size=1)
emb_model = HailoInfer('osnet_x1_0.hef', batch_size=1)
dep_model = HailoInfer('scdepthv3.hef', batch_size=1)
# dep_model = HailoInfer('depth_anything_v2--224x224_quant_hailort_multidevice_1.hef', batch_size=1)

yolo_queue = queue.Queue()
emb_queue = queue.Queue()
dep_queue = queue.Queue()

yolo_input_shape = yolo_model.get_input_shape()
emb_input_shape = emb_model.get_input_shape()
dep_shape = dep_model.get_input_shape()


def l2_normalize(x, eps=1e-6):
    norm = np.linalg.norm(x)
    if norm < eps:
        return x
    return (x / norm).astype(np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def yolo_callback(bindings_list, output_queue, **kwargs):
        # print('yolo callback!')
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
        # print(result)
        output_queue.put(result)


    yolo_frame = cv2.resize(frame, (yolo_input_shape[1], yolo_input_shape[0]))
    yolo_frame = np.ascontiguousarray(yolo_frame)
    dep_frame = cv2.resize(frame, (dep_shape[1], dep_shape[0]))
    dep_frame = np.ascontiguousarray(dep_frame)
    yolo_callback_fn = partial(
        yolo_callback, 
        output_queue = yolo_queue
    )
    dep_callback_fn = partial(
        yolo_callback, 
        output_queue = dep_queue
    )
    yolo_model.run([yolo_frame], yolo_callback_fn)
    dep_model.run([dep_frame], dep_callback_fn)
    # yolo_model.last_infer_job.wait(10000) # wait for yolo ## no need, wait for queue
    # yolo_model.close()

    data = yolo_queue.get() # use queue.Get as sync method
    depth_data = dep_queue.get()
    yolo_queue.task_done()
    # print('got data')
    detections = extract_detections(frame, data, config)  #should return dict with boxes, classes, scores
    img_out = frame.copy()
    img_out = draw_detections(detections, img_out, labels) # draw yolo detection boxes

    # print(detections)
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    num_detections = detections['num_detections']
    classes = detections['detection_classes']

    def emb_callback(bindings_list, output_queue, detection: dict, **kwargs):
        # print('emb callback!')
        # print(detection)
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                raw = bindings.output().get_buffer()
            else:
                raw = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
        # print(result)
        if isinstance(raw, dict):
            # take first output array
            arr = next(iter(raw.values()))
        else:
            arr = raw
        
        emb = np.asarray(arr).flatten()
        emb = l2_normalize(emb)
        output_queue.put((detection, emb))


    objs = 0
    for i in range(num_detections):
        box = boxes[i]
        score = scores[i]
        class_id = classes[i]
        class_ = labels[class_id]
        if class_ != 'person':
            continue

        emb_callback_fn = partial(
            emb_callback,
            output_queue = emb_queue,
            detection = { 'class': class_, 'score': score, 'box': box}
        )

        # print(class_, box, score)
        # ymin, xmin, ymax, xmax = map(int, box)
        xmin, ymin, xmax, ymax = map(int, box)

        crop = frame[ymin:ymax, xmin:xmax]
        try:
            crop = cv2.resize(crop, (emb_input_shape[1], emb_input_shape[0]))
        except:
            continue

        objs += 1
        crop = np.ascontiguousarray(crop)
        emb_model.run([crop], emb_callback_fn)
        # emb_data = emb_queue.get()
        # dets_for_tracker = [[*box, score]]
        # print(dets_for_tracker)

        # online_targets = tracker.update(np.array(dets_for_tracker), emb_data)
        # print(online_targets) 

        # break # only one crop for now. 
        ## perhaps do batch processing on all boxes 
        # somehow consolidate all embedding results 

    # print('# of persons:', objs)
    if objs:
        embs = [emb_queue.get() for _ in range(objs)]

        # for det, emb in embs:
        #     print(np.linalg.norm(emb), emb[:5], det['class'], det['score'])

        # print(embs)
        # visualize
        dets_for_tracker = []
        embs_for_tracker = []
        for i in range(objs):
            det, emb = embs[i]
            xmin, ymin, xmax, ymax = det['box']
            score = det['score']
            box = det['box']
            embs_for_tracker.append(emb)
            dets_for_tracker.append([*box, score])
            # dets_for_tracker.append([xmin, ymin, xmax, ymax, score])
        
        online_targets = tracker.update(np.array(dets_for_tracker), embs_for_tracker)
        # print(online_targets)
        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            # color = tuple(id_to_color(classes[best_idx]).tolist())  # color based on class
            color = (255, 255, 255)
            if best_idx is None:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
                                track.score * 100.0, color, track=True)
            else:
                draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
                                track.score * 100.0, color, track=True)

    dep_out = depth_data
    dep_out = ((dep_out / depth_data.max()) * 255).astype(np.uint8)
    dep_out = cv2.applyColorMap(dep_out, cv2.COLORMAP_JET)
    dep_out = cv2.resize(dep_out, (frame.shape[1], frame.shape[0]))
        # depth_scaled = ((result / result.max()) * 255).astype(np.uint8)
        # result = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        # result = cv2.resize(result, (810, 1080))
    img_out = np.hstack((img_out, dep_out))
    cv2.imshow('output', img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # emb_model.close()
        break

    # yolo_result = inference_result_handler(frame, data, get_labels(str(Path(__file__).parent.parent / "common" / "coco.txt")), config)
    # # print(yolo_result)
    # yolo_result = cv2.cvtColor(yolo_result, cv2.COLOR_BGR2RGB)
    # cv2.imshow('yolo', yolo_result)
    # cv2.waitKey(1)

yolo_model.close()
emb_model.close()
dep_model.close()
# --- Load image ---
# image_path = "bus.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# image = image[350:950, 30:220]

# if image is None:
#     raise FileNotFoundError(f"Image not found: {image_path}")

# # Convert to RGB if needed
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # --- Preprocess if required by the model ---
# # Example: resize to model input size
# batch_size = 1
# # model = HailoInfer('scdepthv3.hef', batch_size=batch_size)
# # model = HailoInfer('fast_depth.hef', batch_size=batch_size)
# model = HailoInfer('repvgg_a0_person_reid_512.hef', batch_size=batch_size)


# input_infos = model.hef.get_input_vstream_infos()
# # print(input_infos[0].name, input_infos[0].shape, input_infos[0].format.type)
# input_shape = model.get_input_shape()  
# print('shapes', input_shape)
# image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))  # HWC -> W,H
# # Convert to float32 and transpose to CHW if model expects it
# # image_preprocessed = image_resized.astype(np.float32).transpose(2, 0, 1)
# image_preprocessed = np.ascontiguousarray(image_resized)


# # --- Run inference ---
# model.run([image_preprocessed], callback_fn)
# model.close()

# while True:
#     time.sleep(0.1)
