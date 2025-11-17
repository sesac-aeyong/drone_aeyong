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
from common.tracker.bot_sort import LongTermBoTSORT, BoTSORT
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from object_detection_post_process import draw_detections, extract_detections, inference_result_handler, find_best_matching_detection_index, id_to_color, draw_detection
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, visualize, FrameRateTracker
from scipy.optimize import linear_sum_assignment


config = load_json_file('config.json')
labels = get_labels(str(Path(__file__).parent.parent / "common" / "coco.txt"))
tracker_config = config.get("visualization_params", {}).get("tracker", {})

tracker = BoTSORT(
    max_age=30,
    min_hits=3, 
    use_reid=True,
    iou_threshold=0.3
)

tracker = LongTermBoTSORT(
    tracker,
    embedding_threshold=0.3,
    reid_max_age=6000,
    iou_threshold=0.4,
    max_displacement=30
)

yolo_model = HailoInfer('yolov8n.hef', batch_size=1)
emb_model = HailoInfer('repvgg_a0_person_reid_2048.hef', batch_size=1)
dep_model = HailoInfer('scdepthv3.hef', batch_size=1)

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


def run(frame):
    def yolo_callback(bindings_list, output_queue, **kwargs):
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
        output_queue.put(result)

    yolo_frame = cv2.resize(frame, (yolo_input_shape[1], yolo_input_shape[0]))
    yolo_frame = np.ascontiguousarray(yolo_frame)
    dep_frame = cv2.resize(frame, (dep_shape[1], dep_shape[0]))
    dep_frame = np.ascontiguousarray(dep_frame)
    
    yolo_callback_fn = partial(yolo_callback, output_queue=yolo_queue)
    dep_callback_fn = partial(yolo_callback, output_queue=dep_queue)
    
    yolo_model.run([yolo_frame], yolo_callback_fn)
    dep_model.run([dep_frame], dep_callback_fn)

    data = yolo_queue.get()
    depth_data = dep_queue.get()
    yolo_queue.task_done()
    
    detections = extract_detections(frame, data, config)
    img_out = frame.copy()
    img_out = draw_detections(detections, img_out, labels)

    boxes = detections['detection_boxes']  # [ymin, xmin, ymax, xmax] format
    scores = detections['detection_scores']
    num_detections = detections['num_detections']
    classes = detections['detection_classes']

    def emb_callback(bindings_list, output_queue, detection: dict, **kwargs):
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
        
        if isinstance(raw, dict):
            arr = next(iter(raw.values()))
        else:
            arr = raw
        
        emb = np.asarray(arr).flatten()
        emb = l2_normalize(emb)
        output_queue.put((detection, emb))

    objs = 0
    for i in range(num_detections):
        box = boxes[i]  # [ymin, xmin, ymax, xmax]
        score = scores[i]
        class_id = classes[i]
        class_ = labels[class_id]
        
        if class_ != 'person':
            continue

        emb_callback_fn = partial(
            emb_callback,
            output_queue=emb_queue,
            detection={'class': class_, 'score': score, 'box': box}
        )

        ymin, xmin, ymax, xmax = map(int, box)
        crop = frame[ymin:ymax, xmin:xmax]
        
        try:
            crop = cv2.resize(crop, (emb_input_shape[1], emb_input_shape[0]))
        except:
            continue

        objs += 1
        crop = np.ascontiguousarray(crop)
        emb_model.run([crop], emb_callback_fn)

    rets = []
    if objs:
        embs = [emb_queue.get() for _ in range(objs)]

        dets_for_tracker = []
        embs_for_tracker = []
        
        for i in range(objs):
            det, emb = embs[i]
            box = det['box']  # [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = box
            score = det['score']
            
            embs_for_tracker.append(emb)
            # Tracker expects [x1, y1, x2, y2, score]
            dets_for_tracker.append([xmin, ymin, xmax, ymax, score])
        
        online_targets = tracker.update(np.array(dets_for_tracker), embs_for_tracker)

        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr  # [x1, y1, x2, y2] format
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            
            # Convert to [ymin, xmin, ymax, xmax] for matching
            track_box_yx = [ymin, xmin, ymax, xmax]
            best_idx = find_best_matching_detection_index(track_box_yx, boxes)

            if best_idx is not None:
                cls_idx = classes[best_idx] if classes[best_idx] is not None else -1
                label_name = labels[cls_idx] if cls_idx >= 0 else "unknown"
                # Return in [xmin, ymin, xmax, ymax] format for easier use
                rets.append((track_id, label_name, track.score, xmin, ymin, xmax, ymax))
            else:
                continue
                
    return rets, depth_data


def close():
    yolo_model.close()
    emb_model.close()
    dep_model.close()


if __name__ == '__main__':
    fps = [0] * 10
    cap = cv2.VideoCapture(0)
    
    while True:
        ct = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        dets, dep = run(frame)
        fh, fw = frame.shape[:2]
        dep = cv2.resize(dep, (fw, fh))

        for det in dets:
            tid, label, score, x1, y1, x2, y2 = det
            # Convert [x1, y1, x2, y2] to [ymin, xmin, ymax, xmax] for draw_detection
            draw_detection(
                frame,
                [y1, x1, y2, x2],
                [label, f"ID {tid}"],
                score=score * 100.0,
                color=(255, 255, 255),
                track=True
            )
            
        dt = time.time() - ct
        fps_cur = 1 / dt
        fps.append(fps_cur)
        fps = fps[-10:]
        
    close()