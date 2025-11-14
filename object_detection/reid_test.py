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
from common.tracker.bot_sort import LongTermBoTSORT, BoTSORT
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from object_detection_post_process import draw_detections, extract_detections, inference_result_handler, find_best_matching_detection_index, id_to_color, draw_detection
from common.toolbox import init_input_source, get_labels, load_json_file, preprocess, visualize, FrameRateTracker
from scipy.optimize import linear_sum_assignment


# cap = cv2.VideoCapture(0)
config = load_json_file('config.json')
labels = get_labels(str(Path(__file__).parent.parent / "common" / "coco.txt"))
tracker_config = config.get("visualization_params", {}).get("tracker", {})
# tracker = BYTETracker(SimpleNamespace(**tracker_config))
# tracker = BoTSORT(max_age=600, min_hits=3, use_reid=True, iou_threshold=0.3)
# tracker = LongTermBoTSORT(tracker)  # replaces previous tracker reference
# Use these more conservative parameters
tracker = BoTSORT(
    max_age=30,
    min_hits=3, 
    use_reid=True,
    iou_threshold=0.3
)

tracker = LongTermBoTSORT(
    tracker,
    embedding_threshold=0.3,    # Stricter embedding match
    reid_max_age=6000,           # Longer memory
    iou_threshold=0.4,         # Higher IoU for re-ID
    max_displacement=30        # Tighter spatial constraint
)
yolo_model = HailoInfer('yolov8n.hef', batch_size=1)
# emb_model = HailoInfer('repvgg_a0_person_reid_512.hef', batch_size=1)
emb_model = HailoInfer('repvgg_a0_person_reid_2048.hef', batch_size=1)
# emb_model = HailoInfer('osnet_x1_0.hef', batch_size=1)
dep_model = HailoInfer('scdepthv3.hef', batch_size=1)
# dep_model = HailoInfer('depth_anything_v2_vits.hef', batch_size=1)
# dep_model = HailoInfer('depth_anything_vits.hef', batch_size=1)
# dep_model = HailoInfer('fast_depth.hef', batch_size=1)
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

def run(frame):
    

# while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break

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

    # depx_frame = np.zeros((dep_shape[1], dep_shape[0], 4), dtype=np.uint8)
    # depx_frame[:, :, :3] = dep_frame

    # dep_frame = np.ascontiguousarray(depx_frame)
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
    # depth_data = dep_frame.copy()
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
        # print(emb)
        # print()
        # print(l2_normalize(emb))
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
    rets = []
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
        # return online_targets, depth_data
        # rets = []

        for track in online_targets:
            track_id = track.track_id
            x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            # color = tuple(id_to_color(classes[best_idx]).tolist())  # color based on class
            color = (255, 255, 255)
            # if best_idx is None:
            #     draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
            #                     track.score * 100.0, color, track=True)
            # else:
                # draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
                #                 track.score * 100.0, color, track=True)
            # rets.append((track_id, labels[classes[best_idx]], track.score, xmin, ymin, xmax, ymax))

            # cls_idx = classes[best_idx] if classes[best_idx] is not None else -1
            # label_name = labels[cls_idx] if cls_idx >= 0 else "unknown"
            # rets.append((track_id, label_name, track.score, xmin, ymin, xmax, ymax))

            if best_idx is not None:
                cls_idx = classes[best_idx] if classes[best_idx] is not None else -1
                label_name = labels[cls_idx] if cls_idx >= 0 else "unknown"
                rets.append((track_id, label_name, track.score, xmin, ymin, xmax, ymax))
            else:
                # no valid detection, skip
                continue
        # print(online_targets)
        # for track in online_targets:
        #     track_id = track.track_id
        #     x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right)
        #     xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            
        #     best_idx = find_best_matching_detection_index(track.tlbr, boxes)
        #     # color = tuple(id_to_color(classes[best_idx]).tolist())  # color based on class
        #     color = (255, 255, 255)
        #     if best_idx is None:
        #         draw_detection(img_out, [xmin, ymin, xmax, ymax], f"ID {track_id}",
        #                         track.score * 100.0, color, track=True)
        #     else:
        #         draw_detection(img_out, [xmin, ymin, xmax, ymax], [labels[classes[best_idx]], f"ID {track_id}"],
        #                         track.score * 100.0, color, track=True)
    return rets, depth_data

    # dep_out = depth_data
    # dep_out = ((dep_out / depth_data.max()) * 255).astype(np.uint8)
    # dep_out = cv2.applyColorMap(dep_out, cv2.COLORMAP_JET)
    # dep_out = cv2.resize(dep_out, (frame.shape[1], frame.shape[0]))
    #     # depth_scaled = ((result / result.max()) * 255).astype(np.uint8)
    #     # result = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    #     # result = cv2.resize(result, (810, 1080))
    # img_out = np.hstack((img_out, dep_out))
    # cv2.imshow('output', img_out)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     # emb_model.close()
    #     return

    # yolo_result = inference_result_handler(frame, data, get_labels(str(Path(__file__).parent.parent / "common" / "coco.txt")), config)
    # # print(yolo_result)
    # yolo_result = cv2.cvtColor(yolo_result, cv2.COLOR_BGR2RGB)
    # cv2.imshow('yolo', yolo_result)
    # cv2.waitKey(1)

def close():
    yolo_model.close()
    emb_model.close()
    dep_model.close()


if __name__ == '__main__':
    fps = [0] * 10
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    while True:
        ct = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        dets, dep = run(frame)
        fh, fw = frame.shape[:2]
        dep = cv2.resize(dep, (fw, fh))

        # rets.append((track_id, labels[classes[best_idx]], track.score, xmin, ymin, xmax, ymax))

        for det in dets:
            tid, label, score, x1, y1, x2, y2 = det
            draw_detection(
                frame,
                [x1, y1, x2, y2],
                [label, f"ID {tid}"],  # second element used as bottom_text when track=True
                score=score * 100.0,
                color=(255, 255, 255),
                track=True
            )
        dt = time.time() - ct
        fps_cur = 1 / dt
        fps.append(fps_cur)
        fps = fps[-10:]
        # print(f'fps: {fps_cur:.1f}, avg: {sum(fps) / 10:.1f}')

        # print(dep.shape)
        # dep_norm = cv2.normalize(dep, None, 0, 255, cv2.NORM_MINMAX)
        # dep_color = cv2.cvtColor(dep_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # view = np.hstack((frame, dep_color))
        # cv2.imshow('result', view)
        # # print('ok')
        # # cv2.imshow('tello feed', frame)
        # cv2.imshow('depth', dep)
        # cv2.imshow('feed', frame)
        
        # dep_pred = dep.squeeze()
        # if 'min_val' not in locals():
        #     min_val, max_val = dep_pred.min(), dep_pred.max()
        # else:
        #     min_val = 0.9 * min_val + 0.1 * dep_pred.min()
        #     max_val = 0.9 * max_val + 0.1 * dep_pred.max()

        # dep_norm = np.clip((dep_pred - min_val) / (max_val - min_val), 0, 1)

        # # dep_norm = (dep_pred - dep_pred.min()) / (dep_pred.max() - dep_pred.min())
        # dep_norm = np.clip((dep_pred - min_val) / (max_val - min_val), 0, 1)

        # dep_uint8 = (dep_norm * 255).astype(np.uint8)
        # dep_vis = cv2.applyColorMap(255 - dep_uint8, cv2.COLORMAP_JET)
        # cv2.imshow('relative depth', dep_vis)
        # cv2.imshow('test', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    close()