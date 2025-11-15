from concurrent.futures import ThreadPoolExecutor
from functools import partial
import queue
import time
from types import SimpleNamespace
from typing import Tuple
import numpy as np
from common.hailo_inference import HailoInfer
from tracker.bot_sort import BoTSORT
from settings import settings as S
import cv2

### TODO:: create and reuse buffer instead of creating lists every frame


# def l2_norm(x, eps=1e-10):
#     norm = np.linalg.norm(x)
#     if norm < eps:
#         return x
#     return (x / norm).astype(np.float32)


class Hailo():
    """
    Hailo pipeline class.

    Hailo.run(frame):
    frame -> vis_model -> emb_model -> (dets, depth)
          \\_ dep_model              /
    """
    def __init__(self):
        ct = time.time()
        print('loading Hailo models')
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model)
        self.emb_m = HailoInfer(S.embed_model, output_type='FLOAT32')
        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.dm_shape = tuple(reversed(self.dep_m.get_input_shape()[:2]))

        self.vis_q = queue.Queue()
        self.dep_q = queue.Queue()
        self.emb_q = queue.Queue()

        self.vis_cb = partial(_callback, output_queue = self.vis_q)
        self.dep_cb = partial(_callback, output_queue = self.dep_q)

        self.tracker = BoTSORT(args = SimpleNamespace(
            track_high_thresh=S.track_high_emb_confidence,
            track_low_thresh=S.track_low_emb_confidence,
            new_track_thresh=S.track_new_threshold,
            track_buffer=S.track_buffer,
            proximity_thresh=0.8,
            appearance_thresh=0.5,
            match_thresh=0.8,
            mot20=False,
            with_reid=True,  # set True if you have ReID embeddings
            cmc_method='sparseOptFlow',  # for GMC
            name='BoTSORT',
            ablation=False
        )) # do configs 

        self._emb_buf = np.zeros((S.max_vis_detections, S._emb_out_size), dtype=np.float32)
        print(f'done loading hailo models, took {time.time() - ct:.1f} seconds')

    def run(self, frame: np.ndarray) -> Tuple[np.array, np.ndarray]:
        """
        frame is expected to be BGR format and in (S.frame_height, S.frame_width)
        """
        # prepare frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis_frame = np.ascontiguousarray(cv2.resize(frame, self.vm_shape))
        dep_frame = np.ascontiguousarray(cv2.resize(frame, self.dm_shape))

        # run vis and dep models
        self.vis_m.run([vis_frame], self.vis_cb)
        self.dep_m.run([dep_frame], self.dep_cb)

        # wait for both models to finish and enque
        vision = self.vis_q.get()

        detections = _extract_detections(frame, vision)

        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        num_detections = detections['num_detections']
        classes = detections['detection_classes']

        # collect only person class in detections. Perhaps finetune only for person class?
        persons = [i for i in range(num_detections) if classes[i] == 0] # class 0 is person on coco.txt 
        # perhaps detect knife or weapons? 

        ### thread embedding part to do cv2 ops while waiting
        ### perhaps try multiprocess if not performant enough.
        ### is this fruitless?

        embeddings = [np.zeros((S._emb_out_size,), dtype=np.float32) for _ in range(num_detections)]

        with ThreadPoolExecutor(max_workers=S.max_emb_threads) as executor:
            futures = []
            for i in persons:
                if scores[i] < S.min_emb_confidence: # don't try to embed when confidence is low.
                    continue
                x1, y1, x2, y2 = self._safe_box(boxes[i])
                crop = frame[y1:y2, x1:x2]
                if crop.size < S.min_emb_cropsize:
                    continue
                
                crop = cv2.resize(crop, self.em_shape)
                crop = np.ascontiguousarray(crop)
                futures.append(executor.submit(self._submit_embedding, crop, i))
        

        ### Tried batch processing, it's actually slower than threaded async. 
        # person_crops = []
        # det_ids = []
        # for i in persons:
        #     if scores[i] < S.min_emb_confidence:
        #         continue
        #     x1, y1, x2, y2 = self._safe_box(boxes[i])
        #     crop = frame[y1:y2, x1:x2]
        #     if crop.size < S.min_emb_cropsize:
        #         continue
        #     # crop = np.ascontiguousarray(cv2.resize(crop, self.em_shape))
        #     crop = cv2.resize(crop, self.em_shape)
        #     person_crops.append(crop)
        #     det_ids.append(i)
        
        # self.emb_m.run(person_crops, 
        #                partial(_batch_callback, 
        #                        output_queue = self.emb_q,
        #                        det_ids = det_ids
        #                        ))
        # detections[i] == embeddings[i] for ease of use. 
        # embeddings = [None] * num_detections
        # print(self.emb_m.hef.get_output_vstream_infos())

        # for _ in range(len(person_crops)):
        #     det_id, emb = self.emb_q.get()
        #     embeddings[det_id] = emb

        for _ in range(len(futures)):
            det_id, emb = self.emb_q.get() # will block here until all embedding results come back.
            embeddings[det_id] = emb

        targets = self.tracker.update(
            np.array([[*box, score] for box, score in zip(boxes, scores)]), # _extract_detections already took care of min conf. 
            frame,
            embeddings
            )
        
        rets = []
        for track in targets:
            t_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            
            rets.append((t_id, 'person', track.score, x1, y1, x2, y2))

        return rets, self.dep_q.get()
        

    def _submit_embedding(self, crop: np.ndarray, det_id:int):
        self.emb_m.run([crop], 
                       partial(_callback, output_queue = self.emb_q, det_id = det_id))


    def _safe_box(self, box: list) -> tuple[int, int, int, int]:
        """
        expetcs str box in shape of y1, x1, y2, y1 
        returns bound safe x1, y1, x2, y2. 
        """
        y1, x1, y2, x2 = map(int, box)
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(S.frame_height, y2)
        x2 = min(S.frame_width, x2)

        return x1, y1, x2, y2
    
    def close(self):
        self.vis_m.close()
        self.dep_m.close()
        self.emb_m.close()

    def __del__(self):
        self.close()






def _callback(bindings_list, output_queue, **kwargs) -> None:
    result = None

    for bindings in bindings_list:
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(
                    bindings.output(name).get_buffer(), axis=0
                )
                for name in bindings._output_names
            }
    if 'det_id' not in kwargs: # vis/dep callback
        output_queue.put_nowait(result)
        return
    # embedding callback
    # L2 vectorize
    # result = l2_norm(np.asarray(result).flatten())
    result = np.asarray(result).flatten()

    output_queue.put_nowait((kwargs.get('det_id'), result))

# def _batch_callback(bindings_list, output_queue, det_ids, **kwargs) -> None:
#     for det_id, bindings in zip(det_ids, bindings_list):
#         result = bindings.output().get_buffer()
#         result = np.asarray(result).flatten()
#         output_queue.put_nowait((det_id, result))











### from object_detection_postprocessing.py



def _denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box


def _extract_detections(image: np.ndarray, detections: list) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    score_threshold = S.min_vis_score_threshold
    max_detections = S.max_vis_detections

    #values used for scaling coords and removing padding
    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                denorm_bbox = _denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    #sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    #take top max_boxes
    top_detections = all_detections[:max_detections]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }

