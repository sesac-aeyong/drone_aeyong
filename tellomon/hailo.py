from concurrent.futures import ThreadPoolExecutor
from functools import partial
import queue
import threading
import time
from types import SimpleNamespace
from typing import List, Tuple
import numpy as np
import cv2

from common.hailo_inference import HailoInfer
from tracker.bot_sort import BoTSORT
from settings import settings as S
from yolo_tools import extract_detections


class Hailo():
    """
    Hailo pipeline class.

    Hailo.run(frame):
    frame -> vis_model -> emb_model -> (dets, depth, yolobox)
          \\_ dep_model              /
    """
    def __init__(self):
        ct = time.time()
        print('loading Hailo models')
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
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

        # optimizations
        self.executor = ThreadPoolExecutor(max_workers=S.max_emb_threads)
        self.emb_buf = np.zeros((S.max_vis_detections, S._emb_out_size), dtype=np.float32)
        """embeddings buffer"""
        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""
        self.dep_fb = np.empty((self.dm_shape[1], self.dm_shape[0], 3), dtype=np.uint8)
        """depth model framebuffer"""
        self.crop_fbs = [np.empty((self.em_shape[1], self.em_shape[0], 3), dtype=np.uint8)
                          for _ in range(S.max_emb_threads)]
        """crop framebuffer per thread"""

        self._thread_local = threading.local()
        print(f'done loading hailo models, took {time.time() - ct:.1f} seconds')

    def run(self, frame: np.ndarray) -> Tuple[np.array, np.ndarray, List]:
        """
        frame is expected to be BGR format and in (S.frame_height, S.frame_width)
        output is detections, depth, list of yolo boxes
        """
        # prepare frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.resize(frame, self.vm_shape, self.vis_fb, interpolation=cv2.INTER_LINEAR)
        cv2.resize(frame, self.dm_shape, self.dep_fb, interpolation=cv2.INTER_LINEAR)

        # run vis and dep models
        self.vis_m.run([self.vis_fb], self.vis_cb)
        self.dep_m.run([self.dep_fb], self.dep_cb)

        # wait for vision model to run embeddings on.
        vision = self.vis_q.get()
        del vision[1:] # 0 is person, remove all other classes.

        detections = extract_detections(frame, vision)

        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        num_detections = detections['num_detections']
        classes = detections['detection_classes']

        # collect only person class in detections. Perhaps finetune only for person class?
        # persons = [i for i in range(num_detections) if classes[i] == 0] # class 0 is person on coco.txt 
        # perhaps detect knife or weapons? 

        ### thread embedding part to do cv2 ops while waiting
        ### perhaps try multiprocess if not performant enough.
        ### is this fruitless?

        futures = []
        emb_ids = []
        for i in range(num_detections):
            if scores[i] < S.min_emb_confidence: # don't try to embed when confidence is low.
                continue
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            crop = frame[y1:y2, x1:x2]
            if crop.size < S.min_emb_cropsize:
                continue
            buf = self._get_thread_buffer()
            cv2.resize(crop, self.em_shape, buf, interpolation=cv2.INTER_LINEAR)
            futures.append(self.executor.submit(self._submit_embedding, buf, i))
            emb_ids.append(i)
        
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
            self.emb_buf[det_id] = emb

        targets = self.tracker.update(
            np.array([[*box, score] for box, score in zip(boxes, scores)]), # _extract_detections already took care of min conf. 
            frame,
            self.emb_buf[:num_detections]
            )
        
        rets = []
        for track in targets:
            t_id = track.track_id
            x1, y1, x2, y2 = track.tlbr
            
            rets.append((t_id, 'person', track.score, x1, y1, x2, y2))

        # clean up used embedding buffers 
        for i in emb_ids:
            self.emb_buf[i].fill(0)

        return rets, self.dep_q.get(), boxes
        
    def _get_thread_buffer(self):
        if not hasattr(self._thread_local, 'buf'):
            # pick a buffer based on thread id
            tid = threading.get_ident() % S.max_emb_threads
            self._thread_local.buf = self.crop_fbs[tid]
        return self._thread_local.buf


    def _submit_embedding(self, crop: np.ndarray, det_id:int):
        self.emb_m.run([crop], 
                       partial(_callback, output_queue = self.emb_q, det_id = det_id))


    def _safe_box(self, box: list) -> tuple[int, int, int, int]:
        """
        expetcs str box in shape of y1, x1, y2, y2 
        returns bound safe x1, y1, x2, y2. 
        """
        x1, y1, x2, y2 = map(int, box)
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
        result = bindings.output().get_buffer()
    if 'det_id' not in kwargs: # vis/dep callback
        output_queue.put_nowait(result)
        return
    # embedding callback
    # L2 vectorize
    result = np.asarray(result).flatten() ## BoT-SORT normalizes vector, skip norm.

    output_queue.put_nowait((kwargs.get('det_id'), result))

# def _batch_callback(bindings_list, output_queue, det_ids, **kwargs) -> None:
#     for det_id, bindings in zip(det_ids, bindings_list):
#         result = bindings.output().get_buffer()
#         result = np.asarray(result).flatten()
#         output_queue.put_nowait((det_id, result))










