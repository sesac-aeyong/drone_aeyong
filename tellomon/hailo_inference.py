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
# from tracker.bot_sort import BoTSORT
from tracker.utils.gallery_io import load_gallery
from tracker.tracker_botsort import BoTSORT, LongTermBoTSORT
from settings import settings as S
from yolo_tools import extract_detections


GALLERY_PATH = "cache/longterm_gallery.npy" 



class HailoRun():
    """
    Hailo pipeline class.

    Hailo.run(frame):
    frame -> vis_model -> emb_model -> (dets, depth, yolobox)
          \\_ dep_model              /
    """
    def load(self):
        self.loaded = True
        ct = time.time()
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model, output_type='UINT16')
        self.emb_m = HailoInfer(S.embed_model, output_type='UINT8') # dequantize on host to save bandwidth

        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.dm_shape = tuple(reversed(self.dep_m.get_input_shape()[:2]))

        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""
        self.dep_fb = np.empty((self.dm_shape[1], self.dm_shape[0], 3), dtype=np.uint8)
        """depth model framebuffer"""
        print(f'done loading hailo models, took {time.time() - ct:.1f} seconds')


    def __init__(self):
        print('hailo init')
        self.loaded = False
        self.vis_m = None
        self.dep_m = None
        self.emb_m = None

        self.vm_shape = None
        self.em_shape = None
        self.dm_shape = None

        self.vis_q = queue.Queue()
        self.dep_q = queue.Queue()
        self.emb_q = queue.Queue()

        self.vis_cb = partial(_callback, output_queue = self.vis_q)
        self.dep_cb = partial(_callback, output_queue = self.dep_q)

        base_tracker = BoTSORT()
        self.tracker = LongTermBoTSORT(base_tracker)

        gallery = load_gallery(GALLERY_PATH)
        if len(gallery) > 0:
            self.tracker.gallery = gallery
            self.tracker.next_identity = max(gallery.keys()) + 1
            print("[LT-GAL] start AGAIN with saved gallery")
        else:
            self.tracker.gallery = {}
            self.tracker.next_identity = 1
            print("[LT-GAL] NEW start with empty gallery")


    def letterbox_buffer(self, src, dst, new_shape=640, color=(114,114,114)):
        """
        src: input image (H, W, 3)
        dst: preallocated buffer (new_shape, new_shape, 3)
        returns: scale, (left, top)
        """

        h, w = src.shape[:2]
        scale = min(new_shape / w, new_shape / h)
        nh, nw = int(h * scale), int(w * scale)

        # Fill destination with padding color
        dst[:] = color

        # Compute padding
        top = (new_shape - nh) // 2
        left = (new_shape - nw) // 2

        # Directly resize into the subregion of dst
        roi = dst[top:top+nh, left:left+nw]
        cv2.resize(src, (nw, nh), dst=roi, interpolation=cv2.INTER_LINEAR)

        return scale, (left, top)
    

    def run(self, frame: np.ndarray) -> Tuple[np.array, np.ndarray, List]:
        """
        frame is expected to be BGR format and in (S.frame_height, S.frame_width)
        output is detections, depth, list of yolo boxes
        """
        # prepare frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.letterbox_buffer(frame, self.vis_fb)
        cv2.resize(frame, self.dm_shape, self.dep_fb, interpolation=cv2.INTER_LINEAR)

        # run vis and dep models
        self.vis_m.run([self.vis_fb], self.vis_cb)
        self.dep_m.run([self.dep_fb], self.dep_cb)

        # wait for vision model to run embeddings on.
        vision = self.vis_q.get()
        depth = self.dep_q.get()
        del vision[1:] # 0 is person, remove all other classes.
        # Perhaps add knife or some weapon types?

        detections = extract_detections(frame, vision)

        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        num_detections = detections['num_detections']
        classes = detections['detection_classes']

        emb_ids = [] 
        emb_crops = np.empty((num_detections, self.em_shape[1], self.em_shape[0], 3), dtype=np.uint8)
        crop_count = 0
        for i in range(num_detections):
            if scores[i] < S.min_emb_confidence:
                continue 
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            crop = frame[y1:y2, x1:x2]
            if crop.size < S.min_emb_cropsize:
                continue
            crop = cv2.resize(crop, self.em_shape, interpolation=cv2.INTER_LINEAR)
            emb_ids.append(i)
            emb_crops[crop_count] = crop
            crop_count += 1

        embeddings = [None] * num_detections

        if emb_ids:
            self.emb_m.run(emb_crops[:crop_count], partial(_batch_callback, output_queue = self.emb_q))
            _embs = self.emb_q.get()
            for i, det_id in enumerate(emb_ids):
                embeddings[det_id] = self._emb_deq_norm(_embs[i])

        targets = self.tracker.update(
            np.array([[*box, score] for box, score in zip(boxes, scores)]), 
            embeddings
            )
        
        rets = []
        for track in targets:
            t_id = getattr(track, 'identity_id', track.track_id)
            x1, y1, x2, y2 = track.last_bbox_tlbr
            
            rets.append((t_id, 'person', track.score, x1, y1, x2, y2))
        

        depth = self._dep_deq(depth)
        depth_norm = cv2.normalize(depth.squeeze(), None, 0, 255, cv2.NORM_MINMAX)

        return rets, depth_norm, boxes
        
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

    def _deq(self, model, data):
        out = model.infer_model.outputs[0]
        qi = out.quant_infos[0]
        return (data - qi.qp_zp) * qi.qp_scale

    def _emb_deq_norm(self, emb):
        emb = np.asarray(emb).flatten()
        deq = self._deq(self.emb_m, emb)
        # return deq 
        norm = np.linalg.norm(deq)
        return deq / norm if norm > 0 else deq
    
    def _dep_deq(self, data):
        deq = self._deq(self.dep_m, data)

        depth = np.exp(-deq)
        depth = 1.0 / (1.0 + depth)
        depth = 1.0 / (depth * 10.0 + 0.009)
        return depth
    
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
    output_queue.put_nowait((kwargs.get('det_id'), result))
    

def _batch_callback(bindings_list, output_queue, **kwargs) -> None:
    """
    batch callback, single output
    """
    # print('batch')
    result = []
    for bindings in bindings_list:
        data = bindings.output().get_buffer()
        
        # data = np.asarray(data).flatten()
        result.append(data)

    output_queue.put_nowait(result)










