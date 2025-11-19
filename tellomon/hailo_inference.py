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
        print('emb model:', S.embed_model)
        print('pos model:', S.pose_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.emb_m = HailoInfer(S.embed_model, output_type='UINT8') # dequantize on host to save bandwidth
        self.pos_m = HailoInfer(S.pose_model)

        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.pm_shape = tuple(reversed(self.pos_m.get_input_shape()[:2]))

        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""


    def __init__(self):
        print('hailo init')
        self.loaded = False
        self.vis_m = None
        self.emb_m = None
        self.pos_m = None

        self.vm_shape = None
        self.em_shape = None
        self.pm_shape = None

        self.vis_q = queue.Queue()
        self.emb_q = queue.Queue()
        self.pos_q = queue.Queue()

        self.vis_cb = partial(_callback, output_queue = self.vis_q)

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


    def letterbox_buffer(self, src: np.ndarray, dst, new_shape=640, color=(114,114,114)):
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

        # run vis model
        self.vis_m.run([self.vis_fb], self.vis_cb)

        # wait for vision model to finish 
        vision = self.vis_q.get()
        del vision[1:] # 0 is person, remove all other classes.
        # Perhaps add knife or some weapon types?

        boxes, scores, classes, num_detections = extract_detections(frame, vision)

        emb_ids = [] 
        emb_crops = np.empty((num_detections, self.em_shape[1], self.em_shape[0], 3), dtype=np.uint8)
        emb_crop_count = 0
        for i in range(num_detections):
            if scores[i] < S.min_emb_confidence:
                continue 
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            crop = frame[y1:y2, x1:x2]
            if crop.size < S.min_emb_cropsize:
                continue
            crop = cv2.resize(crop, self.em_shape, interpolation=cv2.INTER_LINEAR)
            emb_ids.append(i)
            emb_crops[emb_crop_count] = crop
            emb_crop_count += 1

        pos_ids = []
        pos_crops = np.empty((num_detections, self.pm_shape[1], self.pm_shape[0], 3), dtype=np.uint8)
        pos_crop_count = 0
        for i in range(num_detections):
            if scores[i] < S.min_pos_confidence:
                continue
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            crop = frame[y1:y2, x1:x2]
            if crop.size < S.min_emb_cropsize:
                continue
            crop = cv2.resize(crop, self.pm_shape, interpolation=cv2.INTER_LINEAR)
            pos_ids.append(i)
            pos_crops[pos_crop_count] = crop
            pos_crop_count += 1


        embeddings = [None] * num_detections
        poses = [None] * num_detections

        if emb_ids:
            self.emb_m.run(emb_crops[:emb_crop_count], partial(_batch_callback, output_queue = self.emb_q))
            _embs = self.emb_q.get()
            for i, det_id in enumerate(emb_ids):
                embeddings[det_id] = self._emb_deq_norm(_embs[i])

        if pos_ids:
            self.pos_m.run(pos_crops[:pos_crop_count], partial(_batch_callback, output_queue = self.pos_q))
            _poses = self.pos_q.get()
            for i, det_id in enumerate(pos_ids):
                poses[det_id] = self._deq(self.pos_m, _poses[i])

        targets = self.tracker.update(
            np.array([[*box, score] for box, score in zip(boxes, scores)]), 
            embeddings
            )
        
        rets = []
        for track in targets:
            t_id = getattr(track, 'identity_id', track.track_id)
            x1, y1, x2, y2 = track.last_bbox_tlbr
            
            rets.append((t_id, 'person', track.score, x1, y1, x2, y2))
        


        return rets, poses, boxes
        
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
        self.emb_m.close()
        self.pos_m.close()

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










