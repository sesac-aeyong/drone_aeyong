from concurrent.futures import ThreadPoolExecutor
from functools import partial
import queue
import threading
import time
from typing import List, Tuple
import numpy as np
import cv2

from common.hailo_inference import HailoInfer
from tracker.tracker_botsort import BoTSORT, LongTermBoTSORT, ThiefTracker
from settings import settings as S
from yolo_tools import extract_detections



class HailoRun():
    """
    Hailo pipeline class.

    Hailo.run(frame):
    frame -> vis_model -> emb_model -> (dets, depth, yolobox)
          \\_ dep_model              /
    """

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
        self.longterm_tracker = LongTermBoTSORT(base_tracker)

        # Active tracker pointer -> initially longterm
        self.active_tracker = self.longterm_tracker
        self._thief_tracker = None  # created when entering thief mode
        self.thief_id = 0

        self.executor = ThreadPoolExecutor(max_workers=S.max_emb_threads)
        self._thread_local = threading.local()


    def load(self):
        self.loaded = True
        ct = time.time()
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model, output_type='UINT16')
        self.emb_m = HailoInfer(S.embed_model, output_type='UINT8') 

        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.dm_shape = tuple(reversed(self.dep_m.get_input_shape()[:2]))

        # buffers
        self.emb_buf = np.zeros((S.max_vis_detections, S._emb_out_size), dtype=np.float32)
        """embeddings buffer"""
        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""
        self.dep_fb = np.empty((self.dm_shape[1], self.dm_shape[0], 3), dtype=np.uint8)
        """depth model framebuffer"""
        print(f'done loading hailo models, took {time.time() - ct:.1f} seconds')



    def enter_thief_mode(self, thief_id: int) -> bool:
        """
        특정 ID에 대해 ThiefTracker 활성화.
        gallery에서 임베딩을 가져와 active_tracker를 ThiefTracker로 전환.
        """
        if thief_id not in self.longterm_tracker.gallery:
            print(f"[THIEF] ERROR: identity_id {thief_id} not found in gallery")
            return False

        self.thief_id = thief_id
        thief_embs = self.longterm_tracker.gallery[thief_id]["gal_embs"]

        # ThiefTracker 초기화
        self._thief_tracker = ThiefTracker(thief_embs=thief_embs)
        self.active_tracker = self._thief_tracker
        
        print("[HailoRun] Entered thief mode (ThiefTracker active)")
        
        return True
    

    def exit_thief_mode(self) -> None:
        """Switch back to LongTermBoTSORT"""
        self.active_tracker = self.longterm_tracker
        self._thief_tracker = None
        self.thief_id = 0
        print("[HailoRun] Exited thief mode (LongTermBoTSORT active)")


    def letterbox_buffer(self, src, dst, new_shape=640, color=(114,114,114)) -> Tuple[int, Tuple]:
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

        # Resize into a temporary buffer on the fly
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        top = (new_shape - nh) // 2
        left = (new_shape - nw) // 2

        # Copy resized into letterboxed area of dst
        dst[top:top+nh, left:left+nw] = resized

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

        boxes, scores, _, num_detections = extract_detections(frame, vision)

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

        return rets, depth, boxes
    

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
    result = np.asarray(result).flatten() 
    output_queue.put_nowait((kwargs.get('det_id'), result))


def _batch_callback(bindings_list, output_queue, **kwargs) -> None:
    """
    batch callback, single output
    """
    result = []
    for bindings in bindings_list:
        data = bindings.output().get_buffer()
        result.append(data)

    output_queue.put_nowait(result)








