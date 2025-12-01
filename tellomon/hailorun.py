# hailorun.py
from functools import partial
import queue
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
    frame -> vis_model -> emb_model -> (dets, depth)
          \\_ dep_model(thief)        /
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
        self.dm_scale = None

        self.vis_q = queue.Queue()
        self.dep_q = queue.Queue()
        self.emb_q = queue.Queue()
        self.pos_q = queue.Queue()

        self.vis_cb = partial(_callback, output_queue = self.vis_q)
        self.dep_cb = partial(_callback, output_queue = self.dep_q)

        base_tracker = BoTSORT()
        self.longterm_tracker = LongTermBoTSORT(base_tracker)

        # Active tracker pointer -> initially longterm
        self.active_tracker = self.longterm_tracker
        self._thief_tracker = None  # created when entering thief mode
        self.thief_id = 0


    def load(self):
        self.loaded = True
        ct = time.time()
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
        print('pose model:', S.pose_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model, output_type='UINT16')
        self.emb_m = HailoInfer(S.embed_model, output_type='FLOAT32') 
        self.pos_m = HailoInfer(S.pose_model, output_type='FLOAT32')

        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.dm_shape = tuple(reversed(self.dep_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.pm_shape = tuple(reversed(self.pos_m.get_input_shape()[:2]))

        # buffers
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
    

    def run(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        frame is expected to be BGR format and in (S.frame_height, S.frame_width)
        output is detections, depth, list of yolo boxes
        """
        # if frame.shape[0] < 500 or frame.shape[1] < 600:
        #     print('[HailoRun] Skipping wrong inputs')
        #     return [], np.zeros((self.dm_shape[0], self.dm_shape[1], 1)), []
        # prepare and run vis and dep models
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.letterbox_buffer(frame, self.vis_fb)
        self.vis_m.run([self.vis_fb], self.vis_cb)

        # depth model only when thief mode is active
        depth_ran = False
        if self.thief_id > 0:
            self.dep_fb[...] = cv2.resize(frame, self.dm_shape, interpolation=cv2.INTER_LINEAR)
            self.dep_m.run([self.dep_fb], self.dep_cb)
            depth_ran = True

        # wait for vision model to run embeddings on.
        vision = self.vis_q.get()
        del vision[1:] # 0 is person, remove all other classes.
        # Perhaps add knife or some weapon types?

        boxes, scores, _, num_detections = extract_detections(frame, vision)

        emb_ids = [] 
        emb_crops = []
        for i in range(num_detections):
            if scores[i] < S.min_emb_confidence:
                continue 
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            if (y2 - y1) <= 0 or (x2 - x1) <= 0 or ((y2 - y1) * (x2 - x1)) < S.min_emb_cropsize:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], self.em_shape, interpolation=cv2.INTER_LINEAR)
            emb_ids.append(i)
            emb_crops.append(crop)

        embeddings = [None] * num_detections

        # pos_ids = []
        # pos_crops = []
        # for i in range(num_detections):
        #     if scores[i] < 0.5: # S.min_pos_confidence
        #         continue
        #     x1, y1, x2, y2 = self._safe_box(boxes[i])
        #     if (y2 - y1) <= 0 or (x2 - x1) <= 0 or ((y2 - y1) * (x2 - x1)) < 50: #S.min_pos_cropsize:
        #         continue
        #     crop = cv2.resize(frame[y1:y2, x1:x2], self.pm_shape, interpolation=cv2.INTER_LINEAR)
        #     pos_ids.append((i, x1, y1, x2 - x1, y2 - y1))
        #     pos_crops.append(crop)

        # poses = [None] * num_detections
        # frame_vis = frame.copy()
        # if pos_ids:
        #     self.pos_m.run(np.stack(pos_crops, axis=0), partial(_pose_callback, output_queue = self.pos_q, info = pos_ids))
        #     _poses = self.pos_q.get()

#             for _i, (det_id, *_) in enumerate(pos_ids):
#                 keypoints = _poses[_i]
#                 limbs = [
#     (0, 1), (0, 2),       # Nose → eyes
#     (1, 3), (2, 4),       # Eyes → ears
#     (0, 5), (0, 6),       # Nose → shoulders
#     (5, 7), (7, 9),       # Left arm
#     (6, 8), (8, 10),      # Right arm
#     (5, 11), (6, 12),     # Shoulders → hips
#     (11, 12),             # Hip line
#     (11, 13), (13, 15),   # Left leg
#     (12, 14), (14, 16)    # Right leg
# ]               
#                 conf_thresh = 60
#                 for i, (x, y, conf) in enumerate(keypoints):
#                     if conf > conf_thresh:
#                         cv2.circle(frame_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
#                         cv2.putText(frame_vis, str(i), (int(x)+2,int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)

#                 # Draw limbs
#                 for a, b in limbs:
#                     if keypoints[a][2] > conf_thresh and keypoints[b][2] > conf_thresh:
#                         pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
#                         pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
#                         cv2.line(frame_vis, pt1, pt2, (255, 0, 0), 2)

#                 cv2.imshow('crop', frame_vis)
#                 cv2.waitKey(1)
                
        if emb_ids:
            self.emb_m.run(np.stack(emb_crops, axis=0), partial(_batch_callback, output_queue = self.emb_q))
            _embs = self.emb_q.get()
            for i, det_id in enumerate(emb_ids):
                embeddings[det_id] = self._emb_norm(_embs[i])

        targets = self.active_tracker.update(
            np.array([[*box, score] for box, score in zip(boxes, scores)]), 
            embeddings
            )

        
        rets = []
        for track in targets:            
            # ① iid는 절대 덮어쓰지 말고 트래커 산출값 유지
            iid = getattr(track, 'identity_id', None)
            det = {
                "identity_id": iid,
                "confidence": float(track.score),
                "bbox": list(map(int, track.last_bbox_tlbr)),
                "class": "person",
            }
                
            # 도둑 모드라면 부가 정보 표시용 필드 추가
            if (self.thief_id != 0):
                det["thief_dist"] = float(getattr(track, "thief_dist", 1.0))
                det["thief_cos_dist"] = float(getattr(self.active_tracker, "thief_cos_dist", getattr(S, "thief_cos_dist", 0.3)))
                x1, y1, x2, y2 = map(int, track.last_bbox_tlbr)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[0], x2)
                y2 = min(frame.shape[1], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    self.pos_m.run([cv2.resize(crop, self.pm_shape, interpolation=cv2.INTER_LINEAR)],
                                    partial(_pose_callback, output_queue = self.pos_q, info = (0, x1, y1, x2 - x1, y2 - y1)))
                    det["pose"] = self.pos_q.get()
            rets.append(det)
        
        if self.thief_id and depth_ran:
            return rets, cv2.resize(self.dep_q.get(), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST), []
        return rets, np.zeros_like(frame), []
    
            
    def _safe_box(self, box: list) -> tuple[int, int, int, int]:
        """
        returns bound safe x1, y1, x2, y2. 
        """
        x1, y1, x2, y2 = map(int, box)
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(S.frame_height, y2)
        x2 = min(S.frame_width, x2)

        return x1, y1, x2, y2


    def _emb_norm(self, emb):
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    
    
    def close(self):
        self.vis_m.close()
        self.dep_m.close()
        self.emb_m.close()

    def __del__(self):
        self.close()

        
    def _deq(self, model, data):
        out = model.infer_model.outputs[0]
        qi = out.quant_infos[0]
        return (data - qi.qp_zp) * qi.qp_scale



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
    batch callback, single output for embedding vectors
    """
    result = []
    for bindings in bindings_list:
        data = bindings.output().get_buffer()
        result.append(np.asarray(data).flatten())

    output_queue.put_nowait(result)


def _pose_callback(bindings_list, output_queue, info, **kwargs) -> None:
    """
    pose callback, single output
    """

    for bindings in bindings_list:
        data = bindings.output().get_buffer()
        keypoints = []
        _, _x, _y, _w, _h = info
                
        for k in range(17):
            hm = data[:, :, k]            # single keypoint heatmap
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            scale_x = (_w) / hm.shape[1] / 4  # hm.shape is 64x48
            scale_y = (_h) / hm.shape[0] / 4
            keypoints.append((float(x * 4 * scale_x + _x), 
                              float(y * 4 * scale_y + _y), 
                              float(hm[y, x])))          # (x, y) in heatmap coordinates
        
        output_queue.put_nowait(keypoints)




