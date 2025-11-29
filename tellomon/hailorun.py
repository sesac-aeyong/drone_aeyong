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
        self.dm_scale = None

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
        
        S.laser_x_pixels = np.array([85, 63.35, 48.5, 42.45, 40.1, 38])
        S.laser_y_pixels = np.array([38, 26, 17.2, 14.35, 12.9, 12])
        S.laser_distances = np.array([30, 60, 120, 180, 240, 300])


    def load(self):
        self.loaded = True
        ct = time.time()
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model, output_type='UINT16')
        self.emb_m = HailoInfer(S.embed_model, output_type='FLOAT32') 

        self.vm_shape = tuple(reversed(self.vis_m.get_input_shape()[:2]))
        self.em_shape = tuple(reversed(self.emb_m.get_input_shape()[:2]))
        self.dm_shape = tuple(reversed(self.dep_m.get_input_shape()[:2]))

        # buffers
        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""
        self.dep_fb = np.empty((self.dm_shape[1], self.dm_shape[0], 3), dtype=np.uint8)
        """depth model framebuffer"""
        print(f'done loading hailo models, took {time.time() - ct:.1f} seconds')

        self.vis_to_dep_ratio = (self.dm_shape[1] / self.vm_shape[1],
                                 self.dm_shape[0] / self.vm_shape[0])

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
    

    def run(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray, List]:
        """
        frame is expected to be BGR format and in (S.frame_height, S.frame_width)
        output is detections, depth, list of yolo boxes
        """
        if frame.shape[0] != S.frame_height or frame.shape[1] != S.frame_width:
            # print('[HailoRun] Skipping wrong inputs')
            return [], np.zeros((self.dm_shape[0], self.dm_shape[1], 1)), []
        # prepare and run vis and dep models
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.letterbox_buffer(frame, self.vis_fb)
        self.vis_m.run([self.vis_fb], self.vis_cb)
        self.dep_fb[...] = cv2.resize(frame, self.dm_shape, interpolation=cv2.INTER_LINEAR)
        self.dep_m.run([self.dep_fb], self.dep_cb)

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
            rets.append(det)
        
        depth = self.recover_depth(frame, self.dep_q.get())

        return rets, depth, boxes
    
            
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


    def recover_depth(self, frame, dep_rel_quant) -> np.array:
        """
        returns depth_abs recovered from laser point
        """
        ### dequantize and apply hailo's post process
        qi = self.dep_m.infer_model.outputs[0].quant_infos[0]
        dep_rel = (dep_rel_quant.astype(np.float32) - qi.qp_zp) * qi.qp_scale
        dep_rel = 1 / (1 + np.exp(-dep_rel))
        dep_rel = 1.0 / (dep_rel * 10.0 + 0.009)
        # normalize
        # dep_rel = cv2.normalize(dep_rel, None, 0, 255, cv2.NORM_MINMAX)
        dep_rel = cv2.resize(dep_rel, (self.vm_shape[0], self.vm_shape[1]), interpolation=cv2.INTER_NEAREST)


        laser_roi = frame[S.laser_roi_y1:S.laser_roi_y2, S.laser_roi_x1:S.laser_roi_x2]
        laser_roi_b = cv2.cvtColor(laser_roi, cv2.COLOR_BGR2GRAY)
        laser_roi_b = cv2.GaussianBlur(laser_roi_b, (3, 3), 0)
        laser_roi_hsv = cv2.cvtColor(laser_roi, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(laser_roi_hsv)
        # cv2.imshow('laser roi', laser_roi)
        edges = cv2.Canny(laser_roi_b, S.laser_canny_lower_threshold, S.laser_canny_high_threshold)
        # cv2.imshow('edges', edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # frame_vis = frame.copy()
        # cv2.imshow('frame', frame_vis)
        # cv2.waitKey(1)
        
        for cnt in contours:
            #skip big contours
            if cv2.contourArea(cnt) > S.laser_dot_size_threshold:
                continue
            _m = cv2.moments(cnt)
            if _m['m00'] == 0:
                continue

            peri = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity < S.laser_circularity_threshold:  # adjustable
                continue
            (_x, _y), r = cv2.minEnclosingCircle(cnt)

            cx_crop = _m['m10'] / _m['m00']
            cy_crop = _m['m01'] / _m['m00']
            ax, ay = S.laser_roi_x1 + int(cx_crop), S.laser_roi_y1 + int(cy_crop)
            dx, dy = int(ax * self.vis_to_dep_ratio[1]), int(ay * self.vis_to_dep_ratio[0])
            dx = np.clip(dx, 0, self.dm_shape[1] - 1)
            dy = np.clip(dy, 0, self.dm_shape[0] - 1)

            # cv2.circle(frame_vis, (ax, ay), 9, (0, 255, 255), 1)
            # cv2.imshow('h', h)
            # cv2.imshow('s', s)
            # cv2.imshow('v', v)
            red_mask = cv2.inRange(laser_roi_hsv, S.red_mll, S.red_mlu)
            red_mask |= cv2.inRange(laser_roi_hsv, S.red_mul, S.red_muu)
            red_mask = cv2.dilate(red_mask, np.ones((3, 3), np.uint8), 1)
            # cv2.imshow('rm', red_mask)
            laser_dotm = np.zeros_like(laser_roi_b)
            cv2.circle(laser_dotm, (int(cx_crop), int(cy_crop)), int(r) + 3, 255, -1)
            # cv2.imshow('laser_dotm', laser_dotm)
            laser_dotm = cv2.bitwise_and(red_mask, laser_dotm)
            # cv2.imshow('bwa', laser_dotm)
            # cv2.circle(frame_vis, (ax, ay), 1, (0, 0, 255), -1)
            # cv2.putText(frame_vis, f'cx:{cx_crop:.2f} cy:{cy_crop:.2f}', (ax, ay + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            if cv2.countNonZero(laser_dotm) <= 0:
                continue
            # print('laser contourarea:', cv2.contourArea(cnt))

            laser_abs_depth = float(S.laser_distf(cy_crop))
            # cv2.putText(frame_vis, f'depth: {laser_abs_depth:.1f}CM', (ax, ay + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            # cv2.imshow('vis', frame_vis) # imshows left for later debugging
            # cv2.waitKey(999999)
            # pick and average nearby values 
            patch = dep_rel[max(0, dy - 1):dy + 2, max(0, dx - 1):dx + 2]
            laser_rel_depth = np.mean(patch)
            # laser_rel_depth = dep_rel[dy, dx]
            self.dm_scale = laser_abs_depth / laser_rel_depth
            # print(self.dm_scale, laser_abs_depth, laser_rel_depth)
            # print(self.dm_scale * dep_rel[dy, dx])
            break

        # print(self.dm_scale)
        if self.dm_scale:
            return dep_rel * self.dm_scale
        else:
            return dep_rel
        


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

