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

        self.optical_flow = OpticalFlowEgoMotion()


    def load(self):
        self.loaded = True
        ct = time.time()
        print('vis model:', S.vis_model)
        print('dep model:', S.depth_model)
        print('emb model:', S.embed_model)
        self.vis_m = HailoInfer(S.vis_model)
        self.dep_m = HailoInfer(S.depth_model, output_type='FLOAT32')
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
        
        depth_raw = self.dep_q.get()
        depth = self._process_depth(depth_raw)

        # 배경 마스크 생성 (YOLO detection 영역 제외)
        h, w = frame.shape[:2]
        background_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        for i in range(num_detections):
            if scores[i] < 0.3:
                continue
            x1, y1, x2, y2 = map(int, boxes[i])
            # Dilate detection box
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            background_mask[y1:y2, x1:x2] = 0
        
        # Optical flow 계산
        flow_result = self.optical_flow.compute_sparse_optical_flow(
            frame, background_mask
        )
        
        ego_velocity = None
        has_flow = False

        if flow_result is not None:
            good_new, good_old = flow_result
            ego_velocity = self.optical_flow.estimate_ego_velocity(
                good_new, good_old, depth
            )
            has_flow = True
        
        optical_flow_data = {
            'ego_velocity': ego_velocity,  # [vx, vy] in m/s
            'background_mask': background_mask,
            'has_flow': has_flow 
        }
        
        return rets, depth, boxes, optical_flow_data
    
    def _process_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        """
        SC-DepthV3 raw output을 실제 거리(미터)로 변환
        
        SC-DepthV3는 log-space disparity를 출력:
        - 음수 값이 정상
        - exp(-depth_raw)로 변환 필요
        """
        # 디버깅 (필요시 주석 처리)
        #print(f"[DEPTH DEBUG] Raw depth: min={np.min(depth_raw):.4f}, "
        #    f"max={np.max(depth_raw):.4f}, mean={np.mean(depth_raw):.4f}")
        
        # Log-space disparity → Disparity 변환
        # SC-DepthV3: disparity = exp(-depth_raw)
        disparity = np.exp(-depth_raw)
        
        # Disparity → Depth 변환
        # depth = 1 / disparity (역수 관계)
        # 하지만 직접 역수는 불안정하므로 정규화 후 처리
        
        # 정규화 [0, 1]
        disp_min = np.min(disparity)
        disp_max = np.max(disparity)
        disp_normalized = (disparity - disp_min) / (disp_max - disp_min + 1e-6)
        
        # Normalized disparity → Depth
        # 높은 disparity (1에 가까움) = 가까운 거리
        # 낮은 disparity (0에 가까움) = 먼 거리
        depth = 1.0 / (disp_normalized + 0.01)  # 0.01로 0 나누기 방지
        
        DEPTH_SCALE = 1.4
        depth = depth * DEPTH_SCALE
        
        # 유효 범위로 클리핑
        depth = np.clip(depth, 0.1, 10.0)
        
        # 디버깅 (필요시 주석 처리)
        print(f"[DEPTH DEBUG] Processed depth: min={np.min(depth):.4f}, "
            f"max={np.max(depth):.4f}, mean={np.mean(depth):.4f}")
        
        return depth
                
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
    

    def _deq(self, model, data):
        out = model.infer_model.outputs[0]
        qi = out.quant_infos[0]
        return (data - qi.qp_zp) * qi.qp_scale


    def _emb_norm(self, emb):
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    
    
    def close(self):
        self.vis_m.close()
        self.dep_m.close()
        self.emb_m.close()

    def __del__(self):
        self.close()

    def extract_target_depth(self, depth_map: np.ndarray, target_bbox: Tuple[int, int, int, int]):
        """
        특정 타겟의 depth 값을 추출 (배경 영향 최소화)
        """
        if target_bbox is None:
            return None
        
        x1, y1, x2, y2 = map(int, target_bbox)
        
        # BBox 유효성 검사
        h, w = depth_map.shape[:2]
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None
        
        # ✨ 중앙 60% 영역만 사용 (배경 제거)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        center_ratio = 0.6
        
        x1_c = int(x1 + bbox_w * (1 - center_ratio) / 2)
        y1_c = int(y1 + bbox_h * (1 - center_ratio) / 2)
        x2_c = int(x2 - bbox_w * (1 - center_ratio) / 2)
        y2_c = int(y2 - bbox_h * (1 - center_ratio) / 2)
        
        # 영역이 너무 작으면 원본 사용
        if (x2_c - x1_c) < 10 or (y2_c - y1_c) < 10:
            x1_c, y1_c, x2_c, y2_c = x1, y1, x2, y2
        
        # 타겟 영역 추출
        target_depth_region = depth_map[y1_c:y2_c, x1_c:x2_c]
        
        if target_depth_region.size == 0:
            return None
        
        # ✨ 이상치 제거
        depths_flat = target_depth_region.flatten()
        
        # 1. 유효 범위 필터링
        valid_depths = depths_flat[(depths_flat > 0.5) & (depths_flat < 8.0)]
        
        if len(valid_depths) == 0:
            return None
        
        # 2. 중앙값 기준 ±2m 이내만 사용
        median_depth = np.median(valid_depths)
        filtered_depths = valid_depths[np.abs(valid_depths - median_depth) < 2.0]
        
        if len(filtered_depths) == 0:
            return median_depth
        
        # 3. 최종 중앙값 반환
        return float(np.median(filtered_depths))


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

