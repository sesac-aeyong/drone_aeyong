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
from yolo_tools import extract_detections, draw_detection



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

        # buffers
        self.emb_buf = np.zeros((S.max_vis_detections, S._emb_out_size), dtype=np.float32)
        """embeddings buffer"""
        self.vis_fb = np.empty((self.vm_shape[1], self.vm_shape[0], 3), dtype=np.uint8)
        """vision model framebuffer"""
        self.dep_fb = np.empty((self.dm_shape[1], self.dm_shape[0], 3), dtype=np.uint8)
        """depth model framebuffer"""
        self.crop_fbs = [np.empty((self.em_shape[1], self.em_shape[0], 3), dtype=np.uint8)
                          for _ in range(S.max_emb_threads)]
        """crop framebuffer per thread"""
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
        self.longterm_tracker = LongTermBoTSORT(base_tracker)

        # Active tracker pointer -> initially longterm
        self.active_tracker = self.longterm_tracker
        self._thief_tracker = None  # created when entering thief mode
        self.thief_id = 0

        self.executor = ThreadPoolExecutor(max_workers=S.max_emb_threads)
        self._thread_local = threading.local()

    def enter_thief_mode(self, thief_id: int):
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

    def exit_thief_mode(self):
        """Switch back to LongTermBoTSORT"""
        self.active_tracker = self.longterm_tracker
        self._thief_tracker = None
        self.thief_id = 0
        print("[HailoRun] Exited thief mode (LongTermBoTSORT active)")

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
        del vision[1:] # 0 is person, remove all other classes.

        detections = extract_detections(frame, vision)

        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        num_detections = detections['num_detections']

        futures = []
        emb_ids = []
        for i in range(num_detections):
            if scores[i] < S.min_emb_confidence: # don't try to embed when confidence is low.
                continue 
            x1, y1, x2, y2 = self._safe_box(boxes[i])
            w = max(0, x2 - x1) ; h = max(0, y2 - y1) ; area = w * h
            if area < S.min_emb_cropsize:            # don't try to embed when area is small.
                continue
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, self.em_shape, interpolation=cv2.INTER_LINEAR)
            futures.append(self.executor.submit(self._submit_embedding, crop, i))
            emb_ids.append(i)

        embeddings = [None] * num_detections

        for _ in range(len(emb_ids)):
            det_id, emb = self.emb_q.get() # will block here until all embedding results come back.
            embeddings[det_id] = self._emb_deq_norm(emb)

        # Convert now_dets to np.array shape (N,5)
        now_dets = np.array([[*box, score] for box, score in zip(boxes, scores)], dtype=np.float32)

        # Call active tracker (either LongTermBoTSORT or ThiefTracker)
        targets = self.active_tracker.update(now_dets, embeddings)

        rets = []
        for track in targets:
            # 1) 항상 단기 추적 ID는 BoTSORT의 것을 유지
            tid = int(getattr(track, 'track_id', -1))
            
            # 2) identity/visible 결정
            if self.thief_id != 0:
                # ThiefTracker 모드
                iid = int(self.thief_id)   # 도둑 모드: 고정
                vid  = iid          # 도둑 모드: 항상 숫자 노출
            else:
                # LongTermBoTSORT 모드
                iid = getattr(track, 'identity_id', None)
                vid  = getattr(track, 'identity_visible', None)  # 갤러리 조건 충족 시에만 표시
            x1, y1, x2, y2 = map(int, track.last_bbox_tlbr)
            rets.append({
                    'track_id': tid,          # ← 단기 추적 ID로 고정
                    'identity_id': iid,       # ← 장기 ID는 별도 필드로
                    'identity_visible': vid,  # ← 화면 라벨
                    'class': 'person',
                    'confidence': float(track.score),
                    'bbox': [x1, y1, x2, y2]
                })


        depth = self._dep_deq(self.dep_q.get())
        # print(depth[160,128])
        depth_norm = cv2.normalize(depth.squeeze(), None, 0, 255, cv2.NORM_MINMAX)
        # depth_uint8 = depth_norm.astype(np.uint8)
        # depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        # depth_color = cv2.resize(depth_color, (640, 480))
        # cv2.imshow("Relative Depth", depth_color)

        # print(depth)

        return rets, depth, boxes

    def draw_detections_on_frame(self, frame, detections, target_track_id=None):
        """
        프레임에 감지 결과 그리기
        
        Args:
            frame: RGB 이미지
            detections: 감지된 객체 리스트 (bbox in [x1, y1, x2, y2] format)
            target_track_id: 추적 중인 타겟의 track_id (빨간색으로 표시)
        
        Returns:
            annotated_frame: 감지 결과가 그려진 프레임
        """
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        for det in detections:
            label = [det['class']]
            score = float(det['confidence']) * 100.0
            x1, y1, x2, y2 = det['bbox']  # [x1, y1, x2, y2] format
            
            tid = det['track_id']                     # 단기 추적 ID
            iid = det.get('identity_id', None)        # 장기 ID (없으면 폴백)
            vis = det.get('identity_visible', None)   # 화면 표시용
            
            # 도둑 모드면 표시 폴백
            if self.thief_id != 0 and vis is None:
                vis = self.thief_id  # 화면 라벨 강제
                
            # 타깃 강조 조건
            if self.thief_id != 0:
                is_target = (vis == self.thief_id)
            else:
                is_target = (tid == target_track_id)
            color = (0, 0, 255) if is_target else (255, 255, 255)  # 추적 중인 타겟이면 빨간색, 아니면 흰색

            draw_detection(
                annotated_frame, [x1, y1, x2, y2],
                label, score, color, True,
                identity_visible=vis,
            )

            # 추적 중인 타겟이면 중심점도 그리기
            if is_target:
                # bbox를 프레임 범위 내로 클리핑
                x1_clipped = max(0, min(x1, w - 1))
                y1_clipped = max(0, min(y1, h - 1))
                x2_clipped = max(0, min(x2, w - 1))
                y2_clipped = max(0, min(y2, h - 1))
                
                # 유효한 bbox인지 확인
                if x2_clipped > x1_clipped and y2_clipped > y1_clipped:
                    # 클리핑된 bbox의 중심점 계산
                    center_x = int((x1_clipped + x2_clipped) / 2)
                    center_y = int((y1_clipped + y2_clipped) / 2)
                    
                    # 중심점이 프레임 내부에 있을 때만 그리기
                    if 0 <= center_x < w and 0 <= center_y < h:
                        cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 0, 0), -1)
                        cv2.circle(annotated_frame, (center_x, center_y), 15, (255, 0, 0), 2)
        
        return annotated_frame

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
    # L2 vectorize
    result = np.asarray(result).flatten() 

    output_queue.put_nowait((kwargs.get('det_id'), result))