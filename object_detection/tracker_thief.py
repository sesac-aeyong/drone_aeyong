# tracker_thief.py
"""
도둑(이미 선택된 identity_id) 한 명만 추적 + 갤러리 업데이트 트래커.

전제:
  - 도둑 찾기 모드(LongTermBoTSORT)에서 identity 갤러리를 충분히 모아둔 상태.
  - 도둑 ID가 선택되면, 해당 ID의 gal_embs 리스트만 따로 모아서
      thief_embs = [emb1, emb2, ...] 또는 (K, D) array
    형태로 이 트래커에 넘긴다.

Pipeline:

1. YOLO → now_dets   (각각 now_bbox_tlbr, now_score, now_cls)
2. crop → OVReID → now_emb(t)
3. ThiefTracker.update(now_dets, now_embs) 호출
   내부 동작:
     - TrackState.predict() → pred_bbox_tlbr
     - pred_bbox_tlbr & 도둑 갤러리(thief_embs)와의 cos_dist 기반으로
       "도둑일 가능성이 높은" detection만 매칭
     - TrackState.update()로 칼만 보정 / 상태 갱신
     - cos_dist가 적당한 범위이면 갤러리에 new emb 추가 (도둑 갤러리 업데이트)
4. 화면에는 반환된 Track들을 그대로 빨간 박스로 overlay (논리상 0 또는 1개).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.metrics import iou_bbox, cosine_distance, min_cos_dist_to_list
from utils.track_state import TrackState


class ThiefTracker:
    """
    선택된 도둑 identity 하나만 추적하는 단일 타깃 트래커.

    매 프레임(t) 절차:
      1) 모든 TrackState.predict() 호출 (last → pred)
      2) YOLO now_dets + ReID now_embs 에 대해:
           - 도둑 임베딩(thief_embs)과의 코사인 거리로 "도둑 후보" 필터링
           - IoU(pred vs now) + cos_dist(thief) 를 합친 cost로 매칭
      3) 매칭된 detection에 대해 TrackState.update()
      4) 도둑일 확률(thief_prob)이 충분히 높으면 갤러리에 new emb 추가
    """
        
    def __init__(self, thief_embs,
                 max_kf_life=30,         # 관측 없이 예측만 허용할 최대 프레임 수
                 min_match_frames=3,     # 연속 매칭 몇 프레임부터 “진짜 트랙”으로 인정할지                 
                 iou_gate=0.1,           # 예측 bbox vs 현재 bbox 최소 IoU               
                 cos_weight=2.0,         # cost에 들어가는 cos_dist(thief) 가중치
                 
                 thief_cos_dist=0.2,              # 도둑 갤러리와의 최대 코사인 거리 (2차 필터링)
                 max_memory=100,                  # 도둑 갤러리에 저장할 최대 임베딩 수
                 conf_thresh=0.5,                 # YOLO score 이 이상인 dets만 도둑 후보로 인정 (1차 필터링)
                 iou_no_overlap=0.1,              # 다른 yolo bbox와 IoU가 이 값 이하일 때만 저장
                 gal_update_min_cos_dist=0.10,    # 너무 비슷하면 추가 안 함 (중복 방지)
                 gal_update_max_cos_dist=0.30,):  # 너무 다르면 추가 안 함 (오인 방지)
        
        """
        Parameters
        ----------
        thief_embs : list[np.ndarray] or np.ndarray
            도둑 identity에 대한 갤러리 임베딩 리스트.
            - shape (K, D) 또는 길이 K 리스트 [emb1, emb2, ...]
        """

        # 도둑 갤러리 임베딩을 내부 통일 포맷(list[np.ndarray])으로 저장
        if isinstance(thief_embs, np.ndarray):
            if thief_embs.ndim == 1:
                self.thief_embs = [thief_embs.astype(np.float32)]
            else:
                self.thief_embs = [e.astype(np.float32) for e in thief_embs]
        else:
            self.thief_embs = [np.asarray(e, dtype=np.float32) for e in thief_embs]

        # 각 emb별 match count (쓰인 정도)
        self.thief_match_counts = [0 for _ in self.thief_embs]
        
        # TrackState 생성 시 넘겨줄 공통 하이퍼파라미터
        self.max_kf_life      = max_kf_life
        self.min_match_frames = min_match_frames

        # 매칭 cost / gate 파라미터
        self.iou_gate = float(iou_gate)
        self.cos_weight = float(cos_weight)
        self.thief_cos_dist = float(thief_cos_dist)
        self.max_memory = int(max_memory)
        self.conf_thresh = float(conf_thresh)
        self.iou_no_overlap = float(iou_no_overlap)
        self.gal_update_min_cos_dist = float(gal_update_min_cos_dist)
        self.gal_update_max_cos_dist = float(gal_update_max_cos_dist)

        # 내부 상태
        self.tracks = []   # TrackState 객체 리스트 (보통 0 또는 1개)
        self.next_id = 1   # 새 TrackState에 부여할 track_id (의미상 항상 도둑)

    # ----------------- 유틸: 도둑 갤러리 거리 -----------------
    
    def _should_add_thief_emb(self, track, thief_cos_dist, all_dets):
        """
        새 임베딩을 갤러리에 추가할지 여부만 판단.
        - 실제로 append/교체는 _add_thief_emb()에서 수행.
        """
        
        # 다른 yolo bbox들과 겹치는지 체크 (occlusion 가능성 있으면 스킵)
        for det in all_dets:
            det_box = det[:4]
            iou_val = iou_bbox(track.last_bbox_tlbr, det_box)
            if iou_val > self.iou_no_overlap and iou_val < 0.9:
                return False
    
        # 기존 gal_emb들과의 거리 검사 (너무 비슷 or 너무 다르면 제외)
        if thief_cos_dist < self.gal_update_min_cos_dist:
            return False
        if thief_cos_dist > self.gal_update_max_cos_dist:
            return False

        return True
    
    # ----------------- 유틸: 갤러리 업데이트 여부 판단 -----------------

    def _add_thief_emb(self, emb):
        """
        도둑 갤러리(self.thief_embs)에 임베딩 추가.

        - 갤러리 크기 < max_memory:
            → 그냥 append
        - 갤러리 크기 >= max_memory:
            → 전체 N개 중 앞쪽 절반(0 ~ N/2-1)만 후보로 보고,
               그 중 match_count가 가장 적은 emb를 교체
               (동률이면 인덱스가 가장 작은 = 가장 오래된 것 우선 교체)
        """
        new_emb = emb.copy().astype(np.float32)

        n = len(self.thief_embs)
        if n < self.max_memory:
            self.thief_embs.append(new_emb)
            self.thief_match_counts.append(0)
            print(f"[THIEF-GAL] append new emb, size={len(self.thief_embs)}/{self.max_memory}")
            return

        # 꽉 찼을 때: 교체 로직 발동! 앞쪽 절반만 후보로 사용
        half = max(1, n // 2)  # 최소 1은 보장
        candidate_counts = self.thief_match_counts[:half]

        min_count = min(candidate_counts)
        remove_local_idx = candidate_counts.index(min_count)  # 0 ~ half-1
        remove_idx = remove_local_idx  # 실제 인덱스도 동일 (0~half-1 구간)

        print(f"[THIEF-GAL] replace idx={remove_idx} "
              f"(old_count={self.thief_match_counts[remove_idx]}) with new emb")

        self.thief_embs[remove_idx] = new_emb
        self.thief_match_counts[remove_idx] = 0

    def _register_gallery_match(self, emb):
        """
        현재 트랙 임베딩 emb가
        갤러리(thief_embs) 중 어떤 벡터와 가장 가까운지 찾고,
        거리가 thief_cos_dist 이하이면 해당 emb의 match_count += 1
        """
        if emb is None or len(self.thief_embs) == 0:
            return

        dists = [cosine_distance(emb, g) for g in self.thief_embs]
        best_idx = int(np.argmin(dists))
        best_dist = dists[best_idx]

        if best_dist <= self.thief_cos_dist:
            self.thief_match_counts[best_idx] += 1
            
    # ----------------- cost matrix -----------------

    def compute_cost_matrix(self, now_dets, now_embs):
        """
        현재 프레임 YOLO detection vs 이전 프레임 TrackState(pred 상태) 사이의 cost matrix 계산.

        도둑 전용 cost 정의:
          cost(d, t) = (1 - IoU(Track.kf_bbox_tlbr, now_bbox_d))
                       + cos_weight * min_cos_dist(now_emb_d, thief_embs)

        - 여기서는 Track의 last_emb 대신 "도둑 갤러리(thief_embs)"와의 거리를 사용한다.
        """
        N = len(now_dets)
        M = len(self.tracks)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for det_idx in range(N):
            now_bbox = now_dets[det_idx, :4]
            now_emb  = None if now_embs is None else now_embs[det_idx]

            # 도둑 갤러리와의 최소 거리
            cos_dist = min_cos_dist_to_list(now_emb, self.thief_embs)

            for track_idx in range(M):
                last_track = self.tracks[track_idx]

                # 1) 위치 기반 IoU cost (예측 위치(pred) vs 현재 bbox(now))
                iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                cost = 1.0 - iou_score

                # 2) 도둑 갤러리와의 거리 cost
                cost += cos_dist * self.cos_weight

                cost_matrix[det_idx, track_idx] = cost

        return cost_matrix

    # ----------------- 메인 update -----------------

    def update(self, now_dets, now_embs=None):
        """
        한 프레임(t)의 YOLO 결과(now_dets, now_embs)를 받아
        “도둑 한 명”에 대한 TrackState 목록을 반환.

        Parameters
        ----------
        now_dets : np.ndarray
            shape (N,5) 배열, 각 행은 [x1, y1, x2, y2, score]
        now_embs : list(np.ndarray) 또는 None
            길이 N 리스트, 각 detection에 대한 현재 프레임 ReID 임베딩.

        Returns
        -------
        online_tracks : List[TrackState]
            - 도둑이라고 판단된 TrackState 리스트 (보통 길이 0 또는 1)
            - 각 TrackState에는 다음 필드가 추가됨:
                * thief_dist : 도둑 갤러리와 최소 코사인 거리
        """
        # 0) 이전 프레임 Track들을 t 프레임으로 Kalman 예측 (last → pred)
        for last_track in self.tracks:
            last_track.predict()

        # now_dets shape 정규화
        if now_dets is None:
            now_dets = np.zeros((0, 5), dtype=np.float32)
        now_dets = np.asarray(now_dets, dtype=np.float32)

        if now_dets.ndim == 1:
            if now_dets.size == 0:
                now_dets = now_dets.reshape(0, 5)
            else:
                now_dets = now_dets.reshape(1, -1)

        num_now_dets = len(now_dets)

        # detection이 하나도 없는 프레임인 경우:
        if num_now_dets == 0:
            removed_indices = []
            for idx, last_track in enumerate(self.tracks):
                if last_track.mark_missed():
                    removed_indices.append(idx)
            for idx in reversed(removed_indices):
                self.tracks.pop(idx)

            return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]

        # YOLO confidence 필터
        now_scores = now_dets[:, 4]
        valid_inds = np.where(now_scores >= self.conf_thresh)[0]

        if len(valid_inds) == 0:
            # 신뢰도 낮은 프레임 → 도둑 감지 안 된 것으로 처리
            removed_indices = []
            for idx, last_track in enumerate(self.tracks):
                if last_track.mark_missed():
                    removed_indices.append(idx)
            for idx in reversed(removed_indices):
                self.tracks.pop(idx)

            return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]

        # 도둑 후보 detection만 서브셋으로 사용
        dets_valid = now_dets[valid_inds]
        embs_valid = None
        if now_embs is not None:
            embs_valid = [now_embs[i] for i in valid_inds]

        # =========================================
        # 1단계: 도둑 후보 dets vs 기존 TrackState
        # =========================================
        matches = []          # (global_now_det_idx, track_idx)
        matched_det  = set()
        matched_track = set()

        if len(self.tracks) > 0:
            cost = self.compute_cost_matrix(dets_valid, embs_valid)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                global_idx = valid_inds[r]
                track_idx  = c

                now_bbox   = now_dets[global_idx, :4]
                now_score  = now_dets[global_idx, 4]
                now_emb    = None if now_embs is None else now_embs[global_idx]
                last_track = self.tracks[track_idx]

                # IoU gate (pred vs now)
                iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                if iou_score < self.iou_gate:
                    continue

                # 도둑 갤러리와의 거리 gate
                cos_dist = min_cos_dist_to_list(now_emb, self.thief_embs)
                if cos_dist > self.thief_cos_dist:
                    continue

                matches.append((global_idx, track_idx))
                matched_det.add(global_idx)
                matched_track.add(track_idx)

        # 매칭되지 않은 det 중에서 “도둑일 가능성이 높은 것”만 새 Track 후보
        unmatched_det_indices = [idx for idx in valid_inds if idx not in matched_det]

        thief_like_det_indices = []
        if now_embs is not None:
            for idx in unmatched_det_indices:
                emb = now_embs[idx]
                cos_dist = min_cos_dist_to_list(emb, self.thief_embs)
                if cos_dist <= self.thief_cos_dist:
                    thief_like_det_indices.append(idx)
        else:
            # emb 없으면 thief_cos_dist를 쓸 수 없으니, 우선 YOLO conf만으로 후보 판단
            thief_like_det_indices = unmatched_det_indices

        # ==============================
        # 매칭된 TrackState 업데이트 (last ← now)
        # ==============================
        for det_idx, track_idx in matches:
            now_bbox  = now_dets[det_idx, :4]
            now_score = now_dets[det_idx, 4]
            now_emb   = None if now_embs is None else now_embs[det_idx]

            self.tracks[track_idx].update(
                now_bbox_tlbr=now_bbox,
                score=now_score,
                now_emb=now_emb,
            )

        # ==============================
        # 매칭 안 된 TrackState 정리 (Kalman 수명 기준)
        # ==============================
        all_track_indices      = set(range(len(self.tracks)))
        unmatched_tracks_final = [t for t in all_track_indices if t not in matched_track]

        removed_indices = []
        for track_idx in unmatched_tracks_final:
            if self.tracks[track_idx].mark_missed():
                removed_indices.append(track_idx)
        for idx in reversed(removed_indices):
            self.tracks.pop(idx)

        # ==============================
        # 도둑 후보 det → 새 TrackState 생성
        # ==============================
        for det_idx in thief_like_det_indices:
            # 이미 트랙이 하나라도 있으면, 도둑은 한 명이라 가정하고 추가 생성은 생략
            if len(self.tracks) > 0:
                break

            now_bbox  = now_dets[det_idx, :4]
            now_score = now_dets[det_idx, 4]
            now_emb   = None if now_embs is None else now_embs[det_idx]

            new_track = TrackState(
                last_bbox_tlbr=now_bbox,
                track_id=self.next_id,
                score=now_score,
                emb=now_emb,
                max_kf_life=self.max_kf_life,
                min_match_frames=self.min_match_frames,
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # ==============================
        # 도둑 거리 계산 + 갤러리 업데이트
        # ==============================
        for t in self.tracks:
            last_emb = t.last_emb
            cos_dist = min_cos_dist_to_list(last_emb, self.thief_embs)

            # TrackState에 부가 정보로 저장 → draw/drone 제어에서 활용
            t.thief_dist = cos_dist

            # 갤러리 업데이트 여부 판단 (cos_dist 기준)
            if last_emb is not None and self._should_add_thief_emb(t, cos_dist, now_dets):
                self._add_thief_emb(last_emb)

            # 실제로 어느 갤러리 벡터가 사용됐는지 카운트
            if last_emb is not None:
                self._register_gallery_match(last_emb)

            # 디버그 로그
            print(
                f"[THIEF] track_id={t.track_id:3d} "
                f"conf={t.score:.2f} "
                f"kf_life={t.kf_life:2d} "
                f"match_frames={t.match_frames:2d} "
                f"cos_dist={cos_dist:.3f} "
                f"gal_size={len(self.thief_embs)}/{self.max_memory}"
            )

        # ==============================
        # 최종 반환
        # ==============================
        # “충분히 연속 매칭(frame_conf=True)” 이면서
        # “이번 프레임 기준으로 너무 오래 사라지지 않은(kf_life <= 3)” Track만 사용
        return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]
