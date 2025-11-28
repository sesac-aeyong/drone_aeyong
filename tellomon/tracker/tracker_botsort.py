# tracker_botsort.py
"""
Track이 이전 프레임 상태 저장 
→ YOLO/REID가 현재 프레임 상태(now) 뽑음 
→ BoTSORT가 pred vs now 비교해서 track_id 유지/부여 
→ LongTermBoTSORT가 각 Track의 last_emb를 갤러리 gal_emb들과 비교해서 identity_id 부여
"""
import numpy as np
from scipy.optimize import linear_sum_assignment  # pip install scipy 필요
from tracker.utils.metrics import iou_bbox, min_cos_dist_to_list, cosine_distance
from tracker.utils.track_state import TrackState 

class BoTSORT: # 이전 프레임 상태(Track: pred, last_emb) ↔ 현재 프레임 상태(YOLO now_bbox + REID now_emb)를 비교해서 **track_id**를 부여
    """
    매 프레임(t)에서 수행 절차:
      1) 모든 Track.predict() 호출
      2) 예측된 kf_bbox_tlbr vs YOLO now_dets + ReID now_embs 로 cost matrix 생성
      3) high_yolo_thresh 이상 now_dets ↔ 모든 Track 매칭 (Hungarian 후보 뽑고 gate로 거르기)
      4) 남은 Track ↔ [low_yolo_thresh, high_yolo_thresh) 구간 now_dets 매칭 연결
      5) 최종 매칭된 쌍에 대해 Track.update(now_bbox_tlbr, now_score, now_emb) 호출
      6) 이번 프레임에서도 끝내 매칭 안 된 Track에 대해서는 Track.mark_missed() 호출
      7) 끝까지 매칭 안 된 "high_yolo" now_dets → 새 Track으로 생성
    """

    def __init__(self,
                 max_kf_life=30,          # 관측 없이 예측만 허용할 최대 프레임 수
                 min_match_frames=3,      # 연속 매칭 몇 프레임부터 “진짜 트랙”으로 인정할지
                 iou_gate=0.1,            # IoU 기준 최소값
                 cos_gate=0.3,            # ReID 거리 기준 최대값 (None 이면 사용 안 함)
                 cos_weight=2.0,          # cost에 들어가는 ReID 거리 가중치
                 high_yolo_thresh=0.7,    # 새 Track 생성에 쓸 최소 YOLO score
                 low_yolo_thresh=0.5):    # 기존 Track 연결에만 쓸 YOLO score 하한

        # Track 생성 시 넘겨줄 공통 하이퍼파라미터
        self.max_kf_life      = max_kf_life
        self.min_match_frames = min_match_frames

        # 매칭 cost / gate 파라미터
        self.iou_gate    = float(iou_gate)
        self.cos_gate   = cos_gate   # None 이면 ReID gate는 생략
        self.cos_weight = float(cos_weight)

        # ByteTrack 스타일: YOLO confidence 분리
        self.high_yolo_thresh = float(high_yolo_thresh)
        self.low_yolo_thresh  = float(low_yolo_thresh)

        # 내부 상태
        self.tracks = []   # Track 객체 리스트 (last_* + kf_* 들을 들고 있는 친구들)
        self.next_id = 1   # 새 Track에 부여할 track_id

    # ----------------- 유틸 함수들 -----------------

    def compute_cost_matrix(self, now_dets, now_embs):
        """
        현재 프레임 YOLO detection vs 이전 프레임 Track(pred 상태) 사이의 cost matrix 계산.

        Parameters
        ----------
        now_dets : np.ndarray
            shape (N,5), [x1, y1, x2, y2, score]
        now_embs : list[np.ndarray] or None
            길이 N, 각 detection의 현재 프레임 ReID 임베딩.

        cost(d, t) = (1 - IoU(Track.kf_bbox_tlbr, now_bbox_d))
                     + cos_weight * cosine_distance(Track.last_emb, now_emb_d)
                     (단, 임베딩이 둘 다 있을 때만 ReID 항 추가)
        """
        N = len(now_dets)
        M = len(self.tracks)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for det_idx in range(N):
            now_bbox = now_dets[det_idx, :4]
            now_emb  = None if now_embs is None else now_embs[det_idx]

            for track_idx in range(M):
                last_track = self.tracks[track_idx]

                # 1) 위치 기반 IoU cost (예측 위치(pred) vs 현재 bbox(now))
                iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                cost = 1.0 - iou_score

                # 2) ReID 기반 거리 cost (임베딩이 둘 다 있을 때만)
                if last_track.last_emb is not None and now_emb is not None:
                    cos_dist = cosine_distance(last_track.last_emb, now_emb)
                    cost += cos_dist * self.cos_weight

                cost_matrix[det_idx, track_idx] = cost

        return cost_matrix

    # ----------------- 메인 update -----------------

    def update(self, now_dets, now_embs=None):
        """
        한 프레임(t)의 YOLO 결과(now_dets, now_embs)를 받아 BoTSORT 상태를 갱신.

        ByteTrack 스타일 2단계 매칭:
          1단계: high_yolo_thresh 이상 now_dets ↔ 모든 Track 매칭
          2단계: 남은 Track ↔ [low_yolo_thresh ~ high_yolo_thresh) 구간 now_dets 매칭
                 (여기서는 새 Track 생성 없이 연결만 수행)

        Parameters
        ----------
        now_dets : np.ndarray
            shape (N,5) 배열, 각 행은 [x1, y1, x2, y2, score]
        now_embs : list(np.ndarray) 또는 None
            길이 N 리스트, 각 detection에 대한 현재 프레임 ReID 임베딩.
        """
        # 0) 이전 프레임 Track들을 t 프레임으로 Kalman 예측 (last → pred)
        for last_track in self.tracks:
            last_track.predict()

        # now_dets shape 정규화 (0개 / 1개 예외 처리)
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
            # Kalman 예측만 한 상태에서, 너무 오래 안 보인 Track들을 정리
            removed_indices = []
            for idx, last_track in enumerate(self.tracks):
                if last_track.mark_missed():
                    removed_indices.append(idx)
            for idx in reversed(removed_indices):
                self.tracks.pop(idx)

            # 여전히 살아있는 것 중에서:
            # - frame_conf == True (충분히 연속 매칭된 트랙)
            # - kf_life <= 3 (지금 프레임 기준으로 너무 오래 사라지지 않은)
            return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]

        now_scores = now_dets[:, 4]

        # ByteTrack: high / low confidence 분리
        high_yolo_inds = np.where(now_scores >= self.high_yolo_thresh)[0]
        low_yolo_inds  = np.where(
            (now_scores >= self.low_yolo_thresh) &
            (now_scores <  self.high_yolo_thresh)
        )[0]

        # =========================================
        # 1단계: high_yolo dets vs 모든 Track
        # =========================================
        matches = []          # (global_now_det_idx, track_idx)
        matched_now_det = set()
        matched_track   = set()

        if len(self.tracks) == 0:
            # 아직 Track이 하나도 없으면 → high_yolo det 전체가 새 Track 후보
            unmatched_high_yolo = list(high_yolo_inds)
            unmatched_tracks    = []
        else:
            unmatched_tracks = list(range(len(self.tracks)))

            if len(high_yolo_inds) > 0:
                now_dets_high = now_dets[high_yolo_inds]

                # subset용 now_embs 리스트 준비
                now_embs_high = None
                if now_embs is not None:
                    now_embs_high = [now_embs[i] for i in high_yolo_inds]

                cost_high = self.compute_cost_matrix(now_dets_high, now_embs_high)
                row_ind, col_ind = linear_sum_assignment(cost_high)

                for r, c in zip(row_ind, col_ind):
                    global_now_idx = high_yolo_inds[r]
                    track_idx      = c

                    now_bbox   = now_dets[global_now_idx, :4]
                    last_track = self.tracks[track_idx]

                    # IoU gate (pred vs now)
                    iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                    if iou_score < self.iou_gate:
                        continue

                    # ReID gate (옵션: cos_gate가 None이면 스킵)
                    if self.cos_gate is not None and now_embs is not None:
                        now_emb = now_embs[global_now_idx]
                        if last_track.last_emb is not None and now_emb is not None:
                            cos_dist = cosine_distance(last_track.last_emb, now_emb)
                            if cos_dist > self.cos_gate:
                                continue

                    matches.append((global_now_idx, track_idx))
                    matched_now_det.add(global_now_idx)
                    matched_track.add(track_idx)

            # 1단계 이후 아직 안 붙은 high_yolo det / Track 정리
            unmatched_high_yolo = [d for d in high_yolo_inds if d not in matched_now_det]
            unmatched_tracks    = [t for t in unmatched_tracks if t not in matched_track]

        # =========================================
        # 2단계: 남은 Track vs low_yolo dets (연결만, 새 Track 생성 X)
        # =========================================
        if len(unmatched_tracks) > 0 and len(low_yolo_inds) > 0:
            now_dets_low = now_dets[low_yolo_inds]
            now_embs_low = None
            if now_embs is not None:
                now_embs_low = [now_embs[i] for i in low_yolo_inds]

            cost_low_full = self.compute_cost_matrix(now_dets_low, now_embs_low)
            # columns만 unmatched_tracks에 해당하는 것만 남김
            cost_low = cost_low_full[:, unmatched_tracks]  # (len(low), len(unmatched_tracks))

            row2, col2 = linear_sum_assignment(cost_low)

            for r, c in zip(row2, col2):
                global_now_idx = low_yolo_inds[r]
                track_idx      = unmatched_tracks[c]

                now_bbox   = now_dets[global_now_idx, :4]
                last_track = self.tracks[track_idx]

                # IoU gate (pred vs now)
                iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                if iou_score < self.iou_gate:
                    continue

                # ReID gate
                if self.cos_gate is not None and now_embs is not None:
                    now_emb = now_embs[global_now_idx]
                    if last_track.last_emb is not None and now_emb is not None:
                        cos_dist = cosine_distance(last_track.last_emb, now_emb)
                        if cos_dist > self.cos_gate:
                            continue

                matches.append((global_now_idx, track_idx))
                matched_now_det.add(global_now_idx)
                matched_track.add(track_idx)

        # 최종 unmatched Track / now_det 정리
        all_track_indices      = set(range(len(self.tracks)))
        unmatched_tracks_final = [t for t in all_track_indices if t not in matched_track]

        # 새 Track 생성은 “high_yolo 중에서도 끝까지 매칭되지 않은” now_det만 사용
        new_track_now_det_indices = unmatched_high_yolo

        # ==============================
        # 매칭된 Track 업데이트 (last ← now)
        # ==============================
        for now_idx, track_idx in matches:
            now_bbox  = now_dets[now_idx, :4]
            now_score = now_dets[now_idx, 4]
            now_emb   = None if now_embs is None else now_embs[now_idx]

            self.tracks[track_idx].update(
                now_bbox_tlbr=now_bbox,
                score=now_score,
                now_emb=now_emb,
            )

        # ==============================
        # 매칭 안 된 Track 정리 (Kalman 수명 기준)
        # ==============================
        removed_indices = []
        for track_idx in unmatched_tracks_final:
            if self.tracks[track_idx].mark_missed():
                removed_indices.append(track_idx)
        for idx in reversed(removed_indices):
            self.tracks.pop(idx)

        # ==============================
        # high_yolo 남은 now_det → 새 Track 생성
        # ==============================
        for now_idx in new_track_now_det_indices:
            now_bbox  = now_dets[now_idx, :4]
            now_score = now_dets[now_idx, 4]
            now_emb   = None if now_embs is None else now_embs[now_idx]

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
        # 최종 반환
        # ==============================
        # “충분히 연속 매칭(frame_conf=True)” 이면서
        # “이번 프레임 기준으로 너무 오래 사라지지 않은(kf_life <= 3)” Track만 화면에 보이게
        return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]



class LongTermBoTSORT: # BoTSORT가 이어놓은 각 track의 last_emb을 갤러리 gal_emb와 비교해서 **identity_id**를 부여
    """
    각 Track.last_emb 를 identity 갤러리의 gal_emb들과 비교해서 identity_id 할당
    갤러리는 한 번 신중하게 저장 후 업데이트 금지
    """
    def __init__(self, botsort_tracker,
        gal_match_cos_dist=0.3,          # 기존 ID 재사용 한계 
        max_memory=20,                   # 전체 identity 갯수 상한
        max_gal_emb_per_id=10,           # ID 하나당 gal_emb 최대 개수
        conf_thresh=0.7,                 # YOLO score 이 이상일 때만 prototype 후보로 인정
        iou_no_overlap=0.1,              # 다른 Track과 IoU가 이 값 이하일 때만 prototype 저장 허용
        gal_update_min_cos_dist=0.15,    # 기존 gal_emb들과의 최소 거리 < 이면 너무 비슷 → 안 넣음
        gal_update_max_cos_dist=0.25,):  # 기존 gal_emb들과의 최소 거리 > 이면 너무 다름 → 안 넣음
    
        # 단기 추적기 (BoTSORT 인스턴스)
        self.tracker = botsort_tracker

        # ID 매칭 기준 (Track.last_emb ↔ gallery.gal_embs)
        self.gal_match_cos_dist = gal_match_cos_dist

        # 전체 메모리 상한
        self.max_memory = max_memory

        # identity 갤러리: identity_id -> {"gal_embs": [..]}
        self.gallery = {}
        self.next_identity = 1

        # prototype(= gal_emb) 저장 정책
        self.max_gal_emb_per_id = max_gal_emb_per_id
        self.conf_thresh = conf_thresh
        self.iou_no_overlap = iou_no_overlap
        self.gal_update_min_cos_dist = gal_update_min_cos_dist
        self.gal_update_max_cos_dist = gal_update_max_cos_dist

    # ================== 유사 갤러리 합치기 ==================

    def _merge_galleries(self, dist_thresh: float = 0.25):
        def min_cos_dist_between_lists(list1, list2):
            return min(min_cos_dist_to_list(e1, list2) for e1 in list1)
        """
        Merge galleries by comparing all embeddings in galleries.
        """
        active_ids = list(self.gallery.keys())
        merged = {}

        for i in range(len(active_ids)):
            id_i = active_ids[i]
            if id_i in merged:
                continue
            embs_i = self.gallery[id_i]['gal_embs']
            if not embs_i:
                continue

            for j in range(i + 1, len(active_ids)):
                id_j = active_ids[j]
                if id_j in merged:
                    continue
                embs_j = self.gallery[id_j]['gal_embs']
                if not embs_j:
                    continue

                # compute min cosine distance between any embedding pair
                min_dist = min_cos_dist_between_lists(embs_i, embs_j)
                if min_dist < dist_thresh:
                    # merge: append embeddings from j to i
                    self.gallery[id_i]['gal_embs'].extend(embs_j)
                    # keep only last N embeddings
                    self.gallery[id_i]['gal_embs'] = self.gallery[id_i]['gal_embs'][-self.max_gal_emb_per_id:]
                    merged[id_j] = id_i

        # remove merged galleries
        for o_id, new_id in merged.items():
            if o_id in self.gallery:
                print(f'Merged {o_id} into {new_id}')
                del self.gallery[o_id]

    # ================== ID 매칭 / prototype 추가 로직 ==================

    def _assign_identity(self, last_emb, active_identity_ids):
        """
        Track.last_emb 를 받아서,
          - gallery 안의 gal_emb들과 비교 → 가장 가까운 identity_id 찾기
          - 최소 거리 <= gallery_match_threshold 이면 그 ID 재사용
          - 아니면 새 identity_id 발급

        active_identity_ids:
          - 이번 프레임에 이미 사용된 ID 리스트 (한 프레임 내 중복 방지용)
        """
        # 1) 갤러리가 비었거나, 이 트랙에 last_emb가 없으면 → ID 판단 보류(None)
        if last_emb is None or len(self.gallery) == 0:
            return None
        
        best_id = None
        best_cos_dist = self.gal_match_cos_dist  # 이 값보다 가까워야 매칭 인정

        for mem_id, info in self.gallery.items():
            if mem_id in active_identity_ids:
                continue  # 한 프레임 안에서 ID 중복 사용 금지

            gal_emb_list = info.get("gal_embs", [])
            if not gal_emb_list:
                continue

            cos_dist = min_cos_dist_to_list(last_emb, gal_emb_list)

            if cos_dist < best_cos_dist:
                best_cos_dist = cos_dist
                best_id = mem_id

        # threshold 안에 들어온 갤러리 ID가 없으면 → 아직 새 ID 발급하지 않음(None 반환)
        return best_id

    def _should_add_gal_emb(self, identity_id, track, cand_emb, all_tracks):
        """
        cand_emb 를 identity_id의 gal_emb로 추가할지 여부 판단.

        조건:
          - cand_emb(None) → 추가 X
          - track.score >= conf_thresh (YOLO confidence 충분히 높을 때만)
          - 다른 트랙들과 IoU(last_bbox_tlbr 기준)가 iou_no_overlap 이하 (겹치지 않을 때만)
          - identity_id에 이미 max_gal_emb_per_id 개수만큼 저장되어 있으면 추가 X
          - 기존 gal_emb들과의 거리:
              * min_cos_dist <  gal_update_min_cos_dist  → 거의 같은 포즈/상태 → 굳이 추가 X
              * min_cos_dist >  gal_update_max_cos_dist → 너무 다른 벡터 → 잘못된 매칭일 가능성 높음 → 추가 X
        """
        if cand_emb is None:
            return False

        # YOLO confidence 체크
        if getattr(track, "score", 0.0) < self.conf_thresh:
            return False

        # 다른 Track들과 겹치는지 체크 (occlusion 가능성 있으면 스킵)
        for other in all_tracks:
            if other is track:
                continue
            iou_val = iou_bbox(track.last_bbox_tlbr, other.last_bbox_tlbr)
            if iou_val > self.iou_no_overlap:
                return False

        # 갤러리에 이미 있는 gal_emb 개수 확인 (identity_id=None이면 '비어있다'로 취급)
        if identity_id is None:
            gal_emb_list = []
        else:
            gal_emb_list = self.gallery.get(identity_id, {}).get("gal_embs", [])

        if len(gal_emb_list) >= self.max_gal_emb_per_id:
            # 이미 identity_id 당 허용한 최대 개수만큼 저장됨 → 더 안 넣음 (업데이트 금지!)
            return False

        if not gal_emb_list:
            # 첫 gal_emb는 위 조건만 통과하면 허용
            return True

        # 기존 gal_emb들과의 거리 검사
        min_cos_dist = min_cos_dist_to_list(cand_emb, gal_emb_list)
        if min_cos_dist < self.gal_update_min_cos_dist:
            # 거의 같은 포즈/상태 → 새 벡터 추가할 필요 없음
            return False
        if min_cos_dist > self.gal_update_max_cos_dist:
            # 너무 다른 벡터 → 잘못된 매칭일 가능성 높음
            return False

        return True

    def _add_gal_emb(self, identity_id, cand_emb):
        """
        실제로 gallery[identity_id]["gal_embs"]에 cand_emb를 복사해서 추가.
        """
        info = self.gallery.setdefault(identity_id, {"gal_embs": []})
        info["gal_embs"].append(cand_emb.copy())

    # ================== 메인 update ==================

    def update(self, detections: np.ndarray, embeddings: list):
        """
        LongTermBoTSORT의 메인 엔트리.

        Parameters
        ----------
        detections : np.ndarray
            shape (N,5) = [x1, y1, x2, y2, score]
        embeddings : list[np.ndarray] or None
            길이 N, YOLO 각 bbox에 대응하는 현재 프레임 ReID 임베딩 (now_emb).
            → 이 값은 BoTSORT.update(now_dets, now_embs) 에 그대로 넘겨짐.

        반환값
        ------
        online_tracks : List[Track]
            - BoTSORT가 t 프레임까지 추적을 마친 Track 리스트
            - 각 Track 에는 .identity_id 가 추가됨
        """
        # 1) BoTSORT 단기 추적으로 Track 리스트 갱신 (last_* 업데이트 포함)
        online_tracks = self.tracker.update(detections, embeddings)

        # 이번 프레임에서 이미 사용된 identity_id들 (중복 방지용)
        active_identity_ids = set()

        for track in online_tracks:
            # 이 프레임 기준 “해당 사람의 대표 벡터”는 Track.last_emb (RepVGG 결과 한 장)
            last_emb = track.last_emb

            # 2) 기존 갤러리와 매칭 시도 (성공 시 그 ID 사용)
            identity_id = self._assign_identity(last_emb, active_identity_ids)

            if identity_id is not None:
                # 3-a) 기존 ID에 대해 갤러리 업데이트 시도
                if self._should_add_gal_emb(identity_id, track, last_emb, online_tracks):
                    self._add_gal_emb(identity_id, last_emb)
                active_identity_ids.add(identity_id)
                track.identity_id = identity_id
            else:
                # 3-b) 기존 매칭 실패: "저장 조건"을 만족하면 그 순간에만 새 ID 생성+저장
                if self._should_add_gal_emb(None, track, last_emb, online_tracks):
                    new_id = self.next_identity
                    self.next_identity += 1
                    self.gallery.setdefault(new_id, {"gal_embs": []})
                    self._add_gal_emb(new_id, last_emb)
                    track.identity_id = new_id
                    active_identity_ids.add(new_id)
                else:
                    track.identity_id = None

        # 4) 메모리 관리 – 이번 프레임에 쓰이지 않은 오래된 identity 일부 제거 (선택)
        if len(self.gallery) > self.max_memory:
            unused_ids = [iid for iid in self.gallery.keys()
                          if iid not in active_identity_ids]
            # 너무 많이 넘친 만큼만 앞에서부터 제거
            over = max(0, len(self.gallery) - self.max_memory)
            for iid in unused_ids[:over]:
                self.gallery.pop(iid, None)

        # 디버그: 한 프레임 안에서 identity 중복이 있으면 경고
        ids_this_frame = [getattr(t, "identity_id", t.track_id) for t in online_tracks]
        if len(ids_this_frame) != len(set(ids_this_frame)):
            print("[WARN] duplicate identity in this frame:", ids_this_frame)

        # self._merge_galleries()
        return online_tracks
    
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

        # =========================================
        # 1단계: 도둑 후보 dets vs 기존 TrackState
        # =========================================
        matches = []          # (global_now_det_idx, track_idx)
        matched_det  = set()
        matched_track = set()

        if len(self.tracks) == 1:
            track_idx  = 0
            last_track = self.tracks[track_idx]

            best_cost = None
            best_global_idx = None

            # valid_inds 안에서만 후보 탐색
            for global_idx in valid_inds:
                now_bbox = now_dets[global_idx, :4]
                now_emb  = None if now_embs is None else now_embs[global_idx]

                # 1) IoU 게이트: 예측 위치(pred) vs 현재 bbox(now)
                iou_score = iou_bbox(now_bbox, last_track.kf_bbox_tlbr)
                if iou_score < self.iou_gate:
                    continue

                # 2) 도둑 갤러리와의 거리 게이트
                if now_emb is None:
                    continue
                cos_dist = min_cos_dist_to_list(now_emb, self.thief_embs)
                if cos_dist > self.thief_cos_dist:
                    continue

                # 3) cost = (1 - IoU) + cos_weight * cos_dist
                cost = (1.0 - iou_score) + self.cos_weight * cos_dist

                if (best_cost is None) or (cost < best_cost):
                    best_cost = cost
                    best_global_idx = global_idx

            # 조건을 만족하는 detection이 하나라도 있으면 매칭 등록
            if best_global_idx is not None:
                matches.append((best_global_idx, track_idx))
                matched_det.add(best_global_idx)
                matched_track.add(track_idx)

        # 매칭되지 않은 det 중에서 “도둑일 가능성이 높은 것”만 새 Track 후보
        unmatched_det_indices = [idx for idx in valid_inds if idx not in matched_det]

        thief_like_det_indices = []
        if now_embs is not None:
            for idx in unmatched_det_indices:
                emb = now_embs[idx]
                if emb is None:
                    continue
                cos_dist = min_cos_dist_to_list(emb, self.thief_embs)
                if cos_dist <= self.thief_cos_dist:
                    thief_like_det_indices.append(idx)
        else:
            # 도둑 모드는 임베딩 기반이 핵심 → 임베딩 없으면 생성 건너뜀
            thief_like_det_indices = []

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
            # print(
            #     f"[THIEF] track_id={t.track_id:3d} "
            #     f"conf={t.score:.2f} "
            #     f"kf_life={t.kf_life:2d} "
            #     f"match_frames={t.match_frames:2d} "
            #     f"cos_dist={cos_dist:.3f} "
            #     f"gal_size={len(self.thief_embs)}/{self.max_memory}"
            # )

        # ==============================
        # 최종 반환
        # ==============================
        # “충분히 연속 매칭(frame_conf=True)” 이면서
        # “이번 프레임 기준으로 너무 오래 사라지지 않은(kf_life <= 3)” Track만 사용
        return [t for t in self.tracks if t.frame_conf and t.kf_life <= 3]