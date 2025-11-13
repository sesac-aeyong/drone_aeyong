# tracker_botsort.py
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment  # pip install scipy 필요

class Track: # 이 사람은 어디고, 얼마 동안 잘 보였고, 어떤 모양/임베딩을 가진 애냐
    """
    “사람 한 명”에 대한 로컬 상태
    Kalman 필터로 위치/크기를 예측/보정
    ReID 임베딩들을 저장해서 평균 feature 제공
    time_since_update, hit_streak 등 수명 관리
    """
    def __init__(self, tlbr, track_id, score, emb=None, max_age=30, min_hits=3):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.track_id = track_id
        self.score = float(score)
        self.max_age = max_age
        self.min_hits = min_hits

        self.time_since_update = 0
        self.hit_streak = 0
        self.history = deque(maxlen=max_age)
        self.confirmed = False

        # BoT-SORT style embedding gallery
        self.embeddings = []
        if emb is not None:
            self.embeddings.append(emb)

        # === Kalman filter 상태 초기화 ===
        # 상태벡터 x = [cx, cy, w, h, vx, vy]^T
        cx, cy, w, h = self._tlbr_to_cxcywh(self.tlbr)
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32)

        # 상태전이 행렬 (dt=1 가정)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # 관측행렬: z = [cx, cy, w, h]^T
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # 공분산 / 잡음 (값은 대충 reasonable 수준)
        self.P = np.eye(6, dtype=np.float32) * 10.0      # 초기 불확실성
        self.Q = np.eye(6, dtype=np.float32) * 1e-2      # 시스템 잡음
        self.R = np.eye(4, dtype=np.float32) * 1.0       # 관측 잡음

    # --- bbox <-> 상태 변환 유틸 ---

    @staticmethod
    def _tlbr_to_cxcywh(tlbr):
        x1, y1, x2, y2 = tlbr
        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return float(cx), float(cy), float(w), float(h)

    @staticmethod
    def _cxcywh_to_tlbr(cx, cy, w, h):
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    # --- Kalman predict / correct ---

    def predict(self):
        """다음 프레임 위치 예측 (measurement 없이)"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx, cy, w, h = self.x[:4, 0]
        self.tlbr = self._cxcywh_to_tlbr(cx, cy, w, h)

        # 업데이트 안 된 프레임 카운트 증가
        self.time_since_update += 1

    def _correct_kf(self, tlbr_meas):
        """새 detection 박스로 Kalman 보정"""
        cx, cy, w, h = self._tlbr_to_cxcywh(tlbr_meas)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        # y = z - Hx
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 / 공분산 업데이트
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # tlbr 갱신
        cx, cy, w, h = self.x[:4, 0]
        self.tlbr = self._cxcywh_to_tlbr(cx, cy, w, h)

        # 방금 업데이트 됐으니 0으로
        self.time_since_update = 0

    # --- 원래 로직 수정 ---

    def update(self, tlbr, score, emb=None):
        """새 detection으로 트랙 갱신 (Kalman + ReID)"""
        # 1) 칼만 보정으로 위치/크기 업데이트
        self._correct_kf(tlbr)

        # 2) 나머지 메타 정보
        self.score = float(score)
        self.hit_streak += 1
        self.history.append(self.tlbr.copy())

        # 🔥 임베딩 갤러리 업데이트: → '처음 한 번만' 저장. 이후에는 그대로 둔다.
        if emb is not None and not self.embeddings:
            # 첫 프레임 또는 첫 유효 임베딩일 때만
            self.embeddings.append(emb)

        if self.hit_streak >= self.min_hits:
            self.confirmed = True

    def mark_missed(self):
        """
        이번 프레임에 detection과 매칭 안 된 경우:
        - time_since_update는 predict()에서 이미 +1 됨
        - 여기서는 '삭제할지 여부'만 판단
        """
        return self.time_since_update > self.max_age

    def get_feature(self):
        """갤러리 평균 임베딩 반환 (없으면 None)"""
        if self.embeddings:
            return np.mean(self.embeddings, axis=0)
        return None


class BoTSORT: # 헝가리안 + IoU + (Track 내부의 고정 임베딩) 으로 프레임 간 트랙을 이어서 track_id 를 유지
    """
    프레임 간 단기 MOT 추적기
    매 프레임:
        모든 track Kalman predict()
        detector 출력 박스들과 IoU + ReID 기반 cost matrix 생성
        Hungarian 매칭
        매칭된 track은 Track.update()로 Kalman 보정/임베딩 업데이트
        매칭 안 된 track은 age 증가 후 삭제
        매칭 안 된 detection은 새 track 생성
    최종적으로 track_id 기준의 “현재 프레임 트랙들” 반환
    """
    def __init__(self, max_age=60, min_hits=3, use_reid=True, 
                 iou_threshold=0.2, reid_weight=2.0, reid_gate=0.3, 
                 high_thresh=0.7, low_thresh=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_reid = use_reid
        self.iou_threshold = float(iou_threshold)
        self.reid_weight = float(reid_weight)
        self.reid_gate = reid_gate
        self.high_thresh = high_thresh    # 새 트랙 생성용
        self.low_thresh  = low_thresh     # 기존 트랙 연결용

        self.tracks = []
        self.next_id = 1

    # ----------------- 유틸 함수들 -----------------

    def iou(self, bb_test, bb_gt):
        """두 박스의 IoU 계산 (tlbr 포맷)"""
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

        o = wh / (area_test + area_gt - wh + 1e-16)
        return float(o)

    def compute_cost_matrix(self, dets, embs):
        """
        dets: (N,5) [x1,y1,x2,y2,score]
        embs: list/array of N embeddings or None
        """
        N = len(dets)
        M = len(self.tracks)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for d in range(N):
            det = dets[d]
            for t in range(M):
                track = self.tracks[t]
                iou_score = self.iou(det[:4], track.tlbr)
                cost = 1.0 - iou_score

                if self.use_reid and embs is not None and len(embs) > d:
                    track_feat = track.get_feature()
                    if track_feat is not None:
                        dist = np.linalg.norm(track_feat - embs[d])
                        cost += dist * self.reid_weight

                cost_matrix[d, t] = cost

        return cost_matrix

    # ----------------- 메인 update -----------------

    def update(self, dets, embs=None):
        """
        ByteTrack 스타일 2단계 매칭:
          1단계: high_conf dets vs 모든 track  → 매칭 + 새 track 생성
          2단계: 남은 track vs low_conf dets → 연결만, 새 track 생성은 금지
        dets: [[x1,y1,x2,y2,score], ...]
        embs: len(dets) 와 동일한 리스트 (또는 None)
        """
        # 0) 기존 트랙 Kalman 예측
        for trk in self.tracks:
            trk.predict()

        # numpy array normalize
        if dets is None:
            dets = np.zeros((0, 5), dtype=np.float32)
        dets = np.asarray(dets, dtype=np.float32)

        if dets.ndim == 1:
            if dets.size == 0:
                dets = dets.reshape(0, 5)
            else:
                dets = dets.reshape(1, -1)

        num_dets = len(dets)

        # ReID 안 쓰면 embs도 무시
        if not self.use_reid:
            embs = None

        # detection이 하나도 없으면: age 증가 후 삭제만
        if num_dets == 0:
            removed = []
            for i, trk in enumerate(self.tracks):
                if trk.mark_missed():
                    removed.append(i)
            for i in reversed(removed):
                self.tracks.pop(i)
            return [t for t in self.tracks if t.confirmed and t.time_since_update <= 1]

        scores = dets[:, 4]

        # ★ ByteTrack: high / low 분리
        high_inds = np.where(scores >= self.high_thresh)[0]
        low_inds  = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]

        # 편의를 위해 subset용 emb 배열 생성
        def subset_emb(idx_list):
            if embs is None:
                return None
            return [embs[i] for i in idx_list]

        # =========================================
        # 1단계: high_conf dets vs 모든 tracks
        # =========================================
        matches = []              # (global_det_idx, track_idx)
        matched_det = set()
        matched_trk = set()

        if len(self.tracks) == 0:
            # 트랙이 하나도 없으면 → high_conf dets로만 새 트랙 생성
            unmatched_high = list(high_inds)
            unmatched_tracks = []
        else:
            unmatched_tracks = list(range(len(self.tracks)))

            if len(high_inds) > 0:
                dets_high = dets[high_inds]
                embs_high = subset_emb(high_inds)

                cost_high = self.compute_cost_matrix(dets_high, embs_high)
                row_ind, col_ind = linear_sum_assignment(cost_high)

                for r, c in zip(row_ind, col_ind):
                    global_d = high_inds[r]
                    trk_idx  = c

                    # IoU 게이트
                    iou_score = self.iou(dets[global_d, :4], self.tracks[trk_idx].tlbr)
                    if iou_score < self.iou_threshold:
                        continue

                    # ReID 게이트
                    if (
                        self.use_reid and embs is not None and
                        self.reid_gate is not None and
                        global_d < len(embs)
                    ):
                        track_feat = self.tracks[trk_idx].get_feature()
                        if track_feat is not None:
                            dist = np.linalg.norm(track_feat - embs[global_d])
                            if dist > self.reid_gate:
                                continue

                    matches.append((global_d, trk_idx))
                    matched_det.add(global_d)
                    matched_trk.add(trk_idx)

            # 1단계 이후 아직 안 붙은 high det / tracks 정리
            unmatched_high = [d for d in high_inds if d not in matched_det]
            unmatched_tracks = [t for t in unmatched_tracks if t not in matched_trk]

        # =========================================
        # 2단계: 남은 tracks vs low_conf dets (연결만, 새 track 생성 X)
        # =========================================
        if len(unmatched_tracks) > 0 and len(low_inds) > 0:
            dets_low = dets[low_inds]
            embs_low = subset_emb(low_inds)

            # 전체 트랙 기준 cost 계산 후, 사용하고 싶은 트랙 column만 슬라이스
            cost_low_full = self.compute_cost_matrix(dets_low, embs_low)
            # columns만 unmatched_tracks에 해당하는 것만 남김
            cost_low = cost_low_full[:, unmatched_tracks]  # shape: (len(low_inds), len(unmatched_tracks))

            row2, col2 = linear_sum_assignment(cost_low)

            for r, c in zip(row2, col2):
                global_d = low_inds[r]
                trk_idx  = unmatched_tracks[c]

                # IoU 게이트
                iou_score = self.iou(dets[global_d, :4], self.tracks[trk_idx].tlbr)
                if iou_score < self.iou_threshold:
                    continue

                # ReID 게이트
                if (
                    self.use_reid and embs is not None and
                    self.reid_gate is not None and
                    global_d < len(embs)
                ):
                    track_feat = self.tracks[trk_idx].get_feature()
                    if track_feat is not None:
                        dist = np.linalg.norm(track_feat - embs[global_d])
                        if dist > self.reid_gate:
                            continue

                matches.append((global_d, trk_idx))
                matched_det.add(global_d)
                matched_trk.add(trk_idx)

        # 최종 unmatched track / det 정리
        all_track_indices = set(range(len(self.tracks)))
        unmatched_tracks_final = [t for t in all_track_indices if t not in matched_trk]

        # ★ 새 트랙은 "high_conf 중에서도 끝까지 매칭 안 된 것"만 사용
        new_track_det_indices = unmatched_high

        # ==============================
        # 매칭된 트랙 업데이트
        # ==============================
        for d_idx, t_idx in matches:
            emb_d = embs[d_idx] if (embs is not None and d_idx < len(embs)) else None
            self.tracks[t_idx].update(dets[d_idx, :4], dets[d_idx, 4], emb_d)

        # ==============================
        # 매칭 안 된 트랙 age 증가 & 삭제
        # ==============================
        removed_tracks = []
        for t_idx in unmatched_tracks_final:
            if self.tracks[t_idx].mark_missed():
                removed_tracks.append(t_idx)
        for t_idx in reversed(removed_tracks):
            self.tracks.pop(t_idx)

        # ==============================
        # high_conf 남은 detection → 새 트랙 생성
        # ==============================
        for d_idx in new_track_det_indices:
            emb_d = embs[d_idx] if (embs is not None and d_idx < len(embs)) else None
            new_track = Track(
                dets[d_idx, :4],
                self.next_id,
                dets[d_idx, 4],
                emb_d,
                max_age=self.max_age,
                min_hits=self.min_hits,
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # 최종 반환
        return [t for t in self.tracks if t.confirmed and t.time_since_update <= 1]



class LongTermBoTSORT: # 각 track의 초기 임베딩과 “갤러리에 이미 저장된 고정 임베딩들” 을 비교해서... 비슷하면 ID 재사용 vs 다르면 새 ID 부여 후 갤러리 저장
    """
    BoTSORT 위에 얹는 “장기 ID 레이어” → 갤러리는 한 번 신중하게 저장 후 업데이트 금지
    단기 트랙의 ReID feature를 갤러리와 비교해서:
        같으면 기존 identity_id 재사용
        다르면 새 identity_id 부여
    최종적으로 track.identity_id를 단기 트랙에 붙여서 반환
        main에서는 이 identity를 화면에 표시해서 “나갔다 와도, 겹쳐도 가능하면 같은 번호 유지”를 노리는 구조
    """
    def __init__(self, bot_sort_tracker, embedding_threshold=0.1,       # ID 매칭용 threshold (feat vs gallery)
                 max_memory=1000, max_proto_per_id=5,                   # ID 하나당 갤러리에 저장할 임베딩 개수 
                 conf_thresh=0.7, iou_no_overlap=0.1,                   # YOLO conf 이상 & IOU 겹침 이하일 때만 인정 
                 proto_min_dist=0.02, proto_max_dist=0.06,              # 기존과 0.02 이하로 차이나면 같아서 업뎃X, 0.06 이상 차이나면 다른 사람이라 업뎃X
                 ):
        self.tracker = bot_sort_tracker           # BoTSORT 인스턴스
        self.embedding_threshold = embedding_threshold
        self.max_memory = max_memory

        # identity 갤러리: identity_id -> "embs": [ .. ]
        self.gallery = {}
        self.next_identity = 1
        self.max_proto_per_id = max_proto_per_id
        self.conf_thresh = conf_thresh
        self.iou_no_overlap = iou_no_overlap
        self.proto_min_dist = proto_min_dist
        self.proto_max_dist = proto_max_dist
        
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 1.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb + 1e-6))

    def _min_cos_dist_to_list(self, feat, emb_list):
        """feat vs emb_list 중 최소 코사인 거리"""
        if feat is None or not emb_list:
            return 1.0
        dists = [self.cosine_distance(feat, e) for e in emb_list]
        return min(dists)

    def _assign_identity(self, feat, active_identity_ids):
        """
        feat와 가장 가까운 gallery ID를 찾고,
        embedding_threshold 이하면 그 ID 재사용, 아니면 새 ID 생성.
        active_identity_ids: 이번 프레임에 이미 쓰인 ID (한 프레임 내 중복 방지)
        """
        # 갤러리가 비어있거나 feat 없음 → 무조건 새 ID
        if feat is None or len(self.gallery) == 0:
            identity_id = self.next_identity
            self.next_identity += 1
            # ★ 여기서는 갤러리에 바로 넣지 않는다 (조건 체크는 나중에 따로)
            self.gallery.setdefault(identity_id, {"embs": []})
            return identity_id

        best_id = None
        best_dist = self.embedding_threshold

        for mem_id, info in self.gallery.items():
            if mem_id in active_identity_ids:
                continue
            emb_list = info.get("embs", [])
            if not emb_list:
                continue
            dist = self._min_cos_dist_to_list(feat, emb_list)
            print(f"[GALLERY] id={mem_id} dist={dist:.3f}")
            if dist < best_dist:
                best_dist = dist
                best_id = mem_id

        if best_id is None:
            # 비슷한 ID 없음 → 새 ID 부여
            identity_id = self.next_identity
            self.next_identity += 1
            self.gallery.setdefault(identity_id, {"embs": []})
            return identity_id
        else:
            return best_id


    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return float(inter / (area_a + area_b - inter + 1e-6))

    def _should_add_proto(self, identity_id, track, feat, all_tracks):
        """
        갤러리에 이 feat를 identity_id의 프로토타입으로 추가할지 여부 판단.

        조건:
          - feat None → 추가 X
          - conf(=track.score) >= conf_thresh
          - 다른 트랙과 IoU < iou_no_overlap (겹치지 않을 때만)
          - 이미 max_proto_per_id 개수만큼 저장되어 있으면 더 이상 추가 X
          - 기존 프로토타입과의 거리:
              너무 비슷(proto_min_dist 미만) → 새로 안 넣음
              너무 다름(proto_max_dist 초과) → 위험하니 안 넣음
        """
        if feat is None:
            return False

        # YOLO confidence 체크
        if getattr(track, "score", 0.0) < self.conf_thresh:
            return False

        # 다른 사람과 겹치는지 체크
        for other in all_tracks:
            if other is track:
                continue
            iou_val = self._iou(track.tlbr, other.tlbr)
            if iou_val > self.iou_no_overlap:
                # 꽤 겹친다고 판단 → occlusion 가능성 있음
                return False

        # 갤러리에 이미 있는 프로토타입 개수 확인
        info = self.gallery.setdefault(identity_id, {"embs": []})
        emb_list = info["embs"]

        if len(emb_list) >= self.max_proto_per_id:
            # 이미 최대 개수만큼 저장됨 → 더 안 넣음 (업데이트 금지!)
            return False

        if not emb_list:
            # 첫 프로토타입은 위 조건만 통과하면 허용
            return True

        # 기존 프로토타입들과의 거리 검사
        min_dist = self._min_cos_dist_to_list(feat, emb_list)
        if min_dist < self.proto_min_dist:
            # 거의 같은 포즈/상태 → 굳이 추가 X
            return False
        if min_dist > self.proto_max_dist:
            # 너무 다른 벡터 → 잘못된 매칭 가능성 높음
            return False

        return True

    def _add_proto(self, identity_id, feat):
        info = self.gallery.setdefault(identity_id, {"embs": []})
        info["embs"].append(feat.copy())


    def update(self, detections: np.ndarray, embeddings: list):
        """
        detections: [N,5] (x1,y1,x2,y2,score)
        embeddings: 길이 N, 각 요소는 L2 정규화된 임베딩 (또는 None)
        반환: base BoTSORT의 Track 리스트 (각 track에 identity_id 속성 추가)
        """
        # 1) BoTSORT로 단기 추적
        online_tracks = self.tracker.update(detections, embeddings)

        # 이번 프레임에서 이미 사용된 identity_id들 (중복 방지)
        active_identity_ids = set()

        for track in online_tracks:
            feat = track.get_feature()

            # 2) 항상 갤러리 vs feat로 ID 결정
            identity_id = self._assign_identity(feat, active_identity_ids)

            # 3) "한 번만, 아주 신중하게" 갤러리에 추가할지 판단
            if self._should_add_proto(identity_id, track, feat, online_tracks):
                self._add_proto(identity_id, feat)

            # 이번 프레임 중복 방지
            active_identity_ids.add(identity_id)

            # track 객체에 표시용 ID 저장
            track.identity_id = identity_id

        # 4) 메모리 관리 (선택 사항) – 이번 프레임에 쓰이지 않은 오래된 ID 일부 제거
        if len(self.gallery) > self.max_memory:
            unused_ids = [iid for iid in self.gallery.keys()
                          if iid not in active_identity_ids]
            for iid in unused_ids[: max(0, len(self.gallery) - self.max_memory)]:
                self.gallery.pop(iid, None)

        # 디버그: 프레임 내 ID 중복 여부 확인
        ids_this_frame = [getattr(t, "identity_id", t.track_id) for t in online_tracks]
        if len(ids_this_frame) != len(set(ids_this_frame)):
            print("[WARN] duplicate identity in this frame:", ids_this_frame)

        return online_tracks
