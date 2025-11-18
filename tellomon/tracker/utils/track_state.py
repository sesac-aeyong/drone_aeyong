import numpy as np
from collections import deque

class TrackState:  # 이전(t-1) 프레임에서의 위치/임베딩으로 kalman 예측
    """
    “사람 한 명”에 대한 로컬 **상태** 버퍼.

    - last_bbox_tlbr : 마지막 Kalman 보정값 (t-1 기준 “진짜 위치”)
    - kf_bbox_tlbr   : t-1 상태로 t 프레임을 Kalman 예측한 bbox (pred)
    - last_emb       : 마지막으로 매칭된 프레임의 임베딩 (t-1 기준 최신 emb)
    - kf_life        : 보정 없이 예측만 한 프레임 수
    - match_frames   : 연속 매칭된 프레임 수
    - frame_conf     : match_frames >= min_match_frames인지 여부
    """

    def __init__(self, last_bbox_tlbr, track_id, score, emb, max_kf_life, min_match_frames):
        # ---- 기본 메타 ----
        self.last_bbox_tlbr = np.array(last_bbox_tlbr, dtype=np.float32)  # 마지막 보정 결과
        self.kf_bbox_tlbr = self.last_bbox_tlbr.copy()                  # 첫 프레임 초기값 = last

        self.track_id = track_id
        self.score    = float(score)

        # Kalman 수명 / “확실한 트랙” 판단 파라미터
        # (보통 BoTSORT에서 생성할 때 넘겨줌)
        self.max_kf_life      = max_kf_life        # kf_life > max_kf_life → 삭제
        self.min_match_frames = min_match_frames   # match_frames ≥ 이 값 → frame_conf=True

        # ---- 상태 관리 ----
        self.kf_life      = 0                       # 관측 없이 예측만 한 프레임 수
        self.match_frames = 0                       # 연속 매칭 프레임 수
        self.history      = deque(maxlen=max_kf_life)
        self.frame_conf   = False                   # 예전 confirmed

        # ---- ReID 임베딩 (이전 프레임까지의 최신 것 한 장) ----
        self.last_emb = emb   # None 이거나 (D,) 벡터

        # ---- Kalman Filter 초기화 ----
        # 상태벡터 x = [cx, cy, w, h, vx, vy]^T
        cx, cy, w, h = self._bbox_tlbr_to_cxcywh(self.last_bbox_tlbr)
        self.kf = np.array([[cx], [cy], [w], [h], [0.0], [0.0]],
                           dtype=np.float32)

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

        # 공분산 / 잡음
        self.P = np.eye(6, dtype=np.float32) * 10.0   # 초기 불확실성: 작을수록 예측값 신뢰
        self.Q = np.eye(6, dtype=np.float32) * 0.1    # 시스템 잡음: 작을수록 등속도 가정 신뢰   → 크게: 운동이 불규칙하니까 관측을 더 따라가라
        self.R = np.eye(4, dtype=np.float32) * 1.0    # 관측 잡음: 작을수록 yolo 신뢰         → 크게: 관측을 믿을수없으니 예측을 더 따라가라

    # ================== bbox <-> 상태 변환 유틸 ==================

    @staticmethod
    def _bbox_tlbr_to_cxcywh(bbox_tlbr):
        x1, y1, x2, y2 = bbox_tlbr
        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return float(cx), float(cy), float(w), float(h)

    @staticmethod
    def _cxcywh_to_bbox_tlbr(cx, cy, w, h):
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    # ================== Kalman predict / correct ==================

    def predict(self):
        """
        1프레임 뒤 위치 예측 (관측 없이).

        - kf        : F @ kf
        - kf_bbox_tlbr 를 예측 결과로 갱신
        - kf_life  += 1
        - last_bbox_tlbr 는 '마지막 보정값'으로 그대로 유지
        """
        self.kf = self.F @ self.kf
        self.P  = self.F @ self.P @ self.F.T + self.Q

        cx, cy, w, h = self.kf[:4, 0]
        self.kf_bbox_tlbr = self._cxcywh_to_bbox_tlbr(cx, cy, w, h)

        # 관측 없이 예측만 했으므로 수명 +1
        self.kf_life += 1

    def _correct_kf(self, now_bbox_tlbr):
        """
        새 detection bbox(now_bbox_tlbr)로 Kalman 보정.

        - kf / P 업데이트
        - last_bbox_tlbr 를 '보정된 값'으로 갱신
        - kf_bbox_tlbr 도 last_bbox_tlbr 로 동기화
        - kf_life 를 0으로 리셋
        """
        cx, cy, w, h = self._bbox_tlbr_to_cxcywh(now_bbox_tlbr)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        # y = z - Hx
        y = z - (self.H @ self.kf)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 상태 / 공분산 업데이트
        self.kf = self.kf + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        # 보정된 bbox_tlbr → last_bbox_tlbr 로 저장
        cx, cy, w, h = self.kf[:4, 0]
        self.last_bbox_tlbr = self._cxcywh_to_bbox_tlbr(cx, cy, w, h)
        self.kf_bbox_tlbr = self.last_bbox_tlbr.copy()

        # 이제 막 관측으로 보정했으므로 0으로 리셋
        self.kf_life = 0

    # ================== 업데이트 & 수명 관리 ==================

    def update(self, now_bbox_tlbr, score, now_emb=None):
        """
        BoTSORT에서 detection과 매칭된 뒤 호출.

        - Kalman 보정 (now_bbox_tlbr 사용)
        - score / match_frames / history 갱신
        - last_emb 를 이번 프레임 now_emb 로 교체
        - match_frames ≥ min_match_frames → frame_conf = True
        """
        # 1) 칼만 보정 (last → now)
        self._correct_kf(now_bbox_tlbr)

        # 2) 메타 정보 갱신
        self.score = float(score)
        self.match_frames += 1
        self.history.append(self.last_bbox_tlbr.copy())

        # 3) 최신 emb로 교체 (직전 프레임 emb는 덮어씀)
        if now_emb is not None:
            self.last_emb = now_emb

        # 4) 충분히 오래 안정적으로 매칭되었으면 “확신”
        if self.match_frames >= self.min_match_frames:
            self.frame_conf = True

    def mark_missed(self):
        """
        이번 프레임에 detection과 매칭 안 된 경우:
        - kf_life가 max_kf_life를 넘으면 True (삭제 대상)
        """
        return self.kf_life > self.max_kf_life