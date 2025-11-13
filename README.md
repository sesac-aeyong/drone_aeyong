## 1. 📌 Pipeline Overview

YOLOv8 감지 → ReID 임베딩 → BoTSORT 단기 추적 → LongTermBoTSORT 장기 ID 재사용
- 목표: 같은 사람에게 **일관된 track_id + identity_id**를 유지하는 사람 추적 파이프라인


## 2. 📦 Class Structure

### Class Track
"한 사람에 대한 Kalman + 단일(1회) 임베딩 저장용 구조체"

YOLO가 낸 bbox(tlbr)를 입력받아 Kalman state로 변환
x = [cx, cy, w, h, vx, vy]

매 프레임 Kalman predict → update

추적 상태 플래그:
- time_since_update
- hit_streak
- confirmed

유틸:
- self.embeddings → 최초 등장 시 딱 1번 임베딩 저장
(장기 갤러리에서 ID 재사용할 때만 의미 있음)
- get_feature() → embeddings의 평균을 대표벡터로 반환
- mark_missed() → max_age 이상 안보이면 track 삭제


### Class BoTSORT
"프레임 간 단기 연결 (Kalman + Hungarian + IoU + ReID)"

입력: YOLO bbox + ReID embedding

Matching Logic
1. High-confidence detections만 먼저 매칭 (conf > high_thresh)
2. Kalman filter로 모든 기존 track을 predict
3. Yolo dets + ReID embs 거리 기반 cost matrix 생성 
    - 많이 겹칠수록(IoU ↑), feature가 비슷할수록(거리 ↓) → 비용이 낮다(좋은 매칭)
4. Hungarian algorithm으로 최소 비용 매칭
    - 매칭됨 → update
    - 매칭 안됨 →
        - bbox가 high_conf면 새 track 생성
        - low_conf면 2차 매칭 시도 (low_thresh ~ high_thresh)


### Class LongTermBoTSORT
"갤러리 기반 장기 ID 재사용 & 갱신"

🔹 ID 재사용 결정 _assign_identity()

새 Track의 feature가 다음 조건이면 새 ID 발급:
- feat is None
- 갤러리가 비어 있음
- 모든 갤러리 ID와의 거리 best_dist ≥ embedding_threshold → “누구와도 비슷하지 않음 = 새로운 사람”

그 외에는 가장 가까운 ID 재사용

🔹 갤러리 프로토타입 저장 _should_add_proto()

해당 track의 임베딩을 갤러리에 추가하는 조건: 
- YOLO conf ≥ conf_thresh                   → 흐린 프레임(롤링셔터 등) 배제
- IoU ≤ iou_no_overlap                      → 다른 사람과 붙어 있는 경우 배제
- proto 개수 < max_proto_per_id              → 너무 많이 저장 금지
- proto_min_dist < dist < proto_max_dist    → 너무 비슷(중복)하지 않고, 너무 다르지도 않을 때만 저장



## 3. ▶️ 실행 방법
python main_xpu.py --display



## 4. 🧪 Known Behaviors

👍 잘 되는 상황
- 사람이 1명만 있을 때 포즈 변경, 뒷모습, 얼굴 가려도 ID 유지

⚠️ 잘 안 되는 상황
- ?


❓️ 테스트 필요한 상황
- 프레임 드랍(저 FPS) / 저해상도에서 ReID 성능 유지되는지(라베파 올렸을때 가정..)
- 완전 다른 배경/조명 조건에서 같은 사람을 재탐지했을 때 ID 재사용이 되는지
- 두 사람이 겹쳐 지나갈 때 ID가 뒤바뀌는지 여부