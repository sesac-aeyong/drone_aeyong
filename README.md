## 1. 📌 Pipeline Overview

YOLOv8 Detection → ReID Embedding → **BoTSORT 단기 추적(track_id)** → **LongTermBoTSORT 장기 ID(identity_id)**  
- 최종 목표: **동일 인물에 대해 장시간 일관된 track_id + identity_id 부여**
- BoTSORT는 프레임 간 단기 연결(칼만필터 + IoU + ReID)
- LongTermBoTSORT는 과거 임베딩 갤러리를 이용해 **동일 인물 ID 재사용**


---

## 2. 📦 Class Structure

### **Class Track**
“사람 한 명”의 **로컬 상태 버퍼 + 칼만 필터**

**저장 역할**
- `last_bbox_tlbr` : 마지막 보정된 실제 위치(t-1)
- `kf_bbox_tlbr`   : t 기준 칼만 예측 위치(pred)
- `last_emb` : 마지막으로 매칭된 프레임의 ReID 임베딩
- `kf_life` : 관측 없이 예측만 한 프레임 수
- `match_frames` : 연속 매칭된 프레임 수
- `frame_conf` : `match_frames ≥ min_match_frames` 이면 True → 화면에 표시 가능

**칼만 상태**
- x = [cx, cy, w, h, vx, vy]^T  
- predict() → 관측 없이 1프레임 예측  
- update() → detection으로 보정  

**주요 기능**
- `predict()` : t-1 → t 위치 예측  
- `update(now_bbox, score, now_emb)`  
  - 칼만 보정  
  - 최신 bbox/score/embedding 저장  
  - match_frames 증가  
- `mark_missed()`  
  - `kf_life > max_kf_life` 면 삭제 대상


---

### **Class BoTSORT**
YOLO + ReID 기반 **단기 추적기 → track_id 관리**

### 🔹 매 프레임 동작 요약
1. **모든 Track.predict()**
2. now_dets vs predicted_track 위치 + 임베딩 기반 cost matrix 구성
3. **1단계: high confidence dets(≥ high_thresh) ↔ 모든 Track 매칭**
4. **2단계: 남은 Track ↔ low confidence dets 매칭**
5. 매칭된 Track.update()
6. 매칭되지 않은 Track은 kf_life 증가 → 오래되면 제거
7. 끝까지 매칭 안 된 high_yolo det → **새 Track 생성**

### 🔹 Cost 구성
cost = (1 - IoU(pred_bbox, now_bbox))
+ reid_weight * L2(last_emb, now_emb) (둘 다 존재할 때만)



### 🔹 출력
다음 조건을 만족한 Track만 반환:
- `frame_conf == True` (연속 매칭으로 안정적)
- `kf_life <= 1` (이번 프레임 기준으로 거의 사라지지 않음)


---

### **Class LongTermBoTSORT**
ReID embedding 기반 **장기 identity_id 관리 + 갤러리 메모리**

단기 track_id는 프레임 중간에 바뀔 수 있으므로,  
**identity_id를 통해 오랜 시간 같은 사람임을 보장**.

### 🔹 주요 개념
- **gallery:**  
  `{ identity_id: { "gal_embs": [prototype embeddings...] } }`

- 새 사람 등장 시:
  - 갤러리에 유사한 embedding 없음 → 새 identity_id 발급
- 이미 있던 사람이라면:
  - 갤러리 벡터와의 cosine distance가 `threshold` 이내 → ID 재사용

### 🔹 Prototype 저장 조건 (매우 Conservative)
Track 임베딩을 갤러리에 저장하려면:

- YOLO score ≥ `conf_thresh`
- 다른 Track과 IoU ≤ `iou_no_overlap`  → occlusion 제거
- 기존 prototype 수 < `max_gal_emb_per_id`
- 기존 prototype과의 최소 거리:
  - 너무 비슷하면(`min_dist < min_dist_thr`) → 중복이므로 불가
  - 너무 다르면(`min_dist > max_dist_thr`) → 잘못된 ID일 확률 → 불가

**→ 안전한 프로토타입만 갤러리에 저장**

### 🔹 출력
BoTSORT online_tracks에  
`.identity_id` 필드를 추가한 Track 리스트 반환


---

## 3. ▶️ 실행 방법
```bash
python main_xpu.py --display

- YOLO + ReID 모델 로딩
- LongTermBoTSORT가 track_id + identity_id 부여
- 화면에 bbox + (track_id, identity_id) 표시
```


---

## 4. 🧪 Known Behaviors

👍 잘 되는 상황

- 한 명일 때: 포즈 변화, 뒷모습, 부분 가림에서도 ID 유지
- 두 명일 때: 겹친채 지나가도 ㄱㅊ
- 오랜 시간 사라졌다가 재탐지되어도 갤러리를 이용해 identity_id 재사용됨


⚠️ 취약한 상황
- 매우 낮은 해상도 / 임베딩 품질 저하
- 1명 나가고 다른 사람들어오면 기존 사람으로 인식? 
- 프레임 드랍 발생 시 칼만 예측 불확실성 커짐


❓ 테스트 필요
- FPS가 매우 낮을 때 ReID 품질 유지?
- 조명/배경 크게 바뀌어도 identity_id 재사용 잘 되는지
- 겹침 후 분리되는 상황에서 swap 방지 능력