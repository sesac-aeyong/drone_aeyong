# 🚁 가랏! in텔로몬

* AI 기반 경찰 보조 드론 시스템으로, 드론이 공중에서 특정 인물이나 상황을 탐지·식별·추적하여 현장 경찰의 임무 수행을 지원하는 것을 목표로 한다

## 🧩 High Level Design

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/6848a1d4-7808-4f92-977d-f69b887c925c" />


YOLOv11 Detection → ReID Embedding → **BoTSORT 단기 추적(track_id)** → **LongTermBoTSORT 장기 ID(identity_id)**  
- 최종 목표: **동일 인물에 대해 장시간 일관된 track_id + identity_id 부여**
- BoTSORT는 프레임 간 단기 연결(칼만필터 + IoU + ReID)
- LongTermBoTSORT는 과거 임베딩 갤러리를 이용해 **동일 인물 ID 재사용**

## 🔗 Clone code

```shell
git clone https://github.com/sesac-aeyong/drone_aeyong.git
```


## 🧑‍💻 Members

  | Name | Role |
  |----|----|
  | 김대용 | Project lead, AI developer, Depth 모델 설계 및 구현 |
  | 김민성 | Embedded system developer, Hailo10H 적용, 시스템 아키텍처 설계 및 구현|
  | 여정인 | AI developer, ReID 모델 구축, Hailo10H 적용, 드론 제어 로직 구현 |
  | 윤영진 | AI developer, AI modeling, Hailo10H 적용, 드론 제어 로직 구현 |
  | 정지훈 | Frontend developer, 사용자 인터페이스를 정의 및 구현 |

## 🗂️ Project Structure
```shell
.
├── README.md
├── requirements.txt                      # dependencies 설치    
└── tellomon/
    ├── common/                                   # 전처리, 이미지 로드 등 각종 함수
    ├── hailorun.py                               # hailo 모델 추론 파이프라인
    ├── main.py                                   # 프로그램 진입지점
    ├── models/                                   # yolo, depth 등 .hef 모델
    ├── patches.py                                # djitellopy 부분 수정
    ├── settings.py                               # 설정 파일
    ├── telloapp/                                 # flask 앱 
    │   ├── __init__.py
    │   ├── app_tools.py
    │   ├── routes.py
    │   ├── tello_web_server.py
    │   └── templates/                            # flask용 html
    ├── tracker/                                  # ReID 추적 모델
    └── yolo_tools.py                             # yolo 추론 결과 표시

```

## ✅ Prerequisites

- **HailoRT 5.0.0+**

## ✅ Hardware
- **Hailo 10H**
- **DJI Tello**

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## ▶️ Steps to run

```shell
cd drone_aeyong
source .venv/bin/activate

cd tellomon
python main.py

# 텔로 와이파이 연결 후
# http://localhost:5000 웹 서버에서 '연결' 클릭해서 드론 영상 연결

# 영상에서 추적 원하는 타겟 클릭해서 추적 시작
```

## 🎬️ Output

<img width="1819" height="1049" alt="Image" src="https://github.com/user-attachments/assets/57dc855f-3c55-400b-b688-be05149f4548" />
<img width="985" height="908" alt="Image" src="https://github.com/user-attachments/assets/66f2559e-2470-4917-98a0-78d2873ff179" />
<img width="1410" height="947" alt="Image" src="https://github.com/user-attachments/assets/40297dda-3840-4f14-be9c-760772f26dd8" />

## Appendix

### **[프로젝트 노션 페이지](https://www.notion.so/2a2ff23adb8e80f3885cd4e247a615bc?source=copy_link)**

### **[프로젝트 기획서](docs/2025SeSAC_Hackathon_AIservice_aeyong.pdf)** 
