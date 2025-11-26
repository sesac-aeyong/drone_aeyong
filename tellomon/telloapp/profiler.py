# profiler.py
from collections import deque
from time import perf_counter_ns
from dataclasses import dataclass, field
import os

def now_ns():
    return perf_counter_ns()  # monotonic, 시스템시간 변경 영향 없음

@dataclass
class FPSMeter:
    window: int = 120
    times_ns: deque = field(default_factory=lambda: deque(maxlen=120))
    def tick(self):
        self.times_ns.append(now_ns())
    def fps(self):
        if len(self.times_ns) < 2: return 0.0
        dt = (self.times_ns[-1] - self.times_ns[0]) / 1e9
        return (len(self.times_ns)-1) / dt if dt > 0 else 0.0

@dataclass
class LatencyMeter:
    # 각 구간 이름별 누적(ns)
    sums: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)  # 구간 시작 시각
    def start(self, name): self.last[name] = now_ns()
    def stop(self, name):
        t = now_ns() - self.last.get(name, now_ns())
        self.sums[name] = self.sums.get(name, 0) + t
        self.counts[name] = self.counts.get(name, 0) + 1
        return t
    def avg_ms(self, name):
        c = self.counts.get(name, 0)
        return (self.sums.get(name, 0)/c)/1e6 if c else 0.0

# 프레임에 싣는 경량 타임스탬프 컨테이너
def new_trace():
    ns = now_ns()
    return {
        "ts_cam_recv_ns": ns,   # 드론 프레임 수신 시각 (Pi)
        "marks": {},            # 추론 구간별 타임스탬프(ns)
        "_logged": False        # 중복 로깅 방지용
    }
def mark(trace, key):
    trace["marks"][key] = now_ns()

# ============================================================
# CSV LOGGING (자동 지연 측정 저장)
# ============================================================
CSV_PATH = "/tmp/drone_latency.csv"

def _ensure_csv_header():
    """CSV 파일이 없으면 헤더 생성"""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w") as f:
            f.write(
                "ts_cam_recv_ns,ts_frame_recv_ns,ts_yolo_done_ns,"
                "ts_reid_done_ns,ts_depth_done_ns,ts_jpeg_start_ns,"
                "ts_jpeg_done_ns,total_latency_ms\n"
            )


def log_trace_to_csv(trace: dict):
    """
    한 프레임의 trace를 CSV로 기록
    (여기서 total = JPEG 완료 시각 - 카메라 수신 시각)
    """
    if trace.get("_logged", False):
        return  # 한 프레임 중복 기록 방지

    _ensure_csv_header()

    cam = trace["ts_cam_recv_ns"]
    m = trace["marks"]

    # 존재하지 않는 mark는 None 처리
    frecv  = m.get("ts_frame_recv_ns")
    yolo   = m.get("ts_yolo_done_ns")
    reid   = m.get("ts_reid_done_ns")
    depth  = m.get("ts_depth_done_ns")
    jst    = m.get("ts_jpeg_start_ns")
    jdone  = m.get("ts_jpeg_done_ns")

    # E2E latency (JPEG 완료 기준)
    total_ms = (jdone - cam) / 1e6 if jdone else 0.0

    with open(CSV_PATH, "a") as f:
        f.write(
            f"{cam},{frecv},{yolo},{reid},{depth},{jst},{jdone},{total_ms:.3f}\n"
        )

    trace["_logged"] = True