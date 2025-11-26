# ============================================================
# profiler.py  (FULL VERSION)
# - 프레임 단위 trace
# - 일반모드/도둑모드 기록
# - YOLO/ReID/DEPTH 등 모든 스테이지 지연
# - JPEG/HTTP 전송 레이턴시
# - CSV 자동 저장
# ============================================================

from collections import deque
from time import perf_counter_ns
from dataclasses import dataclass, field
import os

CSV_PATH = "/tmp/drone_latency.csv"


# ============================================================
# Timestamp util
# ============================================================
def now_ns():
    return perf_counter_ns()  # monotonic, 안정적


# ============================================================
# Simple FPS meter (원하면 UI에서만 사용)
# ============================================================
@dataclass
class FPSMeter:
    window: int = 120
    times_ns: deque = field(default_factory=lambda: deque(maxlen=120))

    def tick(self):
        self.times_ns.append(now_ns())

    def fps(self):
        if len(self.times_ns) < 2:
            return 0.0
        dt = (self.times_ns[-1] - self.times_ns[0]) / 1e9
        return (len(self.times_ns) - 1) / dt if dt > 0 else 0.0


# ============================================================
# Latency meter for each stage
# ============================================================
@dataclass
class LatencyMeter:
    sums: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    def start(self, name):
        self.last[name] = now_ns()

    def stop(self, name):
        t = now_ns() - self.last.get(name, now_ns())
        self.sums[name] = self.sums.get(name, 0) + t
        self.counts[name] = self.counts.get(name, 0) + 1
        return t

    def avg_ms(self, name):
        c = self.counts.get(name, 0)
        return (self.sums.get(name, 0) / c) / 1e6 if c else 0.0


# ============================================================
# TRACE OBJECT
# ============================================================
def new_trace():
    ns = now_ns()
    return {
        "ts_cam_recv_ns": ns,      # Pi에서 원본 프레임 받은 시각
        "mode": "normal",          # normal / thief
        "thief_id": "",
        "marks": {},               # 모든 시점(GPU/Hailo/HTTP) 저장
        "_logged": False           # 중복 기록 방지
    }


def mark(trace, key):
    trace["marks"][key] = now_ns()


# ============================================================
# CSV HEADER GENERATOR
# ============================================================
def _ensure_csv_header():
    if os.path.exists(CSV_PATH):
        return

    header = (
        "mode,thief_id,"
        "ts_cam_recv_ns,ts_frame_recv_ns,ts_infer_start_ns,"
        "ts_yolo_done_ns,ts_reid_done_ns,ts_depth_done_ns,ts_infer_done_ns,"
        "ts_jpeg_start_ns,ts_jpeg_done_ns,ts_http_send_ns,"
        "yolo_lat_ns,reid_pre_lat_ns,reid_lat_ns,depth_lat_ns,"
        "infer_loop_ms,total_to_jpeg_ms,http_send_ms,total_to_http_ms\n"
    )
    with open(CSV_PATH, "w") as f:
        f.write(header)


# ============================================================
# MAIN CSV LOGGER
# ============================================================
def log_trace_to_csv(trace: dict):
    if trace.get("_logged", False):
        return

    _ensure_csv_header()

    cam = trace["ts_cam_recv_ns"]
    m = trace["marks"]

    # --- timestamps ---
    frecv = m.get("ts_frame_recv_ns")
    infs  = m.get("ts_infer_start_ns")
    yolo  = m.get("ts_yolo_done_ns")
    reid  = m.get("ts_reid_done_ns")
    depth = m.get("ts_depth_done_ns")
    inend = m.get("ts_infer_done_ns")
    jst   = m.get("ts_jpeg_start_ns")
    jdone = m.get("ts_jpeg_done_ns")
    http  = m.get("ts_http_send_ns")

    # --- stage latencies ---
    yolo_lat  = m.get("yolo_lat_ns")
    reid_pre  = m.get("reid_pre_lat_ns")
    reid_lat  = m.get("reid_lat_ns")
    depth_lat = m.get("depth_lat_ns")

    # --- meta ---
    mode     = trace.get("mode", "")
    thief_id = trace.get("thief_id", "")

    # --- derived ---
    infer_loop_ms = ((inend - infs) / 1e6) if (inend and infs) else 0.0
    total_to_jpeg_ms = ((jdone - cam) / 1e6) if (jdone and cam) else 0.0
    http_send_ms = ((http - jdone) / 1e6) if (http and jdone) else 0.0
    total_to_http_ms = ((http - cam) / 1e6) if (http and cam) else 0.0

    row = (
        f"{mode},{thief_id},"
        f"{cam},{frecv},{infs},"
        f"{yolo},{reid},{depth},{inend},"
        f"{jst},{jdone},{http},"
        f"{yolo_lat if yolo_lat is not None else ''},"
        f"{reid_pre if reid_pre is not None else ''},"
        f"{reid_lat if reid_lat is not None else ''},"
        f"{depth_lat if depth_lat is not None else ''},"
        f"{infer_loop_ms:.3f},"
        f"{total_to_jpeg_ms:.3f},"
        f"{http_send_ms:.3f},"
        f"{total_to_http_ms:.3f}\n"
    )

    with open(CSV_PATH, "a") as f:
        f.write(row)

    trace["_logged"] = True
