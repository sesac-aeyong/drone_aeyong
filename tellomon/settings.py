from dataclasses import dataclass, field
from scipy.interpolate import Rbf


import numpy as np
"""
Usage:
from settings import settings [as S]
"""

@dataclass
class Settings:
    """
    Tellomon default settings. Please modify config.json 
    """
    """Hailo"""
    vis_model: str = 'models/yolov11s.hef'
    embed_model: str = 'models/repvgg_a0_person_reid_2048.hef'
    _emb_out_size: int = 2048 if '2048' in embed_model else 512 
    """Hmm. Perhaps there is a way to get this info"""
    depth_model: str = 'models/scdepthv3.hef'

    max_vis_detections: int = 30
    """vision model max detections"""
    min_vis_score_threshold: float = 0.8
    """vision model confidence threshold"""
    min_emb_confidence: float = min_vis_score_threshold
    """minimum vis_model confidence to update embedding"""
    max_emb_threads: int = 4
    """max number of threads to run embedding on"""
    min_emb_cropsize: int = 200
    """minimum crop size to run embedding on"""

    laser_canny_lower_threshold: int = 50
    laser_canny_high_threshold: int = 90
    laser_roi_x1: int = 120
    laser_roi_x2: int = 250
    laser_roi_y1: int = 490
    laser_roi_y2: int = 550
    laser_circularity_threshold: float = 0.7
    """1.0: perfect circle"""       
    laser_x_pixels: np.ndarray = field(default_factory=lambda: np.array([84, 62.0, 48.0, 43, 41, 39.5]))
    laser_y_pixels: np.ndarray = field(default_factory=lambda: np.array([43, 31, 21.7, 20, 18.2, 17]))
    laser_distances: np.ndarray = field(default_factory=lambda: np.array([30, 60, 120, 180, 240, 300]))
    laser_rbf: Rbf = field(init=False)

    # Precompute linear model coefficients
    _laser_dat: np.ndarray = field(init=False)
    laser_coeffs: np.ndarray = field(init=False)
    red_mll: np.ndarray = field(default_factory= lambda: np.array(np.array([0, 0, 50])))
    red_mlu: np.ndarray = field(default_factory= lambda: np.array(np.array([15, 255, 255])))
    red_mul: np.ndarray = field(default_factory= lambda: np.array(np.array([168, 0, 50])))
    red_muu: np.ndarray = field(default_factory= lambda: np.array(np.array([180, 255, 255])))


    ###WIP
    track_high_emb_confidence: float = min_emb_confidence
    """high confidence threshold tracker expects to get embeddings of"""
    track_low_emb_confidence: float = 0.4
    """this probably is not used"""
    track_new_threshold: float = min_emb_confidence + 0.1
    """threshold for vision model confidence to create a new track"""
    track_buffer: int = 600
    """# of frames lost tracks will be kept"""

    """General"""
    frame_width: int = 960
    frame_height: int = 720

    """Tello"""
    # tello_id:str = '5FD7AB' # blue tello 
    # tello_id:str = '5A6D18' # white tello
    tello_id: str = None
    tello_retry_count: int = 1
    tello_response_timeout:int = 3
    tello_vs_port: int = 11111
    tello_ip: str = '192.168.10.1'
    """set tello_id to None for automatic connection"""

    """TelloWebServer"""
    # tello_ws_stream_on_off_timeout: int = 3
    """time in seconds to wait for streamon/streamoff command"""


    def __post_init__(self):
        self.laser_rbf = Rbf(self.laser_x_pixels, self.laser_y_pixels, self.laser_distances, function='multiquadric')
    #     self._laser_dat = np.column_stack([self.laser_x_pixels, self.laser_y_pixels, np.ones(len(self.laser_x_pixels))])
    #     self.laser_coeffs = np.linalg.lstsq(self._laser_dat, self.laser_distances, rcond=None)[0]

settings = Settings()

