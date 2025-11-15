from dataclasses import dataclass
import json
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
    embed_model: str = 'models/repvgg_a0_person_reid_512.hef'
    _emb_out_size: int = 2048 if '2048' in embed_model else 512 
    """Hmm. Perhaps there is a way to get this info"""
    depth_model: str = 'models/scdepthv3.hef'

    max_vis_detections: int = 30
    """vision model max detections"""
    min_vis_score_threshold: float = 0.5
    """vision model confidence threshold"""
    min_emb_confidence: float = 0.5
    """minimum vis_model confidence to update embedding"""
    max_emb_threads: int = 4
    """max number of threads to run embedding on"""
    min_emb_cropsize: int = 50
    """minimum crop size to run embedding on"""


    ###WIP
    track_high_emb_confidence: float = min_emb_confidence
    """high confidence threshold trackre expects to get embeddings of"""
    track_low_emb_confidence: float = 0.4
    """"""
    track_new_threshold: float = 0.6
    """threshold for vision model confidence to create a new track"""
    track_buffer: int = 600
    """# of frames lost tracks will be kept"""


    """General"""
    frame_width: int = 960
    frame_height: int = 720

    def __post_init__(self): #
        try:
            with open('config.json') as f:
                sjson:dict = json.load(f)
                for k, v in sjson.items():
                    if k.startswith('_'):
                        continue
                    if hasattr(self, k):
                        setattr(self, k, v)
                    else:
                        print(f'Unknown config key {k} with value {v} was ignored.')
        except FileNotFoundError:
            print('failed to load config.json, using defaults!')
        except json.JSONDecodeError:
            print('failed to convert config.json to json. Using defaults!')

settings = Settings()

