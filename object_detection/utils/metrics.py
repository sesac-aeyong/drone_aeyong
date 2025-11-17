import numpy as np
from typing import Tuple


def iou_bbox(a: np.ndarray, b: np.ndarray) -> float:
    """
    tlbr 두 박스의 IoU
    a, b: (4,) [x1,y1,x2,y2]
    """
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    코사인 거리 = 1 - cos(a,b)
    """
    if a is None or b is None:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb + 1e-6))

def min_cos_dist_to_list(cand_emb, gallery_embs, default=1.0):
    """
    cand_emb vs gallery_embs(list/ndarray) 중 최소 코사인 거리.
    - cand_emb is None 이거나, gallery가 비어 있으면 default 반환 (기본 1.0)
    """
    if cand_emb is None:
        return default
    if gallery_embs is None:
        return default

    # list, tuple, np.ndarray 다 허용
    # np.ndarray 인 경우 (K, D) / (D,) 모두 처리
    if isinstance(gallery_embs, np.ndarray):
        if gallery_embs.ndim == 1:
            gallery_list = [gallery_embs]
        else:
            gallery_list = [g for g in gallery_embs]
    else:
        gallery_list = list(gallery_embs)

    if len(gallery_list) == 0:
        return default

    return min(cosine_distance(cand_emb, g) for g in gallery_list)


def bbox_center(box: np.ndarray) -> Tuple[float, float]:
    """
    tlbr 박스 중심 (cx, cy)
    """
    x1, y1, x2, y2 = box
    return float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)
