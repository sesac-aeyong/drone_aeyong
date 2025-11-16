# gallery_io.py
import os
import numpy as np
from typing import Dict, Any


def gallery_to_serializable(gallery: Dict[int, dict]) -> Dict[int, np.ndarray]:
    """
    내부 gallery(dict) -> 저장 가능한 dict(id -> 2D array) 로 변환.
    - gallery: {id: {"gal_embs": [emb1, emb2, ...]}, ...}
    - 반환: {id: np.array(shape=(K, D)), ...}
    """
    serializable = {}

    for iid, info in gallery.items():
        gal_emb_list = info.get("gal_embs", [])
        if not gal_emb_list:
            continue
        arr = np.stack(gal_emb_list, axis=0)  # (K, D)
        serializable[int(iid)] = arr

    return serializable


def serializable_to_gallery(data: Dict[Any, np.ndarray]) -> Dict[int, dict]:
    """
    저장된 dict(id -> 2D array) -> 내부 gallery(dict) 복원.
    - data: {id: np.array(shape=(K, D)), ...}
    - 반환: {id: {"gal_embs": [emb1, emb2, ...]}, ...}
    """
    gallery = {}
    for iid, arr in data.items():
        iid_int = int(iid)
        gal_emb_list = [emb.copy() for emb in arr]
        gallery[iid_int] = {"gal_embs": gal_emb_list}
    return gallery


def save_gallery(path: str, gallery: Dict[int, dict]):
    """
    gallery(dict)를 path(.npy)에 저장.
    """
    serializable = gallery_to_serializable(gallery)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, serializable, allow_pickle=True)
    print(f"[GAL-IO] Saved gallery with {len(serializable)} identities → {os.path.abspath(path)}")


def load_gallery(path: str) -> Dict[int, dict]:
    """
    path(.npy)에서 gallery(dict)를 읽어 복원.
    - 파일이 없으면 빈 dict 반환.
    """
    if not os.path.exists(path):
        print(f"[GAL-IO] gallery file not found: {path}")
        return {}

    data = np.load(path, allow_pickle=True).item()
    gallery = serializable_to_gallery(data)
    print(f"[GAL-IO] Loaded gallery with {len(gallery)} identities from {os.path.abspath(path)}")
    return gallery
