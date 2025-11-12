import numpy as np
from collections import deque
from types import SimpleNamespace

class Track:
    def __init__(self, tlbr, track_id, score, emb=None, max_age=30):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.track_id = track_id
        self.score = score
        self.emb = emb
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.max_age = max_age
        self.history = deque(maxlen=max_age)
        self.confirmed = False

    def update(self, tlbr, score, emb=None):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = score
        if emb is not None:
            self.emb = emb
        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append(self.tlbr)
        if self.hit_streak >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.time_since_update += 1
        if self.time_since_update > self.max_age:
            return True
        return False

class BYTETracker:
    def __init__(self, config: SimpleNamespace):
        self.max_age = getattr(config, 'max_age', 30)
        self.min_hits = getattr(config, 'min_hits', 3)
        self.use_reid = getattr(config, 'use_reid', True)
        self.tracks = []
        self.next_id = 1

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                  + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return o

    def associate_detections_to_tracks(self, dets, embs):
        """
        dets: Nx5 array [xmin, ymin, xmax, ymax, score]
        embs: list of N embeddings or None
        """
        if len(self.tracks) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(dets)), np.empty((0,), dtype=int)

        iou_matrix = np.zeros((len(dets), len(self.tracks)), dtype=np.float32)

        for d, det in enumerate(dets):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self.iou(det[:4], track.tlbr)

        # optionally include embedding distance
        if self.use_reid and embs is not None:
            for d, emb in enumerate(embs):
                for t, track in enumerate(self.tracks):
                    if track.emb is not None:
                        dist = np.linalg.norm(track.emb - emb)
                        # convert distance to pseudo-iou for matching
                        iou_matrix[d, t] *= np.exp(-dist)

        matched_indices = []
        unmatched_dets = list(range(len(dets)))
        unmatched_tracks = list(range(len(self.tracks)))

        if iou_matrix.size > 0:
            for d in range(len(dets)):
                t = np.argmax(iou_matrix[d])
                if iou_matrix[d, t] > 0.3:  # IOU threshold
                    matched_indices.append([d, t])
                    unmatched_dets.remove(d)
                    unmatched_tracks.remove(t)

        return np.array(matched_indices), np.array(unmatched_dets), np.array(unmatched_tracks)

    def update(self, dets, embs=None):
        """
        dets: Nx5 array [xmin, ymin, xmax, ymax, score]
        embs: list of N embeddings (optional, for use_reid)
        """
        dets = np.array(dets)
        embs = embs if self.use_reid else None

        matches, unmatched_dets, unmatched_tracks = self.associate_detections_to_tracks(dets, embs)

        # update matched tracks
        for m in matches:
            d, t = m
            emb = embs[d] if embs is not None else None
            self.tracks[t].update(dets[d,:4], dets[d,4], emb)

        # mark unmatched tracks as missed
        removed_tracks = []
        for t in unmatched_tracks:
            if self.tracks[t].mark_missed():
                removed_tracks.append(t)

        # remove old tracks
        for t in reversed(removed_tracks):
            self.tracks.pop(t)

        # create new tracks for unmatched detections
        for d in unmatched_dets:
            emb = embs[d] if embs is not None else None
            new_track = Track(dets[d,:4], self.next_id, dets[d,4], emb, max_age=self.max_age)
            self.next_id += 1
            self.tracks.append(new_track)

        # return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.confirmed or not self.use_reid]
        return confirmed_tracks
