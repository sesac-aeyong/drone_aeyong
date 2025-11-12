import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, tlbr, track_id, score, emb=None, max_age=30):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.track_id = track_id
        self.score = score
        self.max_age = max_age
        self.time_since_update = 0
        self.hit_streak = 0
        self.history = deque(maxlen=max_age)
        self.confirmed = False

        # BoT-SORT style embedding gallery
        self.embeddings = []
        if emb is not None:
            self.embeddings.append(emb)

    def update(self, tlbr, score, emb=None):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = score
        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append(self.tlbr)

        if emb is not None:
            if not self.embeddings:
                self.embeddings.append(emb)
            else:
                # EMA: smooth previous embedding with new
                alpha = 0.5
                self.embeddings[-1] = alpha * emb + (1 - alpha) * self.embeddings[-1]

        if self.hit_streak >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.time_since_update += 1
        return self.time_since_update > self.max_age

    def get_feature(self):
        if self.embeddings:
            # average of gallery
            return np.mean(self.embeddings, axis=0)
        return None

class BoTSORT:
    def __init__(self, max_age=30, min_hits=3, use_reid=True, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_reid = use_reid
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1


    def merge_similar_tracks(self, similarity_thresh=0.4):
        """
        Merge tracks with very similar embeddings. Keep the oldest track ID.
        """
        merged = set()
        N = len(self.tracks)
        for i in range(N):
            if i in merged:
                continue
            track_i = self.tracks[i]
            feat_i = track_i.get_feature()
            if feat_i is None:
                continue
            for j in range(i+1, N):
                if j in merged:
                    continue
                track_j = self.tracks[j]
                feat_j = track_j.get_feature()
                if feat_j is None:
                    continue
                dist = np.linalg.norm(feat_i - feat_j)
                if dist < similarity_thresh:
                    # merge j into i (keep i's ID)
                    track_i.embeddings.extend(track_j.embeddings)
                    track_i.embeddings = track_i.embeddings[-100:]  # keep max gallery size
                    merged.add(j)
                    print(f"Merging track {track_j.track_id} into {track_i.track_id}")

        # remove merged tracks
        self.tracks = [t for idx, t in enumerate(self.tracks) if idx not in merged]

        
    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                  + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-16)
        return o

    def compute_cost_matrix(self, dets, embs):
        N = len(dets)
        M = len(self.tracks)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for d, det in enumerate(dets):
            for t, track in enumerate(self.tracks):
                iou_score = self.iou(det[:4], track.tlbr)
                cost = 1 - iou_score
                if self.use_reid and embs is not None:
                    track_feat = track.get_feature()
                    if track_feat is not None:
                        dist = np.linalg.norm(track_feat - embs[d])
                        cost += dist * 0.8  # combine IOU & feature distance
                cost_matrix[d, t] = cost
        return cost_matrix

    def update(self, dets, embs=None):
        dets = np.array(dets)
        embs = embs if self.use_reid else None

        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(dets)))
            unmatched_tracks = []
            matches = []
        else:
            cost_matrix = self.compute_cost_matrix(dets, embs)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = []
            unmatched_dets = list(range(len(dets)))
            unmatched_tracks = list(range(len(self.tracks)))

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > 1 - self.iou_threshold:
                    continue
                matches.append([r, c])
                unmatched_dets.remove(r)
                unmatched_tracks.remove(c)

        # update matched tracks
        for d, t in matches:
            emb = embs[d] if embs is not None else None
            self.tracks[t].update(dets[d, :4], dets[d, 4], emb)

        # mark unmatched tracks as missed
        removed_tracks = []
        for t in unmatched_tracks:
            if self.tracks[t].mark_missed():
                removed_tracks.append(t)
        for t in reversed(removed_tracks):
            self.tracks.pop(t)

        # create new tracks for unmatched detections
        for d in unmatched_dets:
            emb = embs[d] if embs is not None else None
            new_track = Track(dets[d, :4], self.next_id, dets[d, 4], emb, max_age=self.max_age)
            self.next_id += 1
            self.tracks.append(new_track)

        self.merge_similar_tracks(similarity_thresh=0.4)
        # return confirmed tracks
        # return [t for t in self.tracks if t.confirmed or not self.use_reid]
        return [t for t in self.tracks if t.confirmed and t.time_since_update <= 1]
