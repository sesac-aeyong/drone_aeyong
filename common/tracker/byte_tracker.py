import numpy as np
from .kalman_filter import KalmanFilter
from .matching import Matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feature=None):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

        # ReID embedding
        self.smooth_feat = None
        self.curr_feat = None
        if feature is not None:
            self.update_feature(feature)

    def update_feature(self, feat, alpha=0.9):
        """Exponential moving average feature update."""
        feat = feat / np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_feature(new_track.curr_feat)

    def update(self, new_track, frame_id):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_feature(new_track.curr_feat)

    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert tlwh → tlbr."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh → (cx, cy, aspect, h)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # active tracks
        self.lost_stracks = []     # lost tracks
        self.removed_stracks = []  # removed tracks
        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, embeddings=None):
        """
        output_results: Nx5 (x1,y1,x2,y2,score)
        embeddings: [N x feat_dim] (optional)
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # --- create detections (with embeddings if available) ---
        detections = []
        if len(dets) > 0:
            for i, (tlbr, s) in enumerate(zip(dets, scores_keep)):
                feat = None
                if embeddings is not None and i < len(embeddings):
                    feat = embeddings[i]
                detections.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, feature=feat))

        """Separate unconfirmed and active tracks"""
        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked_stracks).append(track)

        """Step 1: First association (high-score detections)"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # compute distances (IoU + appearance)
        iou_dists = Matching.iou_distance(strack_pool, detections)
        app_dists = Matching.embedding_distance(strack_pool, detections)
        alpha = 0.4
        dists = Matching.fuse_motion_appearance(iou_dists, app_dists, alpha=alpha)

        matches, u_track, u_detection = Matching.linear_assignment(dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """Step 2: Second association (low-score detections)"""
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)
        ] if len(dets_second) > 0 else []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        iou_dists = Matching.iou_distance(r_tracked_stracks, detections_second)
        app_dists = Matching.embedding_distance(r_tracked_stracks, detections_second)
        dists = Matching.fuse_motion_appearance(iou_dists, app_dists, alpha=alpha)

        matches, u_track, u_detection_second = Matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Step 3: Handle unconfirmed tracks"""
        detections = [detections[i] for i in u_detection]
        iou_dists = Matching.iou_distance(unconfirmed, detections)
        app_dists = Matching.embedding_distance(unconfirmed, detections)
        dists = Matching.fuse_motion_appearance(iou_dists, app_dists, alpha=alpha)
        matches, u_unconfirmed, u_detection = Matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """Step 4: Initialize new tracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """Step 5: Cleanup"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


# ---------- Utility functions ----------

def joint_stracks(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = Matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


# ---------- Extend Matching with ReID distance ----------

def _add_reid_support_to_matching():
    def embedding_distance(tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                if t.smooth_feat is None or d.curr_feat is None:
                    cost[i, j] = 1.0
                else:
                    cost[i, j] = 1.0 - np.dot(t.smooth_feat, d.curr_feat)
        return cost

    def fuse_motion_appearance(iou_dists, app_dists, alpha=0.6):
        if iou_dists.size == 0:
            return app_dists
        if app_dists.size == 0:
            return iou_dists
        return alpha * iou_dists + (1 - alpha) * app_dists

    Matching.embedding_distance = staticmethod(embedding_distance)
    Matching.fuse_motion_appearance = staticmethod(fuse_motion_appearance)


_add_reid_support_to_matching()
