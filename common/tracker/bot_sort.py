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
        self.deleted = False

        self.embeddings = []
        if emb is not None:
            self.embeddings.append(emb)

    def update(self, tlbr, score, emb=None):
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = score
        self.time_since_update = 0
        self.hit_streak += 1
        self.history.append(self.tlbr)

        if score >= 0.85:
            # print('update embedding for', self.track_id)
            if emb is not None:
                if not self.embeddings:
                    self.embeddings.append(emb)
                else:
                    alpha = 0.9
                    self.embeddings[-1] = alpha * emb + (1 - alpha) * self.embeddings[-1]

        if self.hit_streak >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.time_since_update += 1
        return self.time_since_update > self.max_age

    def get_feature(self):
        if self.embeddings:
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
        self.frame_count = 0

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        return wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                     + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-16)

    def update(self, dets, embs=None):
        self.frame_count += 1
        dets = np.array(dets)
        embs = embs if self.use_reid else None

        # === STAGE 1: MATCH EXISTING TRACKS WITH BETTER COST MATRIX ===
        if len(self.tracks) == 0:
            matches = []
            unmatched_dets = list(range(len(dets)))
            unmatched_tracks = []
        else:
            # Use BETTER cost matrix with both IoU and appearance
            cost_matrix = self.compute_balanced_cost_matrix(dets, embs)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matches = []
            unmatched_dets = list(range(len(dets)))
            unmatched_tracks = list(range(len(self.tracks)))
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.8:  # Reasonable threshold
                    matches.append([r, c])
                    unmatched_dets.remove(r)
                    unmatched_tracks.remove(c)

        # Update matched tracks
        for d, t in matches:
            emb = embs[d] if embs is not None else None
            self.tracks[t].update(dets[d, :4], dets[d, 4], emb)

        # === STAGE 2: BETTER RE-ID FOR UNMATCHED TRACKS ===
        if self.use_reid and embs is not None and len(unmatched_tracks) > 0:
            reid_matches = []
            
            # Try to re-ID ALL unmatched tracks (not just confirmed ones)
            for t_idx in unmatched_tracks[:]:
                track = self.tracks[t_idx]
                if track.time_since_update > 20:  # Don't re-ID tracks lost for too long
                    continue
                    
                track_feat = track.get_feature()
                if track_feat is None:
                    continue
                    
                best_det = None
                best_cost = float('inf')
                
                for d_idx in unmatched_dets[:]:
                    det_feat = embs[d_idx]
                    cost = np.linalg.norm(track_feat - det_feat)
                    
                    # MORE REASONABLE re-ID threshold
                    if cost < 0.4 and cost < best_cost:  # Less strict!
                        iou_score = self.iou(dets[d_idx, :4], track.tlbr)
                        if iou_score > 0.05:  # Very loose spatial consistency
                            best_det = d_idx
                            best_cost = cost
                
                if best_det is not None:
                    reid_matches.append([best_det, t_idx])
                    unmatched_dets.remove(best_det)
                    unmatched_tracks.remove(t_idx)
                    print(f"RE-ID: Track {track.track_id} recovered (cost={best_cost:.3f})")
            
            # Apply re-ID matches
            for d, t in reid_matches:
                emb = embs[d] if embs is not None else None
                self.tracks[t].update(dets[d, :4], dets[d, 4], emb)

        # === STAGE 3: BETTER TRACK MANAGEMENT ===
        removed_tracks = []
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            if track.mark_missed():
                # Much more conservative removal
                if track.time_since_update > self.max_age:
                    # Only remove if very new AND unconfirmed
                    if track.hit_streak < 10 and not track.confirmed:
                        track.deleted = True
                        removed_tracks.append(t_idx)
                        print(f"REMOVING: Track {track.track_id} (hits={track.hit_streak})")
        
        for t_idx in reversed(removed_tracks):
            self.tracks.pop(t_idx)

        # === STAGE 4: CREATE NEW TRACKS (WITH BETTER CHECK) ===
        new_tracks_created = 0
        for d_idx in unmatched_dets:
            emb = embs[d_idx] if embs is not None else None
            
            # Check if this detection is similar to any existing track
            similar_track_id = self.find_similar_track(dets[d_idx, :4], emb)
            if similar_track_id is None:
                # No similar track exists - create new one
                new_track = Track(dets[d_idx, :4], self.next_id, dets[d_idx, 4], emb, max_age=self.max_age)
                self.next_id += 1
                self.tracks.append(new_track)
                new_tracks_created += 1
                print(f"NEW TRACK: ID {new_track.track_id} created")
            else:
                print(f"SKIPPED: Detection similar to Track {similar_track_id}")
        
        if new_tracks_created > 0:
            print(f"TOTAL TRACKS: {len(self.tracks)}")

        # === STAGE 5: SAFE MERGING - PREVENT ID STEALING ===
        self.merge_only_obvious_duplicates()

        # === STAGE 6: VALIDATION ===
        self.validate_track_assignments()

        # Return active tracks
        online_targets = [t for t in self.tracks if t.time_since_update <= 2]
        return online_targets

    def compute_balanced_cost_matrix(self, dets, embs):
        """Cost matrix that strongly protects confirmed tracks"""
        N = len(dets)
        M = len(self.tracks)
        cost_matrix = np.ones((N, M), dtype=np.float32) * 10.0  # High default cost
        
        for d, det in enumerate(dets):
            for t, track in enumerate(self.tracks):
                if track.deleted:
                    continue
                    
                iou_score = self.iou(det[:4], track.tlbr)
                iou_cost = 1.0 - iou_score
                
                appearance_cost = 0.0
                if self.use_reid and embs is not None and embs[d] is not None:
                    track_feat = track.get_feature()
                    if track_feat is not None:
                        dist = np.linalg.norm(track_feat - embs[d])
                        appearance_cost = min(dist * 0.5, 1.0)
                
                if track.confirmed:
                    # HEAVILY protect confirmed tracks from bad matches
                    if iou_score < 0.3:  # Low IoU
                        total_cost = 5.0  # Very high cost - prevents bad matches
                    elif appearance_cost > 0.3:  # Bad appearance match
                        total_cost = 3.0  # High cost
                    else:
                        # Good match - normal cost
                        total_cost = iou_cost * 0.4 + appearance_cost * 0.6
                else:
                    # Normal cost for unconfirmed tracks
                    total_cost = iou_cost * 0.7 + appearance_cost * 0.3
                
                cost_matrix[d, t] = total_cost
        
        return cost_matrix

    def find_similar_track(self, bbox, emb):
        """Find if a similar track already exists"""
        for track in self.tracks:
            if track.deleted:
                continue
                
            iou_score = self.iou(bbox, track.tlbr)
            
            # High spatial overlap = definitely similar
            if iou_score > 0.5:
                return track.track_id
                
            # Similar appearance + some spatial overlap = similar
            if (emb is not None and track.get_feature() is not None and 
                track.time_since_update <= 5):  # Only recent tracks
                appearance_dist = np.linalg.norm(emb - track.get_feature())
                if appearance_dist < 0.4 and iou_score > 0.1:
                    return track.track_id
                    
        return None

    def merge_only_obvious_duplicates(self):
        """Merge tracks safely - NEVER steal IDs from confirmed tracks"""
        i = 0
        merged_any = False
        while i < len(self.tracks):
            track_i = self.tracks[i]
            if track_i.deleted:
                i += 1
                continue
            
            j = i + 1
            while j < len(self.tracks):
                track_j = self.tracks[j]
                if track_j.deleted:
                    j += 1
                    continue
                    
                iou_score = self.iou(track_i.tlbr, track_j.tlbr)
                
                # ONLY merge if extremely high overlap AND both are active
                if (iou_score > 0.8 and 
                    track_i.time_since_update <= 1 and 
                    track_j.time_since_update <= 1):
                    
                    # === SAFE MERGING RULES - PREVENT ID STEALING ===
                    keep, remove = None, None
                    
                    # Rule 1: Never merge two confirmed tracks
                    if track_i.confirmed and track_j.confirmed:
                        # Two confirmed tracks with high overlap - this shouldn't happen
                        print(f"CONFLICT: Two confirmed tracks {track_i.track_id} and {track_j.track_id} with high IoU")
                        j += 1
                        continue
                    
                    # Rule 2: Always keep confirmed tracks over unconfirmed
                    elif track_i.confirmed and not track_j.confirmed:
                        keep, remove = track_i, track_j
                    elif track_j.confirmed and not track_i.confirmed:
                        keep, remove = track_j, track_i
                    
                    # Rule 3: If both unconfirmed, keep the one with more history
                    elif not track_i.confirmed and not track_j.confirmed:
                        if track_i.hit_streak >= track_j.hit_streak:
                            keep, remove = track_i, track_j
                        else:
                            keep, remove = track_j, track_i
                    
                    # Rule 4: If same confirmation status, keep older ID
                    else:
                        if track_i.track_id < track_j.track_id:
                            keep, remove = track_i, track_j
                        else:
                            keep, remove = track_j, track_i
                    
                    if keep is not None and remove is not None and keep.track_id != remove.track_id:
                        print(f"MERGE: Track {remove.track_id} -> {keep.track_id} (iou={iou_score:.3f}, keep_confirmed={keep.confirmed}, remove_confirmed={remove.confirmed})")
                        
                        # Transfer embeddings to keep the appearance history
                        if remove.embeddings:
                            keep.embeddings.extend(remove.embeddings)
                        
                        remove.deleted = True
                        self.tracks.pop(j)
                        merged_any = True
                        continue
                        
                j += 1
            i += 1
        
        self.tracks = [t for t in self.tracks if not t.deleted]
        return merged_any

    def validate_track_assignments(self):
        """Validate that track assignments make sense"""
        for i in range(len(self.tracks)):
            for j in range(i + 1, len(self.tracks)):
                track_i = self.tracks[i]
                track_j = self.tracks[j]
                
                if (track_i.confirmed and track_j.confirmed and 
                    track_i.time_since_update == 0 and track_j.time_since_update == 0):
                    
                    iou_score = self.iou(track_i.tlbr, track_j.tlbr)
                    if iou_score > 0.5:
                        # Two confirmed tracks shouldn't have high overlap
                        print(f"WARNING: Two confirmed tracks {track_i.track_id} and {track_j.track_id} with IoU {iou_score:.3f}")
                        
                        # Check if this might be an ID swap
                        if (len(track_i.history) > 1 and len(track_j.history) > 1):
                            prev_i = track_i.history[-2]
                            prev_j = track_j.history[-2]
                            curr_i = track_i.tlbr
                            curr_j = track_j.tlbr
                            
                            # If tracks crossed, it might be a swap
                            if (self.iou(curr_i, prev_j) > 0.3 and 
                                self.iou(curr_j, prev_i) > 0.3):
                                print(f"POTENTIAL SWAP DETECTED between {track_i.track_id} and {track_j.track_id}")



class LongTermBoTSORT:
    """
    BoTSORT wrapper with safe persistent memory re-ID.
    - Prevents ID stealing
    - Allows reappearing tracks to regain original ID
    - Combines within-frame merge + memory re-ID
    """
    def __init__(self, bot_sort_tracker, embedding_threshold=0.45,
                 reid_max_age=30, iou_threshold=0.3, max_displacement=50):
        self.tracker = bot_sort_tracker
        self.embedding_threshold = embedding_threshold
        self.reid_max_age = reid_max_age
        self.iou_threshold = iou_threshold
        self.max_displacement = max_displacement
        self.persistent_memory = {}  # track_id -> (embedding, last_box, last_frame)
        self.frame_idx = 0

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            return 1.0
        return 1.0 - np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))

    @staticmethod
    def iou(bb1, bb2):
        xx1 = max(bb1[0], bb2[0])
        yy1 = max(bb1[1], bb2[1])
        xx2 = min(bb1[2], bb2[2])
        yy2 = min(bb1[3], bb2[3])
        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        wh = w * h
        return wh / ((bb1[2]-bb1[0])*(bb1[3]-bb1[1]) + (bb2[2]-bb2[0])*(bb2[3]-bb2[1]) - wh + 1e-16)

    @staticmethod
    def box_distance(bb1, bb2):
        c1 = [(bb1[0]+bb1[2])/2, (bb1[1]+bb1[3])/2]
        c2 = [(bb2[0]+bb2[2])/2, (bb2[1]+bb2[3])/2]
        return np.linalg.norm(np.array(c1) - np.array(c2))
    

    def update(self, detections: np.ndarray, embeddings: list):
        self.frame_idx += 1
        online_targets = self.tracker.update(detections, embeddings)

        # Purge old memory
        to_delete = [tid for tid, (_, _, last_frame) in self.persistent_memory.items()
                    if self.frame_idx - last_frame > self.reid_max_age]
        for tid in to_delete:
            self.persistent_memory.pop(tid, None)

        # Update memory with confirmed tracks
        for track in online_targets:
            if track.confirmed:
                feat = track.get_feature()
                if feat is not None:
                    self.persistent_memory[track.track_id] = (feat, track.tlbr.copy(), self.frame_idx)

        # SAFE memory re-ID with stricter rules
        unconfirmed_tracks = [t for t in online_targets if not t.confirmed]
        available_memory_ids = list(self.persistent_memory.keys())
        
        # Remove memory IDs that are currently active and confirmed
        active_confirmed_ids = {t.track_id for t in online_targets if t.confirmed}
        available_memory_ids = [mid for mid in available_memory_ids if mid not in active_confirmed_ids]
        
        # Create cost matrix for matching
        cost_matrix = np.ones((len(unconfirmed_tracks), len(available_memory_ids))) * 1000
        
        for i, track in enumerate(unconfirmed_tracks):
            track_feat = track.get_feature()
            if track_feat is None:
                continue
                
            for j, mem_id in enumerate(available_memory_ids):
                mem_emb, mem_box, last_frame = self.persistent_memory[mem_id]
                
                # Stricter temporal constraint
                if self.frame_idx - last_frame < 5:  # Very recent, be careful
                    continue
                    
                # Multiple similarity measures
                emb_dist = self.cosine_distance(track_feat, mem_emb)
                iou_score = self.iou(track.tlbr, mem_box)
                spatial_dist = self.box_distance(track.tlbr, mem_box)
                print(emb_dist)
                # Combined cost with weights
                if (emb_dist < self.embedding_threshold and 
                    iou_score > 0.1 and 
                    spatial_dist < self.max_displacement):
                    
                    cost = (emb_dist * 0.6 + 
                        (1 - iou_score) * 0.3 + 
                        min(spatial_dist / self.max_displacement, 1.0) * 0.1)
                    cost_matrix[i, j] = cost

        # Hungarian matching for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.8:  # Stricter overall threshold
                track = unconfirmed_tracks[i]
                mem_id = available_memory_ids[j]
                
                # Final safety check
                conflict = any(t.track_id == mem_id for t in online_targets if t.confirmed)
                if not conflict:
                    track.track_id = mem_id
                    track.confirmed = True
                    # Update memory with current state
                    self.persistent_memory[mem_id] = (track.get_feature(), track.tlbr.copy(), self.frame_idx)

        return online_targets