import torch
import torch.nn.functional as F
import numpy as np


class SemanticRelinker:
    def __init__(
        self,
        sim_threshold=0.985,
        ttl=45,
        ema_beta=0.83,
        spatial_gate=0.11,
        min_lost_frames=2,
        min_iou=0.0,
        mahalanobis_threshold=6.6,
        debug=False,
    ):
        self.sim_threshold = sim_threshold
        self.ttl = ttl
        self.ema_beta = ema_beta
        self.spatial_gate = spatial_gate
        self.min_lost_frames = min_lost_frames
        self.min_iou = min_iou
        self.mahalanobis_threshold = mahalanobis_threshold
        self.debug = debug
        self.alias = {}
        self.features = {}
        self.last_seen = {}
        self.last_boxes = {}
        self.motion = {}
        self.stats = {
            "attempts": 0,
            "accepted": 0,
            "reject_age": 0,
            "reject_assigned": 0,
            "reject_spatial": 0,
            "reject_mahalanobis": 0,
            "reject_similarity": 0,
            "new_ids": 0,
        }
        self.accept_sims = []
        self.accept_ious = []
        self.accept_center_dists = []
        self.accept_mahas = []

    def _normalize(self, embedding):
        return F.normalize(embedding.float(), dim=0)

    def _spatial_metrics(self, box, old_box, w, h):
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        ocx = (old_box[0] + old_box[2]) * 0.5
        ocy = (old_box[1] + old_box[3]) * 0.5
        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
        center_norm = dist / max(w, h)

        ix1, iy1 = max(box[0], old_box[0]), max(box[1], old_box[1])
        ix2, iy2 = min(box[2], old_box[2]), min(box[3], old_box[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        old_area = max(0.0, old_box[2] - old_box[0]) * max(0.0, old_box[3] - old_box[1])
        iou = inter / (area + old_area - inter + 1e-6)
        return center_norm, iou

    def update_motion_snapshots(self, snapshots):
        for snap in snapshots:
            canonical = self.alias.get(snap.obj_id, snap.obj_id)
            self.motion[canonical] = snap

    def _measurement(self, box):
        w = max(1e-6, box[2] - box[0])
        h = max(1e-6, box[3] - box[1])
        return np.array(
            [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, w / h, h],
            dtype=np.float32,
        )

    def _mahalanobis(self, box, snapshot):
        state = np.asarray(snapshot.state[:4], dtype=np.float32)
        cov = np.asarray(snapshot.covariance, dtype=np.float32).reshape(8, 8)[:4, :4]
        h = max(float(state[3]), 1e-6)
        pos_std = h / 20.0
        r = np.diag([pos_std**2, pos_std**2, 1e-2, pos_std**2]).astype(np.float32)
        s = cov + r
        residual = self._measurement(box) - state
        try:
            solved = np.linalg.solve(s, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(s) @ residual
        return float(residual @ solved)

    def resolve(self, raw_id, embedding, box, score, frame_id, w, h, assigned):
        if embedding is None:
            return self.alias.get(raw_id, raw_id)

        emb = self._normalize(embedding)
        if raw_id not in self.alias:
            self.stats["attempts"] += 1
            best_id, best_sim = None, self.sim_threshold
            best_iou, best_center, best_maha = 0.0, 0.0, 0.0
            for cid, old_emb in self.features.items():
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if cid in assigned:
                    self.stats["reject_assigned"] += 1
                    continue
                if age < self.min_lost_frames or age > self.ttl:
                    self.stats["reject_age"] += 1
                    continue
                center_norm, iou = self._spatial_metrics(box, self.last_boxes[cid], w, h)
                if center_norm > self.spatial_gate or iou < self.min_iou:
                    self.stats["reject_spatial"] += 1
                    continue
                maha = 0.0
                if self.mahalanobis_threshold > 0.0:
                    snapshot = self.motion.get(cid)
                    if snapshot is None:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                    maha = self._mahalanobis(box, snapshot)
                    if maha > self.mahalanobis_threshold:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                sim = torch.dot(emb, old_emb).item()
                if sim > best_sim:
                    best_id, best_sim = cid, sim
                    best_iou, best_center, best_maha = iou, center_norm, maha
                else:
                    self.stats["reject_similarity"] += 1
            if best_id is not None:
                self.stats["accepted"] += 1
                self.accept_sims.append(best_sim)
                self.accept_ious.append(best_iou)
                self.accept_center_dists.append(best_center)
                self.accept_mahas.append(best_maha)
                self.alias[raw_id] = best_id
            else:
                self.stats["new_ids"] += 1
                self.alias[raw_id] = raw_id

        canonical = self.alias[raw_id]
        old = self.features.get(canonical)
        updated = emb if old is None else F.normalize(self.ema_beta * old + (1.0 - self.ema_beta) * emb, dim=0)
        self.features[canonical] = updated.detach()
        self.last_seen[canonical] = frame_id
        self.last_boxes[canonical] = box
        assigned.add(canonical)
        return canonical

    def report(self):
        print("🔁 Semantic Relink Report:")
        print(
            "  attempts={attempts} accepted={accepted} new_ids={new_ids} "
            "reject_age={reject_age} reject_assigned={reject_assigned} "
            "reject_spatial={reject_spatial} reject_mahalanobis={reject_mahalanobis} "
            "reject_similarity={reject_similarity}".format(**self.stats)
        )
        if self.accept_sims:
            print(
                f"  accepted mean_sim={np.mean(self.accept_sims):.3f} "
                f"mean_iou={np.mean(self.accept_ious):.3f} "
                f"mean_center_norm={np.mean(self.accept_center_dists):.3f} "
                f"mean_maha={np.mean(self.accept_mahas):.3f}"
            )
