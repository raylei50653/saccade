import torch
import torch.nn.functional as F


class OutputAppearanceBank:
    def __init__(
        self,
        *,
        max_samples: int,
        min_score: float,
        min_consistency: float,
    ) -> None:
        self.max_samples = max(1, max_samples)
        self.min_score = min_score
        self.min_consistency = min_consistency
        self.samples: dict[int, list[tuple[float, int, torch.Tensor]]] = {}

    def update(
        self,
        track_id: int,
        embedding: torch.Tensor | None,
        *,
        score: float,
        frame_id: int,
    ) -> None:
        if embedding is None or score < self.min_score:
            return
        emb = F.normalize(embedding.detach().float(), dim=0)
        samples = self.samples.setdefault(track_id, [])
        samples.append((score, frame_id, emb))
        samples.sort(key=lambda item: (item[0], item[1]), reverse=True)
        del samples[self.max_samples :]

    def update_many(
        self,
        updates: list[tuple[int, torch.Tensor | None, float, int]],
    ) -> None:
        touched_track_ids: set[int] = set()
        for track_id, embedding, score, frame_id in updates:
            if embedding is None or score < self.min_score:
                continue
            emb = F.normalize(embedding.detach().float(), dim=0)
            samples = self.samples.setdefault(track_id, [])
            samples.append((score, frame_id, emb))
            touched_track_ids.add(track_id)

        for track_id in touched_track_ids:
            samples = self.samples.get(track_id, [])
            samples.sort(key=lambda item: (item[0], item[1]), reverse=True)
            del samples[self.max_samples :]

    def count(self, track_id: int) -> int:
        return len(self.samples.get(track_id, []))

    def consistency(self, track_id: int) -> float:
        samples = self.samples.get(track_id, [])
        if len(samples) < 2:
            return 1.0
        stacked = torch.stack([sample[2] for sample in samples])
        cosines = stacked @ stacked.T
        n = len(samples)
        return float((cosines.sum() - n) / max(n * (n - 1), 1))

    def representative(self, track_id: int) -> torch.Tensor | None:
        samples = self.samples.get(track_id, [])
        if not samples:
            return None
        return F.normalize(
            torch.stack([sample[2] for sample in samples]).mean(dim=0), dim=0
        )

    def similarity(self, a: int, b: int) -> float | None:
        a_emb = self.representative(a)
        b_emb = self.representative(b)
        if a_emb is None or b_emb is None:
            return None
        return float(torch.dot(a_emb, b_emb).item())
