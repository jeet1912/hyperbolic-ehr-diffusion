from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TrajDataset(Dataset):
    """Dataset of trajectories of visits (lists of ICD code indices)."""

    def __init__(self, trajs: Sequence[Sequence[Sequence[int]]], max_len: int, pad_idx: int):
        self.trajs = trajs
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.trajs)

    def __getitem__(self, idx: int) -> List[List[int]]:  # type: ignore[override]
        traj = list(self.trajs[idx])
        if len(traj) >= self.max_len:
            return [list(v) for v in traj[: self.max_len]]
        pad_visits = [[self.pad_idx]] * (self.max_len - len(traj))
        return [list(v) for v in traj] + pad_visits


def make_collate_fn(pad_idx: int) -> Callable:
    """Return a collate function that flattens visits and tracks masks."""

    def collate_fn(batch: Sequence[Sequence[Sequence[int]]]):
        B = len(batch)
        L = len(batch[0])

        flat_visits: List[torch.Tensor] = []
        visit_mask: List[List[bool]] = []

        for traj in batch:
            row_mask: List[bool] = []
            for visit_codes in traj:
                v = torch.tensor(visit_codes, dtype=torch.long)
                flat_visits.append(v)

                if len(visit_codes) == 1 and visit_codes[0] == pad_idx:
                    row_mask.append(False)
                else:
                    row_mask.append(True)

            visit_mask.append(row_mask)

        visit_mask_tensor = torch.tensor(visit_mask, dtype=torch.bool)
        return flat_visits, B, L, visit_mask_tensor

    return collate_fn


def build_visit_tensor(
    visit_enc, flat_visits: Sequence[torch.Tensor], B: int, L: int, dim: int, device: torch.device
) -> torch.Tensor:
    """Encode visits into a [B, L, dim] tensor using the provided encoder."""
    visit_enc.eval()
    with torch.no_grad():
        visit_vecs = visit_enc(flat_visits).to(device)  # [B*L, dim]
    return visit_vecs.view(B, L, dim)
