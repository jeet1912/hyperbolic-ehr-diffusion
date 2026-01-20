import pickle
from typing import Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class MimicDataset(Dataset):
    """
    Lightweight wrapper around the mimic_hf_cohort.pkl file produced by data/create.py.
    Each sample is a list of visits, and each visit is a list of code indices.
    """

    def __init__(self, pkl_path: str):
        print(f"[MIMIC] Loading {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.x = data["x"]
        self.y = data["y"]
        self.code_map = data["code_map"]
        self.vocab_size = data["vocab_size"]
        self.subject_id = data["subject_id"]

        print(f"[MIMIC] Patients: {len(self.x)} | Vocab size: {self.vocab_size}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[List[List[int]], int]:
        return self.x[idx], self.y[idx]


def make_pad_collate(vocab_size: int) -> Callable:
    """
    Returns a collate function that converts jagged patient sequences into
    multi-hot padded tensors.
    """

    def pad_collate(batch: Sequence[Tuple[List[List[int]], int]]):
        batch_x, batch_y = zip(*batch)

        max_visits = max(len(p) for p in batch_x)
        max_codes = max((len(v) for p in batch_x for v in p), default=0)
        if max_codes == 0:
            max_codes = 1

        padded_x = torch.zeros(len(batch_x), max_visits, vocab_size, dtype=torch.float32)
        mask = torch.zeros(len(batch_x), max_visits, dtype=torch.float32)

        for i, patient in enumerate(batch_x):
            for j, visit in enumerate(patient):
                mask[i, j] = 1.0
                for code in visit:
                    if 0 <= code < vocab_size:
                        padded_x[i, j, code] = 1.0

        tensor_y = torch.tensor(batch_y, dtype=torch.float32)
        return padded_x, tensor_y, mask

    return pad_collate