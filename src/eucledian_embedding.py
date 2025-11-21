import torch
import torch.nn as nn

class EuclideanCodeEmbedding(nn.Module):
    def __init__(self, num_codes: int, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_codes, dim)

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(code_ids)


class EuclideanVisitEncoder(nn.Module):
    def __init__(self, code_embedding: EuclideanCodeEmbedding):
        super().__init__()
        self.code_embedding = code_embedding

    def forward(self, code_ids_batch):
        """
        code_ids_batch: list of 1D LongTensors (variable length), len = B*T
        Returns: tensor [B*T, dim] in Euclidean space.
        """
        visit_vecs = []
        for ids in code_ids_batch:
            x = self.code_embedding(ids)      # [k, d]
            visit_vec = x.mean(dim=0)         # [d]
            visit_vecs.append(visit_vec)
        return torch.stack(visit_vecs, dim=0)
