import torch
import torch.nn as nn

class EuclideanCodeEmbedding(nn.Module):
    def __init__(self, num_codes: int, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_codes, dim)

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(code_ids)


class EuclideanVisitEncoder(nn.Module):
    def __init__(self, code_embedding: EuclideanCodeEmbedding, pad_idx: int):
        super().__init__()
        self.code_embedding = code_embedding
        self.pad_idx = pad_idx

    def forward(self, code_ids_batch):
        visit_vecs = []
        d = self.code_embedding.emb.embedding_dim
        device = self.code_embedding.emb.weight.device

        for ids in code_ids_batch:
            ids = ids[ids != self.pad_idx]
            if ids.numel() == 0:
                visit_vecs.append(torch.zeros(d, device=device))
                continue

            x = self.code_embedding(ids)    # [k, d]
            visit_vec = x.mean(dim=0)       # [d]
            visit_vecs.append(visit_vec)

        return torch.stack(visit_vecs, dim=0)
