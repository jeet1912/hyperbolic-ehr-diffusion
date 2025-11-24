"""Utility snippets to sanity-check hyperbolic components with a single forward pass.

Run this file directly with ``python -m src.verify_hyperbolic_components`` to see
shapes and a couple of tensor slices printed to stdout.
"""

import torch

try:  # allow running via "python src/..." or "python -m src...."
    from hyperbolic_embeddings import HyperbolicCodeEmbedding, HyperbolicVisitEncoder
    from decoders import HyperbolicDistanceDecoder
except ModuleNotFoundError:  # pragma: no cover - fallback when src is package
    from src.hyperbolic_embeddings import HyperbolicCodeEmbedding, HyperbolicVisitEncoder
    from src.decoders import HyperbolicDistanceDecoder


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_dummy_visits(num_visits: int, max_len: int, num_real_codes: int, pad_idx: int, device):
    rng = torch.Generator().manual_seed(1234)
    visits = []
    for _ in range(num_visits - 1):
        length = int(torch.randint(1, max_len + 1, (1,), generator=rng))
        codes = torch.randint(0, num_real_codes, (length,), generator=rng)
        visits.append(codes.to(device))
    # include a padded-only visit to exercise the empty-path logic
    visits.append(torch.tensor([pad_idx], dtype=torch.long, device=device))
    return visits


def verify_visit_encoder(device: torch.device):
    print("[Check] HyperbolicVisitEncoder")
    num_real_codes = 64
    pad_idx = num_real_codes
    dim = 16

    code_emb = HyperbolicCodeEmbedding(num_codes=num_real_codes + 1, dim=dim, c=1.0).to(device)
    visit_enc = HyperbolicVisitEncoder(code_emb, pad_idx=pad_idx).to(device)

    flat_visits = _build_dummy_visits(num_visits=5, max_len=4, num_real_codes=num_real_codes, pad_idx=pad_idx, device=device)
    latents = visit_enc(flat_visits)

    print(f"  Latent shape: {latents.shape}")
    print(f"  Example latent[0][:5]: {latents[0][:5].tolist()}")
    return latents, code_emb


def verify_distance_decoder(device: torch.device, latents: torch.Tensor, code_emb: HyperbolicCodeEmbedding):
    print("[Check] HyperbolicDistanceDecoder")
    num_real_codes = code_emb.emb.size(0) - 1
    freq = torch.ones(num_real_codes, device=device)
    decoder = HyperbolicDistanceDecoder(
        code_embedding=code_emb.emb,
        manifold=code_emb.manifold,
        code_freq=freq,
    ).to(device)

    logits = decoder(latents)
    print(f"  Logit shape: {logits.shape}")
    print(f"  Logits[0][:5]: {logits[0][:5].tolist()}")


def main():
    torch.manual_seed(0)
    device = _detect_device()
    print(f"Using device: {device}")
    latents, code_emb = verify_visit_encoder(device)
    verify_distance_decoder(device, latents, code_emb)


if __name__ == "__main__":
    main()
