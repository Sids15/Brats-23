"""Effective receptive field (ERF) measurement (roadmap S4.1).

The ERF is how much of the input *actually* influences a given output voxel -- usually far
smaller than the theoretical receptive field (Luo et al., 2016). We measure it by
back-propagating a single central output voxel to the input and looking at how spatially
spread-out the input gradient is. This is the x-axis of the money figure (ERF vs
faithfulness, S4.1).

Returns an *effective radius* in voxels: the gradient-magnitude-weighted RMS distance from
the center. Larger radius = the model integrates context from farther away.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


@torch.enable_grad()
def measure_erf(
    model: nn.Module,
    input_size: tuple[int, int, int],
    in_channels: int = 4,
    out_channel: int = 0,
    n_samples: int = 4,
    device: torch.device | None = None,
    seed: int = 0,
) -> float:
    """Gradient-based effective receptive radius (voxels), averaged over random inputs.

    Args:
        model: the network (set to eval; grads still flow to the input).
        input_size: spatial size of the probe volume, e.g. ``(128, 128, 128)``.
        out_channel: which output region channel to probe (0=WT, 1=TC, 2=ET).
        n_samples: number of random input volumes to average over (stability).
    """
    device = device or next(model.parameters()).device
    model.eval()

    center = tuple(s // 2 for s in input_size)
    # Precompute squared distance of every voxel from the center.
    grids = np.meshgrid(*[np.arange(s) for s in input_size], indexing="ij")
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    dist2_t = torch.tensor(dist2, dtype=torch.float32, device=device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    radii: list[float] = []
    for _ in range(n_samples):
        x = torch.randn(1, in_channels, *input_size, generator=gen).to(device).requires_grad_(True)
        out = model(x)
        out[(0, out_channel, *center)].backward()
        if x.grad is None:
            continue
        weight = x.grad.detach().abs().sum(dim=1)[0]  # (X, Y, Z): sensitivity per input voxel
        total = weight.sum()
        if total <= 0:
            continue
        radii.append(float(torch.sqrt((weight * dist2_t).sum() / total)))
    return float(np.mean(radii)) if radii else float("nan")
