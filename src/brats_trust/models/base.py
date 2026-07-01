"""Shared modeling primitives and constants."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

IN_CHANNELS = 4
OUT_CHANNELS = 3


def _align_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad/crop x's spatial dims to match ref's, so skip-concat works for any input size.

    Pooling floor-divides odd dimensions while transposed conv doubles them, so an
    arbitrary-sized (e.g. brain-cropped) volume can produce off-by-one skip mismatches.
    """
    diffs = [r - s for s, r in zip(x.shape[2:], ref.shape[2:])]
    if any(d != 0 for d in diffs):
        pad: list[int] = []
        for d in reversed(diffs):
            pad += [0, max(d, 0)]
        x = F.pad(x, pad)
        x = x[:, :, : ref.shape[2], : ref.shape[3], : ref.shape[4]]
    return x


def _norm_act(channels: int) -> nn.Sequential:
    # InstanceNorm is the standard choice for small-batch 3D medical segmentation.
    return nn.Sequential(nn.InstanceNorm3d(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ConvBlock(nn.Module):
    """Standard double 3D convolution (the default scaffold block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            _norm_act(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            _norm_act(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable double conv: changes receptive field while keeping params/FLOPs
    comparable to :class:`ConvBlock` (roadmap S4 Probe 3 control)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2

        def ds(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(cin, cin, kernel_size, padding=pad, groups=cin, bias=False),  # depthwise
                nn.Conv3d(cin, cout, 1, bias=False),                                    # pointwise
                _norm_act(cout),
            )

        self.net = nn.Sequential(ds(in_ch, out_ch), ds(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DepthwiseSeparableBlock}


def estimate_flops(model: nn.Module, input_size: tuple[int, ...]) -> float:
    """Estimate FLOPs (Multiply-Accumulates * 2) for a forward pass.

    Uses temporary forward hooks on Conv3D, ConvTranspose3D, and Linear modules.
    Fails gracefully returning 0.0 if the dummy pass fails.
    """
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        out_shape = output.shape
        batch_size = out_shape[0]
        out_features = out_shape[1]
        out_dims = out_shape[2:]
        
        kernel_size = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        
        # Multiply-accumulates (MACs) per output voxel
        macs_per_voxel = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2] // groups
        # FLOPs = 2 * MACs (mul + add)
        spatial_vol = 1
        for d in out_dims:
            spatial_vol *= d
        layer_flops = batch_size * out_features * spatial_vol * macs_per_voxel * 2
        flops += int(layer_flops)

    def linear_hook(module, input, output):
        nonlocal flops
        out_features = module.out_features
        in_features = module.in_features
        batch_elements = output.numel() // out_features
        layer_flops = batch_elements * in_features * out_features * 2
        flops += int(layer_flops)

    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy = torch.zeros(*input_size, device=device)
    
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy)
    except Exception:
        flops = 0
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return float(flops)
