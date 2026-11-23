"""Factorized readout with a channel-wise batchnorm between the spatial mask and the feature mixing.

This mirrors the legacy adaptation readout (shared spatial convolution -> TemporalBatchNorm over the
output channels -> linear channel mixing -> softplus). Keeping the batchnorm as a real layer (rather
than folding it into the feature weights) means it stays live: its running statistics can be
re-estimated when the upstream activations shift, e.g. when gain control is switched on.
"""

from typing import Any, Iterable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from openretina.modules.readout.factorized import FactorizedReadout
from openretina.modules.readout.multi_readout import MultiReadoutBase


class NormalizedFactorizedReadout(FactorizedReadout):
    """FactorizedReadout with a channel-wise BatchNorm applied to the masked activations."""

    def __init__(self, *args: Any, batchnorm_affine: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        channels = self.in_shape[0]
        self.channel_norm = nn.BatchNorm1d(channels, affine=batchnorm_affine)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.apply_constraints()
        batch, channels, time, height, width = x.shape
        x_flat = x.view(batch, channels, time, -1).permute(0, 2, 1, 3)  # (B, T, C, HW)
        masked = torch.matmul(x_flat, self.mask_weights)  # (B, T, C, N)
        masked = masked.permute(0, 1, 3, 2)  # (B, T, N, C)

        # BatchNorm over the channel dimension, normalizing per channel across (batch, time, neurons).
        normed = self.channel_norm(masked.reshape(-1, channels)).reshape(masked.shape)

        out = (normed * self.feature_weights.T.unsqueeze(0).unsqueeze(0)).sum(dim=3)  # (B, T, N)
        if self.readout_bias:
            out = out + self.bias
        return F.softplus(out)


class MultiNormalizedFactorizedReadout(MultiReadoutBase):
    """Multi-session version of :class:`NormalizedFactorizedReadout`."""

    _base_readout_cls = NormalizedFactorizedReadout

    def __init__(
        self,
        in_shape: tuple[int, int, int, int],
        n_neurons_dict: dict[str, int],
        mask_l1_reg: float,
        weights_l1_reg: float,
        laplace_mask_reg: float,
        readout_bias: bool = False,
        batchnorm_affine: bool = False,
        weights_constraint: Literal["abs", "norm", "absnorm"] | None = None,
        mask_constraint: Literal["abs"] | None = None,
        init_mask: Optional[torch.Tensor] = None,
        init_weights: Optional[torch.Tensor] = None,
        init_scales: Optional[Iterable[Iterable[float]]] = None,
        readout_reg_avg: bool = False,
        mean_activity_dict: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__(
            in_shape=in_shape,
            n_neurons_dict=n_neurons_dict,
            mask_size=in_shape[2:],
            mask_l1_reg=mask_l1_reg,
            weights_l1_reg=weights_l1_reg,
            laplace_mask_reg=laplace_mask_reg,
            readout_bias=readout_bias,
            batchnorm_affine=batchnorm_affine,
            weights_constraint=weights_constraint,
            mask_constraint=mask_constraint,
            init_mask=init_mask,
            init_weights=init_weights,
            init_scales=init_scales,
            readout_reg_avg=readout_reg_avg,
            mean_activity_dict=mean_activity_dict,
        )
