"""Conditional diffusion model for the multi-scale Fisher estimator.

A ``UNet2DConditionModel`` (from ``diffusers``) operating on ``C``-channel,
``(Hp, Wp)`` images, conditioned on an ``N``-dimensional population response
through a small SiLU MLP that maps ``r`` to a single cross-attention token.

Generalised from the paper's single-channel 64x64 setup to arbitrary channel
count and (padded) spatial size, so open-retina stimuli (e.g. 2-channel 18x16
padded to 20x16) fit directly. Classifier-free-guidance dropout on ``r`` is
applied in the training loop, not here, so inference / G(x) estimation stays
deterministic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel


class ConditioningMLP(nn.Module):
    """``[B, N] -> [B, 1, embedding_dim]`` single cross-attention token."""

    def __init__(self, n_neurons: int, embedding_dim: int, n_layers: int = 3):
        super().__init__()
        dims = [n_neurons] + [embedding_dim] * (n_layers - 1)
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.SiLU())
        modules.append(nn.Linear(embedding_dim, embedding_dim))
        self.mlp = nn.Sequential(*modules)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.mlp(r).unsqueeze(1)


class NeuralConditionedUNet(nn.Module):
    """Conditional UNet ``eps_theta(x_t, t, r)`` for open-retina stimuli.

    Parameters
    ----------
    in_channels:
        stimulus channel count ``C``.
    sample_size:
        ``(Hp, Wp)`` padded spatial size (each divisible by ``2**(len(block_channels)-1)``).
    n_neurons:
        conditioning vector dimension ``N``.
    block_channels:
        UNet block widths; a single cross-attention block sits at the deepest scale.
    """

    def __init__(
        self,
        in_channels: int,
        sample_size: tuple[int, int],
        n_neurons: int,
        block_channels: tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        cross_attention_dim: int = 256,
        embedding_dim: int = 256,
        observation_noise_sigma: float = 0.22,
        cfg_dropout_rate: float = 0.1,
    ):
        super().__init__()
        n_blocks = len(block_channels)
        down_blocks = ["DownBlock2D"] * (n_blocks - 1) + ["CrossAttnDownBlock2D"]
        up_blocks = ["CrossAttnUpBlock2D"] + ["UpBlock2D"] * (n_blocks - 1)

        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=tuple(block_channels),
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
            cross_attention_dim=cross_attention_dim,
        )
        self.conditioning = ConditioningMLP(n_neurons, embedding_dim)
        self.n_neurons = int(n_neurons)
        self.in_channels = int(in_channels)
        self.observation_noise_sigma = float(observation_noise_sigma)
        self.cfg_dropout_rate = float(cfg_dropout_rate)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """``x_t``: ``[B, C, Hp, Wp]``, ``t``: ``[B]``, ``r``: ``[B, N]`` -> ``[B, C, Hp, Wp]``."""
        return self.unet(x_t, t, encoder_hidden_states=self.conditioning(r)).sample


def build_scheduler(
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DDPMScheduler:
    """DDPM scheduler with the paper's linear beta schedule (epsilon prediction)."""
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type="epsilon",
    )
