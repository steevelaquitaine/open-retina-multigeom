"""Train the conditional diffusion model ``eps_theta(x_t, t, r)``.

Standard DDPM epsilon-MSE objective on natural stimulus frames, conditioned on
population responses generated on-the-fly by the (frozen) retina encoder
``f_phi``. Two recipe details from the paper are essential for a stable G(x)
estimator:

* observation noise injected into the condition each step (``r_tilde = r + sigma * xi``);
* classifier-free-guidance dropout (zero the condition with prob ``cfg_dropout``)
  so the null-token pass used by the estimator is meaningful.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


def _cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_conditional_diffusion(
    unet,
    encoder,
    dataset: Dataset,
    scheduler,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 500,
    sigma: float = 0.22,
    cfg_dropout: float = 0.1,
    grad_clip: float = 1.0,
    device: torch.device | str = "cuda",
    num_workers: int = 4,
    encoder_batch_size: int | None = None,
    log_every: int = 10,
    noise_model: str = "gaussian",
    poisson_gain: float = 1.0,
) -> dict:
    """Train ``unet`` in place; return a ``{"epoch": [...], "loss": [...]}`` history.

    Responses ``r = f_phi(x)`` are computed on the fly (the encoder is frozen).
    ``encoder_batch_size`` optionally chunks the encoder pass if it is heavier
    than the diffusion batch.
    """
    device = torch.device(device)
    unet = unet.to(device).train()
    encoder = encoder.to(device).eval()
    num_timesteps = scheduler.config.num_train_timesteps

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    history: dict[str, list] = {"epoch": [], "loss": []}

    # Poisson works on the rate scale, so responses must stay RAW until counts are drawn.
    poisson = noise_model == "poisson"

    @torch.no_grad()
    def encode(x: torch.Tensor) -> torch.Tensor:
        bs = encoder_batch_size or x.shape[0]
        fn = encoder.forward_raw if poisson else encoder
        return torch.cat([fn(x[i : i + bs]) for i in range(0, x.shape[0], bs)], dim=0)

    step = 0
    for epoch in range(1, epochs + 1):
        running, n = 0.0, 0
        for batch in loader:
            # dataset may yield either x, or a (x, r) pair with precomputed responses
            if isinstance(batch, (list, tuple)):
                x, r = batch
                x = x.to(device, non_blocking=True)
                r = r.to(device, non_blocking=True)
            else:
                x = batch.to(device, non_blocking=True)  # [B, C, Hp, Wp] diffusion space
                r = encode(x)  # [B, N]

            t = torch.randint(0, num_timesteps, (x.shape[0],), device=device)
            z = torch.randn_like(x)
            x_t = scheduler.add_noise(x, z, t)

            if poisson:
                # r holds the RAW rate lambda. Observe spike counts over the response window
                # (n ~ Poisson(gain * lambda), gain = window frames), convert back to the rate
                # scale, then standardise so the UNet condition keeps its designed scale.
                counts = torch.poisson((poisson_gain * r).clamp(min=0.0))
                r_tilde = encoder.standardize(counts / poisson_gain)
            else:
                r_tilde = r + sigma * torch.randn_like(r)
            if cfg_dropout > 0:
                drop = (torch.rand(x.shape[0], device=device) < cfg_dropout).view(-1, 1)
                r_tilde = torch.where(drop, torch.zeros_like(r_tilde), r_tilde)

            eps_pred = unet(x_t, t, r_tilde)
            loss = F.mse_loss(eps_pred, z)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            for g in optimizer.param_groups:
                g["lr"] = lr * _cosine_with_warmup(step, warmup_steps, total_steps)
            optimizer.step()

            running += loss.item()
            n += 1
            step += 1

        epoch_loss = running / max(1, n)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            LOGGER.info("epoch %d/%d  loss=%.4f", epoch, epochs, epoch_loss)

    return history
