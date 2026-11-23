"""Conditional-pair estimator of the multi-scale Fisher metric G(x).

Algorithm 1 of Azeglio et al. (2026), generalised from single-channel 64x64 to
arbitrary ``C``-channel ``(Hp, Wp)`` stimuli. For a clean image ``x`` (diffusion
space, padded), an encoder ``f_phi`` and a conditional denoiser
``eps_theta(x_t, t, r)``, it returns the top-``k`` eigenpairs of

    G(x) = integral_0^inf [phi_tau * J_tau](x) dtau

by Monte-Carlo integration along the log-SNR axis. After Tweedie's identity and
the ``dtau = tau d(log tau)`` change of variable, the integrand reduces to
squared conditional noise-prediction differences::

    D_{(j,s),:} = sqrt(w_j / (2 N_MC)) * ( eps_theta(x_t, t, r_1) - eps_theta(x_t, t, r_2) )

with ``r_1, r_2`` i.i.d. from ``p(r | x_hat(x_t)) = N(f_phi(x_hat), sigma^2 I)``.
The top-``k`` eigenpairs of ``G(x) ~= D^T D`` come from a thin SVD of ``D``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class EstimatorOutput:
    eigenvalues: torch.Tensor  # [k] descending
    eigenvectors: torch.Tensor  # [k, P] rows are eigenvectors of G(x)


def _log_tau_grid(timesteps: Sequence[int], alphas_cumprod: torch.Tensor) -> torch.Tensor:
    t = torch.tensor(list(timesteps), dtype=torch.long, device=alphas_cumprod.device)
    a_bar = alphas_cumprod[t]
    return torch.log((1.0 - a_bar) / a_bar)


def _trapezoid_weights(log_tau: torch.Tensor) -> torch.Tensor:
    w = torch.zeros_like(log_tau)
    w[0] = 0.5 * (log_tau[1] - log_tau[0])
    w[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    if log_tau.shape[0] > 2:
        w[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
    return w


@torch.no_grad()
def conditional_pair_estimate(
    image: torch.Tensor,
    encoder,
    unet,
    alphas_cumprod: torch.Tensor,
    timesteps: Sequence[int],
    n_samples: int = 200,
    batch_size: int = 25,
    sigma: float = 0.22,
    top_k: int = 50,
    device: torch.device | None = None,
    noise_model: str = "gaussian",
    poisson_gain: float = 1.0,
) -> EstimatorOutput:
    """Top-``top_k`` eigenpairs of G(x) for a single image via Algorithm 1.

    Parameters
    ----------
    image:
        ``[1, C, Hp, Wp]`` clean stimulus in diffusion space (padded canvas).
    encoder:
        ``f_phi``: accepts ``[B, C, Hp, Wp]`` and returns ``[B, N]``.
    unet:
        trained ``NeuralConditionedUNet`` (``eps_theta``).
    alphas_cumprod:
        scheduler ``alphas_cumprod`` ``[T]`` on ``device``.
    timesteps:
        integer DDPM steps ``{t_j}`` to integrate over.
    """
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W], got {tuple(image.shape)}")
    device = device or image.device
    image = image.to(device)
    encoder.eval()
    unet.eval()

    _, C, H, W = image.shape
    P = C * H * W
    log_tau = _log_tau_grid(timesteps, alphas_cumprod)
    w = _trapezoid_weights(log_tau)
    timesteps = list(timesteps)

    if n_samples % batch_size != 0:
        raise ValueError(f"n_samples={n_samples} must be divisible by batch_size={batch_size}")
    n_batches = n_samples // batch_size
    n_neurons = encoder.n_neurons

    total_rows = len(timesteps) * n_samples
    D = torch.empty(total_rows, P, dtype=torch.float32)
    row = 0
    null_token = torch.zeros(batch_size, n_neurons, device=device)

    for j, t_j in enumerate(timesteps):
        s_j = float((w[j] / (2.0 * n_samples)).sqrt())
        sqrt_ab = float(alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())
        t_tensor = torch.full((batch_size,), t_j, device=device, dtype=torch.long)

        for _ in range(n_batches):
            z = torch.randn(batch_size, C, H, W, device=device)
            x_t = sqrt_ab * image + sqrt_1m * z

            eps_u = unet(x_t, t_tensor, null_token)
            x_hat = ((x_t - sqrt_1m * eps_u) / sqrt_ab).clamp(-1.0, 1.0)

            if noise_model == "poisson":
                # Two independent spike-count observations of the same rate lambda(x_hat),
                # drawn on the rate scale then standardised for the UNet condition.
                lam = (poisson_gain * encoder.forward_raw(x_hat)).clamp(min=0.0)
                r1 = encoder.standardize(torch.poisson(lam) / poisson_gain)
                r2 = encoder.standardize(torch.poisson(lam) / poisson_gain)
            else:
                mu = encoder(x_hat)  # [B, N]
                r1 = mu + sigma * torch.randn_like(mu)
                r2 = mu + sigma * torch.randn_like(mu)

            delta = (unet(x_t, t_tensor, r1) - unet(x_t, t_tensor, r2)).reshape(batch_size, P)
            D[row : row + batch_size] = (s_j * delta).cpu()
            row += batch_size

    assert row == total_rows
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    k = int(top_k)
    return EstimatorOutput(eigenvalues=(S**2)[:k].clone(), eigenvectors=Vh[:k].clone())
