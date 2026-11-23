"""Closed-form Fisher information baseline J(x).

For the Gaussian observation model ``p(r | x) = N(f_phi(x), sigma^2 I)``,

    J(x) = (1 / sigma^2) * J_{f_phi}(x)^T J_{f_phi}(x),

where ``J_{f_phi}(x) in R^{N x P}`` is the encoder Jacobian at ``x``. Since
``J_{f_phi}`` has rank at most ``N << P``, the top eigenpairs of ``J(x)`` come
directly from the SVD of ``J_{f_phi} = U S V^T`` (eigenvalues ``S^2 / sigma^2``,
eigenvectors the rows of ``V^T``), avoiding the ``P x P`` outer product.

This works for *any* differentiable encoder ``f_phi``; the Jacobian is
materialised row-by-row with reverse-mode autograd (one backward per neuron).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FisherOutput:
    eigenvalues: torch.Tensor  # [k] descending
    eigenvectors: torch.Tensor  # [k, P] rows are eigenvectors of J(x)


def fisher_eigendecomposition(
    image: torch.Tensor,
    encoder,
    sigma: float,
    top_k: int = 50,
    device: torch.device | None = None,
    noise_model: str = "gaussian",
    poisson_gain: float = 1.0,
) -> FisherOutput:
    """Top-``top_k`` eigenpairs of ``J(x) = (1/sigma^2) J_f^T J_f``.

    Parameters
    ----------
    image:
        ``[1, C, Hp, Wp]`` clean stimulus (diffusion space, padded canvas).
    encoder:
        differentiable ``f_phi``: ``[1, C, Hp, Wp] -> [1, N]``.
    """
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W], got {tuple(image.shape)}")
    device = device or next(encoder.parameters()).device
    encoder.eval()

    poisson = noise_model == "poisson"
    x = image.detach().to(device).clone().requires_grad_(True)
    with torch.enable_grad():
        # Poisson Fisher info is defined on the rate scale, so differentiate the raw rate.
        responses = encoder.forward_raw(x) if poisson else encoder(x)  # [1, N]
    n_neurons = responses.shape[1]
    P = x[0].numel()

    J_f = torch.zeros(n_neurons, P, device=device)
    for i in range(n_neurons):
        if x.grad is not None:
            x.grad.zero_()
        responses[0, i].backward(retain_graph=(i < n_neurons - 1))
        J_f[i] = x.grad.detach().reshape(-1).clone()

    if poisson:
        # J = gain * J_f^T diag(1/lambda) J_f  ->  weight row i by sqrt(gain / lambda_i),
        # so the thin SVD of the weighted matrix gives the eigenpairs directly.
        lam = responses.detach()[0].clamp(min=1e-8)
        J_f = J_f * (poisson_gain / lam).sqrt().unsqueeze(1)

    _, S, Vh = torch.linalg.svd(J_f, full_matrices=False)
    eigenvalues = (S**2) if poisson else (S**2) / (sigma**2)
    k = min(top_k, n_neurons)
    return FisherOutput(
        eigenvalues=eigenvalues[:k].detach().cpu().clone(),
        eigenvectors=Vh[:k].detach().cpu().clone(),
    )
