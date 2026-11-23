import torch
import torch.nn as nn
from jaxtyping import Float


class GainControlLayer(nn.Module):
    """
    Temporal gain control layer modelling fast adaptation in the retina.

    Each timepoint is divided by a gain that grows with a leaky-integrated history of the
    (rectified) activity that preceded it:

        A[t] = H * sum_{k<=t} exp(-(t - k + 1) * time_step / tau) * x[k]
        G[t] = 1 / (1 + A[t] ** exp)
        y[t] = G[t] * x[t]

    The temporal integration is computed in closed form with a lower-triangular decay matrix,
    so the forward pass is stateless: the whole history of a clip must be present in the time
    dimension of the input (no cross-batch memory is kept).

    Args:
        H: Initial gain strength (learned).
        tau: Adaptation time constant, in the same units as ``time_step``.
        exp: Exponent applied to the adaptation term.
        time_step: Duration of one timebin.
    """

    tau: torch.Tensor
    exp: torch.Tensor
    time_step: torch.Tensor

    def __init__(self, H: float = 0.0, tau: float = 400.0, exp: float = 1.0, time_step: float = 25.0):
        super().__init__()
        self.H = nn.Parameter(torch.tensor(float(H)))
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("exp", torch.tensor(float(exp)))
        self.register_buffer("time_step", torch.tensor(float(time_step)))

    def _decay_matrix(self, n_timepoints: int, device: torch.device) -> Float[torch.Tensor, "time time"]:
        t = torch.arange(n_timepoints, device=device)
        delay = t.unsqueeze(1) - t.unsqueeze(0)  # delay[t, k] = t - k
        weights = torch.exp(-(delay + 1) * self.time_step / self.tau)
        return torch.tril(weights)  # keep only causal terms k <= t

    def forward(
        self, x: Float[torch.Tensor, "batch channels time height width"]
    ) -> Float[torch.Tensor, "batch channels time height width"]:
        n_timepoints = x.shape[2]
        decay = self._decay_matrix(n_timepoints, x.device)

        # A[..., t, ...] = H * sum_k decay[t, k] * x[..., k, ...]
        adaptation = self.H * torch.einsum("tk,bckhw->bcthw", decay, x)
        gain = 1.0 / (1.0 + adaptation**self.exp)
        return gain * x
