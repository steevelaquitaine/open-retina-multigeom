"""Adapt an open-retina model into a static-image encoder ``f_phi``.

The multi-scale Fisher pipeline needs a map ``f_phi : image -> population
response vector``. open-retina models take a 5-D movie ``[B, C, T, H, W]`` and
return ``[B, T_out, N]``. ``RetinaImageEncoder`` presents a single spatial
stimulus to the model and reduces the response to one vector per image,
handling both regimes:

* **Static image models** (e.g. ``goldin_2022``, ``model_cut_frames == 0``):
  the stimulus is shown for a single frame and the (single) output frame is the
  response.
* **Temporal models** (e.g. ``hoefling_2024``, which eats ``model_cut_frames``
  frames): the stimulus is shown as a flashed presentation ``grey -> image`` and
  the response is the mean over a post-onset window (``[+50, +450] ms``).

Two population definitions are supported through the ``responder`` module:

* **mosaic** -- a proper single-cell-type population built from one model cell by
  displacing its receptive field across a spatial grid (``build_mosaic_encoder``).
  This is the correct construction when a model's readout units are heterogeneous
  cell types (e.g. ``goldin_2022``) rather than one tiled mosaic.
* **session** -- the raw set of all per-session readout units
  (``build_session_encoder``); only a valid "population" when those units share a
  cell type, otherwise a mixed-cell-type comparison baseline.

The encoder consumes stimuli in *diffusion space* (padded, roughly ``[-1, 1]``)
and converts to *model space* internally, so it plugs straight into the
diffusion model and the G(x) estimator.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from openretina.insilico.information_geometry.stimuli import PadSpec, StimulusNormalizer
from openretina.modules.readout.gaussian import PointGaussianReadout

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Responders: movie [B, C, T, H, W] -> responses [B, T_out, N]
# ---------------------------------------------------------------------------
class _SessionResponder(nn.Module):
    """Real session population via the full (frozen) model readout."""

    def __init__(self, model: nn.Module, session_id: str):
        super().__init__()
        self.model = model
        self.session_id = session_id
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        return self.model(movie, data_key=self.session_id)


class _PointMosaicResponder(nn.Module):
    """Exact RF-displacement mosaic via a tiled ``PointGaussianReadout``.

    One chosen cell (a single cell type) is replicated by copying its feature
    weights / RF size and tiling only its readout position ``_mu`` across the
    core feature map. Because the core is convolutional, reading the same filter
    at a new location *is* that cell's receptive field displaced there -- exact,
    with no translation-equivariance assumption and no stimulus-boundary
    artefacts. The model's post-readout nonlinearity is applied to match the
    real model output.
    """

    def __init__(self, core: nn.Module, readout: PointGaussianReadout, nonlinearity):
        super().__init__()
        self.core = core
        self.readout = readout
        self.nonlinearity = nonlinearity
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        core_out = self.core(movie)  # [B, c, T, w, h]
        b, c, t, w, h = core_out.shape
        y = self.readout(core_out.reshape(b * t, c, w, h), sample=False)  # [B*T, N]
        return self.nonlinearity(y).reshape(b, t, -1)  # [B, T, N]


class _TranslationMosaicResponder(nn.Module):
    """Synthetic single-cell-type mosaic via receptive-field displacement.

    One chosen model cell (a single cell type / optic-tract channel) is
    replicated across space by translating the stimulus and reading out the
    *same* base cell. Because the core is convolutional (translation-equivariant
    up to boundaries), shifting the stimulus by ``-delta`` is equivalent to
    displacing that cell's receptive field by ``+delta``. This uses the full
    model end-to-end (including its output nonlinearity) and is model-agnostic.
    """

    def __init__(self, model: nn.Module, session_id: str, base_neuron_idx: int, shifts: torch.Tensor):
        super().__init__()
        self.model = model
        self.session_id = session_id
        self.base_neuron_idx = int(base_neuron_idx)
        self.register_buffer("shifts", shifts.long())  # [N, 2] integer (dy, dx) pixel shifts
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        # movie: [B, C, T, H, W] -> N spatially-shifted copies -> base cell response
        b = movie.shape[0]
        n = self.shifts.shape[0]
        shifted = [
            torch.roll(movie, shifts=(int(dy), int(dx)), dims=(-2, -1))
            for dy, dx in self.shifts
        ]
        stacked = torch.stack(shifted, dim=1).reshape(b * n, *movie.shape[1:])  # [B*N, C, T, H, W]
        out = self.model(stacked, data_key=self.session_id)  # [B*N, T_out, N_all]
        resp = out[..., self.base_neuron_idx]  # [B*N, T_out]
        return resp.reshape(b, n, -1).permute(0, 2, 1)  # [B, T_out, N]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class RetinaImageEncoder(nn.Module):
    """Static-image encoder ``f_phi`` wrapping an open-retina responder.

    Parameters
    ----------
    responder:
        module mapping ``[B, C, T, H, W] -> [B, T_out, N]``.
    normalizer, pad:
        diffusion-space <-> model-space and canvas pad/crop (see ``stimuli.py``).
    n_neurons:
        response dimension (after optional ``neuron_indices`` selection).
    grey_frames, image_frames:
        flashed-presentation frame counts (``grey_frames == 0`` for static models).
    reduce_window:
        ``(w0, w1)`` output-frame slice averaged into the response.
    grey_level:
        model-space grey value (default 0.0 == normalised mean).
    neuron_indices:
        optional subset of the responder's neurons to keep.
    """

    def __init__(
        self,
        responder: nn.Module,
        normalizer: StimulusNormalizer,
        pad: PadSpec,
        n_neurons: int,
        grey_frames: int,
        image_frames: int,
        reduce_window: tuple[int, int],
        grey_level: float = 0.0,
        neuron_indices: torch.Tensor | None = None,
    ):
        super().__init__()
        self.responder = responder
        self.normalizer = normalizer
        self.pad = pad
        self._n_neurons = int(n_neurons)
        self.grey_frames = int(grey_frames)
        self.image_frames = int(image_frames)
        self.w0, self.w1 = int(reduce_window[0]), int(reduce_window[1])
        self.grey_level = float(grey_level)
        if neuron_indices is not None:
            self.register_buffer("neuron_indices", neuron_indices.long())
        else:
            self.neuron_indices = None
        self.register_buffer("resp_mean", None)  # optional per-cell response standardisation
        self.register_buffer("resp_std", None)

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    def set_response_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Standardise responses to ~unit scale so the estimator's sigma is calibrated
        (responses -> (r - mean) / std). Call after measuring stats over training frames."""
        dev = next(self.parameters()).device
        self.resp_mean = mean.detach().clone().to(dev)
        self.resp_std = std.detach().clamp(min=1e-6).clone().to(dev)

    def _movie(self, x_model: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, C, T, H, W]`` (optional grey pre-roll + image)."""
        b, c, h, w = x_model.shape
        image = x_model.unsqueeze(2).expand(b, c, self.image_frames, h, w)
        if self.grey_frames == 0:
            return image
        grey = x_model.new_full((b, c, self.grey_frames, h, w), self.grey_level)
        return torch.cat([grey, image], dim=2)

    def forward_raw(self, x_diff: torch.Tensor) -> torch.Tensor:
        """``[B, C, Hp, Wp]`` -> ``[B, N]`` **raw** response (model units, no standardisation).

        These are the encoder's rate predictions (non-negative: the model's last stage is a
        nonlinearity), i.e. the Poisson rate lambda(x). Needed by the Poisson noise model,
        which must sample counts on the rate scale rather than the standardised scale.
        """
        x_model = self.normalizer.to_model(self.pad.crop(x_diff))
        out = self.responder(self._movie(x_model))  # [B, T_out, N]
        resp = out[:, self.w0 : self.w1, :].mean(dim=1)  # [B, N]
        if self.neuron_indices is not None:
            resp = resp[:, self.neuron_indices]
        return resp

    def standardize(self, resp: torch.Tensor) -> torch.Tensor:
        """Apply the stored response standardisation to raw responses (no-op if unset)."""
        if self.resp_mean is None:
            return resp
        return (resp - self.resp_mean) / self.resp_std

    def forward(self, x_diff: torch.Tensor) -> torch.Tensor:
        """``[B, C, Hp, Wp]`` (diffusion space) -> ``[B, N]`` population response."""
        return self.standardize(self.forward_raw(x_diff))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _resolve_timing(
    model,
    grey_ms: float,
    image_ms: float,
    window_ms: tuple[float, float],
) -> tuple[int, int, tuple[int, int], float]:
    """Return ``(grey_frames, image_frames, (w0, w1), grey_level)`` from ``data_info``.

    Static models (``model_cut_frames == 0`` or no stimulus rate) collapse to a
    single-frame presentation; temporal models use a flashed window.
    """
    di = model.data_info
    cut = int(di.get("model_cut_frames", 0))
    rate_hz = di.get("stimulus_rate_hz", None)

    if cut == 0 or not rate_hz:
        # Static image model: one frame in, one frame out.
        return 0, 1, (0, 1), 0.0

    rate = rate_hz / 1000.0  # frames per ms
    grey_frames = int(round(grey_ms * rate))
    image_frames = int(round(image_ms * rate))
    onset_out = grey_frames - cut
    if onset_out < 0:
        raise ValueError(
            f"grey_ms={grey_ms} -> {grey_frames} grey frames < cut {cut}; increase grey_ms."
        )
    w0 = onset_out + int(round(window_ms[0] * rate))
    w1 = onset_out + int(round(window_ms[1] * rate))
    total_out = grey_frames + image_frames - cut
    if not (0 <= w0 < w1 <= total_out):
        raise ValueError(
            f"response window [{w0}, {w1}) out of range for {total_out} output frames."
        )
    return grey_frames, image_frames, (w0, w1), 0.0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_session_encoder(
    model,
    session_id: str,
    normalizer: StimulusNormalizer,
    pad: PadSpec,
    neuron_indices: torch.Tensor | None = None,
    grey_ms: float = 1200.0,
    image_ms: float = 600.0,
    window_ms: tuple[float, float] = (50.0, 450.0),
) -> RetinaImageEncoder:
    """Encoder conditioned on a model's real per-session population.

    Model-agnostic: works for static image models (``goldin_2022``) and
    temporal models alike. ``neuron_indices`` optionally selects a subset.
    """
    grey_f, image_f, window, grey_lvl = _resolve_timing(model, grey_ms, image_ms, window_ms)
    n_total = int(model.data_info["n_neurons_dict"][session_id])
    n_neurons = len(neuron_indices) if neuron_indices is not None else n_total
    responder = _SessionResponder(copy.deepcopy(model).eval(), session_id)
    return RetinaImageEncoder(
        responder=responder, normalizer=normalizer, pad=pad, n_neurons=n_neurons,
        grey_frames=grey_f, image_frames=image_f, reduce_window=window,
        grey_level=grey_lvl, neuron_indices=neuron_indices,
    ).eval()


def measure_pixel_rf_sigma(
    model,
    session_id: str,
    neuron_idx: int,
    frames: torch.Tensor,
    device: torch.device | str = "cuda",
    n_frames: int = 64,
) -> float:
    """Estimate a cell's receptive-field size in stimulus pixels.

    Averages ``|d response / d pixel|`` over ``n_frames`` natural frames and
    returns the isotropic second-moment std (mean of the x and y stds).
    """
    device = torch.device(device)
    model = model.to(device).eval()
    x = frames[:n_frames].unsqueeze(2).to(device).clone().requires_grad_(True)  # [n,C,1,H,W]
    out = model(x, data_key=session_id)[:, 0, neuron_idx]
    (grad,) = torch.autograd.grad(out.sum(), x)
    rf = grad.abs().mean(dim=0).sum(dim=(0, 1))  # [H, W]
    h, w = rf.shape
    p = (rf / rf.sum().clamp_min(1e-12)).cpu()
    ys, xs = torch.arange(h).float(), torch.arange(w).float()
    cy, cx = (p.sum(1) * ys).sum(), (p.sum(0) * xs).sum()
    sy = ((p.sum(1) * (ys - cy) ** 2).sum()).sqrt()
    sx = ((p.sum(0) * (xs - cx) ** 2).sum()).sqrt()
    return float((sx + sy) / 2)


def _build_point_mosaic_readout(
    base_readout: PointGaussianReadout,
    base_neuron_idx: int,
    grid: torch.Tensor,
) -> PointGaussianReadout:
    """Tile one cell of a ``PointGaussianReadout`` to ``N`` positions (``grid`` [N, 2]).

    Copies the base cell's feature weights, RF covariance and bias; only ``_mu``
    (the readout position) is tiled. Built by surgery on a deep copy so the exact
    ``forward`` / ``sample_grid`` logic is reused.
    """
    idx = base_neuron_idx
    n = grid.shape[0]
    ro = copy.deepcopy(base_readout)
    with torch.no_grad():
        ro.outdims = n
        ro.grid_shape = (1, n, 1, 2)
        ro._mu = nn.Parameter(grid.view(1, n, 1, 2).clone())
        ro.sigma = nn.Parameter(base_readout.sigma.data[:, idx : idx + 1].repeat(1, n, 1, 1).clone())
        ro._features = nn.Parameter(base_readout._features.data[:, :, :, idx : idx + 1].repeat(1, 1, 1, n).clone())
        bias = base_readout.bias.data[idx].repeat(n).clone() if base_readout.bias is not None else None
        if bias is not None:
            ro.bias = nn.Parameter(bias)
    for p in ro.parameters():
        p.requires_grad_(False)
    return ro


def _feature_map_width(model, session_id: str) -> int:
    di_shape = model.data_info["input_shape"]  # (C, H, W)
    c, h, w = di_shape
    dummy = torch.zeros(1, c, 1, h, w, device=next(model.parameters()).device)
    with torch.no_grad():
        return int(model.core(dummy).shape[-1])


def build_mosaic_encoder(
    model,
    session_id: str,
    base_neuron_idx: int,
    normalizer: StimulusNormalizer,
    pad: PadSpec,
    rf_sigma_px: float,
    n_rows: int = 8,
    n_cols: int = 8,
    spacing_sigma: float = 2.0,
    grey_ms: float = 1200.0,
    image_ms: float = 600.0,
    window_ms: tuple[float, float] = (50.0, 450.0),
) -> RetinaImageEncoder:
    """Encoder conditioned on an exact single-cell-type RF-displacement mosaic.

    For ``PointGaussianReadout`` models (e.g. ``goldin_2022``): one cell
    (``base_neuron_idx``) is tiled to an ``n_rows x n_cols`` grid whose spacing is
    ``spacing_sigma`` times the cell's RF size ``rf_sigma_px`` (so neighbouring RFs
    overlap at ``spacing_sigma`` sigma). Positions are set on the readout feature
    map via ``_mu`` (:class:`_PointMosaicResponder`).
    """
    base_readout = model.readout[session_id]
    if not isinstance(base_readout, PointGaussianReadout):
        raise TypeError(
            f"session readout is {type(base_readout).__name__}; the _mu mosaic supports "
            "PointGaussianReadout. Use build_translation_mosaic_encoder for other models."
        )
    n_total = int(model.data_info["n_neurons_dict"][session_id])
    if not (0 <= base_neuron_idx < n_total):
        raise IndexError(f"base_neuron_idx out of range for {n_total} neurons in {session_id}.")

    # RF size (px) -> spacing in normalised _mu coords: 1 norm unit = feat_w/2 feature
    # cells ~= feat_w/2 stimulus px (near 1:1 for valid convs), so d_mu = d_px / (feat_w/2).
    feat_w = _feature_map_width(model, session_id)
    spacing_norm = spacing_sigma * rf_sigma_px / (feat_w / 2.0)
    ys = torch.linspace(-(n_rows - 1) / 2, (n_rows - 1) / 2, n_rows) * spacing_norm
    xs = torch.linspace(-(n_cols - 1) / 2, (n_cols - 1) / 2, n_cols) * spacing_norm
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).clamp(-0.98, 0.98)  # (x, y) order

    mosaic = _build_point_mosaic_readout(base_readout, base_neuron_idx, grid)
    responder = _PointMosaicResponder(
        copy.deepcopy(model.core).eval(), mosaic, model.readout.nonlinearity
    )
    grey_f, image_f, window, grey_lvl = _resolve_timing(model, grey_ms, image_ms, window_ms)
    LOGGER.info("mosaic: %d cells, rf_sigma=%.2f px, spacing=%.2f px (=%.3f norm)",
                grid.shape[0], rf_sigma_px, spacing_sigma * rf_sigma_px, spacing_norm)
    return RetinaImageEncoder(
        responder=responder, normalizer=normalizer, pad=pad, n_neurons=grid.shape[0],
        grey_frames=grey_f, image_frames=image_f, reduce_window=window, grey_level=grey_lvl,
    ).eval()


class _SlidingReadoutResponder(nn.Module):
    """Convolve a fixed-input cell across a *larger* image: one frozen core pass on
    the big image (gain applied inside the core), then slide the cell's readout
    window across the big core feature map at a grid of positions. Each position is
    one mosaic cell reading a distinct patch, so the population tiles a large field
    (avoids the failure where one RF drives the whole map on a small image)."""

    def __init__(self, core: nn.Module, readout: nn.Module, positions: torch.Tensor, win: int):
        super().__init__()
        self.core = core
        self.readout = readout
        self.win = int(win)
        self.corepad = int(win) // 2  # reflect-pad the core map so cell centres reach the field edge
        self.register_buffer("positions", positions.long())  # [N, 2] (row, col) in padded core-map coords
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        co = self.core(movie)  # [B, C, T, Hc, Wc]
        if self.corepad > 0:  # so edge cells tile the border (no dead margin -> no boundary ring)
            p = self.corepad
            b0, c0, t0, hc, wc = co.shape
            co = torch.nn.functional.pad(co.reshape(b0 * c0 * t0, 1, hc, wc), (p, p, p, p), mode="reflect")
            co = co.reshape(b0, c0, t0, hc + 2 * p, wc + 2 * p)
        b, c, t = co.shape[:3]
        w, n = self.win, self.positions.shape[0]
        windows = [co[..., int(i) : int(i) + w, int(j) : int(j) + w] for i, j in self.positions]
        stacked = torch.stack(windows, dim=1).reshape(b * n, c, t, w, w)  # [B*N, C, T, w, w]
        out = self.readout(stacked)  # [B*N, T_out, 1]
        return out.reshape(b, n, -1).permute(0, 2, 1)  # [B, T_out, N]


def build_sliding_readout_mosaic_encoder(
    model,
    session_id: str,
    base_neuron_idx: int,
    normalizer: StimulusNormalizer,
    pad: PadSpec,
    n_rows: int = 8,
    n_cols: int = 8,
    device: str = "cpu",
    grey_ms: float = 500.0,
    image_ms: float = 500.0,
    window_ms: tuple[float, float] = (50.0, 350.0),
    layout: str = "grid",
    nn_spacing_px: float | None = None,
) -> RetinaImageEncoder:
    """Mosaic for a fixed-input factorized-mask cell (e.g. lorenzi) on a LARGE image.

    Feeds the big image (``pad.h`` px) through the conv core once, then slides the
    cell's ``sqrt(mask)`` x ``sqrt(mask)`` readout window across the big core feature
    map at an ``n_rows x n_cols`` grid. ``pad.h`` should be several times the model's
    native input so the cells tile a wide field. Requires ``model.data_info`` timing.
    """
    sr = model.readout[session_id]
    win = int(round(sr.mask_weights.shape[0] ** 0.5))  # readout window (native core-map side)
    c, h, w = model.data_info["input_shape"]
    dummy = torch.zeros(1, c, model.data_info["model_cut_frames"] + 2, pad.h, pad.w, device=device)
    with torch.no_grad():
        core_hw = int(model.core(dummy).shape[-1])  # big core-map side
    padded_hw = core_hw + 2 * (win // 2)  # responder reflect-pads so centres reach the edge
    if layout == "hex" and nn_spacing_px:
        # hexagonal packing: window top-left corners on a hex lattice, nearest-neighbour = nn_spacing_px
        # core-px (rows offset by d/2, row height d*sqrt(3)/2). RF-centre spacing == window spacing.
        d = float(nn_spacing_px)
        row_h = d * (3.0 ** 0.5) / 2.0
        hi = float(padded_hw - win)
        cen: list[tuple[float, float]] = []
        iy = 0
        y = 0.0
        while y <= hi + 1e-6:
            x = (d / 2.0) if (iy % 2) else 0.0
            while x <= hi + 1e-6:
                cen.append((y, x))
                x += d
            y += row_h
            iy += 1
        positions = torch.tensor(cen).round().long()  # [N, 2]
    else:
        ys = torch.linspace(0, padded_hw - win, n_rows).round().long()
        xs = torch.linspace(0, padded_hw - win, n_cols).round().long()
        positions = torch.stack([ys.repeat_interleave(len(xs)), xs.repeat(len(ys))], dim=1)  # [N, 2]

    base_ro = copy.deepcopy(sr)
    with torch.no_grad():  # restrict the deep-copied readout to the single base neuron
        base_ro.mask_weights = nn.Parameter(sr.mask_weights.data[:, base_neuron_idx : base_neuron_idx + 1])
        base_ro.feature_weights = nn.Parameter(sr.feature_weights.data[:, base_neuron_idx : base_neuron_idx + 1])
        if getattr(sr, "bias", None) is not None:
            base_ro.bias = nn.Parameter(sr.bias.data[base_neuron_idx : base_neuron_idx + 1])
    responder = _SlidingReadoutResponder(copy.deepcopy(model.core).eval(), base_ro, positions, win)
    grey_f, image_f, reduce_window, grey_lvl = _resolve_timing(model, grey_ms, image_ms, window_ms)
    LOGGER.info("sliding-readout mosaic: %d cells, big core-map %dx%d, window=%dpx", positions.shape[0],
                core_hw, core_hw, win)
    return RetinaImageEncoder(
        responder=responder, normalizer=normalizer, pad=pad, n_neurons=positions.shape[0],
        grey_frames=grey_f, image_frames=image_f, reduce_window=reduce_window, grey_level=grey_lvl,
    ).eval()


class _MaskShiftResponder(nn.Module):
    """RF-displacement mosaic for a factorized (spatial-mask) readout: one frozen
    core pass (gain control applied inside the core) then a tiled readout whose
    per-neuron spatial mask is the base cell's mask rolled to a grid of positions."""

    def __init__(self, core: nn.Module, readout: nn.Module):
        super().__init__()
        self.core = core
        self.readout = readout
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        return self.readout(self.core(movie))  # [B, T_out, N]


def build_maskshift_mosaic_encoder(
    model,
    session_id: str,
    base_neuron_idx: int,
    normalizer: StimulusNormalizer,
    pad: PadSpec,
    n_rows: int = 8,
    n_cols: int = 8,
    max_shift: int = 6,
    grey_ms: float = 500.0,
    image_ms: float = 500.0,
    window_ms: tuple[float, float] = (50.0, 350.0),
) -> RetinaImageEncoder:
    """RF-displacement mosaic for factorized-mask readouts (e.g. lorenzi adaptation).

    Tiles one cell's spatial mask across ``n_rows x n_cols`` positions (rolled by up
    to ``+/- max_shift`` core-map cells). Requires ``model.data_info`` with
    ``model_cut_frames`` and ``stimulus_rate_hz`` set (for the flashed-image timing).
    """
    sr = model.readout[session_id]
    hw = int(round(sr.mask_weights.shape[0] ** 0.5))  # spatial side of the core map
    idx = base_neuron_idx
    ro = copy.deepcopy(sr)
    with torch.no_grad():
        base_mask = sr.mask_weights.data[:, idx].reshape(hw, hw)
        ys = torch.linspace(-max_shift, max_shift, n_rows).round().long()
        xs = torch.linspace(-max_shift, max_shift, n_cols).round().long()
        masks = [
            torch.roll(base_mask, shifts=(int(dy), int(dx)), dims=(0, 1)).reshape(-1)
            for dy in ys for dx in xs
        ]
        n = len(masks)
        ro.mask_weights = nn.Parameter(torch.stack(masks, dim=1))  # [HW, N]
        ro.feature_weights = nn.Parameter(sr.feature_weights.data[:, idx : idx + 1].repeat(1, n))  # [C, N]
        if getattr(sr, "bias", None) is not None:
            ro.bias = nn.Parameter(sr.bias.data[idx].repeat(n))
    for p in ro.parameters():
        p.requires_grad_(False)
    responder = _MaskShiftResponder(copy.deepcopy(model.core).eval(), ro)
    grey_f, image_f, window, grey_lvl = _resolve_timing(model, grey_ms, image_ms, window_ms)
    LOGGER.info("maskshift mosaic: %d cells, core-map %dx%d, max_shift=%d", n, hw, hw, max_shift)
    return RetinaImageEncoder(
        responder=responder, normalizer=normalizer, pad=pad, n_neurons=n,
        grey_frames=grey_f, image_frames=image_f, reduce_window=window, grey_level=grey_lvl,
    ).eval()


def build_translation_mosaic_encoder(
    model,
    session_id: str,
    base_neuron_idx: int,
    normalizer: StimulusNormalizer,
    pad: PadSpec,
    n_rows: int = 6,
    n_cols: int = 6,
    max_shift: int = 30,
    grey_ms: float = 1200.0,
    image_ms: float = 600.0,
    window_ms: tuple[float, float] = (50.0, 450.0),
) -> RetinaImageEncoder:
    """Fallback mosaic via stimulus translation (model-agnostic, approximate).

    Tiles one cell across ``N = n_rows * n_cols`` positions by rolling the
    stimulus (assumes translation-equivariance; wraps at borders). Prefer
    :func:`build_mosaic_encoder` for ``PointGaussianReadout`` models.
    """
    ys = torch.linspace(-max_shift, max_shift, n_rows).round().long()
    xs = torch.linspace(-max_shift, max_shift, n_cols).round().long()
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    shifts = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=1)
    responder = _TranslationMosaicResponder(
        copy.deepcopy(model).eval(), session_id, base_neuron_idx, shifts
    )
    grey_f, image_f, window, grey_lvl = _resolve_timing(model, grey_ms, image_ms, window_ms)
    return RetinaImageEncoder(
        responder=responder, normalizer=normalizer, pad=pad, n_neurons=shifts.shape[0],
        grey_frames=grey_f, image_frames=image_f, reduce_window=window, grey_level=grey_lvl,
    ).eval()
