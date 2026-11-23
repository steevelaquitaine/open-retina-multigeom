"""Parametric bipolar-cell subunit layer for the adaptation model.

The Difference-of-Gaussians (DoG) subunit layer is a small bank of fixed (or optionally learnable) difference-of-Gaussians
receptive fields — one per output channel (ON / OFF) — each a separable product of a 2D spatial
DoG and a 1D temporal DoG. The resulting 3D kernels are applied to the movie with a single
``conv3d``. Parameters are initialised from measured bipolar-cell fits shipped alongside the code
(``Bipolar_cells_params_inum_ms.npy``).
"""

import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_BC_PARAMS_PATH = pathlib.Path(__file__).resolve().parent / "bipolar_cell_params.npy"

# Temporal grid step, in ms (one temporal bin), used to build the temporal filter support.
_TEMPORAL_STEP_MS = 25


class DoGSubunitLayer(nn.Module):
    """
    Bank of parametric difference-of-Gaussians subunits applied as a single 3D convolution.

    Args:
        spatial_bin_size: (height, width) of each subunit kernel, in pixels.
        temporal_bin_size: Number of frames in each subunit kernel.
        subunit_surround: Whether to include the (spatial and temporal) surround.
        microns_per_px: Spatial scale of the stimulus, used to place the DoG on a physical grid.
        out_channels: Number of subunits / output channels (1 -> ON only, 2 -> ON and OFF).
        stride: Spatial stride of the convolution.
        learn_subunit_amp: Learn the center amplitude/width of the spatial DoG.
        learn_subunit_surround: Learn the spatial center amplitude while fixing the surround.
        learn_subunit_temporal_surround: Learn the temporal surround amplitudes.
        learn_only_temporal_subunit_params: Learn only the temporal DoG parameters.
        learn_all_subunit_params: Learn the full (redundant-free) parameter set.
        learn_bias: Learn the per-channel additive bias.
        bc_params_path: Path to the bipolar-cell parameter ``.npy`` asset.
    """

    def __init__(
        self,
        spatial_bin_size: tuple[int, int],
        temporal_bin_size: int,
        subunit_surround: bool = True,
        microns_per_px: float = 14,
        out_channels: int = 2,
        stride: int = 1,
        learn_subunit_amp: bool = False,
        learn_subunit_surround: bool = False,
        learn_subunit_temporal_surround: bool = False,
        learn_only_temporal_subunit_params: bool = False,
        learn_all_subunit_params: bool = False,
        learn_bias: bool = True,
        surround_init_scale: float = 1.0,
        bc_params_path: str | pathlib.Path = DEFAULT_BC_PARAMS_PATH,
    ):
        super().__init__()
        if out_channels not in (1, 2):
            raise ValueError(f"out_channels must be 1 (ON) or 2 (ON/OFF), got {out_channels}")

        self.spatial_bin_size = spatial_bin_size
        self.temporal_bin_size = temporal_bin_size
        self.subunit_surround = subunit_surround
        self.microns_per_px = microns_per_px
        self.out_channels = out_channels
        self.stride = stride

        self.bc_params = np.load(bc_params_path, allow_pickle=True).item()

        self._init_filter_parameters(
            learn_subunit_amp,
            learn_subunit_surround,
            learn_subunit_temporal_surround,
            learn_only_temporal_subunit_params,
            learn_all_subunit_params,
        )
        # Legacy "new_init": down-scale the (fixed) spatial surround amplitude at initialisation.
        if surround_init_scale != 1.0:
            with torch.no_grad():
                for name in ("B_s_on", "B_s_off"):
                    getattr(self, name).mul_(surround_init_scale)
        self._init_st_grid()
        self.sub_bias = nn.Parameter(torch.zeros(out_channels), requires_grad=learn_bias)

    # ------------------------------------------------------------------ setup
    def _init_st_grid(self) -> None:
        h, w = self.spatial_bin_size
        x_um = np.arange(0, h * self.microns_per_px, self.microns_per_px)
        y_um = np.arange(0, w * self.microns_per_px, self.microns_per_px)
        x1, y1 = np.meshgrid(x_um, y_um)
        self.x1 = nn.Parameter(torch.from_numpy(x1), requires_grad=False)
        self.y1 = nn.Parameter(torch.from_numpy(y1), requires_grad=False)
        self.x_time = nn.Parameter(
            torch.from_numpy(np.arange(0, self.temporal_bin_size * _TEMPORAL_STEP_MS, _TEMPORAL_STEP_MS)),
            requires_grad=False,
        )

    def _set(self, names: list[str], values, requires_grad: bool) -> None:
        for name, value in zip(names, values, strict=True):
            setattr(self, name, nn.Parameter(torch.tensor(value), requires_grad=requires_grad))

    def _init_filter_parameters(
        self,
        learn_subunit_amp: bool,
        learn_subunit_surround: bool,
        learn_subunit_temporal_surround: bool,
        learn_only_temporal_subunit_params: bool,
        learn_all_subunit_params: bool,
    ) -> None:
        learn_all = learn_all_subunit_params
        self._set(["B_c_on", "sig_c_on", "B_s_on", "sig_s_on"], self.bc_params["on_spatial"], learn_all)
        self._set(["B_c_off", "sig_c_off", "B_s_off", "sig_s_off"], self.bc_params["off_spatial"], learn_all)
        temporal_names = [
            "center_x_0_t_{s}",
            "center_x_1_t_{s}",
            "center_B_c_t_{s}",
            "center_sig_c_t_{s}",
            "center_B_s_t_{s}",
            "center_sig_s_t_{s}",
        ]
        surround_names = [n.replace("center_", "surround_") for n in temporal_names]
        for suffix, key in [("on", "on_temporal"), ("off", "off_temporal")]:
            self._set([n.format(s=suffix) for n in temporal_names], self.bc_params[key], learn_all)
            self._set([n.format(s=suffix) for n in surround_names], self.bc_params[key], learn_all)

        if learn_subunit_amp:
            for name in ("B_c_on", "B_c_off", "sig_c_on", "sig_c_off"):
                getattr(self, name).requires_grad = True

        if learn_all:
            # Center amplitudes are redundant with the surround, so keep them fixed.
            for name in ("B_c_on", "B_c_off", "center_B_c_t_on", "center_B_c_t_off"):
                getattr(self, name).requires_grad = False

        if learn_subunit_surround:
            for on, off in [("B_c_on", "B_c_off")]:
                getattr(self, on).requires_grad = True
                getattr(self, off).requires_grad = True
            for name in ("B_s_on", "sig_s_on", "B_s_off", "sig_s_off"):
                getattr(self, name).requires_grad = False

        if learn_subunit_temporal_surround:
            grads = {
                "center_B_c_t_on": False,
                "center_B_s_t_on": True,
                "surround_B_c_t_on": False,
                "surround_B_s_t_on": True,
                "center_B_c_t_off": True,
                "center_B_s_t_off": False,  # inverted for OFF
                "surround_B_c_t_off": True,
                "surround_B_s_t_off": False,
            }
            for name, g in grads.items():
                getattr(self, name).requires_grad = g

        if learn_only_temporal_subunit_params:
            for suffix in ("on", "off"):
                inverted = suffix == "off"
                for prefix in ("center", "surround"):
                    grads = {
                        f"{prefix}_x_0_t_{suffix}": True,
                        f"{prefix}_x_1_t_{suffix}": True,
                        f"{prefix}_B_c_t_{suffix}": inverted,
                        f"{prefix}_B_s_t_{suffix}": not inverted,
                        f"{prefix}_sig_c_t_{suffix}": False,
                        f"{prefix}_sig_s_t_{suffix}": False,
                    }
                    for name, g in grads.items():
                        getattr(self, name).requires_grad = g

    # ------------------------------------------------------------------ filters
    @staticmethod
    def _temporal_dog(x, x_0, x_1, B_c, sig_c, B_s, sig_s) -> Tensor:
        return B_c * torch.exp(-((x - x_0) ** 2) / (2 * sig_c**2)) - B_s * torch.exp(-((x - x_1) ** 2) / (2 * sig_s**2))

    def _spatial_gaussian(self, x_0, y_0, B, sig) -> Tensor:
        return B * torch.exp(-((self.x1 - x_0) ** 2 + (self.y1 - y_0) ** 2) / (2 * sig**2))

    def _channel_kernel(
        self,
        B_c,
        sig_c,
        B_s,
        sig_s,
        c_x0,
        c_x1,
        c_Bc,
        c_sc,
        c_Bs,
        c_ss,
        s_x0,
        s_x1,
        s_Bc,
        s_sc,
        s_Bs,
        s_ss,
    ) -> Tensor:
        h, w = self.spatial_bin_size
        cx, cy = h / 2 * self.microns_per_px, w / 2 * self.microns_per_px
        center_spatial = self._spatial_gaussian(cx, cy, B_c, sig_c)
        surround_spatial = torch.nan_to_num(self._spatial_gaussian(cx, cy, B_s, sig_s), nan=0.0)
        center_temporal = self._temporal_dog(self.x_time, c_x0, c_x1, c_Bc, c_sc, c_Bs, c_ss)
        surround_temporal = self._temporal_dog(self.x_time, s_x0, s_x1, s_Bc, s_sc, s_Bs, s_ss)

        if self.subunit_surround:
            frames = [
                center_spatial * center_temporal[k] - surround_spatial * surround_temporal[k]
                for k in range(center_temporal.shape[0])
            ]
        else:
            frames = [center_spatial * center_temporal[k] for k in range(center_temporal.shape[0])]
        return torch.flip(torch.stack(frames), dims=(0,)).float()

    def _build_kernels(self) -> Tensor:
        on = self._channel_kernel(
            self.B_c_on,
            self.sig_c_on,
            self.B_s_on,
            self.sig_s_on,
            self.center_x_0_t_on,
            self.center_x_1_t_on,
            self.center_B_c_t_on,
            self.center_sig_c_t_on,
            self.center_B_s_t_on,
            self.center_sig_s_t_on,
            self.surround_x_0_t_on,
            self.surround_x_1_t_on,
            self.surround_B_c_t_on,
            self.surround_sig_c_t_on,
            self.surround_B_s_t_on,
            self.surround_sig_s_t_on,
        )
        if self.out_channels == 1:
            return on.unsqueeze(0).unsqueeze(1)
        off = self._channel_kernel(
            self.B_c_off,
            self.sig_c_off,
            self.B_s_off,
            self.sig_s_off,
            self.center_x_0_t_off,
            self.center_x_1_t_off,
            self.center_B_c_t_off,
            self.center_sig_c_t_off,
            self.center_B_s_t_off,
            self.center_sig_s_t_off,
            self.surround_x_0_t_off,
            self.surround_x_1_t_off,
            self.surround_B_c_t_off,
            self.surround_sig_c_t_off,
            self.surround_B_s_t_off,
            self.surround_sig_s_t_off,
        )
        return torch.cat([on.unsqueeze(0), off.unsqueeze(0)]).unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv3d(x, self._build_kernels(), bias=self.sub_bias, stride=(1, self.stride, self.stride))


# Backward compatibility: checkpoints saved before the DoGSubunitLayer rename pickle a
# reference to ``SubunitLayer``, so the old name must stay resolvable for torch.load.
SubunitLayer = DoGSubunitLayer
