"""Core of the adaptation gain-control model.

The core turns a movie into ON/OFF bipolar-cell feature maps and applies fast temporal
adaptation (gain control) to them:

    subunit DoG filters -> BatchNorm3d -> bias -> ReLU -> subunit gain control

The subunit filters are parametric (mostly fixed) difference-of-Gaussians, so the core carries
very few free parameters (the gain strengths, the bias, and optionally the DoG parameters). The
downstream spatial pooling to single neurons is handled by the readout.
"""

import torch
from jaxtyping import Float

from openretina.modules.core.base_core import Core
from openretina.modules.layers.gain_control import GainControlLayer
from openretina.modules.layers.scaling import Bias3DLayer
from openretina.modules.layers.subunit import DoGSubunitLayer


class AdaptationCore(Core):
    """
    Subunit + gain-control feature extractor.

    Args:
        spatial_bin_size: (height, width) of the subunit kernels, in pixels.
        temporal_bin_size: Number of frames in each subunit kernel.
        microns_per_px: Spatial scale of the stimulus.
        out_channels: Number of subunit channels (2 -> ON/OFF).
        subunit_surround: Whether the subunit DoG includes a surround.
        use_batchnorm: Insert a BatchNorm3d between the subunit filters and the bias.
        batchnorm_affine: Whether that BatchNorm3d is affine.
        gain_control_tau / gain_control_exp / gain_control_time_step: Gain control constants.
        gain_control_H: Initial gain strength (0 -> adaptation starts as the identity).
        separate_on_off: Give each channel its own gain-control layer (independent strengths).
        subunit_kwargs: Extra keyword arguments forwarded to ``DoGSubunitLayer`` (e.g. learn_* flags).
    """

    def __init__(
        self,
        spatial_bin_size: tuple[int, int] = (15, 15),
        temporal_bin_size: int = 16,
        microns_per_px: float = 3.5 * 6,
        out_channels: int = 2,
        subunit_surround: bool = True,
        use_batchnorm: bool = True,
        batchnorm_affine: bool = False,
        gain_control_tau: float = 400.0,
        gain_control_exp: float = 1.0,
        gain_control_time_step: float = 25.0,
        gain_control_H: float = 0.0,
        gain_control_trainable: bool = False,
        separate_on_off: bool = True,
        subunit_kwargs: dict | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.separate_on_off = separate_on_off

        self.subunit = DoGSubunitLayer(
            spatial_bin_size=spatial_bin_size,
            temporal_bin_size=temporal_bin_size,
            subunit_surround=subunit_surround,
            microns_per_px=microns_per_px,
            out_channels=out_channels,
            learn_bias=False,  # bias is applied after the (optional) batchnorm, as Bias3DLayer
            **(subunit_kwargs or {}),
        )
        self.norm = torch.nn.BatchNorm3d(out_channels, affine=batchnorm_affine) if use_batchnorm else None
        self.bias = Bias3DLayer(out_channels)
        self.nonlinearity = torch.nn.ReLU()

        def make_gc() -> GainControlLayer:
            return GainControlLayer(
                H=gain_control_H, tau=gain_control_tau, exp=gain_control_exp, time_step=gain_control_time_step
            )

        # One gain-control layer per channel (independent strengths) or a single shared one.
        self.gain_control: torch.nn.Module = (
            torch.nn.ModuleList([make_gc() for _ in range(out_channels)]) if separate_on_off else make_gc()
        )
        # During core training the subunit gain control is off (H frozen at its init value).
        for param in self.gain_control.parameters():
            param.requires_grad = gain_control_trainable

    def _apply_gain_control(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.gain_control, torch.nn.ModuleList):
            channels = [gc(x[:, c : c + 1]) for c, gc in enumerate(self.gain_control)]
            return torch.cat(channels, dim=1)
        return self.gain_control(x)

    def forward(
        self, x: Float[torch.Tensor, "batch 1 time height width"]
    ) -> Float[torch.Tensor, "batch channels time height width"]:
        x = self.subunit(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.bias(x)
        x = self.nonlinearity(x)
        return self._apply_gain_control(x)

    def regularizer(self) -> torch.Tensor | float:
        # The subunit filters are parametric (mostly fixed) and gain control only adds scalar
        # strengths, so the core carries no weight-decay-style regularization of its own.
        return 0.0
