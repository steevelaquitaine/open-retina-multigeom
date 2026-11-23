"""Adaptation gain-control model (Lorenzi et al.).

A core–readout model reproducing fast adaptive mechanisms in the retina:

    core:    subunit DoG filters -> batchnorm -> bias -> ReLU -> subunit gain control
    readout: factorized spatial pooling to single neurons (+ softplus)
    ganglion gain control on the neuron output
"""

import inspect
from typing import Any

import torch
from jaxtyping import Float, Int

from openretina.data_io.lorenzi_adaptation.constants import (
    FRAME_RATE_HZ,
    N_ADAPT_FRAMES,
)
from openretina.data_io.lorenzi_adaptation.metadata import metadata_for_session
from openretina.models.core_readout import BaseCoreReadout
from openretina.modules.core.adaptation_core import AdaptationCore
from openretina.modules.layers.gain_control import GainControlLayer
from openretina.modules.readout.multi_readout import MultiFactorizedReadout
from openretina.modules.readout.normalized_factorized import (
    MultiNormalizedFactorizedReadout,
)


def load_adaptation_model(
    path: str, map_location: str | torch.device = "cpu"
) -> "AdaptationGainControlModel":
    """Load a translated adaptation checkpoint saved as ``{"state_dict", "hyper_parameters"}``.

    ``translate_adaptation_weights`` / ``materialize_translated_weights`` save the model's init
    hyper-parameters plus its ``state_dict`` (not a full Lightning checkpoint), so this rebuilds the
    model from the init hyper-parameters and loads the weights. Gain-control ``exp``/``H`` are restored
    from the ``state_dict`` (they are buffers/parameters), so any baked-in fitted gain is preserved.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    init_params = set(
        inspect.signature(AdaptationGainControlModel.__init__).parameters
    ) - {"self"}
    kwargs = {k: v for k, v in blob["hyper_parameters"].items() if k in init_params}
    model = AdaptationGainControlModel(**kwargs)
    model.load_state_dict(blob["state_dict"])
    return model.eval()


class _BatchMeanLoss(torch.nn.Module):
    """Averages a summed 3d loss over the batch (the legacy ``criterion(...).mean(0).sum()`` reduction)."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(output, target) / output.shape[0]


class AdaptationGainControlModel(BaseCoreReadout):
    def __init__(
        self,
        in_shape: Int[tuple, "channels time height width"],
        n_neurons_dict: dict[str, int],
        # --- core = subunit ---
        spatial_bin_size: tuple[int, int] = (15, 15),
        temporal_bin_size: int = 16,
        microns_per_px: float = 3.5 * 6,
        out_channels: int = 2,
        subunit_surround: bool = True,
        core_use_batchnorm: bool = True,
        core_batchnorm_affine: bool = False,
        subunit_kwargs: dict | None = None,
        # --- gain control ---
        gain_control_tau: float = 400.0,
        gain_control_exp: float = 1.0,
        gain_control_time_step: float = 25.0,
        subunit_gain_control_H: float = 0.0,
        readout_gain_control_H: float = 0.0,
        subunit_gain_control_trainable: bool = False,
        readout_gain_control_trainable: bool = False,
        separate_on_off: bool = True,
        output_scale: float = 1.0,
        # --- readout ---
        readout_mask_l1_reg: float = 0.0,
        readout_weights_l1_reg: float = 0.0,
        readout_laplace_mask_reg: float = 0.0,
        readout_bias: bool = True,
        readout_positive: bool = False,
        readout_batchnorm: bool = False,
        readout_init_scales: tuple[tuple[float, float], tuple[float, float]] = (
            (0.0, 0.01),
            (0.0, 0.01),
        ),
        # --- training ---
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        legacy_loss_reduction: bool = False,
        data_info: dict[str, Any] | None = None,
    ):
        core = AdaptationCore(
            spatial_bin_size=spatial_bin_size,
            temporal_bin_size=temporal_bin_size,
            microns_per_px=microns_per_px,
            out_channels=out_channels,
            subunit_surround=subunit_surround,
            use_batchnorm=core_use_batchnorm,
            batchnorm_affine=core_batchnorm_affine,
            gain_control_tau=gain_control_tau,
            gain_control_exp=gain_control_exp,
            gain_control_time_step=gain_control_time_step,
            gain_control_H=subunit_gain_control_H,
            gain_control_trainable=subunit_gain_control_trainable,
            separate_on_off=separate_on_off,
            subunit_kwargs=subunit_kwargs,
        )

        in_shape_readout = self.compute_readout_input_shape(tuple(in_shape), core)
        mean_activity_dict = (data_info or {}).get("mean_activity_dict")
        readout_kwargs: dict[str, Any] = dict(
            in_shape=in_shape_readout,
            n_neurons_dict=n_neurons_dict,
            mask_l1_reg=readout_mask_l1_reg,
            weights_l1_reg=readout_weights_l1_reg,
            laplace_mask_reg=readout_laplace_mask_reg,
            readout_bias=readout_bias,
            # "positive readout" forbids negative pixels in the spatial mask.
            mask_constraint="abs" if readout_positive else None,
            init_scales=readout_init_scales,
            mean_activity_dict=mean_activity_dict,
        )

        if readout_batchnorm:
            readout: MultiFactorizedReadout | MultiNormalizedFactorizedReadout = (
                MultiNormalizedFactorizedReadout(
                    batchnorm_affine=core_batchnorm_affine, **readout_kwargs
                )
            )
        else:
            readout = MultiFactorizedReadout(**readout_kwargs)

        super().__init__(
            core=core, readout=readout, learning_rate=learning_rate, data_info=data_info
        )
        self._weight_decay = weight_decay

        # Attach the stimulus/response timing to data_info so downstream analysis and plotting can align
        # predictions to the stimulus without injecting it manually. A freshly-built or translated model
        # otherwise has these unset until it has been through a validation loop (which sets model_cut_frames).
        self.data_info.setdefault(
            "model_cut_frames", N_ADAPT_FRAMES
        )  # 15: leading frames the model drops (lag)
        self.data_info.setdefault(
            "frame_rate_hz", FRAME_RATE_HZ
        )  # 40 Hz response bin rate

        # Attach each cell's metadata (experiment, cell_type, test_reliability, model_test_score) under
        # sessions_kwargs, mirroring what the dataloader path stores from ResponsesTrainTestSplit.session_kwargs,
        # so a translated model (which never runs the dataloader) still exposes the cell type in data_info.
        sessions_kwargs = self.data_info.setdefault("sessions_kwargs", {})
        for session in n_neurons_dict:
            cell_metadata = metadata_for_session(session)
            if cell_metadata:
                sessions_kwargs.setdefault(session, cell_metadata)

        # (Minor note): In Lorenzi et al., we averaged the Poisson term over the batch and
        # summed over the time bins (``criterion(...).mean(0).sum()``), whereas openretina sums the whole
        # batch. The summed batch makes a given readout L1 ~batch_size times weaker. This matters if we
        # need to compare the L1 regularization strength betzeen training older and newer training runs.
        if legacy_loss_reduction:
            self.loss = _BatchMeanLoss(self.loss)

        # Readout gain control, applied to the neuron output after the readout.
        self.readout_gain_control = GainControlLayer(
            H=readout_gain_control_H,
            tau=gain_control_tau,
            exp=gain_control_exp,
            time_step=gain_control_time_step,
        )
        self.readout_gain_control.H.requires_grad = readout_gain_control_trainable

        # Fitted output scale A (least squares on mean-centred PSTHs, see fit_adaptation_gain_control): the
        # gain-control score is scale-invariant, so A restores the firing-rate scale. Fixed, never trained.
        # The paper's in-silico analyses (LSTA / arrow plots) were computed without it (i.e. output_scale = 1).
        self.output_scale: torch.Tensor
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def _apply_readout_gain_control(
        self, responses: Float[torch.Tensor, "batch time neurons"]
    ) -> Float[torch.Tensor, "batch time neurons"]:
        # GainControlLayer works on (batch, channels, time, h, w); treat neurons as channels.
        x = responses.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        x = self.readout_gain_control(x)
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels time height width"],
        data_key: str | None = None,
    ) -> Float[torch.Tensor, "batch time neurons"]:
        responses = super().forward(x, data_key)
        return self._apply_readout_gain_control(responses) * self.output_scale

    def configure_optimizers(self):
        # Same optimizer/scheduler as BaseCoreReadout, but with a configurable weight decay (the base
        # uses AdamW's default of 0.01; the legacy training used 0).
        config = super().configure_optimizers()
        for group in config["optimizer"].param_groups:
            group["weight_decay"] = self._weight_decay
        return config

    def prepare_for_gain_control_fit(
        self, layer: str = "core"
    ) -> list[torch.nn.Parameter]:
        """Freeze the whole model except one gain-control layer's strength(s), returned as a list.

        Typically, this can be used for a second training stage or for fine-tuning on new data.
        With the core and readout fixed, only the gain-control
        strength ``H`` is fit on the test set. ``layer="readout"`` fits the ganglion gain control after
        the readout (one ``H``); ``layer="core"`` fits the on/off subunit gain control in the core
        (one ``H`` per channel).
        """
        for param in self.parameters():
            param.requires_grad = False
        if layer == "readout":
            gc_modules = [
                m for n, m in self.named_modules() if n == "readout_gain_control"
            ]
        elif layer == "core":
            gc_modules = [
                m
                for n, m in self.named_modules()
                if n.startswith("core.gain_control") and isinstance(m, GainControlLayer)
            ]
        else:
            raise ValueError(f"layer must be 'readout' or 'core', got {layer!r}")
        h_params = [m.get_parameter("H") for m in gc_modules]
        for h in h_params:
            h.requires_grad = True
        return h_params
