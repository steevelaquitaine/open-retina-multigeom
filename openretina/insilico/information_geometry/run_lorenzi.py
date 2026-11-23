"""Multi-scale Fisher G(x) maps for a Lorenzi adaptation cell (one checkpoint = one cell).

Treats the cell as a temporal cell model (flashed grey->image, response = mean over a
[+50,+350] ms window; gain control comes fitted inside the loaded weights). Builds a
single-cell-type mosaic by displacing the readout's spatial mask, trains a conditional
diffusion model on TinyImageNet@40px (standardised to the model's input stats), and
estimates G(x) at held-out (in-distribution) TinyImageNet test points.

Example:
    python -m openretina.insilico.information_geometry.run_lorenzi \
        --ckpt /path/0-exp7.ckpt --cell 0-exp7 --epochs 8 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from openretina.insilico.information_geometry.diffusion import NeuralConditionedUNet, build_scheduler
from openretina.insilico.information_geometry.estimator import conditional_pair_estimate
from openretina.insilico.information_geometry.fisher import fisher_eigendecomposition
from openretina.insilico.information_geometry.metrics import subspace_alignment
from openretina.insilico.information_geometry.retina_encoder import build_sliding_readout_mosaic_encoder
from openretina.insilico.information_geometry.run_example import _plot_cell_summary, _plot_spectrum
from openretina.insilico.information_geometry.stimuli import (
    NatStimFrameDataset,
    PadSpec,
    PairedFrameDataset,
    StimulusNormalizer,
    load_tinyimagenet_frames,
    precompute_responses,
)
from openretina.insilico.information_geometry.training import train_conditional_diffusion
from openretina.models.adaptation_gain_control import AdaptationGainControlModel

LOGGER = logging.getLogger("information_geometry.lorenzi")

CUT_FRAMES = 15
STIM_RATE_HZ = 40
PIXEL_SIZE_UM = 3.5 * 6  # 21 um/px (per the lorenzi display)


def load_lorenzi(ckpt_path: str, device: str):
    """Load a lorenzi adaptation checkpoint. Returns (model, session, resolution).

    Resolution is per-cell/experiment (read from the checkpoint's in_shape): exp7
    cells are 40x40, exp5 cells 30x30, etc.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck["pytorch-lightning_version"] = "2.0.0"  # patch missing key for load_from_checkpoint
    tf = tempfile.mktemp(suffix=".ckpt")
    torch.save(ck, tf)
    # torch >=2.6 defaults weights_only=True, which rejects the checkpoint's custom openretina
    # classes; force weights_only=False for the load (trusted, own checkpoint).
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    try:
        model = AdaptationGainControlModel.load_from_checkpoint(tf, map_location=device).eval()
    finally:
        torch.load = _orig_load
    session = list(model.readout.keys())[0]
    res = int(ck["hyper_parameters"]["in_shape"][-1])  # native spatial resolution for this cell
    model.data_info = {
        "model_cut_frames": CUT_FRAMES, "stimulus_rate_hz": STIM_RATE_HZ,
        "n_neurons_dict": {session: 1}, "input_shape": (1, res, res),
        "retina_pixel_size_um": PIXEL_SIZE_UM,
        "microns_per_px": float(ck["hyper_parameters"].get("microns_per_px", PIXEL_SIZE_UM)),
    }
    return model, session, res


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-rows", type=int, default=8)
    p.add_argument("--n-cols", type=int, default=8)
    p.add_argument("--mosaic-layout", choices=["grid", "hex"], default="grid",
                   help="square grid (n_rows x n_cols) or hexagonal packing (uses --nn-spacing-um)")
    p.add_argument("--nn-spacing-um", type=float, default=200.0,
                   help="hex nearest-neighbour RF-centre spacing in microns (biological mosaic spacing)")
    p.add_argument("--big-res", type=int, default=0,
                   help="info-map canvas (px); model is convolved across it. 0 => 3x native res")
    p.add_argument("--stim-source", choices=["tinyimagenet", "goldin"], default="tinyimagenet",
                   help="natural-image prior for training + estimation (goldin = wide-field scenes)")
    p.add_argument("--train-images", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--mc-samples", type=int, default=150)
    p.add_argument("--mc-batch", type=int, default=25)
    p.add_argument("--mc-step", type=int, default=40)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-eig-plot", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.22)
    p.add_argument("--noise-model", choices=["gaussian", "poisson"], default="gaussian",
                   help="response noise model. 'poisson' samples spike counts on the encoder's "
                        "rate scale (the models were fit with a Poisson loss, so the output is a rate)")
    p.add_argument("--poisson-gain", type=float, default=None,
                   help="counts are drawn as Poisson(gain*lambda)/gain. Default: the response-window "
                        "length in frames, i.e. model the window average of per-frame Poisson counts. "
                        "Sets the Poisson SNR; gain=1 treats the response as a single Poisson draw")
    p.add_argument("--seed", type=int, default=None,
                   help="seed torch/numpy RNG (diffusion init, shuffling, MC noise) for seed-floor runs")
    p.add_argument("--suffix", default="",
                   help="appended to output filenames (e.g. _s1) so multiple seeds coexist in one dir")
    p.add_argument("--gc-zero", action="store_true",
                   help="ablate gain control: set subunit gain H=0 (NOTE: readout channel_norm is NOT "
                        "recalibrated, so compare only against maps that use the same BN)")
    p.add_argument("--load-ckpt", default=None, help="skip training, load a saved diffusion checkpoint")
    p.add_argument("--finetune", default=None,
                   help="warm-start the UNet from this diffusion.pt then keep training (e.g. adapt the "
                        "TinyImageNet prior to Goldin with few images)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    # Blackwell (sm_100+, e.g. GB10 sm_121) lacks efficient/flash SDPA kernels in current torch:
    # they silently return zeros and G collapses. Force the math backend on those archs only
    # (leaves 4090/A100/H100 on their fast kernels).
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        LOGGER.info("SDPA: forced math backend for sm_%d%d", *torch.cuda.get_device_capability(device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        LOGGER.info("seeded RNG with %d (suffix=%r)", args.seed, sfx)

    model, session, res = load_lorenzi(args.ckpt, args.device)
    LOGGER.info("loaded lorenzi cell %s (session=%s)", args.cell, session)
    if args.gc_zero:
        gc = model.core.gain_control
        gcs = list(gc) if hasattr(gc, "__iter__") else [gc]
        for g in gcs:
            if hasattr(g, "H"):
                g.H.data.zero_()
        LOGGER.info("gain control ABLATED: subunit H=0 (channel_norm NOT recalibrated)")

    big_res = args.big_res or 3 * res  # convolve the native-res model across a larger field
    if args.stim_source == "goldin":
        from openretina.insilico.information_geometry.stimuli import load_goldin_frames_matched
        frames = load_goldin_frames_matched(args.train_images, resize=big_res, split="train")
    else:
        frames = load_tinyimagenet_frames(args.train_images, resize=big_res)
    normalizer = StimulusNormalizer.from_frames(frames)
    pad = PadSpec.to_multiple(big_res, big_res, 4)
    nn_spacing_px = None
    if args.mosaic_layout == "hex":
        stride = getattr(model.core.subunit, "stride", 1)
        stride = stride[0] if isinstance(stride, (tuple, list)) else stride
        um_per_corepx = float(stride) * model.data_info["microns_per_px"]
        nn_spacing_px = args.nn_spacing_um / um_per_corepx
        LOGGER.info("hex mosaic: NN=%.0fum = %.2f core-px (%.1fum/core-px)",
                    args.nn_spacing_um, nn_spacing_px, um_per_corepx)
    encoder = build_sliding_readout_mosaic_encoder(
        model, session, 0, normalizer, pad,
        n_rows=args.n_rows, n_cols=args.n_cols, device=args.device,
        layout=args.mosaic_layout, nn_spacing_px=nn_spacing_px,
    ).to(device)
    LOGGER.info("native res=%d, big canvas=%dx%d, mosaic N=%d, window=[%d,%d)", res, pad.hp, pad.wp,
                encoder.n_neurons, encoder.w0, encoder.w1)

    train_ds = NatStimFrameDataset(frames, normalizer, pad)
    LOGGER.info("precomputing responses for response standardisation ...")
    raw = precompute_responses(encoder, train_ds, batch_size=16, device=device)  # [N, K]
    encoder.set_response_normalization(raw.mean(0), raw.std(0))
    # Poisson draws counts from the rate, so it needs the RAW responses; Gaussian adds noise
    # on the standardised scale.
    poisson = args.noise_model == "poisson"
    poisson_gain = float(args.poisson_gain if args.poisson_gain is not None else (encoder.w1 - encoder.w0))
    responses = raw if poisson else (raw - raw.mean(0)) / raw.std(0).clamp(min=1e-6)
    if poisson:
        LOGGER.info("noise model=poisson, gain=%.1f (window frames); raw rate: min=%.4f med=%.4f max=%.4f",
                    poisson_gain, raw.min(), raw.median(), raw.max())

    unet = NeuralConditionedUNet(
        in_channels=1, sample_size=(pad.hp, pad.wp), n_neurons=encoder.n_neurons,
        observation_noise_sigma=args.sigma,
    ).to(device)
    scheduler = build_scheduler()

    if args.load_ckpt:
        unet.load_state_dict(torch.load(args.load_ckpt, map_location=device)["unet"])
    else:
        if args.finetune:  # warm-start from a prior diffusion.pt (e.g. the TinyImageNet model), then keep training
            unet.load_state_dict(torch.load(args.finetune, map_location=device)["unet"])
            LOGGER.info("fine-tuning: warm-started UNet from %s", args.finetune)
        LOGGER.info("training diffusion on %d %s@%dpx frames ...", len(train_ds), args.stim_source, big_res)
        paired = PairedFrameDataset(train_ds, responses)
        history = train_conditional_diffusion(
            unet, encoder, paired, scheduler,
            epochs=args.epochs, batch_size=args.batch_size, sigma=args.sigma, device=device,
            noise_model=args.noise_model, poisson_gain=poisson_gain,
        )
        torch.save({"unet": unet.state_dict(), "cell": args.cell,
                    "resp_mean": raw.mean(0), "resp_std": raw.std(0),
                    "noise_model": args.noise_model, "poisson_gain": poisson_gain},
                   out_dir / f"diffusion{sfx}.pt")
        LOGGER.info("saved diffusion.pt (final loss %.4f)", history["loss"][-1])

    # ---- estimate at held-out (in-distribution) test points ----
    if args.stim_source == "goldin":
        from openretina.insilico.information_geometry.stimuli import load_goldin_frames_matched
        test_frames = load_goldin_frames_matched(args.n_test, resize=big_res, split="test", augment=False)
    else:
        test_frames = load_tinyimagenet_frames(args.n_test, resize=big_res, offset=50000)
    test_ds = NatStimFrameDataset(test_frames, normalizer, pad)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    timesteps = list(range(args.mc_step, 1000 - args.mc_step + 1, args.mc_step))

    alignments, results = [], []
    for i in range(len(test_ds)):
        x = test_ds[i].unsqueeze(0).to(device)
        g = conditional_pair_estimate(x, encoder, unet, alphas_cumprod, timesteps,
                                      n_samples=args.mc_samples, batch_size=args.mc_batch,
                                      sigma=args.sigma, top_k=args.top_k, device=device,
                                      noise_model=args.noise_model, poisson_gain=poisson_gain)
        j = fisher_eigendecomposition(x, encoder, sigma=args.sigma, top_k=args.top_k, device=device,
                                      noise_model=args.noise_model, poisson_gain=poisson_gain)
        # J's rank is the mosaic size, so it can hold fewer modes than G; clamp the
        # diagnostic to whichever is smaller rather than failing the whole run.
        k_al = min(args.top_k, g.eigenvectors.shape[0], j.eigenvectors.shape[0])
        align = subspace_alignment(g.eigenvectors, j.eigenvectors, top_k=k_al)
        alignments.append(align)
        results.append((i, x.cpu(), g, j, align))
        last_g, last_j = g, j
        LOGGER.info("test %d: G/J align = %.3f", i, align)

    mean_align = float(np.mean(alignments))
    tag = f"lorenzi_{args.cell}"
    _plot_cell_summary(out_dir / f"{tag}{sfx}_G.png", results, pad, 1, args.n_eig_plot, mean_align, tag)
    _plot_spectrum(out_dir / f"spectrum{sfx}.png", last_g, last_j)
    torch.save({"cell": args.cell, "canvas": (pad.hp, pad.wp), "channels": 1,
                "noise_model": args.noise_model, "poisson_gain": poisson_gain,
                "G_eigvals": [r[2].eigenvalues for r in results],
                "G_eigvecs": [r[2].eigenvectors for r in results],
                "J_eigvals": [r[3].eigenvalues for r in results],
                "alignments": alignments}, out_dir / f"eigenpairs{sfx}.pt")
    LOGGER.info("done: %s (mean G-vs-J align=%.3f)", out_dir, mean_align)


if __name__ == "__main__":
    main()
