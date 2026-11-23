"""End-to-end example: multi-scale Fisher geometry of an open-retina model.

Loads a retina model, builds a synthetic-mosaic image encoder, trains a
conditional diffusion model on the model's natural stimulus frames, estimates
G(x) (multi-scale Fisher) and J(x) (classical Fisher) on held-out frames, and
saves leading-eigenvector figures for visual inspection.

Example
-------
    python -m openretina.insilico.information_geometry.run_example \
        --stimulus-pkl /path/to/rgc_natstim_18x16_joint_normalized_2024-01-11.pkl \
        --epochs 60 --n-test 6 --out-dir /tmp/ig_run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from openretina.insilico.information_geometry.diffusion import (
    NeuralConditionedUNet,
    build_scheduler,
)
from openretina.insilico.information_geometry.estimator import conditional_pair_estimate
from openretina.insilico.information_geometry.fisher import fisher_eigendecomposition
from openretina.insilico.information_geometry.metrics import subspace_alignment
from openretina.insilico.information_geometry.retina_encoder import (
    build_mosaic_encoder,
    build_session_encoder,
    measure_pixel_rf_sigma,
)
from openretina.insilico.information_geometry.stimuli import (
    NatStimFrameDataset,
    PadSpec,
    PairedFrameDataset,
    StimulusNormalizer,
    load_goldin_frames,
    load_natstim_frames,
    load_tinyimagenet_frames,
    precompute_responses,
)
from openretina.insilico.information_geometry.training import train_conditional_diffusion
from openretina.models import load_core_readout_from_remote

LOGGER = logging.getLogger("information_geometry.run")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="goldin_2022_mouse")
    p.add_argument("--population", choices=["session", "mosaic"], default="mosaic",
                   help="synthetic single-cell-type RF mosaic (correct for goldin), or the "
                        "raw all-neuron session set (mixed cell types; comparison only)")
    p.add_argument("--stimulus-pkl", default=None,
                   help="rgc_natstim_*.pkl (temporal mouse models); goldin uses its own mirror")
    p.add_argument("--goldin-specie", default="mouse", choices=["mouse", "axolotl"])
    p.add_argument("--train-tinyimagenet", type=int, default=0,
                   help="if >0, train the diffusion model on N TinyImageNet images "
                        "(standardised to goldin stats) instead of the ~3k goldin frames; "
                        "estimation still uses the goldin test images")
    p.add_argument("--session", default=None, help="session id (default: first)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="/tmp/ig_run")
    # mosaic (population=mosaic only)
    p.add_argument("--base-neuron", type=int, default=16, help="index of the cell type to tile")
    p.add_argument("--n-rows", type=int, default=8)
    p.add_argument("--n-cols", type=int, default=8)
    p.add_argument("--spacing-sigma", type=float, default=2.0,
                   help="mosaic centre spacing in units of the cell's RF sigma")
    # diffusion training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--precompute-batch", type=int, default=16,
                   help="frame batch for response precompute (small: mosaic expands x N cells)")
    p.add_argument("--block-channels", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--pad-multiple", type=int, default=4)
    p.add_argument("--resolution", type=int, default=None,
                   help="run the whole pipeline at this square resolution (e.g. 64 for native "
                        "TinyImageNet, no upsampling; goldin frames resized to match). Default: native.")
    # estimation
    p.add_argument("--n-test", type=int, default=6)
    p.add_argument("--test-indices", default=None,
                   help="comma-separated goldin test-frame indices to estimate on (overrides --n-test); "
                        "e.g. 0,1,8,19,24,5 to curate background+object frames")
    p.add_argument("--test-source", choices=["goldin", "tinyimagenet"], default="goldin",
                   help="estimate G at goldin test frames, or at held-out TinyImageNet frames "
                        "(in-distribution for the TinyImageNet diffusion prior)")
    p.add_argument("--mc-samples", type=int, default=200)
    p.add_argument("--mc-batch", type=int, default=25)
    p.add_argument("--mc-step", type=int, default=40)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--sigma", type=float, default=0.22)
    p.add_argument("--n-eig-plot", type=int, default=4)
    p.add_argument("--load-ckpt", default=None, help="skip training, load a saved unet checkpoint")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for training+estimation; also appended to output filename tag")
    return p.parse_args()


def _resize_frames(frames, res):
    """Resize [N, C, H, W] frames to (res, res) if needed (bilinear)."""
    if res is None or frames.shape[-1] == res:
        return frames
    return torch.nn.functional.interpolate(frames, size=(res, res), mode="bilinear", align_corners=False)


def _load_frames(args, split):
    if args.stimulus_pkl:
        frames = load_natstim_frames(args.stimulus_pkl, split)
    elif args.model.startswith("goldin"):
        frames = load_goldin_frames(args.goldin_specie, split)
    else:
        raise ValueError("provide --stimulus-pkl for non-goldin models")
    return _resize_frames(frames, args.resolution)


def build_all(args):
    device = torch.device(args.device)
    model = load_core_readout_from_remote(args.model, args.device)
    session = args.session or list(model.data_info["n_neurons_dict"])[0]

    if args.train_tinyimagenet > 0:
        res = args.resolution or 108
        frames = load_tinyimagenet_frames(args.train_tinyimagenet, resize=res)  # native at `res`, goldin stats
        LOGGER.info("training frames: %d TinyImageNet images @ %dpx (standardised to goldin stats)",
                    frames.shape[0], res)
    else:
        frames = _load_frames(args, "train")  # [N, C, H, W]
    c, h, w = frames.shape[1:]
    normalizer = StimulusNormalizer.from_frames(frames)
    pad = PadSpec.to_multiple(h, w, args.pad_multiple)

    if args.population == "mosaic":
        rf_sigma_px = measure_pixel_rf_sigma(model, session, args.base_neuron, frames, device=device)
        LOGGER.info("base cell %d RF sigma = %.2f px", args.base_neuron, rf_sigma_px)
        encoder = build_mosaic_encoder(
            model, session, args.base_neuron, normalizer, pad, rf_sigma_px,
            n_rows=args.n_rows, n_cols=args.n_cols, spacing_sigma=args.spacing_sigma,
        ).to(device)
    else:
        encoder = build_session_encoder(model, session, normalizer, pad).to(device)
    LOGGER.info("encoder: population=%s session=%s N=%d canvas=%dx%d channels=%d window=[%d,%d)",
                args.population, session, encoder.n_neurons, pad.hp, pad.wp, c, encoder.w0, encoder.w1)
    return device, model, session, frames, normalizer, pad, encoder, c


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, model, session, frames, normalizer, pad, encoder, c = build_all(args)
    train_ds = NatStimFrameDataset(frames, normalizer, pad)

    unet = NeuralConditionedUNet(
        in_channels=c, sample_size=(pad.hp, pad.wp), n_neurons=encoder.n_neurons,
        block_channels=tuple(args.block_channels), observation_noise_sigma=args.sigma,
    ).to(device)
    scheduler = build_scheduler()
    ckpt_path = out_dir / "diffusion.pt"

    if args.load_ckpt:
        LOGGER.info("loading unet checkpoint from %s", args.load_ckpt)
        unet.load_state_dict(torch.load(args.load_ckpt, map_location=device)["unet"])
    else:
        LOGGER.info("precomputing responses for %d frames ...", len(train_ds))
        responses = precompute_responses(
            encoder, train_ds, batch_size=args.precompute_batch, device=device
        )
        paired = PairedFrameDataset(train_ds, responses)
        LOGGER.info("training diffusion model for %d epochs ...", args.epochs)
        history = train_conditional_diffusion(
            unet, encoder, paired, scheduler,
            epochs=args.epochs, batch_size=args.batch_size, sigma=args.sigma, device=device,
        )
        torch.save(
            {"unet": unet.state_dict(), "normalizer": normalizer.state_dict(),
             "pad": vars(pad), "history": history,
             "mosaic": {"session": session, "base_neuron": args.base_neuron,
                        "n_rows": args.n_rows, "n_cols": args.n_cols}},
            ckpt_path,
        )
        LOGGER.info("saved checkpoint to %s (final loss %.4f)", ckpt_path, history["loss"][-1])

    # ---- estimate G(x) and J(x) on held-out test frames ----
    if args.test_source == "tinyimagenet":
        # held-out TinyImageNet frames (offset past the training set) -> in-distribution
        # for the TinyImageNet diffusion prior
        res = args.resolution or 108
        test_frames = load_tinyimagenet_frames(args.n_test, resize=res, offset=50000)
        test_ds = NatStimFrameDataset(test_frames, normalizer, pad)
        indices = list(range(min(args.n_test, len(test_ds))))
        LOGGER.info("test source: %d held-out TinyImageNet frames (in-distribution)", len(indices))
    else:
        test_frames = _load_frames(args, "test")
        test_ds = NatStimFrameDataset(test_frames, normalizer, pad)
        if args.test_indices:
            indices = [int(v) for v in args.test_indices.split(",")]
        else:
            indices = list(range(min(args.n_test, len(test_ds))))
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    timesteps = list(range(args.mc_step, 1000 - args.mc_step + 1, args.mc_step))

    alignments, results = [], []
    for idx in indices:
        x = test_ds[idx].unsqueeze(0).to(device)
        g = conditional_pair_estimate(
            x, encoder, unet, alphas_cumprod, timesteps,
            n_samples=args.mc_samples, batch_size=args.mc_batch,
            sigma=args.sigma, top_k=args.top_k, device=device,
        )
        j = fisher_eigendecomposition(x, encoder, sigma=args.sigma, top_k=args.top_k, device=device)
        align = subspace_alignment(g.eigenvectors, j.eigenvectors, top_k=args.top_k)
        alignments.append(align)
        results.append((idx, x.cpu(), g, j, align))
        last_g, last_j = g, j
        LOGGER.info("test idx %d: G/J subspace alignment (k=%d) = %.3f", idx, args.top_k, align)

    mean_align = float(np.mean(alignments))
    LOGGER.info("mean G-vs-J subspace alignment across %d images: %.3f", len(alignments), mean_align)
    tag = f"{args.model}_{args.population}_cell{args.base_neuron}"
    if args.seed is not None:
        tag += f"_seed{args.seed}"
    _plot_cell_summary(out_dir / f"{tag}_G.png", results, pad, c, args.n_eig_plot, mean_align, tag)
    _plot_spectrum(out_dir / "spectrum.png", last_g, last_j)

    # Save raw eigenpairs so plots can be redrawn and cross-cell alignment
    # quantified without re-running the estimator.
    torch.save({
        "model": args.model, "population": args.population, "base_neuron": args.base_neuron,
        "canvas": (pad.hp, pad.wp), "channels": c,
        "test_indices": [r[0] for r in results],
        "images": [r[1] for r in results],
        "G_eigvals": [r[2].eigenvalues for r in results],
        "G_eigvecs": [r[2].eigenvectors for r in results],
        "J_eigvals": [r[3].eigenvalues for r in results],
        "alignments": alignments,
    }, out_dir / "eigenpairs.pt")
    LOGGER.info("done. figures + eigenpairs in %s", out_dir)


def _to_grid(eigvec_row: torch.Tensor, pad: PadSpec, c: int) -> np.ndarray:
    """[P] eigenvector -> [C, H, W] cropped to native stimulus size."""
    v = eigvec_row.reshape(c, pad.hp, pad.wp)
    v = pad.crop(v)  # [C, H, W]
    return v.cpu().numpy()


def _plot_cell_summary(path, results, pad, c, n_eig, mean_align, tag):
    """One figure per cell: each test image is a row (stimulus + top-n_eig G
    eigenvectors). Only the multi-scale metric G(x) is shown as maps (J is local
    RF-scale by construction); J is still summarised via the subspace-alignment
    numbers and the spectrum figure. Channel 0 is shown for multi-channel models."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ch = 0
    n_img = len(results)
    fig, axes = plt.subplots(n_img, n_eig + 1, figsize=(2.0 * (n_eig + 1), 2.0 * n_img),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, (idx, x, g, _j, align) in enumerate(results):
        x_img = pad.crop(x[0]).numpy()  # [C, H, W]
        # informativeness of each eigenvector = eigenvalue share of sum (= share of local MI)
        share = (g.eigenvalues / g.eigenvalues.sum().clamp(min=1e-12)).tolist()
        axes[r, 0].imshow(x_img[ch], cmap="gray")
        axes[r, 0].set_ylabel(f"idx{idx}\nalign={align:.2f}", fontsize=8)
        if r == 0:
            axes[r, 0].set_title("stimulus", fontsize=8)
        for k in range(n_eig):
            vg = _to_grid(g.eigenvectors[k], pad, c)[ch]
            vmax = np.abs(vg).max() + 1e-9
            axes[r, k + 1].imshow(vg / vmax, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[r, k + 1].text(0.04, 0.95, f"{share[k] * 100:.0f}%", transform=axes[r, k + 1].transAxes,
                                fontsize=8, va="top", ha="left", color="black",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
            if r == 0:
                axes[r, k + 1].set_title(f"e{k+1}", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    # Shared colorbar: eigenvectors are sign/scale-free, so each panel is
    # normalised to its own max; the bar shows blue=- / white=0 / red=+.
    sm = ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["−", "0", "+"])
    cbar.set_label("eigenvector weight (per-panel normalised)", fontsize=8)
    fig.suptitle(f"{tag}: G(x) leading eigenvectors per image  (mean G-vs-J align={mean_align:.2f})",
                 fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_spectrum(path, g, j):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(g.eigenvalues.cpu().numpy() / g.eigenvalues.max().item(), "o-", label="G (multi-scale)")
    ax.semilogy(j.eigenvalues.cpu().numpy() / j.eigenvalues.max().item(), "s-", label="J (Fisher)")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("normalised eigenvalue")
    ax.legend(); ax.set_title("Eigenvalue spectra (last test image)")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
