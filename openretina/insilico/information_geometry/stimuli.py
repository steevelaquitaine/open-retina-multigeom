"""Stimulus handling for the information-geometry pipeline.

Three concerns live here, all shared by training, estimation and the Fisher
baseline so the conventions never drift:

* ``StimulusNormalizer`` -- a per-channel affine map between *model space*
  (the units the retina model was trained on, e.g. the joint-normalised
  ``rgc_natstim`` frames) and *diffusion space* (roughly ``[-1, 1]``, the range
  a DDPM variance-preserving schedule expects). The diffusion model and G(x)
  live in diffusion space; the retina encoder converts back to model space
  internally.
* ``PadSpec`` -- centre pad/crop between the native stimulus size ``(H, W)`` and
  a UNet-friendly padded size ``(Hp, Wp)`` (both divisible by the UNet's
  downsampling factor). The diffusion model operates on the padded canvas; the
  encoder crops back before feeding the retina model, so padding pixels carry
  (near-)zero metric weight.
* ``NatStimFrameDataset`` -- single natural frames sampled from an open-retina
  ``rgc_natstim_*.pkl`` stimulus file, in diffusion space and padded.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class StimulusNormalizer:
    """Per-channel affine map model-space <-> diffusion-space.

    ``x_model = x_diff * scale + mean`` and the inverse. ``mean``/``scale`` are
    ``[C]`` tensors. Diffusion space is chosen so that most of the stimulus mass
    falls within ``[-1, 1]``.
    """

    mean: torch.Tensor  # [C]
    scale: torch.Tensor  # [C]

    def to_model(self, x_diff: torch.Tensor) -> torch.Tensor:
        """Diffusion space -> model space. ``x_diff``: ``[..., C, H, W]``."""
        m = self.mean.to(x_diff.device).view(-1, 1, 1)
        s = self.scale.to(x_diff.device).view(-1, 1, 1)
        return x_diff * s + m

    def to_diffusion(self, x_model: torch.Tensor) -> torch.Tensor:
        """Model space -> diffusion space. ``x_model``: ``[..., C, H, W]``."""
        m = self.mean.to(x_model.device).view(-1, 1, 1)
        s = self.scale.to(x_model.device).view(-1, 1, 1)
        return (x_model - m) / s

    @classmethod
    def from_frames(cls, frames: torch.Tensor, coverage: float = 3.0) -> "StimulusNormalizer":
        """Fit from ``frames`` ``[N, C, H, W]`` (model space).

        ``scale = coverage * std`` per channel, so roughly ``+/- coverage``
        standard deviations map into ``[-1, 1]``.
        """
        c = frames.shape[1]
        flat = frames.transpose(0, 1).reshape(c, -1)
        mean = flat.mean(dim=1)
        scale = coverage * flat.std(dim=1).clamp(min=1e-6)
        return cls(mean=mean, scale=scale)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "scale": self.scale.cpu()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "StimulusNormalizer":
        return cls(mean=d["mean"].clone(), scale=d["scale"].clone())


@dataclass
class PadSpec:
    """Centre pad/crop between native ``(h, w)`` and padded ``(hp, wp)``."""

    h: int
    w: int
    hp: int
    wp: int

    @classmethod
    def to_multiple(cls, h: int, w: int, multiple: int) -> "PadSpec":
        """Smallest ``(hp, wp) >= (h, w)`` divisible by ``multiple``."""
        hp = ((h + multiple - 1) // multiple) * multiple
        wp = ((w + multiple - 1) // multiple) * multiple
        return cls(h=h, w=w, hp=hp, wp=wp)

    @property
    def pads(self) -> tuple[int, int, int, int]:
        """``(left, right, top, bottom)`` for ``F.pad``."""
        dh, dw = self.hp - self.h, self.wp - self.w
        top, left = dh // 2, dw // 2
        return left, dw - left, top, dh - top

    def pad(self, x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        """``[..., h, w] -> [..., hp, wp]`` (constant pad, default grey=0)."""
        return F.pad(x, self.pads, mode="constant", value=value)

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., hp, wp] -> [..., h, w]`` (inverse of :meth:`pad`)."""
        left, _, top, _ = self.pads
        return x[..., top : top + self.h, left : left + self.w]


def load_natstim_frames(
    pkl_path: str | Path,
    split: str = "train",
) -> torch.Tensor:
    """Load ``rgc_natstim_*.pkl`` frames as ``[N, C, H, W]`` float32 (model space).

    The file stores ``{"train": (C, N, H, W), "test": (C, N, H, W), ...}``.
    Used by temporal mouse models (e.g. ``hoefling_2024``).
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    arr = np.asarray(obj[split], dtype=np.float32)  # (C, N, H, W)
    frames = torch.from_numpy(arr).permute(1, 0, 2, 3).contiguous()  # (N, C, H, W)
    return frames


def load_goldin_frames(
    specie: str = "mouse",
    split: str = "train",
    base_data_path: str = (
        "https://huggingface.co/datasets/open-retina/open-retina/resolve/main/marre_lab/goldin_2022"
    ),
) -> torch.Tensor:
    """Load Goldin-2022 natural-scene frames as ``[N, C, H, W]`` float32 (model space).

    Downloads / caches the open-retina mirror of the dataset. Frames are the
    same normalised natural-scene images the ``goldin_2022`` model was trained on.
    """
    from openretina.data_io.goldin_2022.stimuli import load_all_stimuli

    sessions = load_all_stimuli(base_data_path, specie=specie, normalize_stimuli=True)
    split_obj = next(iter(sessions.values()))  # (C, T, H, W) arrays
    arr = split_obj.train if split == "train" else split_obj.test_dict["test"]
    arr = np.asarray(arr, dtype=np.float32)  # (C, N, H, W)
    return torch.from_numpy(arr).permute(1, 0, 2, 3).contiguous()  # (N, C, H, W)


def load_tinyimagenet_frames(
    n_images: int = 20000,
    resize: int = 108,
    seed: int = 0,
    clip: tuple[float, float] = (-2.4, 2.5),
    root: str | Path | None = None,
    offset: int = 0,
) -> torch.Tensor:
    """Load TinyImageNet images as ``[N, 1, resize, resize]`` frames matched to the
    goldin model's input statistics.

    Images are grayscaled, resized, then **standardised to mean 0 / std 1** (the
    goldin natural-scene stimuli are z-scored to the same range) and clipped to
    ``clip``, so the goldin encoder receives in-distribution inputs. Provides a
    large, diverse natural-image prior for training the diffusion model (the
    encoder predicts responses to any natural image, so training need not be
    restricted to the ~3k goldin stimulus frames).
    """
    import random as _random

    from openretina.utils.file_utils import get_cache_directory

    root = Path(root) if root else Path(get_cache_directory()) / "data" / "tiny-imagenet-200" / "train"
    paths = sorted(root.glob("*/images/*.JPEG"))
    if not paths:
        raise FileNotFoundError(f"No TinyImageNet JPEGs under {root}")
    _random.Random(seed).shuffle(paths)
    paths = paths[offset : offset + n_images]  # offset lets test frames be held out from training

    # Grayscale (ITU-R 601 luma) -> resize (bilinear) -> [0,1] tensor. Done with PIL so the
    # pipeline carries no torchvision dependency (some ARM/CUDA builds ship a broken torchvision).
    def _load(p) -> torch.Tensor:
        img = Image.open(p).convert("L").resize((resize, resize), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)  # [1, H, W]

    frames = torch.stack([_load(p) for p in paths])  # [N, 1, H, W]
    frames = (frames - frames.mean()) / frames.std().clamp(min=1e-6)  # -> mean 0, std 1
    return frames.clamp(clip[0], clip[1])


def load_goldin_frames_matched(
    n_images: int,
    resize: int = 120,
    split: str = "train",
    seed: int = 0,
    clip: tuple[float, float] = (-2.4, 2.5),
    augment: bool = True,
) -> torch.Tensor:
    """Goldin natural-scene frames as ``[n, 1, resize, resize]``, standardised like the model input.

    A wide-field-scene prior counterpart to :func:`load_tinyimagenet_frames`. Goldin ships only
    ~3160 train frames, so (for training) we augment with horizontal flips + small wrap shifts up to
    ``n_images`` to reach a comparable diffusion-training count. ``augment=False`` (use for the test
    split) returns the real frames untouched.
    """
    frames = load_goldin_frames("mouse", split)  # [N, 1, 108, 108], ~mean0/std1
    frames = F.interpolate(frames, size=(resize, resize), mode="bilinear", align_corners=False)
    if augment and frames.shape[0] < n_images:
        g = torch.Generator().manual_seed(seed)
        base = torch.cat([frames, torch.flip(frames, dims=[-1])], 0)  # + horizontal flip (scene-preserving)
        out = [base]
        while sum(x.shape[0] for x in out) < n_images:
            sx = int(torch.randint(-12, 13, (1,), generator=g))
            sy = int(torch.randint(-12, 13, (1,), generator=g))
            out.append(torch.roll(base, shifts=(sy, sx), dims=(-2, -1)))
        frames = torch.cat(out, 0)[:n_images]
    else:
        frames = frames[:n_images]
    frames = (frames - frames.mean()) / frames.std().clamp(min=1e-6)  # -> mean 0, std 1
    return frames.clamp(clip[0], clip[1])


class NatStimFrameDataset(Dataset):
    """Single natural frames in diffusion space, padded to the UNet canvas.

    Parameters
    ----------
    frames:
        ``[N, C, H, W]`` model-space frames (e.g. from :func:`load_natstim_frames`).
    normalizer:
        model-space <-> diffusion-space map.
    pad:
        centre pad to the UNet canvas.
    """

    def __init__(
        self,
        frames: torch.Tensor,
        normalizer: StimulusNormalizer,
        pad: PadSpec,
    ):
        self.frames = frames
        self.normalizer = normalizer
        self.pad = pad

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        x_model = self.frames[idx]
        x_diff = self.normalizer.to_diffusion(x_model)
        return self.pad.pad(x_diff, value=0.0)  # grey (model mean) -> 0 in diffusion space


class PairedFrameDataset(Dataset):
    """A :class:`NatStimFrameDataset` paired with precomputed responses ``[N, K]``."""

    def __init__(self, frame_dataset: NatStimFrameDataset, responses: torch.Tensor):
        if len(frame_dataset) != responses.shape[0]:
            raise ValueError(
                f"frames ({len(frame_dataset)}) != responses ({responses.shape[0]})"
            )
        self.frame_dataset = frame_dataset
        self.responses = responses

    def __len__(self) -> int:
        return len(self.frame_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.frame_dataset[idx], self.responses[idx]


@torch.no_grad()
def precompute_responses(
    encoder,
    frame_dataset: NatStimFrameDataset,
    batch_size: int = 256,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Evaluate the (frozen) encoder over all frames -> ``[N, K]`` responses."""
    device = torch.device(device)
    encoder = encoder.to(device).eval()
    from torch.utils.data import DataLoader

    loader = DataLoader(frame_dataset, batch_size=batch_size, shuffle=False)
    out = [encoder(x.to(device)).cpu() for x in loader]
    return torch.cat(out, dim=0)
