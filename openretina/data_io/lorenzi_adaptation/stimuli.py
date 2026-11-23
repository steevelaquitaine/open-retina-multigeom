"""Stimulus loading for the flashed-natural-image adaptation dataset (Lorenzi et al.).

Each cell is stored as one ``<cell_id>.npz`` holding a per-cell (STA-centred) frame bank plus, for
every clip, the indices of its adapting and natural image. The contiguous train/test movies are
reconstructed here by repeating those frames (adapting image for ``N_ADAPT_FRAMES``, natural image
for ``N_NAT_FRAMES``), which keeps the on-disk dataset small and free of duplicated frames.
"""

import os

import numpy as np
from jaxtyping import Float

from openretina.data_io.base import MoviesTrainTestSplit, normalize_train_test_movies
from openretina.data_io.lorenzi_adaptation.constants import N_ADAPT_FRAMES, N_NAT_FRAMES
from openretina.utils.file_utils import get_local_file_path


def reconstruct_clips(
    frames: Float[np.ndarray, "n_frames height width"],
    adapt_idx: np.ndarray,
    nat_idx: np.ndarray,
) -> Float[np.ndarray, "channels time height width"]:
    """Expand per-clip (adapt, nat) frame indices into a contiguous, channel-first movie."""
    clips = [
        np.concatenate(
            [np.repeat(frames[a][None], N_ADAPT_FRAMES, axis=0), np.repeat(frames[n][None], N_NAT_FRAMES, axis=0)],
            axis=0,
        )
        for a, n in zip(adapt_idx, nat_idx, strict=True)
    ]
    movie = np.concatenate(clips, axis=0)  # (time, height, width)
    return movie[None].astype(np.float32)  # add channel dimension


def load_stimuli_for_session(npz_path: str | os.PathLike, normalize_stimuli: bool = False) -> MoviesTrainTestSplit:
    data = np.load(npz_path)
    frames = data["frames"]
    train_movie = reconstruct_clips(frames, data["train_adapt_idx"], data["train_nat_idx"])
    test_movie = reconstruct_clips(frames, data["test_adapt_idx"], data["test_nat_idx"])

    if normalize_stimuli:
        train_movie, test_movie, norm = normalize_train_test_movies(train_movie, test_movie)
    else:
        norm = {"norm_mean": None, "norm_std": None}

    return MoviesTrainTestSplit(
        train=train_movie,
        test=test_movie,
        stim_id="flashed_natural_images",
        norm_mean=norm["norm_mean"],
        norm_std=norm["norm_std"],
    )


def load_all_stimuli(
    base_data_path: str | os.PathLike,
    normalize_stimuli: bool = False,
) -> dict[str, MoviesTrainTestSplit]:
    """Load stimuli for every ``<cell_id>.npz`` in ``base_data_path`` (one session per cell)."""
    base_data_path = get_local_file_path(str(base_data_path))
    stimuli_all_sessions = {}
    for entry in sorted(os.scandir(os.fspath(base_data_path)), key=lambda e: e.name):
        if entry.name.endswith(".npz"):
            session = entry.name.removesuffix(".npz")
            print(f"Loading stimuli from {entry.path}")
            stimuli_all_sessions[session] = load_stimuli_for_session(entry.path, normalize_stimuli)
    return stimuli_all_sessions
