"""Response loading for the flashed-natural-image adaptation dataset (Lorenzi et al.).

Each cell's ``.npz`` stores the ganglion-cell PSTH of the whole adaptor + probe sequence per clip
(``N_PSTH_BINS`` bins from adaptor onset; for the test clips, per repeat). Responses are reconstructed
frame-aligned with the reconstructed movies: the clip drops the adaptor's first frame, so frame ``f`` of
a clip is aligned with PSTH bin ``f + 1``. The training loss only uses the last ``N_NAT_FRAMES`` frames
(the natural-image window). The test response is averaged over repeats, with the per-repeat responses
kept in ``test_by_trial`` for oracle / reliability metrics.
"""

import os

import numpy as np
from jaxtyping import Float

from openretina.data_io.base import ResponsesTrainTestSplit
from openretina.data_io.lorenzi_adaptation.constants import CLIP_LENGTH, N_PSTH_BINS
from openretina.data_io.lorenzi_adaptation.metadata import metadata_for_session
from openretina.utils.file_utils import get_local_file_path


def reconstruct_response(psth_per_clip: Float[np.ndarray, "n_clips n_bins"]) -> Float[np.ndarray, " time"]:
    """Align each clip's full-sequence PSTH with the clip frames (bins 1..31) and concatenate over clips."""
    if psth_per_clip.shape[1] != N_PSTH_BINS:
        raise ValueError(
            f"Expected {N_PSTH_BINS} PSTH bins per clip (from adaptor onset), got {psth_per_clip.shape[1]}; "
            "re-export the data with the current export script."
        )
    return psth_per_clip[:, -CLIP_LENGTH:].reshape(-1)


def load_responses_for_session(npz_path: str | os.PathLike, fr_normalization: float = 1.0) -> ResponsesTrainTestSplit:
    data = np.load(npz_path)

    train = reconstruct_response(data["train_psth"])[None] / fr_normalization  # (1 neuron, time)

    by_trial = data["test_psth_by_trial"]  # (n_clips, n_repeats, n_bins)
    n_repeats = by_trial.shape[1]
    test_by_trial = np.stack([reconstruct_response(by_trial[:, r, :]) for r in range(n_repeats)])  # (n_repeats, time)
    test = test_by_trial.mean(axis=0, keepdims=True) / fr_normalization  # (1 neuron, time)
    test_by_trial = test_by_trial[:, None, :] / fr_normalization  # (n_repeats, 1 neuron, time)

    session = os.path.splitext(os.path.basename(os.fspath(npz_path)))[0]
    return ResponsesTrainTestSplit(
        train=train,
        test=test,
        test_by_trial=test_by_trial,
        stim_id="flashed_natural_images",
        session_kwargs=metadata_for_session(session),
    )


def load_all_responses(
    base_data_path: str | os.PathLike,
    fr_normalization: float = 1.0,
) -> dict[str, ResponsesTrainTestSplit]:
    """Load responses for every ``<cell_id>.npz`` in ``base_data_path`` (one session per cell)."""
    base_data_path = get_local_file_path(str(base_data_path))
    responses_all_sessions = {}
    for entry in sorted(os.scandir(os.fspath(base_data_path)), key=lambda e: e.name):
        if entry.name.endswith(".npz"):
            session = entry.name.removesuffix(".npz")
            print(f"Loading responses from {entry.path}")
            responses_all_sessions[session] = load_responses_for_session(entry.path, fr_normalization)
    return responses_all_sessions
