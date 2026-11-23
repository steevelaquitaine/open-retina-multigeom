"""Per-cell and per-experiment metadata for the flashed-natural-image adaptation dataset.

The table ``cell_metadata.csv`` (built by ``openretina_port/scripts/build_cell_metadata.py``) holds, for
every modelled cell:

  * ``experiment``        -- acquisition session the cell belongs to (``exp4`` .. ``exp7``);
  * ``cell_type``         -- functional type from the experiment's rating sheet (empty where untyped);
  * ``test_reliability``  -- cell reliability on the test set (odd/even split-half noise ceiling);
  * ``model_test_score``  -- model performance on the test set (paper Fig 2): mean over test contexts of the
    noise-corrected r2 with the fitted layer-1 gain control, for the released model. For exp6 the published rows of
    54 cells were computed on older stage-1 models; their values here use the released models with the gain refit on
    them with the original protocol; exp4/5/7 are the published values;
  * ``stimulus_size_px``, ``microns_per_px`` -- stimulus geometry (30 px at 3.5*6 um/px; exp7: 40 px at 2.6*6);
  * ``n_grey_frames``     -- grey frames preceding each clip at evaluation;
  * ``H_on``, ``H_off``   -- fitted on/off subunit gain-control strengths (baked into the released weights), as
    stored by the legacy fit: some are slightly negative, which with gain exponent 2 acts as ``|H|``;
  * ``A``                 -- fitted output scale (least squares on mean-centred PSTHs), applied at the output;

Use :func:`metadata_for_session` to attach a cell's metadata (it is stored on
``ResponsesTrainTestSplit.session_kwargs`` by the response loader) and :func:`experiment_metadata`
for the per-session (per-experiment) aggregates.
"""

import csv
import functools
import os
from collections import Counter
from typing import Any

_CSV_PATH = os.path.join(os.path.dirname(__file__), "cell_metadata.csv")


def _to_float(value: str) -> float | None:
    """Parse a CSV cell to float, or ``None`` for a blank (a cell without a fitted score)."""
    return float(value) if value not in ("", "nan") else None


@functools.lru_cache(maxsize=1)
def load_cell_metadata() -> tuple[dict[str, Any], ...]:
    """Load the per-cell metadata table as a tuple of row dicts (cached)."""
    with open(_CSV_PATH, newline="") as f:
        return tuple(
            {
                "session": r["session"],
                "experiment": r["experiment"],
                "cell_type": r["cell_type"] or "",
                "test_reliability": _to_float(r["test_reliability"]),
                "model_test_score": _to_float(r["model_test_score"]),
                "stimulus_size_px": int(r["stimulus_size_px"]),
                "microns_per_px": float(r["microns_per_px"]),
                "n_grey_frames": int(r["n_grey_frames"]),
                "H_on": _to_float(r["H_on"]),
                "H_off": _to_float(r["H_off"]),
                "A": _to_float(r["A"]),
            }
            for r in csv.DictReader(f)
        )


def metadata_for_session(session: str) -> dict[str, Any]:
    """Metadata for one cell/session, or ``{}`` if the cell is not in the table."""
    for row in load_cell_metadata():
        if row["session"] == session:
            return {k: v for k, v in row.items() if k != "session"} | {"cell_type": row["cell_type"] or None}
    return {}


def experiment_metadata() -> dict[str, dict[str, Any]]:
    """Per-experiment (per-session) aggregates: cell count, mean test reliability, mean model test
    score, and the functional cell-type counts."""
    rows = load_cell_metadata()
    out: dict[str, dict[str, Any]] = {}
    for experiment in sorted({r["experiment"] for r in rows}):
        group = [r for r in rows if r["experiment"] == experiment]
        typed = [r["cell_type"] for r in group if r["cell_type"]]
        reliabilities = [r["test_reliability"] for r in group if r["test_reliability"] is not None]
        scores = [r["model_test_score"] for r in group if r["model_test_score"] is not None]
        out[experiment] = {
            "n_cells": len(group),
            "mean_test_reliability": sum(reliabilities) / len(reliabilities) if reliabilities else None,
            "mean_model_test_score": sum(scores) / len(scores) if scores else None,
            "cell_type_counts": dict(Counter(typed).most_common()),
        }
    return out
