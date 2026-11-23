"""Constants for the flashed-natural-image adaptation dataset (Lorenzi et al.).

Each stimulus clip is a flashed adapting image held for ``N_ADAPT_FRAMES`` frames followed by a
flashed natural image held for ``N_NAT_FRAMES`` frames (``CLIP_LENGTH`` frames total, at
``FRAME_RATE_HZ``). The adaptor was shown for 16 frames; the clip drops its first frame. The export
stores the PSTH of the whole sequence from adaptor onset (``N_PSTH_BINS`` bins of ``BIN_SIZE_S``); bin 0
precedes the first clip frame, so the frame-aligned response is bins 1..31.

Test clips are ordered adaptor-major, image-minor, adaptor 1 being grey: exp4/5 = 3 adaptors x 10
images (30 clips); exp6/7 = 6 adaptors x 5 images, skipping the 5 pairs whose adaptor is the probe
image itself (25 clips). This is the legacy order (``test_clip_order`` in the export script).

For evaluation, each clip is preceded by ``N_GREY_FRAMES`` frames of the cell's grey level so the gain
control settles (see ``openretina/cli/fit_adaptation_gain_control.py`` for the scoring protocol).
"""

# Clip geometry, in frames (matches the export in scripts/export_adaptation_data.py).
N_ADAPT_FRAMES = 15
N_NAT_FRAMES = 16
CLIP_LENGTH = N_ADAPT_FRAMES + N_NAT_FRAMES  # 31

# Response binning of the whole adaptor + probe sequence.
BIN_SIZE_S = 0.025
FRAME_RATE_HZ = 1.0 / BIN_SIZE_S  # 40 Hz
N_PSTH_BINS = 32

# Evaluation protocol (legacy run_test / run_test_like_exp6).
N_GREY_FRAMES = 69
GREY_ADAPT_ID = 1  # frame-bank index of the uniform grey adaptor

STIMULUS_SHAPE = (30, 30)  # (height, width) of the STA-centred crops (40x40 for exp7)
