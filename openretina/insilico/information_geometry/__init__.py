"""Multi-scale information geometry of neural population codes for open-retina.

Exploratory port of the diffusion-based multi-scale Fisher metric G(x)
(Azeglio et al., 2026) adapted so that an open-retina retina model can act as
the image-to-response encoder f_phi.

The pipeline for a single spatial stimulus x is:

    1. RetinaImageEncoder wraps a BaseCoreReadout as a static-image encoder
       f_phi : [B, C, H, W] -> [B, N] using a flashed-image presentation
       (grey -> image, response = mean over a post-onset window) and a
       synthetic *mosaic* population built by displacing one model cell's RF.
    2. A conditional diffusion model eps_theta(x_t, t, r) is trained on the
       natural-stimulus frame distribution, conditioned on f_phi responses.
    3. conditional_pair_estimate() computes the top-k eigenpairs of G(x)
       (Algorithm 1 of the paper); fisher_eigendecomposition() the J(x)
       baseline; metrics.py the cross-model similarity measures.

This subpackage is research/exploratory code; it is intentionally not part of
the public, supported openretina API.
"""

from openretina.insilico.information_geometry.estimator import (
    EstimatorOutput,
    conditional_pair_estimate,
)
from openretina.insilico.information_geometry.fisher import (
    FisherOutput,
    fisher_eigendecomposition,
)
from openretina.insilico.information_geometry.metrics import (
    bures_wasserstein,
    subspace_alignment,
)
from openretina.insilico.information_geometry.retina_encoder import (
    RetinaImageEncoder,
    build_mosaic_encoder,
    build_session_encoder,
    build_translation_mosaic_encoder,
    measure_pixel_rf_sigma,
)
from openretina.insilico.information_geometry.stimuli import StimulusNormalizer

__all__ = [
    "RetinaImageEncoder",
    "StimulusNormalizer",
    "build_session_encoder",
    "build_mosaic_encoder",
    "build_translation_mosaic_encoder",
    "measure_pixel_rf_sigma",
    "conditional_pair_estimate",
    "EstimatorOutput",
    "fisher_eigendecomposition",
    "FisherOutput",
    "subspace_alignment",
    "bures_wasserstein",
]
