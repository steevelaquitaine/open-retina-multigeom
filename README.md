# open-retina-multigeom (private)

The part of [open-retina](https://github.com/open-retina/open-retina) that multigeom's
retina tests need (`multigeom.encoders.retina`, the tests in multigeom-tutorials
`tests/test_retina.py`). It is installed as the `openretina` package:

```bash
pip install --no-deps "git+https://github.com/steevelaquitaine/open-retina-multigeom.git"
```

then the few dependencies it imports, keeping your torch:
`pip install jaxtyping hydra-core lightning tenacity opencv-python-headless seaborn moviepy`.

## Where the code comes from

- **open-retina** 1.1.1 (MIT licence, see LICENSE), by the open-retina authors.
- **Baptiste Lorenzi's unpublished extension**, from his local branch
  `dev/baptiste/diffusion_metric` (not on GitHub): the Lorenzi adaptation model
  (`openretina/models/adaptation_gain_control.py`, `openretina/data_io/lorenzi_adaptation/`)
  and the information geometry of the retina (`openretina/insilico/information_geometry/`:
  mosaic encoder, conditional diffusion model, G(x) and Poisson J(x)).

Only the files those tests load are tracked (see `.gitignore`). **This repository is
private: Baptiste's work is unpublished. Do not make it public without his agreement.**

The model weights are not here: the Lorenzi encoder checkpoints and the diffusion
checkpoints are kept with the data.
