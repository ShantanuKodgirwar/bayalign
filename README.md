# bayalign

Efficient Bayesian inference in JAX via geodesic slice sampling on the sphere (GeoSSS) for point-cloud based 3D-2D rigid registration, demonstrated for the cryo-EM data.

## Installation

To install the *locked* dependencies via pip in a virtual environment for a python version >= 3.11,

```bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally, the package is maintained with the fast package manager [uv](https://github.com/astral-sh/uv) and the dependencies can also be installed as,

```bash
uv sync
```

Finally for an editable installation of this package,

```bash
pip install -e .
```
