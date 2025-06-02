# Bayalign
[![PyPI](https://img.shields.io/pypi/v/bayalign)](https://pypi.org/project/bayalign/)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

🛠️ *work-in-progress* 🛠️

*Bayalign* is a lightweight JAX-based library for rigid point cloud registration using efficient Bayesian inference via geodesic slice sampling on the sphere (GeoSSS). See the package [`geosss`](https://github.com/microscopic-image-analysis/geosss) for details.

The package is tailored for any rigid registration problem (currently doesn't support translation estimation) and has mainly been motivated from a scientific application such as Cryo-EM where the goal is to estimate the rotation of a 3D structure that best aligns with noisy or partial 2D projections.

<p align="center">
<img src="https://github.com/ShantanuKodgirwar/bayalign/blob/8fc3448b6f634723146101f0fc01a6e7b7921f96/assets/reg3d2d.png" width="800">
</p>

## Features
- Supports 3D-2D and 3D-3D rigid registration
- GPU acceleration, Automatic differentiation, JIT via JAX
- Fast inference via GeoSSS
- Uses Gaussian Mixture Models (GMM) for scoring the rigid poses
 
## Installation

```bash
pip install bayalign
```

## Quickstart

A basic example of 3D-to-2D registration:

```python
import jax.numpy as jnp

from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import GaussianMixtureModel
from bayalign.inference import ShrinkageSphericalSliceSampler
from bayalign.sphere_utils import sample_sphere

# Define 2D target and 3D source point clouds
target = PointCloud(positions, weights)               # shape (N, 2)
source = RotationProjection(positions, weights)       # shape (M, 3)

# Define a target probability model using GMM
target_pdf = GaussianMixtureModel(target, source, sigma=1.0, k=20)

# Sample from the posterior over 3D rotations (quaternions)
init_q = sample_sphere(random.key(645), d=3)             # initial quaternion (4,)
sampler = ShrinkageSphericalSliceSampler(target_pdf, init_q, seed=123)
samples = sampler.sample(n_samples=100, burnin=0.2)

# Find the best rotation
log_probs = jnp.asarray([target_pdf.log_prob(q) for q in samples])
best_rot = samples[jnp.argmax(log_probs)]
transformed_source = source.transform_positions(best_rot)
```

For 3D-3D registration, use `PointCloud` for both target and source. Check out the [examples](examples/) directory for detailed use cases using synthetic and cryo-EM data. 

To run the examples, you need to install some optional dependencies. Follow one of the methods below to set up your environment.

## Development

Clone the repository and navigate to the root.

```bash
git clone https://github.com/ShantanuKodgirwar/bayalign.git
cd bayalign
```

### Option 1: Using uv (recommended!)

The package `bayalign` and all its *locked* dependencies are maintained by [uv](https://github.com/astral-sh/uv) and can be installed within a virtual environment as:

```bash
uv sync --extra all
```

### Option 2: Using pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Issues

If you encounter any problems, have questions, please feel free to [open an issue](https://github.com/ShantanuKodgirwar/bayalign/issues).
