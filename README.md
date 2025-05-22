# Bayalign
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

🛠️ *work-in-progress* 🛠️

*Bayalign* is a lightweight JAX-based library for Bayesian inference in rigid point cloud registration using Geodesic slice sampling on the sphere (GeoSSS) for inference. See the package [`geosss`](https://github.com/microscopic-image-analysis/geosss) for details.

The package is tailored for scientific applications such as Cryo-EM where the goal is to estimate the rotation of a 3D structure that best aligns with noisy or partial 2D projections.

## Features
- Supports 3D-2D and 3D-3D rigid registration
- GPU acceleration, Automatic differentiation via JAX
- Fast inference via GeoSSS
- Uses Gaussian Mixture Models (GMM) for scoring the rigid poses
 
## Installation

```bash
pip install git+https://github.com/ShantanuKodgirwar/bayalign.git
```

## Quickstart

A basic example of 3D-to-2D registration:

```python
from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import GaussianMixtureModel
from bayalign.inference import ShrinkageSphericalSliceSampler
from bayalign.sphere_utils import sample_sphere

# Define 2D target and 3D source point clouds
target_2d = PointCloud(positions, weights)               # shape (N, 2)
source_3d = RotationProjection(positions, weights)       # shape (M, 3)

# Define a target probability model using GMM
target_pdf = GaussianMixtureModel(target_2d, source_3d, sigma=1.0, k=20)

# Sample from the posterior over 3D rotations (quaternions)
init_q = sample_sphere(random.key(645), d=3)             # initial quaternion (4,)
sampler = ShrinkageSphericalSliceSampler(target_pdf, init_q, seed=123)
samples = sampler.sample(n_samples=100, burnin=0.2)

# Find the best rotation
log_probs = np.array([target_pdf.log_prob(q) for q in samples])
best_rot = samples[np.argmax(log_probs)]
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

Export pip-compatible dependencies:

```bash
uv export --extra all --no-emit-project --no-hashes -o requirements.txt
```

### Option 2: Using pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Issues

If you encounter any problems, have questions, please feel free to [open an issue](https://github.com/ShantanuKodgirwar/bayalign/issues).