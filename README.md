# bayalign
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

🛠️ *work-in-progress* 🛠️

A light-weight JAX-based package for efficient Bayesian inference in rigid point cloud registration using Gaussian mixture modeling. The method is demonstrated on a cryo-EM sub-problem involving the estimation of the 3D structure’s rotation that best aligns with observed 2D projections. 

## Features
- Supports rigid registration between 3D-3D and 3D-2D point clouds.
- Uses kernel correlation or Gaussian mixtures to score candidate poses.
- Enables efficient bayesian inference on the rotation group via geodesic slice sampling on the sphere (GeoSSS), with additional support for HMC and RWMH 

## Installation

```bash
pip install git+https://github.com/ShantanuKodgirwar/bayalign.git
```

## Usage

```python
from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians

# TODO: Add a minimal example to demonstrate the two ways (or atleast one)
# 3D-3D

# 3D-2D
```

## Development

Clone the repository and navigate to the root.

```bash
git clone https://github.com/ShantanuKodgirwar/bayalign.git
cd bayalign
```

### Via pip (legacy workflow)

If you don't want to use [uv](https://github.com/astral-sh/uv), a fast package manager (if you do, see below!), and simply rely on pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt # or via `requirements-gpu.txt` if you have a GPU with cuda12
pip install -e . --no-deps
```

This installs all the dependencies, including the ones necessary for the [examples](examples/) directory.

### Via uv (recommended!)

The package `bayalign` and its *locked* dependencies are maintained by [uv](https://github.com/astral-sh/uv) and can be installed as:

```bash
uv sync --extra all
```
This installs CPU dependencies (including the ones for the examples directory) in a virtual environment. For JAX to use GPU (cuda12), use the flag `all-gpu` instead of `all` in the above command. To activate this environment, 

```bash
source .venv/bin/activate
```

Any dependency changes can be exported to a requirements file (for pip users) with the execution of the helper shell script as 

```bash
./export_requirements.sh
```

