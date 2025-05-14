# bayalign
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

A work-in-progress JAX-based library for efficient Bayesian inference using geodesic slice sampling on the sphere (GeoSSS), designed for rigid registration via Gaussian mixture models (GMM). It is demonstrated on a cryo-EM sub-problem involving estimation of the 3D model's rotation that best aligns with observed 2D projections. 

## Development

Clone the repository and navigate to the root.

```bash
git clone https://github.com/ShantanuKodgirwar/bayalign.git
cd bayalign
```

### Installation with pip (legacy workflow)

If you don't want to use uv (if you do, see below!), and simply rely on pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt # or via `requirements-gpu.txt`
pip install -e . --no-deps
```

> [!NOTE]  
> Feel free to choose the type of your environment (venv/conda) when using pip.

### Installation with uv (recommended!)

The package is maintained with the fast package manager [uv](https://github.com/astral-sh/uv). The package `bayalign` and its *locked* dependencies can be installed in a dedicated virtual environment (only supports venv) with the single command:

```bash
uv sync
```

This provides a default JAX CPU support. For a full list of options,

| Command                   | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `uv sync --extra gpu`     | Installs the package with JAX GPU/CUDA 12 support |
| `uv sync --extra viz`     | Installs with visualization dependencies          |
| `uv sync --extra dev`     | Installs with development tools                   |
| `uv sync --extra all`     | Installs with all dependencies (CPU version)      |
| `uv sync --extra all-gpu` | Installs with all dependencies (GPU version)      |


Additionally for pip users, please export any dependency changes to a requirements file as:

```bash
uv export --extra all --no-emit-project --no-hashes -o requirements-cpu.txt
```

For GPU dependency changes, 

```bash
uv export --extra all-gpu --no-emit-project --no-hashes -o requirements-cpu.txt
```