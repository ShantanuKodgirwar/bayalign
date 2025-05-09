# bayalign
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

Efficient Bayesian inference in JAX via geodesic slice sampling on the sphere (GeoSSS) for point-cloud based 3D-2D rigid registration, demonstrated for the cryo-EM data.

## Development

Clone the repository and navigate to the root.

```bash
git clone https://github.com/ShantanuKodgirwar/bayalign.git
cd bayalign
```

### Installation with uv (recommended!)

The package is maintained with the fast package manager [uv](https://github.com/astral-sh/uv). The package `bayalign` and its *locked* dependencies can be installed in a dedicated virtual environment with the single command:

```bash
uv sync
```

> [!NOTE]  
> If dependencies were changed, make sure to run `uv lock`, followed by exporting these to `requirements.txt` as `uv export --no-emit-project --no-hashes -o requirements.txt` for pip users.

### Installation with pip (legacy workflow)

If you don't want to use uv, and simply rely on pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

> [!NOTE]  
> Feel free to choose the type of your environment (venv/conda) when using pip. However uv only supports venv.
