darkmod - A Model of Dark-Field X-ray Microscopy
======================================================

This package is a forward model for Dark field X-ray Microscopy allowing for simulation of diffraction based image formation from samples that are spatially extended on 3D voxeldated grids.

In addition, it supports reconstruction of defomration gradient tensor fields from multiple reflection data.

This repository is provided as is, and is not intended to be a packaged and indexed python library. The associated journal publication is currenlty under review for Journal of the Mechanics and Physics of Solids.

installation
==============================


```bash
git clone https://github.com/AxelHenningsson/darkmod.git
cd darkmod
pip install -e .
```

Documentation
======================================
Some documentation is found in the docstrings of the modules under `darkmod`.

Chromatic imaging
======================================

The [chromatic imaging guide](docs/source/chromatic.rst) describes energy
scans with fixed lens and detector positions, the ASTRA/CUDA API, model
assumptions and validation commands. Public API docstrings are included in
the Sphinx documentation.

To build the documentation from the repository root:

```bash
pip install -r docs/requirements.txt
PYTHONPATH=. python -m sphinx -b html docs/source docs/build/html
```
