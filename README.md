# Plato
## Thomas Schouten

<!-- [![codecov](https://codecov.io/gh/larsgeb/hmclab/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmclab) [![license](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub releases](https://img.shields.io/badge/download-latest%20release-green.svg)](https://github.com/larsgeb/hmclab/releases/latest) -->

**Plato** is an object-oriented Python package that provide an efficient and standardised workflow to analyse the motions of tectonic plates using the torque balance expected from their reconstructed geometries.

This package leverages [`GPlately`](https://gplates.github.io/gplately/v1.3.0/) to interrogate quantitative plate reconstructions built using [`GPlates`](https://www.gplates.org). `GPlates` is open-source application software offering a novel combination of interactive plate-tectonic reconstructions, geographic information system functionality and raster data visualisation.

The package is based on the algorithm initially published by [Clennett et al. (2023)](https://www.nature.com/articles/s41598-023-37117-w).

- **Website:** https://thomas-schouten.github.io/plato/index.html
<!-- - **Python documentation:** https://python.hmclab.science -->
- **Source code:** https://github.com/thomas-schouten/plato
<!-- - **Docker image:** https://hub.docker.com/repository/docker/larsgebraad/hmclab -->
- **Bug reports:** https://github.com/thomas-schouten/plato/issues

# Installation

To install `Plato`, create a Conda environment and run the following:

```
> $ git clone https://github.com/thomas-schouten/plato.git
> $ cd plato
> $ conda env create -f environment.yml
> $ conda activate plato
> $ pip install -e .
```

This will create a Conda environment with `Plato` and all required dependencies, which should be able to run all notebooks found in
plato/notebooks.

# References

Schouten, T. L. A. (2025). *From puzzling plates to mantle mysteries: New approaches and applications for plate tectonic reconstructions* (Doctoral dissertation, ETH Zurich). ETH Research Collection. https://doi.org/10.3929/ethz-c-000788918
