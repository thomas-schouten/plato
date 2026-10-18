# Plato
## Thomas Schouten

**Plato** is an object-oriented Python package that provide an efficient and standardised workflow to analyse the motions of tectonic plates using the torque balance expected from their reconstructed geometries.

This package leverages [`GPlately`](https://gplates.github.io/gplately/v1.3.0/) to interrogate quantitative plate reconstructions built using [`GPlates`](https://www.gplates.org). `GPlates` is open-source application software offering a novel combination of interactive plate-tectonic reconstructions, geographic information system functionality and raster data visualisation.

The package is based on the algorithm initially published by [Clennett et al. (2023)](https://www.nature.com/articles/s41598-023-37117-w).  The theoretical framework can be found in Appendix B of [Schouten, T. L. A. (2025)](https://doi.org/10.3929/ethz-c-000788918).

- **Website:** https://thomas-schouten.github.io/plato/index.html
- **Source code:** https://github.com/thomas-schouten/plato
- **Bug reports:** https://github.com/thomas-schouten/plato/issues

# Installation

To install `Plato`, create a Conda environment and run the following:

```
> $ git clone https://github.com/thomas-schouten/plato.git
> $ cd plato
> $ conda env create -f environment.yml
> $ conda activate plato_env
> $ pip install -e .
```

This will create a Conda environment with `Plato` and all required dependencies, which should be able to run all notebooks found in
plato/notebooks.

# References

Schouten, T. L. A. (2025). *From puzzling plates to mantle mysteries: New approaches and applications for plate tectonic reconstructions* (Doctoral dissertation, ETH Zurich). ETH Research Collection. https://doi.org/10.3929/ethz-c-000788918

Clennett, E. J., Holt, A. F., Tetley, M. G., Becker, T. W., & Faccenna, C. (2023). Assessing plate reconstruction models using plate driving forces. *Scientific Reports*, 13, 10191. https://doi.org/10.1038/s41598-023-37117-w
