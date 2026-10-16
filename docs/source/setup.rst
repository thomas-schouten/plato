Installation
============

`Plato` requires Python 3.10. Installing with :code:`Conda` is recommended, as it
handles the geospatial dependencies (GDAL, GMT, GPlates) on Windows, macOS and Linux.

Install with Conda
******************

.. code-block:: bash

    > $ git clone https://github.com/thomas-schouten/plato.git
    > $ cd plato
    > $ conda env create -f environment.yml
    > $ conda activate plato_env
    > $ pip install -e .

This creates a Conda environment called :code:`plato_env` with `Plato` and all
required dependencies.

Install with pip only
*********************

If you prefer not to use Conda, you can install `Plato` and its dependencies
into any Python 3.10 environment from inside the cloned repository:

.. code-block:: bash

    > $ pip install -e .

Running the example notebooks
*****************************

The example notebooks are in the :code:`notebooks` folder of the repository and
use the sample data provided in :code:`sample_data`, so no downloads are needed.
Open them in Jupyter or VS Code with the :code:`plato_env` environment selected
as the kernel.