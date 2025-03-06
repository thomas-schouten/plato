# %%
import os
import sys
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextlib
import xarray as xr

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from plato.optimisation import Optimisation
from plato.plate_torques import PlateTorques
from plato.plot import PlotReconstruction

from copy import deepcopy

cm2in = 0.3937008

# Set parameters
# Plate reconstruction
reconstruction_name = "Muller2016" 

# Reconstruction ages of interest
ages = [0]

# Set path to sample data
path = "/Users/thomas/Documents/_Plato/Plato/sample_data/M2016"

# Load seafloor age grids
seafloor_age_grids = {}
for age in ages:
    seafloor_age_grids[age] = xr.open_dataset(f"{path}/seafloor_age_grids/M2016_SeafloorAgeGrid_{age}Ma.nc")

# Set up PlateTorques object
M2016 = PlateTorques(
    reconstruction_name = reconstruction_name, 
    ages = ages, 
    seafloor_age_grids = seafloor_age_grids,
    rotation_file = f"{path}/gplates_files/M2016_rotations_Lr-Hb.rot",
    topology_file = f"{path}/gplates_files/M2016_topologies.gpml",
)
M2016.settings.options["ref"]["Slab suction torque"] = True

M2016.sample_all()
M2016.calculate_all_torques()

# %%
optimise_M2016 = Optimisation(M2016)

# %%
results = optimise_M2016.minimise_residual_torque_v4(plateIDs = [802])

# %%
before = np.log10(M2016.plates.data[36]["ref"].residual_torque_mag/M2016.plates.data[36]["ref"].driving_torque_mag)f# %%
optimise_M2016 = Optimisation(M2016)
optimise_M2016.invert_residual_torque_v4(plateIDs = [902, 904, 909, 911, 918, 926, 901], cases = ["ref"], NUM_ITERATIONS=4, PLOT=True)

# %%
plt.scatter(
    M2016.slabs.data[36]["ref"].lon, 
    M2016.slabs.data[36]["ref"].lat, 
    c = M2016.slabs.data[36]["ref"].slab_pull_constant,
    vmin=0, vmax=.5
)

# %%
plt.scatter(
    M2016.slabs.data[36]["ref"].slab_seafloor_age,
    M2016.slabs.data[36]["ref"].slab_pull_force_mag
)
# %%
M2016.calculate_all_torques()
after = np.log10(M2016.plates.data[36]["ref"].residual_torque_mag/M2016.plates.data[36]["ref"].driving_torque_mag)

# %%
plt.scatter(M2016.plates.data[36]["ref"].plateID, np.abs(after)/np.abs(before))