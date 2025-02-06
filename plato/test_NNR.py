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

path = "/Users/thomas/Documents/_Plato/Plato/sample_data/M2016"

# Load seafloor age grids
seafloor_age_grids = {}
for age in ages:
    seafloor_age_grids[age] = xr.open_dataset(f"{path}/seafloor_age_grids/M2016_SeafloorAgeGrid_{age}Ma.nc")

# Set up PlateTorques object
M2016 = PlateTorques(reconstruction_name = reconstruction_name, ages = ages, seafloor_age_grids = seafloor_age_grids, files_dir="output", rotation_file = f"{path}/gplates_files/M2016_rotations_Lr-Hb.rot")

# %%
M2016.settings.options["ref"]["Reconstructed motions"] = False

M2016.sample_all()
M2016.calculate_all_torques()
M2016.calculate_synthetic_velocity()
M2016.calculate_net_rotation()

# %%
# Set up PlotReconstruction object
plot_M2016 = PlotReconstruction(M2016)
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
plot_M2016.plot_velocity_map(ax, 0, "ref")
# %%
print(M2016.globe.data["ref"].net_rotation_rate)

# %%
# Remove net rotation
M2016.remove_net_rotation()

print(M2016.globe.data["ref"].net_rotation_rate)

# %%
M2016.calculate_net_rotation()
print(M2016.globe.data["ref"].net_rotation_rate)

# %%
M2016.settings.options["ref"]["Minimum plate area"]