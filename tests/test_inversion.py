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
plateIDs = [101]

path = "/Users/thomas/Documents/_Plato/Plato/sample_data/M2016"

# Load seafloor age grids
seafloor_age_grids = {}
for age in ages:
    seafloor_age_grids[age] = xr.open_dataset(f"{path}/seafloor_age_grids/M2016_SeafloorAgeGrid_{age}Ma.nc")

# Set up PlateTorques object
M2016 = PlateTorques(reconstruction_name = reconstruction_name, ages = ages, seafloor_age_grids = seafloor_age_grids, rotation_file = f"{path}/gplates_files/M2016_rotations_Lr-NNR.rot", topology_file = f"{path}/gplates_files/M2016_topologies.gpml", polygon_file = f"{path}/gplates_files/M2016_polygons.gpml")

# Set up PlotReconstruction object
# plot_M2016 = PlotReconstruction(M2016)

# Sample all
M2016.sample_all()

M2016.settings.options["ref"]["Slab suction torque"] = True
M2016.slabs.data[0]["ref"]["slab_suction_constant"] = .1

# Calculate all torques
M2016.calculate_all_torques()

# Save for later
# M2016.save_all()

# %%
# Print initial residual torques
_plate_data = M2016.plates.data[ages[0]]["ref"]
_plate_data = _plate_data[_plate_data["plateID"].isin(plateIDs)]

print(_plate_data["plateID"].values)
print(np.log10(_plate_data["residual_torque_mag"].values/_plate_data["driving_torque_mag"].values))

print(_plate_data["residual_torque_mag"].values, _plate_data["driving_torque_mag"].values)

# Set up Optimisation object
optimisation = Optimisation(M2016)

# Invert for slab pull constants
# for _ in range(20):
# optimisation.invert_residual_torque_v3(plateIDs=902, plot=True)
# %%
# Invert for slab pull constants
optimisation.optimise_slab_pull_coefficient(plateIDs=plateIDs)
optimisation.invert_residual_torque_v5(plateIDs=plateIDs, PLOT=False, NUM_ITERATIONS=2, vmin=-100, vmax=25, step=1.)

# %%
_plate_data = M2016.plates.data[ages[0]]["ref"]
_plate_data = _plate_data[_plate_data["plateID"].isin(plateIDs)]

print(np.log10(_plate_data["residual_torque_mag"].values/_plate_data["driving_torque_mag"].values))

print(_plate_data["residual_torque_mag"].values, _plate_data["driving_torque_mag"].values)

# %%
# Plot slab pull constants
_slab_data = M2016.slabs.data[ages[0]]["ref"]
_slab_data = _slab_data[_slab_data["upper_plateID"].isin(plateIDs)]
plt.scatter(
    _slab_data.lon,
    _slab_data.lat,
    c=_slab_data.slab_suction_constant,
    # vmin=0, vmax=.5,
)
plt.xlim(-180, 180); plt.ylim(-90, 90)
plt.colorbar()
plt.show()

# %%
plt.plot(_slab_data.arc_residual_force_mag)

# %%
# Plot slab pull constants
_slab_data = M2016.slabs.data[ages[0]]["ref"]
# _slab_data = _slab_data[_slab_data["lower_plateID"].isin(plateIDs)]
plt.scatter(
    _slab_data.lon,
    _slab_data.lat,
    c=_slab_data.arc_residual_force_lon,
    # vmin=0,
    # vmax=.5,
)
plt.xlim(-180, 180); plt.ylim(-90, 90)
plt.colorbar()
plt.show()

# %%
_plate_data = M2016.plates.data[ages[0]]["ref"]
_plate_data = _plate_data[_plate_data["plateID"].isin(plateIDs)]

print(np.log10(_plate_data["residual_torque_mag"].values/_plate_data["driving_torque_mag"].values))

print(_plate_data["residual_torque_mag"].values, _plate_data["driving_torque_mag"].values)

# %%
M2016.calculate_all_torques()

# %%
_plate_data = M2016.plates.data[ages[0]]["ref"]
_plate_data = _plate_data[_plate_data["plateID"].isin(plateIDs)]

print(np.log10(_plate_data["residual_torque_mag"].values/_plate_data["driving_torque_mag"].values))

print(_plate_data["slab_suction_torque_mag"].values, _plate_data["driving_torque_mag"].values)
# %%
plt.plot(_slab_data.slab_pull_constant)