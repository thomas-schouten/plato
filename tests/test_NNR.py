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

# Load excel file with settings
settings_file = os.path.join(os.getcwd(), "cases_test.xlsx")

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
    cases_file = settings_file,
    cases_sheet = "Sheet2",
    seafloor_age_grids = seafloor_age_grids,
)
# %%
# M2016.settings.options["ref"]["Reconstructed motions"] = False

M2016.remove_net_rotation(cases = ["nnr", "syn_nnr"])
M2016.sample_all()
M2016.calculate_all_torques(cases = M2016.cases)

# %%
optimise_M2016 = Optimisation(M2016)

optimise_M2016.invert_residual_torque_v4(plateIDs = [901, 909, 911], cases = ["ref", "nnr"], NUM_ITERATIONS=20)

# %%
# Copy the optimised slab pull constants to the synthetic case
for age in M2016.ages:
        M2016.slabs.data[age]["syn"].slab_pull_constant = M2016.slabs.data[age]["ref"].slab_pull_constant.copy()
        M2016.slabs.data[age]["syn_nnr"].slab_pull_constant = M2016.slabs.data[age]["nnr"].slab_pull_constant.copy()

# Calculate synthetic plate velocities
M2016.calculate_all_torques(cases = ["syn", "syn_nnr"], plateIDs = [901, 909, 911])
M2016.calculate_synthetic_velocity(cases = ["syn", "syn_nnr"], plateIDs = [901, 909, 911])
# %%
# Set up PlotReconstruction object
plot_M2016 = PlotReconstruction(M2016)

# Plotting parameters
cm2in = 0.3937008
fig_width = 18*cm2in*2; fig_height_graphs = 8*cm2in*2; fig_height_maps = 10*cm2in*2
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"
title_fontsize = 20
plot_times = [45, 60, 75, 90]
projection = ccrs.Robinson(central_longitude = 160)
annotations = ["a", "b", "c", "d"]

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

ax1 = fig.add_subplot(gs[0, 0], projection=projection)
plot_M2016.plot_velocity_map(ax1, 0, "ref")

ax2 = fig.add_subplot(gs[0, 1], projection=projection)
plot_M2016.plot_velocity_map(ax2, 0, "nnr")

ax3 = fig.add_subplot(gs[1, 0], projection=projection)
plot_M2016.plot_velocity_difference_map(ax3, 0, "nnr", "ref", vmin=-5, vmax=5, vector_scale=1e2)

ax4 = fig.add_subplot(gs[1, 1], projection=projection)
plot_M2016.plot_relative_velocity_difference_map(ax4, 0, "nnr", "ref")

plt.show()

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

ax1 = fig.add_subplot(gs[0, 0], projection=projection)
plot_M2016.plot_velocity_map(ax1, 0, "syn")

ax2 = fig.add_subplot(gs[0, 1], projection=projection)
plot_M2016.plot_velocity_map(ax2, 0, "syn_nnr")

ax3 = fig.add_subplot(gs[1, 0], projection=projection)
plot_M2016.plot_velocity_difference_map(ax3, 0, "syn", "ref", vmin=-5, vmax=5, vector_scale=1e2)

ax4 = fig.add_subplot(gs[1, 1], projection=projection)
plot_M2016.plot_velocity_difference_map(ax4, 0, "syn_nnr", "nnr", vmin=-5, vmax=5, vector_scale=1e2)

plt.show()

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps/2), dpi=300)
gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.2)

k = 0
for case in M2016.cases:
    if "nnr" in case:
        continue

    ax = fig.add_subplot(gs[0, k], projection=projection)
    plot_M2016.plot_basemap(ax)
    plot_M2016.plot_reconstruction(
        ax,
        0,
        coastlines_facecolour = "lightgrey",
        coastlines_edgecolour = "lightgrey",
        coastlines_linewidth = 0,
        plate_boundaries_linewidth = 1,
    )
    slab_data = M2016.slabs.data[0][case]
    slab_data = slab_data[slab_data.lower_plateID.isin([901, 909, 911])]
    sc = ax.scatter(
        slab_data.lon,
        slab_data.lat,
        c = slab_data.slab_pull_constant,
        s = 30,
        vmin = 0,
        vmax = .5,
        cmap = "cmc.nuuk",
        transform = ccrs.PlateCarree(),
    )

    k += 1

plt.show()
# %%
_slab_data = M2016.slabs.data[0]
for case in M2016.cases:
    if "syn" in case:
        continue

    _slab_data[case] = _slab_data[case][_slab_data[case].lower_plateID.isin([901, 909, 911])]

plt.scatter(
    _slab_data["ref"].slab_pull_constant,
    _slab_data["nnr"].slab_pull_constant
)

# %%
_plate_data = M2016.plates.data[0]["syn_nnr"]
_plate_data = _plate_data[_plate_data.plateID.isin([901, 909, 911])]
print(_plate_data.velocity_rms_mag)
print(_plate_data.spin_rate_rms_mag)
# print(M2016.plates.data[0]["ref"].residual_torque_mag - M2016.plates.data[0]["nnr"].residual_torque_mag)

# %%
# Remove net rotation
M2016.remove_net_rotation()

print(M2016.globe.data["ref"].net_rotation_rate)

# %%
M2016.calculate_net_rotation()
print(M2016.globe.data["ref"].net_rotation_rate)

# %%
M2016.settings.options["ref"]["Minimum plate area"]

# %%
plt.scatter(
    M2016.points.data[0]["ref"].lon,
    M2016.points.data[0]["ref"].lat,
    c= M2016.points.data[0]["nnr"]. - M2016.points.data[0]["ref"].mantle_drag_force_mag,
)