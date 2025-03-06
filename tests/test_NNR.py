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

# Load the continental grids
continental_grids = {}
for age in ages:
    continental_grids[age] = xr.open_dataset(f"{path}/continental_grids/M2016_ContinentalGrid_{age}Ma.nc")

# Set up PlateTorques object
M2016 = PlateTorques(
    reconstruction_name = reconstruction_name, 
    ages = ages, 
    cases_file = settings_file,
    cases_sheet = "Sheet2",
    seafloor_age_grids = seafloor_age_grids,
    continental_grids = continental_grids,
    rotation_file = f"{path}/gplates_files/M2016_rotations_Lr-Hb.rot",
    topology_file = f"{path}/gplates_files/M2016_topologies.gpml",
)

# %%
# print(M2016.plates.data[0]["syn_nnr"][M2016.plates.data[0]["syn_nnr"].plateID == 901].pole_lat)

M2016.sample_all()
M2016.calculate_all_torques()
M2016.calculate_synthetic_velocity()
# print("ref")
M2016.remove_net_rotation(cases = ["nnr", "syn_nnr"])
# print("syn")
# %%
plt.scatter(M2016.plates.data[0]["syn_nnr"].pole_lon, M2016.plates.data[0]["syn_nnr"].pole_lat, c=M2016.plates.data[0]["syn_nnr"].pole_angle)
plt.colorbar()
M2016.remove_net_rotation(cases = ["syn_nnr"], VERSION=1)

# %%
plt.scatter(M2016.plates.data[0]["syn_nnr"].pole_lon, M2016.plates.data[0]["syn_nnr"].pole_lat, c=M2016.plates.data[0]["syn_nnr"].pole_angle)
plt.colorbar()
# %%
# M2016.calculate_mantle_drag_torque(cases = ["syn_ref", "syn_nnr"])

# %%
print(M2016.plates.data[0]["syn_nnr"][M2016.plates.data[0]["syn_nnr"].plateID == 901].pole_lat)
# %%
plt.scatter(
    M2016.points.data[0]["ref"].lon,
    M2016.points.data[0]["ref"].lat,
    c = M2016.points.data[0]["syn_ref"].velocity_mag/M2016.points.data[0]["syn_nnr"].velocity_mag,
)
plt.colorbar()
plt.show()

# %%
plt.plot(M2016.points.data[0]["syn_ref"].velocity_mag)

# %%
M2016.calculate_synthetic_velocity(cases = ["syn_ref", "syn_nnr"], VERSION=2)

# %%
# M2016.remove_net_rotation(cases = ["syn_nnr"], VERSION=2)
plt.scatter(
    M2016.points.data[0]["ref"].lon,
    M2016.points.data[0]["ref"].lat,
    c = M2016.points.data[0]["syn_nnr"].velocity_mag-M2016.points.data[0]["syn_ref"].velocity_mag,
)
plt.colorbar()

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

# plateIDs = [901, 909, 911]

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

ax1 = fig.add_subplot(gs[0, 0], projection=projection)
ax1.set_title("Moving hotspot reference frame", fontsize=title_fontsize)
vels = plot_M2016.plot_velocity_map(ax1, ages[0], "ref")#, plateIDs = plateIDs)

ax2 = fig.add_subplot(gs[0, 1], projection=projection)
ax2.set_title("No-net-rotation reference frame", fontsize=title_fontsize)
plot_M2016.plot_velocity_map(ax2, ages[0], "nnr")#, plateIDs = plateIDs)

ax3 = fig.add_subplot(gs[1, 0], projection=projection)
ax3.set_title("Plate speed difference", fontsize=title_fontsize)
vels_diff = plot_M2016.plot_velocity_difference_map(ax3, ages[0], "ref", "nnr", vmin=-5, vmax=5, vector_scale=1e2)#, plateIDs = plateIDs, vmin=-5, vmax=5, vector_scale=1e2)

ax4 = fig.add_subplot(gs[1, 1], projection=projection)
ax4.set_title("Relative plate speed difference", fontsize=title_fontsize)
rel_vels_diff = plot_M2016.plot_relative_velocity_difference_map(ax4, ages[0], "ref", "nnr")#, plateIDs = plateIDs)

for i in range(2):
    lon = M2016.globe.data["ref"].net_rotation_pole_lon.values[0]
    lat = M2016.globe.data["ref"].net_rotation_pole_lat.values[0]
    lon = (lon + 180) % 360 if i == 0 else lon
    lat = -lat if i == 0 else lat
    ax3.scatter(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        s=100,
        c="k",
        marker="*",
        zorder=1e2,
    )
    ax4.scatter(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        s=100,
        c="k",
        marker="*",
        zorder=1e2,
    )

# Create a new grid for the colorbar
cax1 = fig.add_axes([0.162, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar1 = plt.colorbar(vels[0], cax=cax1, orientation="horizontal", extend="max", extendfrac=25e-3)

# Set colorbar label
cbar1.set_label("Plate speed [cm/a]", labelpad=5)

# Create a new grid for the colorbar
cax2 = fig.add_axes([0.412, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar2 = plt.colorbar(vels_diff[0], cax=cax2, orientation="horizontal", extend="both", extendfrac=25e-3)

# Set colorbar label
cbar2.set_label("Plate speed difference [cm/a]", labelpad=5)

# Create a new grid for the colorbar
cax3 = fig.add_axes([0.662, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar3 = plt.colorbar(rel_vels_diff[0], cax=cax3, orientation="horizontal", extend="both", extendfrac=25e-3)

# Set colorbar label
cbar3.set_label("Log10(relative plate speed difference)", labelpad=5)

plt.show()

# %%
M2016.options["syn_nnr"]["Reconstructed motions"]


# %%
optimise_M2016 = Optimisation(M2016)
# optimise_M2016.invert_residual_torque_v4(plateIDs = [901], cases = ["ref"], NUM_ITERATIONS=5, PLOT=True)

# %%
optimise_M2016.invert_residual_torque_v4(plateIDs = [901, 909, 911], cases = ["ref", "nnr"], NUM_ITERATIONS=4)#, PARALLEL_MODE=False)

# %%
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
# Set up PlotReconstruction object
plot_M2016 = PlotReconstruction(M2016)

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps/2))#, dpi=300)
gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.2)

k = 0
for case in M2016.cases:
    if "syn" in case:
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
rec_v_rms = {901: [], 909: [], 911: []}
rec_omega_rms = {901: [], 909: [], 911: []}
for plateID in [901, 909, 911]:
    rec_v_rms[plateID].append(
        M2016.plates.data[0]["ref"][M2016.plates.data[0]["ref"].plateID == plateID].velocity_rms_mag.values[0]
    )
    rec_omega_rms[plateID].append(
        M2016.plates.data[0]["ref"][M2016.plates.data[0]["ref"].plateID == plateID].spin_rate_rms_mag.values[0]
    )

num_iterations = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]#, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946]
syn_v_rms = {901: [], 909: [], 911: []}
syn_omega_rms = {901: [], 909: [], 911: []}
residual_torque = {901: [], 909: [], 911: []}
driving_torque = {901: [], 909: [], 911: []}

for num_iter in num_iterations:
    optimise_M2016.invert_residual_torque_v4(plateIDs = [901, 909, 911], cases = ["ref"], NUM_ITERATIONS=num_iter)
    for case in M2016.cases:
        if case == "syn" or "syn" in case:
            continue

        M2016.slabs.data[0][f"syn_{case}"].slab_pull_constant = M2016.slabs.data[0][case].slab_pull_constant

    M2016.calculate_all_torques(cases = ["syn_ref"], plateIDs = [901, 909, 911])
    M2016.calculate_synthetic_velocity(cases = ["syn_ref"], plateIDs = [901, 909, 911])

    for plateID in [901, 909, 911]:
        syn_v_rms[plateID].append(
            M2016.plates.data[0]["syn_ref"][M2016.plates.data[0]["syn_ref"].plateID == plateID].velocity_rms_mag.values[0]
        )
        syn_omega_rms[plateID].append(
            M2016.plates.data[0]["syn_ref"][M2016.plates.data[0]["syn_ref"].plateID == plateID].spin_rate_rms_mag.values[0]
        )
        residual_torque[plateID].append(
            M2016.plates.data[0]["ref"][M2016.plates.data[0]["ref"].plateID == plateID].residual_torque_mag.values[0]
        )
        driving_torque[plateID].append(
            M2016.plates.data[0]["ref"][M2016.plates.data[0]["ref"].plateID == plateID].driving_torque_mag.values[0]
        )

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10*cm2in*2, 10*cm2in*2))
for plateID, plate_name in zip([901, 909, 911], ["Pacific", "Cocos", "Nazca"]):
    ax1.plot(
        num_iterations,
        np.log10(
            np.array(residual_torque[plateID])/
            np.array(driving_torque[plateID]),
        ),
        label=f"{plate_name}",
        # c="r",
    )
    ax2.plot(
        num_iterations,
        np.array(syn_v_rms[plateID])/np.array(rec_v_rms[plateID][0]),
        label=f"{plate_name}",
        # c="g",
    )
    ax3.plot(
        num_iterations,
        np.array(syn_omega_rms[plateID])/np.array(rec_v_rms[plateID][0]),
        label=f"{plate_name}",
        # c="b",
    )

for ax in [ax1, ax2, ax3]:
    ax.set_xscale("log")
    ax.grid(ls=":")
    ax.set_xlim([num_iterations[0], num_iterations[-1]])
    ax.vlines(4, ymin=-1e99, ymax=1e99, color="k", linestyle="--")

ax1.set_ylim([-20, 1])
ax2.set_ylim([0.5, 2])
ax3.set_ylim([0, 0.05])
ax3.legend()
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xlabel("Number of iterations")
ax1.set_ylabel("Normalised residual\ndriving torque")
ax2.set_ylabel("Synthetic velocity/\nReconstructed velocity")
ax3.set_ylabel("Synthetic spin rate/\nReconstructed velocity")

# %%
optimise_M2016.invert_residual_torque_v4(plateIDs = [901], cases = ["ref"], NUM_ITERATIONS=5, PLOT=True)
# %%
plt.plot(np.array(residual_torque[901])/driving_torque[901])
plt.yscale("log")


# %%
plt.scatter(
    M2016.points.data[0]["ref"].lon,
    M2016.points.data[0]["ref"].lat,
    c = M2016.points.data[0]["ref"].velocity_mag - M2016.points.data[0]["nnr"].velocity_mag,
    cmap = "RdBu_r",
)
plt.colorbar()

# %%
plt.scatter(
    M2016.slabs.data[0]["ref"].slab_pull_constant,
    M2016.slabs.data[0]["nnr"].slab_pull_constant,
)
# %%
M2016.calculate_synthetic_velocity(cases = ["syn", "syn_nnr"])

# %%
M2016.plates.data[0]["nnr"].
# %%
plt.scatter(
    M2016.plates.data[0]["ref"].area,
    M2016.plates.data[0]["syn"].velocity_rms_mag/M2016.plates.data[0]["ref"].velocity_rms_mag,
)
plt.yscale("log")
plt.xscale("log")
# plt.plot(np.linspace(0, 1e14, 1000), 5e11/np.linspace(0, 1e14, 1000)+1, c="k", linestyle="--")
plt.show()

# %%
plt.scatter(
    M2016.plates.data[0]["ref"].area,
    M2016.plates.data[0]["ref"].residual_torque_mag/M2016.plates.data[0]["syn"].driving_torque_mag,
)
plt.yscale("log")
plt.xscale("log")
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
plateIDs = [901, 909, 911]
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

ax1 = fig.add_subplot(gs[0, 0], projection=projection)
ax1.set_title("Moving hotspot reference frame", fontsize=title_fontsize)
vels = plot_M2016.plot_velocity_map(ax1, 100, "ref")#, plateIDs = plateIDs)

ax2 = fig.add_subplot(gs[0, 1], projection=projection)
ax2.set_title("No-net-rotation reference frame", fontsize=title_fontsize)
plot_M2016.plot_velocity_map(ax2, 100, "nnr")#, plateIDs = plateIDs)

ax3 = fig.add_subplot(gs[1, 0], projection=projection)
ax3.set_title("Plate speed difference", fontsize=title_fontsize)
vels_diff = plot_M2016.plot_velocity_difference_map(ax3, 100, "ref", "nnr", vmin=-5, vmax=5, vector_scale=1e2)#, plateIDs = plateIDs, vmin=-5, vmax=5, vector_scale=1e2)

ax4 = fig.add_subplot(gs[1, 1], projection=projection)
ax4.set_title("Relative plate speed difference", fontsize=title_fontsize)
rel_vels_diff = plot_M2016.plot_relative_velocity_difference_map(ax4, 100, "ref", "nnr")#, plateIDs = plateIDs)

for i in range(2):
    lon = M2016.globe.data["ref"].net_rotation_pole_lon.values[0]
    lat = M2016.globe.data["ref"].net_rotation_pole_lat.values[0]
    lon = (lon + 180) % 360 if i == 0 else lon
    lat = -lat if i == 0 else lat
    ax3.scatter(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        s=100,
        c="k",
        marker="*",
        zorder=1e2,
    )
    ax4.scatter(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        s=100,
        c="k",
        marker="*",
        zorder=1e2,
    )

# Create a new grid for the colorbar
cax1 = fig.add_axes([0.162, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar1 = plt.colorbar(vels[0], cax=cax1, orientation="horizontal", extend="max", extendfrac=25e-3)

# Set colorbar label
cbar1.set_label("Plate speed [cm/a]", labelpad=5)

# Create a new grid for the colorbar
cax2 = fig.add_axes([0.412, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar2 = plt.colorbar(vels_diff[0], cax=cax2, orientation="horizontal", extend="both", extendfrac=25e-3)

# Set colorbar label
cbar2.set_label("Plate speed difference [cm/a]", labelpad=5)

# Create a new grid for the colorbar
cax3 = fig.add_axes([0.662, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar3 = plt.colorbar(rel_vels_diff[0], cax=cax3, orientation="horizontal", extend="both", extendfrac=25e-3)

# Set colorbar label
cbar3.set_label("Log10(relative plate speed difference)", labelpad=5)

plt.show()

# %%
optimise_M2016 = Optimisation(M2016)

optimise_M2016.invert_residual_torque_v4(plateIDs = [901, 909, 911], cases = ["ref", "nnr"], NUM_ITERATIONS=5)

# %%
# optimise_M2016.invert_residual_torque_v4(plateIDs = [901, 909, 911], cases = ["ref", "nnr"], NUM_ITERATIONS=5)

# %%
# Copy the optimised slab pull constants to the synthetic case
for age in M2016.ages:
        M2016.slabs.data[age]["syn_ref"].slab_pull_constant = M2016.slabs.data[age]["ref"].slab_pull_constant.copy()
        M2016.slabs.data[age]["syn_nnr"].slab_pull_constant = M2016.slabs.data[age]["nnr"].slab_pull_constant.copy()

# Calculate synthetic plate velocities
M2016.calculate_all_torques(cases = ["syn_ref", "syn_nnr"], plateIDs = [901, 909, 911])
M2016.calculate_synthetic_velocity(cases = ["syn_ref", "syn_nnr"], plateIDs = [901, 909, 911])

# %%
# Create a figure and gridspec
fig = plt.figure(figsize=(fig_width, fig_height_maps), dpi=300)
gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.2)

ax1 = fig.add_subplot(gs[0, 0], projection=projection)
ax1.set_title("Synthetic velocities (based on MH)", fontsize=title_fontsize)
plot_M2016.plot_velocity_map(ax1, 0, "syn_ref")#, plateIDs = plateIDs)

ax2 = fig.add_subplot(gs[0, 1], projection=projection)
ax2.set_title("Synthetic velocities (based on NNR)", fontsize=title_fontsize)
vels = plot_M2016.plot_velocity_map(ax2, 0, "syn_nnr")#, plateIDs = plateIDs)

ax3 = fig.add_subplot(gs[1, 0], projection=projection)
ax3.set_title("Synthetic - reconstructed velocity (MH)", fontsize=title_fontsize)
vels_diff = plot_M2016.plot_velocity_difference_map(ax3, 0, "syn_ref", "ref", vmin=-5, vmax=5)#, plateIDs = plateIDs, vmin=-5, vmax=5, vector_scale=1e2)

ax4 = fig.add_subplot(gs[1, 1], projection=projection)
ax4.set_title("Synthetic - reconstructed velocity (NNR)", fontsize=title_fontsize)
plot_M2016.plot_velocity_difference_map(ax4, 0, "syn_nnr", "nnr", vmin=-5, vmax=5)#, plateIDs = plateIDs, vmin=-5, vmax=5, vector_scale=1e2)

# Create a new grid for the colorbar
cax1 = fig.add_axes([0.215, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar1 = plt.colorbar(vels[0], cax=cax1, orientation="horizontal", extend="max", extendfrac=25e-3)

# Set colorbar label
cbar1.set_label("Plate speed [cm/a]", labelpad=5)

# Create a new grid for the colorbar
cax2 = fig.add_axes([0.615, 0.06, 0.2, 0.02])

# Create a colorbar below the subplots
cbar2 = plt.colorbar(vels_diff[0], cax=cax2, orientation="horizontal", extend="both", extendfrac=25e-3)

# Set colorbar label
cbar2.set_label("Plate speed difference [cm/a]", labelpad=5)

plt.show()


# %%
_slab_data = M2016.slabs.data[0]
for case in M2016.cases:
    if "syn" in case:
        continue

    _slab_data[case] = _slab_data[case][_slab_data[case].lower_plateID.isin([901, 909, 911])]

fig, ax = plt.subplots()
ax.scatter(
    _slab_data["ref"].slab_pull_constant,
    _slab_data["nnr"].slab_pull_constant
)
ax.set_xlabel("Slab pull constant (ref)")
ax.set_ylabel("Slab pull constant (nnr)")
ax.grid(ls=":")

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
print(M2016.globe.data["nnr"].net_rotation_rate)

# %%
M2016.settings.options["ref"]["Minimum plate area"]

# %%
plt.scatter(
    M2016.points.data[0]["ref"].lon,
    M2016.points.data[0]["ref"].lat,
    c= M2016.points.data[0]["nnr"]. - M2016.points.data[0]["ref"].mantle_drag_force_mag,
)