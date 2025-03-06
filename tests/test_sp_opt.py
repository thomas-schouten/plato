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
# ages = np.arange(0, 51, 5)
ages = [80]

# Set directory to save the results
# results_dir = "01-Results"

path = "/Users/thomas/Documents/_Plato/Plato/sample_data/M2016"

# Load seafloor age grids
seafloor_age_grids = {}
for age in ages:
    seafloor_age_grids[age] = xr.open_dataset(f"{path}/seafloor_age_grids/M2016_SeafloorAgeGrid_{age}Ma.nc")

continental_grids = {}
for age in ages:
    continental_grids[age] = xr.open_dataset(f"{path}/continental_grids/M2016_ContinentalGrid_{age}Ma.nc")

# Set up PlateTorques object
M2016 = PlateTorques(reconstruction_name = reconstruction_name, ages = ages, seafloor_age_grids = seafloor_age_grids, continental_grids = continental_grids)

# Set up PlotReconstruction object
plot_M2016 = PlotReconstruction(M2016)

# %%
M2016.sample_all()
M2016.calculate_all_torques()
fig, ax = plt.subplots(figsize = (18*cm2in, 18*cm2in), subplot_kw={"projection": ccrs.Orthographic(central_longitude=-100, central_latitude=45)})
# ax.set_extent([120, 300, 10, 90], crs=ccrs.PlateCarree())
ax.set_global()
im, qu = plot_M2016.plot_torques_map(ax, age=0, minimum_plate_area=7.5e12, vector_scale=5e26, vector_linewidth=.5, plateIDs=101)
ax.gridlines(draw_labels=True)
plt.show()
# %%
# Calculate all torques
# M2016.settings.options["ref"]["Continental keels"] = True
M2016.settings.options["ref"]["Slab suction torque"] = True
M2016.sample_all()
M2016.calculate_slab_suction_torque()
M2016.calculate_residual_torque()
# Plot torques map
fig, ax = plt.subplots(figsize = (18*cm2in, 18*cm2in), subplot_kw={"projection": ccrs.Orthographic(central_longitude=-100, central_latitude=45)})
# ax.set_extent([120, 300, 10, 90], crs=ccrs.PlateCarree())
ax.set_global()
im, qu = plot_M2016.plot_torques_map(ax, age=0, minimum_plate_area=7.5e12, vector_scale=5e26, vector_linewidth=.5, plateIDs=101)
ax.gridlines(draw_labels=True)
plt.show()
# %%
_data = M2016.plates.data[0]["ref"].copy()
# print(_data.lower_plateID.unique())
# _data = _data[_data["lower_plateID"] == 101]
# fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
# ax.scatter(
#     _data["lon"],
#     _data["lat"],
#     c=_data["slab_pull_force_mag"],
#     transform = ccrs.PlateCarree(),
# )
# ax.set_global()
# ax.coastlines()

# %%
fig, ax = plt.subplots(figsize = (18*cm2in, 12*cm2in), subplot_kw={"projection": ccrs.Robinson(central_longitude=160)}, dpi=300)
im, qu = plot_M2016.plot_velocity_map(ax, age=80)
fig.colorbar(im, orientation = "horizontal", label="Speed [cm/a]", shrink=.8)
plt.show()
# %%

# %%
_data.residual_force_mag
# %%
M2016.plates.data[0]["ref"]["plateID"]
# %%
M2016.plates.data[0]["ref"]["slab_pull_torque_mag"]
# %%
M2016.plates.data[0]["ref"]["slab_suction_torque_mag"]
# %%
M2016.settings.options["ref"]["Slab suction torque"] = True
M2016.settings.options["ref"]["Slab suction constant"] = .5
M2016.sample_all()
M2016.calculate_all_torques()

optimise_M2016 = Optimisation(M2016)
optimise_M2016.minimise_residual_torque_v4(plateIDs=[901,911,201,101])
# %%
plot_age = 0
plt.scatter(
    M2016.slabs.data[plot_age]["ref"].slab_pull_force_mag,
    M2016.slabs.data[plot_age]["ref"].slab_suction_force_mag,
)
print(np.mean(M2016.slabs.data[plot_age]["ref"].slab_pull_force_mag/M2016.slabs.data[plot_age]["ref"].slab_suction_force_mag))
# %%
plot_age = 0
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
p=ax.scatter(
    M2016.slabs.data[plot_age]["ref"].lon,
    M2016.slabs.data[plot_age]["ref"].lat,
    c=M2016.slabs.data[plot_age]["ref"].slab_suction_force_mag,
    # vmin=0, vmax=1,
    transform = ccrs.PlateCarree()
)
fig.colorbar(p, orientation="horizontal")
ax.quiver(
    M2016.slabs.data[plot_age]["ref"].lon,
    M2016.slabs.data[plot_age]["ref"].lat,
    M2016.slabs.data[plot_age]["ref"].slab_suction_force_lon,
    M2016.slabs.data[plot_age]["ref"].slab_suction_force_lat,
    # vmin=0, vmax=1,
    transform = ccrs.PlateCarree()
)
ax.set_global()
plt.show()

# %%
M2016.calculate_slab_pull_torque()
M2016.calculate_gpe_torque()
for i in range(0,3):
    # Sample seafloor ages
    M2016.sample_seafloor_ages()

    # Sample LAB depth
    M2016.calculate_lab_depths()
    
    if i > 0:
        M2016.settings.options["ref"]["Continental keels"] = True
    else:
        M2016.settings.options["ref"]["Continental keels"] = False
    
    if i > 1:
        M2016.settings.options["ref"]["Slab suction torque"] = True
        M2016.calculate_slab_suction_torque()
    else:
        M2016.settings.options["ref"]["Slab suction torque"] = False

    print(i, M2016.settings.options["ref"]["Continental keels"], M2016.settings.options["ref"]["Slab suction torque"])    
    
    M2016.calculate_mantle_drag_torque()

    optimise_M2016 = Optimisation(M2016)
    optimise_M2016.minimise_residual_torque_v4(plateIDs=[901, 911, 201, 101])
# %%

# %%
M2016.plates.data[0]["ref"].mantle_drag_torque_mag

# %%
plt.scatter(
    M2016.points.data[0]["ref"]["lon"],
    M2016.points.data[0]["ref"]["lat"],
    c=M2016.points.data[0]["ref"]["mantle_drag_force_mag"],
)
plt.colorbar()
# M2016.points.data[0]["ref"]["mantle_drag_force_mag"]

# %%
for index, row in M2016.plates.data[0]["ref"].iterrows():
    print(row.plateID, row.mean_LAB_depth)

# %%
print(np.sum(M2016.plates.data[0]["ref"]["mean_LAB_depth"] * M2016.plates.data[0]["ref"]["area"]) / np.sum(M2016.plates.data[0]["ref"]["area"]))
# %%
M2016.calculate_gpe_torque()
M2016.points.data[0]["ref"]["lithospheric_mantle_thickness"]
# %%
# plt.scatter(
#     M2016.points.data[0]["ref"]["lon"],
#     M2016.points.data[0]["ref"]["lat"],
#     c=M2016.points.data[0]["ref"]["lithospheric_mantle_thickness"] + M2016.points.data[0]["ref"]["crustal_thickness"],
#     cmap="viridis",
#     vmin=1.2e5, vmax=2.5e5
# )
plt.scatter(
    M2016.points.data[0]["ref"]["lon"],
    M2016.points.data[0]["ref"]["lat"],
    c=M2016.points.data[0]["ref"]["LAB_depth"],
    cmap="viridis",
    vmin=1.2e5, vmax=2.5e5
)
plt.colorbar()
plt.show()

# %%
plt.scatter(
    M2016.points.data[0]["ref"]["lon"],
    M2016.points.data[0]["ref"]["lat"],
    c=M2016.points.data[0]["ref"]["LAB_depth"],
    cmap="viridis",
    vmin=0, vmax=2.5e5
)
plt.colorbar()
plt.show()

# %%

# Calculate torques
M2016.calculate_all_torques()

driving_torque_ref = M2016.extract_data_through_time(ages = ages[0], var = "driving_torque_mag")[901].values[0]
residual_torque_ref = M2016.extract_data_through_time(ages = ages[0], var = "residual_torque_mag")[901].values[0]

x_values = np.arange(10., 12., .1)
# results = []
colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(x_values)))

# Initialise empty lists
driving_torque = []; residual_torque = []
driving_torque_opt = []; residual_torque_opt = []

with contextlib.redirect_stdout(open(os.devnull, 'w')):

    for i, x in enumerate(x_values):
        # Start with a fresh copy
        M2016.calculate_all_torques()
        _data = {}
        for age in ages:
            # Select data
            _data[age] = {}
            _data[age]["ref"] = M2016.slabs.data[age]["ref"].copy()

            _data[age]["ref"]["slab_pull_force_mag"] -= (_data[age]["ref"]["residual_force_lat"] *_data[age]["ref"]["slab_pull_force_lat"] + _data[age]["ref"]["residual_force_lon"] *_data[age]["ref"]["slab_pull_force_lon"]) * 10**-x

            _data[age]["ref"]["slab_pull_force_lat"] = np.cos(np.deg2rad(_data[age]["ref"]["trench_normal_azimuth"])) * _data[age]["ref"]["slab_pull_force_mag"]
            _data[age]["ref"]["slab_pull_force_lon"] = np.sin(np.deg2rad(_data[age]["ref"]["trench_normal_azimuth"])) * _data[age]["ref"]["slab_pull_force_mag"]
            
            M2016.plates.calculate_torque_on_plates(_data, ages = age, torque_var="slab_pull")

            M2016.calculate_driving_torque(ages = age, plateIDs = [901])
            M2016.calculate_residual_torque(ages = age, plateIDs = [901])

            # Extract the driving and residual torques
            driving_torque_opt.append(M2016.extract_data_through_time(ages = age, var = "driving_torque_mag")[901].values[0])
            residual_torque_opt.append(M2016.extract_data_through_time(ages = age, var = "residual_torque_mag")[901].values[0])

# Convert the lists to numpy arrays
# driving_torque = np.array(driving_torque); residual_torque = np.array(residual_torque)
driving_torque_opt = np.array(driving_torque_opt); residual_torque_opt = np.array(residual_torque_opt)

# results = np.log10(residual_torque_ref/driving_torque_ref) - np.log10(residual_torque_opt/driving_torque_opt)

plt.plot(residual_torque_ref*np.ones_like(driving_torque_opt)/driving_torque_ref*np.ones_like(residual_torque_opt), label="Reference")
plt.plot(residual_torque_opt/driving_torque_opt, label="Optimised")
plt.xlabel("x value")
plt.ylabel("Normalised residual torque")
plt.legend()
plt.show()

# %%
print(x_values[np.argmin(residual_torque_opt/driving_torque_opt)])

# %%
M2016_v2 = PlateTorques(reconstruction_name = reconstruction_name, ages = np.arange(0, 51, 5), files_dir="output")
M2016_v2.sample_seafloor_ages()

M2016_v2.calculate_all_torques()

driving_torque_ref_v2 = M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "driving_torque_mag")
residual_torque_ref_v2 = M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "residual_torque_mag")

optimise_M2016_v2 = Optimisation(M2016_v2)

optimise_M2016_v2.optimise_slab_pull_coefficient()
optimise_M2016_v2.optimise_torques()

# %%
driving_torque_opt_v3 = optimise_M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "driving_torque_mag")
residual_torque_opt_v3 = optimise_M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "residual_torque_mag")
# %%

driving_torque_opt_v2 = {}
residual_torque_opt_v2 = {}

x_values = np.arange(9.5, 12., .1)
for x in x_values:
    print(f"Optimising for x = {x}")
    _data = {}

    # Calculate the torques the normal way
    M2016_v2.calculate_slab_pull_torque()
    M2016_v2.calculate_driving_torque()
    M2016_v2.calculate_residual_torque()

    # Modify the slab pull forces
    for age in M2016_v2.ages:
        _data[age] = {}
        _data[age]["ref"] = M2016_v2.slabs.data[age]["ref"].copy()

        _data[age]["ref"]["slab_pull_force_mag"] -= (_data[age]["ref"]["residual_force_lat"] *_data[age]["ref"]["slab_pull_force_lat"] + _data[age]["ref"]["residual_force_lon"] *_data[age]["ref"]["slab_pull_force_lon"]) * 10**-x

        _data[age]["ref"]["slab_pull_force_lat"] = np.cos(np.deg2rad(_data[age]["ref"]["trench_normal_azimuth"])) * _data[age]["ref"]["slab_pull_force_mag"]
        _data[age]["ref"]["slab_pull_force_lon"] = np.sin(np.deg2rad(_data[age]["ref"]["trench_normal_azimuth"])) * _data[age]["ref"]["slab_pull_force_mag"]
    
    # Calculate the torques with the modified slab pull forces
    M2016_v2.plates.calculate_torque_on_plates(_data, np.arange(0, 51, 5), torque_var="slab_pull")
    M2016_v2.calculate_driving_torque(np.arange(0, 51, 5))
    M2016_v2.calculate_residual_torque(np.arange(0, 51, 5))

    # Extract the driving and residual torques
    driving_torque_opt_v2[x] = M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "driving_torque_mag")
    residual_torque_opt_v2[x] = M2016_v2.extract_data_through_time(ages = np.arange(0, 51, 5), var = "residual_torque_mag")

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

cm2in = 0.3937008
plate_names = ["Pacific", "Nazca", "South America", "Farallon"]
plateIDs = [901, 911, 201, 902]
colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(x_values)))
fig, ax = plt.subplots(figsize=(fig_width, fig_height_graphs), dpi=300)
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.1, hspace=0.2)

k = 0
for m in range(2):
    for j in range(2):
        ax = plt.subplot(gs[m, j])
        ax.set_title(plate_names[k], fontweight="bold")
        ax.plot(M2016_v2.ages, np.log10(residual_torque_ref_v2[plateIDs[k]]/driving_torque_ref_v2[plateIDs[k]]), label="Reference", color="k", lw=2, zorder=100)
        ax.plot(M2016_v2.ages, np.log10(residual_torque_opt_v3[plateIDs[k]]/driving_torque_opt_v3[plateIDs[k]]), label="Optimised", color="k", ls="--", zorder=100)
        
        for i, x in enumerate(residual_torque_opt_v2.keys()):
            if x < 9.5 or x > 12:
                continue
            ax.plot(M2016_v2.ages, np.log10(residual_torque_opt_v2[x][plateIDs[k]]/driving_torque_opt_v2[x][plateIDs[k]]), color=colours[i], label=f"x = {x:.1f}")
        
        ax.grid(ls=":")
        
        ax.set_xlim(0, len(M2016_v2.ages))
        ax.set_ylim(-2, 1)

        ax.set_xticks(np.arange(M2016_v2.ages.min(), M2016_v2.ages.max()+1, 5))
        # ax.set_yticks(np.arange(0, 26, 5))

        ax.set_xticklabels([]) if m == 0 else None
        ax.set_yticklabels([]) if j == 1 else None

        ax.set_xlabel("Age [Ma]") if m == 1 else None
        ax.set_ylabel("Log10(Normalised\nresidual torque)") if j == 0 else None

        ax.annotate(annotations[m], xy=(0, 1.03), xycoords="axes fraction", fontweight="bold")

        k += 1

# Add a legend below the entire figure
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.15))
fig.savefig("/Users/thomas/Documents/_Plato/Reconstruction_analysis/Figures/M2016_preliminary_optimisation_results.png", bbox_inches="tight")
plt.show()

# %%
colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(x_values)))
for i, x in enumerate(residual_torque_opt_v2.keys()):
    if x < 10.8:
        continue
    plt.plot(residual_torque_opt_v2[x][911]/driving_torque_opt_v2[x][911], color=colours[i], label=f"x = {x:.1f}")
plt.plot(residual_torque_ref_v2[911]/driving_torque_ref_v2[911], label="Reference", color="k", lw=2)
plt.legend()
plt.semilogy()
# %%
plt.plot(residual_torque_ref_v2[901]/driving_torque_ref_v2[901], label="Reference", color="r")
plt.plot(residual_torque_opt_v2[901]/driving_torque_opt_v2[901], label="Optimised", color="r", ls="--")
plt.plot(residual_torque_ref_v2[911]/driving_torque_ref_v2[911], color="b")
plt.plot(residual_torque_opt_v2[911]/driving_torque_opt_v2[911], color="b", ls="--")
plt.plot(residual_torque_ref_v2[201]/driving_torque_ref_v2[201], label="Reference", color="g")
plt.plot(residual_torque_opt_v2[201]/driving_torque_opt_v2[201], label="Optimised", color="g", ls="--")
plt.plot(residual_torque_ref_v2[902]/driving_torque_ref_v2[902], color="k")
plt.plot(residual_torque_opt_v2[902]/driving_torque_opt_v2[902], color="k", ls="--")

plt.legend()
plt.semilogy()

# %%
residual_torque_opt_v2