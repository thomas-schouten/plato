# %%
import os
import sys
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import contextlib
import xarray as xr
import pandas as pd
import pygplates as _pygplates

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def mag_azi2lat_lon(
        magnitude,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        azimuth,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
    ):
    """
    Decompose a vector defined by magnitude and azimuth into latitudinal and longitudinal components.

    :param magnitude:   Magnitude of vector.
    :type magnitude:    list, numpy.array, pandas.Series
    :param azimuth:     Azimuth of vector in degrees.
    :type azimuth:      numpy.array, pandas.Series

    :return:            Latitudinal and longitudinal components.
    :rtype:             float or numpy.array, float or numpy.array
    """
    # Ensure inputs are NumPy arrays
    magnitude = np.asarray(magnitude)
    azimuth = np.array(azimuth)

    # Convert azimuth from degrees to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Calculate components
    component_lat = np.cos(azimuth_rad) * magnitude
    component_lon = np.sin(azimuth_rad) * magnitude

    return component_lat, component_lon

def tangent_cartesian2spherical(
        vectors_xyz,## np.ndarray,
        points_lat,#:,# Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        points_lon,#:,# Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        PARALLEL_MODE,#:# bool = False,
    ):
    """
    Convert a vector that is tangent to the surface of a sphere to spherical coordinates.

    

    NOTE:   This function is probably the slowest in Plato.
            It could be sped up by actually implementing the mathematics using numpy.
    """
    def _convert_vector_to_lat_lon(point_lat, point_lon, vector_xyz):
        """
        Convert a vector tangent to the surface of a sphere to latitudinal and longitudinal components.
        """
        # Make PointonSphere
        point = _pygplates.PointOnSphere(point_lat, point_lon)

        # Convert vector to magnitude, azimuth, and inclination
        return np.asarray(
            _pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point, 
                (vector_xyz[0], vector_xyz[1], vector_xyz[2])
            )
        )

    # Ensure inputs are NumPy arrays
    points_lat = np.asarray(points_lat)
    points_lon = np.asarray(points_lon)

    # Initialise result arrays
    vectors_mag = np.zeros_like(points_lat)
    vectors_azi = np.zeros_like(points_lat)

    for i, (point_lat, point_lon, vector_xyz) in enumerate(zip(points_lat, points_lon, vectors_xyz)):
        # Convert vector to magnitude and azimuth
        vectors_mag[i], vectors_azi[i], _ = _convert_vector_to_lat_lon(point_lat, point_lon, vector_xyz)

    # Convert azimuth from radians to degrees
    vectors_azi = np.rad2deg(vectors_azi)
    
    # Convert to latitudinal and longitudinal components
    vectors_lat, vectors_lon = mag_azi2lat_lon(vectors_mag, vectors_azi)

    return vectors_lat, vectors_lon, vectors_mag, vectors_azi

def tangent_spherical2cartesian(
        points_lat,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series], 
        points_lon,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series], 
        vectors_lat,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series], 
        vectors_lon,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series], 
    ):
    """
    Convert a vector tangent to the surface of a sphere to Cartesian coordinates.

    :param points_lat:  latitude of the vector origins in degrees.
    :type points_lat:   list, numpy.array, pandas.Series
    :param points_lon:  longitude in the vector origin in degrees.
    :type points_lon:   list, numpy.array, pandas.Series
    :param vectors_lat: latitudinal vector components.
    :type vectors_lat:  numpy.array, pandas.Series
    :param vectors_lon: longitudinal vector components.
    :type vectors_lon:  numpy.array, pandas.Series

    :return:            vectors in Cartesian coordinates.
    :rtype:             numpy.array
    """
    # Ensure inputs are NumPy arrays
    points_lat = np.asarray(points_lat)
    points_lon = np.asarray(points_lon)

    # Convert lon, lat to radian
    points_lon_rad = np.deg2rad(points_lon)
    points_lat_rad = np.deg2rad(points_lat)

    # Calculate force_magnitude
    vectors_mag = np.linalg.norm([vectors_lat, vectors_lon], axis=0)

    # Calculate theta
    theta = np.empty_like(vectors_lon)
    mask = ~np.logical_or(vectors_lon == 0, np.isnan(vectors_lon), np.isnan(vectors_lat))
    theta[mask] = np.where(
        (vectors_lon[mask] > 0) & (vectors_lat[mask] >= 0),  
        np.arctan(vectors_lat[mask] / vectors_lon[mask]),                          
        np.where(
            (vectors_lon[mask] < 0) & (vectors_lat[mask] >= 0) | (vectors_lon[mask] < 0) & (vectors_lat[mask] < 0),    
            np.pi + np.arctan(vectors_lat[mask] / vectors_lon[mask]),              
            (2*np.pi) + np.arctan(vectors_lat[mask] / vectors_lon[mask])           
        )
    )

    # Calculate force in Cartesian coordinates
    vectors_x = vectors_mag * np.cos(theta) * (-1.0 * np.sin(points_lon_rad))
    vectors_y = vectors_mag * np.cos(theta) * np.cos(points_lon_rad)
    vectors_z = vectors_mag * np.sin(theta) * np.cos(points_lat_rad)

    # Convert to numpy array
    vectors_xyz = np.asarray([vectors_x, vectors_y, vectors_z])

    return vectors_xyz   

def geocentric_spherical2cartesian(
        points_lat,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        points_lon,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        vectors_mag,#: Union[int, float, List[Union[int, float]], np.ndarray, _pandas.Series] = 1
    ):
    """
    Convert latitude and longitude to Cartesian coordinates.

    

    By default, the Cartesian coordinates are calculated on the unit sphere.
    """
    # Ensure inputs are NumPy arrays
    points_lat = np.asarray(points_lat)
    points_lon = np.asarray(points_lon)
    vectors_mag = np.asarray(vectors_mag)

    # Convert to radians
    points_lat_rad = np.deg2rad(points_lat)
    points_lon_rad = np.deg2rad(points_lon)

    # Calculate x, y, z
    x = vectors_mag * np.cos(points_lat_rad) * np.cos(points_lon_rad)
    y = vectors_mag * np.cos(points_lat_rad) * np.sin(points_lon_rad)
    z = vectors_mag * np.sin(points_lat_rad)

    return x, y, z

# %%
# Load data
results_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "output_notebooks", "02-Keels-slab_suction-sediment_subduction"))

points = pd.read_parquet(f"{results_dir}/Points/Points_Muller2016_ref_0Ma.parquet")
plates = pd.read_parquet(f"{results_dir}/Plates/Plates_Muller2016_ref_0Ma.parquet")

n_am_plate = plates[plates.plateID == 901]
n_am_point = points[points.plateID == 901]

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic(central_latitude=45, central_longitude=-110)})
p = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=n_am_point.mantle_drag_force_mag,
    cmap="viridis",
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(p, label="Mantle drag force (N)")
plt.show()

# %%
# Get the mantle drag force and the weights of all points
mantle_drag_force = n_am_point.mantle_drag_force_mag.values
weights = n_am_point.segment_length_lon.values
sum_weights = np.sum(weights)
asthenospheric_thickness = np.where(
    (n_am_point.seafloor_age.isna()) & (n_am_point.LAB_depth > 100e3), 
    200e3 - (n_am_point.LAB_depth-100e3),
    200e3,
)

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic(central_latitude=45, central_longitude=-110)})
p = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=asthenospheric_thickness/1e3,
    cmap="viridis",
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(p, label="Asthenospheric thickness [km]")
plt.show()

# %%
# Convert the positions to Cartesian coordinates
positions_xyz = np.column_stack(geocentric_spherical2cartesian(
    n_am_point.lat.values,
    n_am_point.lon.values,
    1,
))

# Convert the forces to Cartesian coordinates
forces_xyz = tangent_spherical2cartesian(
    n_am_point.lat.values,
    n_am_point.lon.values,
    n_am_point.mantle_drag_force_lat.values * n_am_point.segment_length_lon.values * n_am_point.segment_length_lon.values,
    n_am_point.mantle_drag_force_lon.values * n_am_point.segment_length_lon.values * n_am_point.segment_length_lat.values
)

print("Positions shape:", positions_xyz.shape)
print("Forces shape:", forces_xyz.shape)

torques_xyz = np.cross(positions_xyz.T, forces_xyz, axis=0)

print("Torque shape:", torques_xyz.shape)


# %%

# Get the mantle drag torque in Cartesian coordinates
torque_xyz = np.array([n_am_plate.mantle_drag_torque_x.values[0], n_am_plate.mantle_drag_torque_y.values[0], n_am_plate.mantle_drag_torque_z.values[0]])

# %%
# Calculate the velocity at each point
velocity_xyz = (weights / 1.25e20) * asthenospheric_thickness * np.cross(torque_xyz, positions_xyz, axis=0)
velocity_xyz /= sum_weights

# %%
# Calculate the velocity at each point
N = len(n_am_point.lat.values)  # Number of points (assuming positions_xyz has N rows)

# Function to compute the cross-product matrix for each column of r
def cross_matrix(vec):
    """Convert a 3D vector to a 3x3 skew-symmetric cross-product matrix."""
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

# Apply weights to positions
positions_xyz *= weights[:, None]

# Initialize cross product matrices for each column of positions_xyz
cross_matrices = np.zeros((N, 3, 3))  # Shape (N, 3, 3)
for i in range(N):
    cross_matrices[i] = cross_matrix(positions_xyz[i, :])  # Populate each cross product matrix

# Initialize an array to store the calculated velocity at each point (syn_mantle_drag_force)
syn_mantle_drag_force = np.zeros_like(positions_xyz)  # Shape (N, 3)

# Solve for each column using np.linalg.pinv (pseudo-inverse) for numerical stability
# Since you're using the same torque across all points, you can compute once
torque_i = torque_xyz.reshape(3, 1)  # The same torque vector for each point (3x1)

for i in range(N):
    # Use np.linalg.pinv to compute the pseudo-inverse of the cross product matrix
    # Since we are solving r × v = torque_xyz, we need to invert the cross product matrix and multiply by torque
    syn_mantle_drag_force[i, :] = (np.linalg.pinv(cross_matrices[i]) @ torque_i).flatten()  # Solve for v[i]

# Adjust by sum_weights (check if sum_weights is correctly calculated)
syn_mantle_drag_force /= sum_weights

print("Solved syn_mantle_drag_force shape:", syn_mantle_drag_force.shape)

# %%
# Assuming positions_xyz (N, 3) and torque_xyz (3, N) are defined
N = len(n_am_point.lat.values)  # Number of points (assuming positions_xyz has N rows)

# Function to compute the cross-product matrix for each column of r
def cross_matrix(vec):
    """Convert a 3D vector to a 3x3 skew-symmetric cross-product matrix."""
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

# Apply weights to positions
positions_xyz *= weights[:, None]

# Initialize cross product matrices for each column of positions_xyz
cross_matrices = np.zeros((N, 3, 3))  # Shape (N, 3, 3)
for i in range(N):
    cross_matrices[i] = cross_matrix(positions_xyz[i, :])  # Populate each cross product matrix

# Initialize an array to store the calculated velocity at each point (syn_mantle_drag_force)
syn_mantle_drag_force = np.zeros_like(positions_xyz)  # Shape (N, 3)

# You already have torques_xyz which is the sum of individual torques
# Iterate over the positions and solve for the velocity at each point using pseudo-inverse
for i in range(N):
    # Since we are solving r × v = torques_xyz, use np.linalg.pinv to compute the pseudo-inverse
    # for each cross product matrix and multiply by the torque (same for each point)
    syn_mantle_drag_force[i, :] = (np.linalg.pinv(cross_matrices[i]) @ torques_xyz).flatten()  # Solve for v[i]

# Adjust by sum_weights (check if sum_weights is correctly calculated)
syn_mantle_drag_force /= sum_weights

print("Solved syn_mantle_drag_force shape:", syn_mantle_drag_force.shape)

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic(central_latitude=45, central_longitude=-110)})
q = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=np.log10(np.linalg.norm(syn_mantle_drag_force, axis=1)),# / mantle_drag_force),
    cmap="viridis",
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force")
plt.show()

# %%
print(syn_mantle_drag_force)

# %%

# Convert rotation pole to spherical coordinates
pole_x, pole_y, pole_z = geocentric_spherical2cartesian(
    n_am_plate.pole_lat.values,
    n_am_plate.pole_lon.values,
    n_am_plate.pole_angle.values,
)
print(-n_am_plate.mantle_drag_torque_x.values[0]/pole_x)
print(-n_am_plate.mantle_drag_torque_y.values[0]/pole_y)
print(-n_am_plate.mantle_drag_torque_z.values[0]/pole_z)

# %%
# Load data
results_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "output_notebooks", "02-Keels-slab_suction-sediment_subduction"))

case = "ref"
points = pd.read_parquet(f"{results_dir}/Points/Points_Muller2016_{case}_0Ma.parquet")
plates = pd.read_parquet(f"{results_dir}/Plates/Plates_Muller2016_{case}_0Ma.parquet")

forces_mag = np.zeros_like(points.plateID); forces_lat = np.zeros_like(points.plateID); forces_lon = np.zeros_like(points.plateID)
error = np.zeros_like(points.plateID); lat = np.zeros_like(points.plateID); velocity = np.zeros_like(points.plateID)
for plateID in plates.plateID:
    n_am_plate = plates[plates.plateID == plateID]
    n_am_point = points[points.plateID == plateID]

    # Convert the positions to Cartesian coordinates
    positions_xyz = np.column_stack(geocentric_spherical2cartesian(
        n_am_point.lat.values,
        n_am_point.lon.values,
        6371e3,
    )).T

    # Convert the forces to Cartesian coordinates
    forces_xyz = tangent_spherical2cartesian(
        n_am_point.lat.values,
        n_am_point.lon.values,
        n_am_point.mantle_drag_force_lat.values * n_am_point.segment_length_lon.values * n_am_point.segment_length_lat.values,
        n_am_point.mantle_drag_force_lon.values * n_am_point.segment_length_lon.values * n_am_point.segment_length_lat.values
    )

    # print("Positions shape:", positions_xyz.shape)
    # print("Forces shape:", forces_xyz.shape)

    # Get the mantle drag torque in Cartesian coordinates
    torque_xyz = np.array([n_am_plate.mantle_drag_torque_x.values[0], n_am_plate.mantle_drag_torque_y.values[0], n_am_plate.mantle_drag_torque_z.values[0]])

    torques_xyz = np.cross(positions_xyz, forces_xyz, axis=0)

    total_torque = np.sum(torques_xyz, axis=1)

    # print("Total torque:", total_torque)
    # print("Original total torque:", torque_xyz)

    # Step 1: Compute the cross-product matrices for each position vector
    def cross_matrix(vec):
        """Convert a 3D vector into a 3x3 skew-symmetric cross-product matrix."""
        return np.array([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

    N = len(n_am_point.lat.values)  # Number of points (assuming positions_xyz has N rows)
    cross_matrices = np.array([cross_matrix(positions_xyz[:, i]) for i in range(N)])  # Shape (N, 3, 3)

    # Step 2: Solve for forces_xyz using least squares
    forces_xyz_new = np.zeros((3, N))  # Ensure correct shape (3, N)

    # print("Forces shape:", forces_xyz_new.shape)

    for i in range(N):
        forces_xyz_new[:, i], _, _, _ = np.linalg.lstsq(cross_matrices[i], torque_xyz, rcond=None)

    # Step 3: Scale the result properly
    if "keels" not in case:
        asthenospheric_thicknesses = np.ones_like(n_am_point.LAB_depth.values) * 200e3
    else:
        asthenospheric_thicknesses = np.where(
            (n_am_point.seafloor_age.isna()) & (n_am_point.LAB_depth > 100e3),
            200e3 - (n_am_point.LAB_depth-100e3),
            200e3,
        )
    weights = asthenospheric_thicknesses / np.sum(asthenospheric_thicknesses) / 1.25e20 * n_am_point.segment_length_lon.values / np.sum(n_am_point.segment_length_lon.values) * 6731e3**2 - (268.2*1/(90-np.abs(n_am_point.lat.values))+1)

    # weights = 200e3 / 1.25e20 * 16

    forces_xyz_new *= weights

    # Convert the forces back to spherical coordinates
    forces_sph = tangent_cartesian2spherical(
        forces_xyz_new.T,
        n_am_point.lat.values,
        n_am_point.lon.values,
        PARALLEL_MODE=False,
    )

    forces_lat[points.plateID == plateID] = forces_sph[0]
    forces_lon[points.plateID == plateID] = forces_sph[1]
    forces_mag[points.plateID == plateID] = forces_sph[2]

# Get error
error = points.mantle_drag_force_mag-forces_mag
lat = points.lat
velocity = points.velocity_mag


# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    points.lon,
    points.lat,
    c=forces_mag,
    cmap="viridis",
    # vmin=-1, vmax=3e6,
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Log10(Reconstructed/synthetic mantle drag force)", orientation="horizontal")
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    points.lon,
    points.lat,
    c=np.log10(points.mantle_drag_force_mag/forces_mag),
    cmap="RdBu",
    # vmin=-1, vmax=1,
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Log10(Reconstructed/synthetic mantle drag force)", orientation="horizontal")
plt.show()

# %%
print(np.sum(np.isnan(forces_lat)))

# %%
print(np.nanmean(error))
print(np.nanmedian(error))

plt.scatter(error, lat, s=1)
plt.show()
plt.scatter(error, velocity, s=1, c=lat)

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    points.lon,
    points.lat,
    c=forces_mag,
    cmap="viridis",
    s=4,
    vmin=0, vmax=2.5e6,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force", orientation="horizontal")
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    points.lon,
    points.lat,
    c=points.mantle_drag_force_mag,
    cmap="viridis",
    s=4,
    vmin=0, vmax=2.5e6,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force", orientation="horizontal")
plt.show()

# %%
prefactor=268.2
error = np.array(error)
lat = np.array(lat)
plt.scatter(error, velocity, s=1)
# plt.scatter(prefactor*1/(90-np.abs(lat))+1, lat, s=1)
# plt.xlim(0, 4)

# %%
plt.scatter(
    n_am_point.mantle_drag_force_lat,
    forces_lat
)

# %%
from scipy.optimize import curve_fit
# Your data
error = np.array(error)
lat = np.array(lat)

# Remove NaNs
error_filtered = error[~np.isnan(lat) & ~np.isnan(error)]
lat_filtered = lat[~np.isnan(lat) & ~np.isnan(error)]

# Define an exponential function or Gaussian-like model
def exp_fit(x, a, b, c, d):
    return a * np.exp(b * (x - c)**2) + d

# Perform the fit using curve_fit
params, covariance = curve_fit(exp_fit, lat_filtered, error_filtered, p0=[1, -0.1, 0, 0])  # Initial guesses for the parameters

# Get the fitted parameters
a, b, c, d = params
print("Fitted parameters:", a, b, c, d)

# Plot the original data and the fitted polynomial
plt.scatter(error, lat, s=1, label="Original Data")
plt.plot(lat, exp_fit(lat, *params), label="Exponential fit", color="red")
# plt.xlim(0, 4)
plt.xlabel('Error')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# %%
prefactor = 268.2
plt.scatter(error, np.abs(lat), s=1, label="Original Data")
plt.scatter(1/(90-np.abs(lat))*prefactor, np.abs(lat), s=1)
# plt.plot(exp_fit(np.abs(lat), 1, 1, 2, 1), lat, label="Exponential fit", color="red")
plt.xlim(0, 4)
plt.xlabel('Error')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# plt.scatter(error-1/(90-np.abs(lat))*prefactor, lat, s=1)

# %%
print(np.nanmean(error-1/(90-np.abs(lat))*prefactor))
# %%
prefactors = np.arange(260, 280, .1)
misfit = []
for prefactor in prefactors:
    # print(prefactor)
    misfit.append(np.abs(np.nanmean(error-prefactor*1/(90-np.abs(lat))+1)))
    # misfit.append(np.abs(np.nanmean(error-prefactor*1/(90-np.abs(lat))+1)))

opt_prefactor = prefactors[np.argmin(misfit)]
print(opt_prefactor)
print(np.min(misfit))
plt.plot(prefactors, misfit)
# plt.scatter(prefactors, opt_prefactor, color="red")
plt.show()
# %%

np.savetxt("error.txt", error)
np.savetxt("lat.txt", lat)
# %%
exponent = 2
error = np.array(error)
lat = np.array(lat)
plt.scatter(velocity, error, s=1)
plt.ylim(0,4)
# plt.scatter(np.abs(lat)**exponent/(90**exponent)+1, lat, s=1)




# %%
print(np.nanmedian(error))
print(np.nanmean(error))
# %%

# for i in range(N):
#     forces_xyz_new[:, i] = np.cross(torque_xyz, positions_xyz[:, i])

# Step 3: Scale the result properly
# weights = n_am_point.segment_length_lon.values * n_am_point.segment_length_lat.values

# print("Weights shape:", weights[:, None].shape)
# print("Forces shape:", forces_xyz_new.shape)

# print("Sum of weights:", np.sum(weights))
# print("Min weight:", np.min(weights), "Max weight:", np.max(weights))

# forces_xyz_new *= weights[:, None].T / np.sum(weights)

# Convert back to spherical coordinates


# # Step 4: Compute recovered torques for verification
# recovered_torques_xyz = np.cross(positions_xyz, forces_xyz_new, axis=0)
# total_recovered_torque = np.sum(recovered_torques_xyz, axis=1)

# print("Total recovered torque:", total_recovered_torque)

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=forces_mag,
    cmap="viridis",
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force")
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=160)})
q = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=n_am_point.mantle_drag_force_mag,
    cmap="viridis",
    s=4,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force")
plt.show()

# %%
print(np.radians(1))

# %%
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Orthographic(central_latitude=-15, central_longitude=-70)})
q = ax.scatter(
    n_am_point.lon,
    n_am_point.lat,
    c=n_am_point.mantle_drag_force_mag/forces_mag,
    cmap="RdBu",
    s=4,
    vmin=.7, vmax=1.3,
    transform=ccrs.PlateCarree(),
)
ax.set_global()
ax.coastlines()
fig.colorbar(q, label="Mantle drag force")
plt.show()

# %%
print(np.median(n_am_point.mantle_drag_force_mag/forces_mag))

# print(*np.pi)
# %%

# Step 2: Create a diagonal weight matrix W (shape: N x N)
W = np.diag(weights)

# Step 3: Convert total torque into a column vector
total_torque = torque_xyz[:, None]  # Shape (3, 1)

# Step 4: Create a matrix to store the torques at each point
# We want to find T_i, the individual torques for each point
# Solve for the individual torques by dividing the total torque appropriately

# Step 5: Solve for the individual torques using least squares or by scaling
# Since W is diagonal, we can directly scale the total torque

# The formula to recover individual torques for each point: 
# T_i = (total_torque * weights) / sum(weights)

# Reshape weights to have shape (N, 1) so that the multiplication can be done element-wise
weights_reshaped = weights  # Shape (N, 1)

# Step 6: Recover the individual torques
recovered_torques_xyz = (total_torque * weights_reshaped) / np.sum(weights)

# Step 7: To check, sum the recovered torques weighted by their respective weights
recovered_total_torque = np.sum(recovered_torques_xyz * weights_reshaped.T, axis=0)

# Step 8: Output results
print("Recovered individual torques (scaled by weights):")
print(recovered_torques_xyz)

print("Recovered total torque:", recovered_total_torque.shape)
print("Original total torque:", total_torque.shape)

plt.scatter(
    recovered_torques_xyz[0],
    torques_xyz[0],
)
plt.show()

plt.scatter(
    recovered_torques_xyz[1],
    torques_xyz[1],
)
plt.show()

plt.scatter(
    recovered_torques_xyz[2],
    torques_xyz[2],
)
plt.show()

# %%
print(weights)
