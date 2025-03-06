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

def geocentric_cartesian2spherical(
        vectors_x,#: Union[int, float, List[Union[int, float]], np.ndarray, _pandas.Series],
        vectors_y,#: Optional[Union[int, float, List[Union[int, float]], np.ndarray, _pandas.Series]] = None,
        vectors_z,#: Optional[Union[int, float, List[Union[int, float]], np.ndarray, _pandas.Series]] = None,
    ):# -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to latitude, longitude, magnitude, and azimuth.
    """
    # If only x is provided as a 1D array, unpack it into x, y, z
    # This makes the function more flexible.
    if vectors_y is None and vectors_z is None:
        vectors_x, vectors_y, vectors_z = vectors_x[0], vectors_x[1], vectors_x[2]

    # Convert integers and floats to lists
    if isinstance(vectors_x, (int, float, np.integer, np.floating)):
        vectors_x = [vectors_x]
    if isinstance(vectors_y, (int, float, np.integer, np.floating)):
        vectors_y = [vectors_y]
    if isinstance(vectors_z, (int, float, np.integer, np.floating)):
        vectors_z = [vectors_z]

    # Ensure x, y, z are NumPy arrays
    vectors_x = np.asarray(vectors_x)
    vectors_y = np.asarray(vectors_y)
    vectors_z = np.asarray(vectors_z)

    # Stack coordinates to handle multiple points
    vectors_xyz = np.column_stack((vectors_x, vectors_y, vectors_z))

    # Calculate magnitude (norm)
    vectors_mags = np.linalg.norm(vectors_xyz, axis=1)

    # Mask for zero or NaN magnitudes
    valid_mask = (vectors_mags > 0) & (~np.isnan(vectors_mags))

    # Initialise result arrays
    vectors_lats = np.zeros_like(vectors_mags)
    vectors_lons = np.zeros_like(vectors_mags)
    vectors_azis = np.zeros_like(vectors_mags)

    # Calculate latitude (in degrees)
    vectors_lats[valid_mask] = np.rad2deg(np.arcsin(vectors_z[valid_mask] / vectors_mags[valid_mask]))

    # Calculate longitude (in degrees)
    vectors_lons[valid_mask] = np.rad2deg(np.arctan2(vectors_y[valid_mask], vectors_x[valid_mask]))

    # Calculate azimuth (in degrees, measured from North in XY plane)
    vectors_azis[valid_mask] = np.rad2deg(np.arctan2(vectors_x[valid_mask], vectors_y[valid_mask]))

    return vectors_lats, vectors_lons, vectors_mags, vectors_azis

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

def haversine_distance(
        points_lat1,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        points_lon1,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        points_lat2,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
        points_lon2,#: Union[List[Union[int, float]], np.ndarray, _pandas.Series],
    ):
    """
    Calculate the great-circle distance between two points on a sphere.
    """
    points_lat1 = np.asarray(points_lat1); points_lon1 = np.asarray(points_lon1)
    points_lat2 = np.asarray(points_lat2); points_lon2 = np.asarray(points_lon2)

    # Convert to radians
    points_lat1 = np.deg2rad(points_lat1)
    points_lon1 = np.deg2rad(points_lon1)
    points_lat2 = np.deg2rad(points_lat2)
    points_lon2 = np.deg2rad(points_lon2)

    # Calculate differences
    delta_lat = points_lat2 - points_lat1
    delta_lon = points_lon2 - points_lon1

    # Calculate great-circle distance
    a = np.sin(delta_lat / 2)**2 + np.cos(points_lat1) * np.cos(points_lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c

def compute_rotation_matrix(v):
    """Compute a rotation matrix that aligns vector v with the z-axis."""
    v = v / np.linalg.norm(v)  # Normalize input vector

    # Define the new z-axis as v
    z_axis = v

    # Choose an arbitrary x-axis (avoid collinearity)
    arbitrary_vector = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(arbitrary_vector, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Compute y-axis
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix: columns are the new basis vectors
    R = np.stack([x_axis, y_axis, z_axis]).T
    return R

# %%
# Load data
results_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "output_notebooks", "02-Keels-slab_suction-sediment_subduction"))

case = "ref"
points = pd.read_parquet(f"{results_dir}/Points/Points_Muller2016_{case}_0Ma.parquet")
plates = pd.read_parquet(f"{results_dir}/Plates/Plates_Muller2016_{case}_0Ma.parquet")

# %%
mags = []; errors = []; distances = []; median_errors = []; syn_md_force_mags = []; ref_md_force_mags = []
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
for plateID in plates.plateID:
    plate = plates[plates.plateID == plateID]
    point = points[points.plateID == plateID]

    # Get mantle drag torque in Cartesian coordinates
    md_torque_sph = geocentric_cartesian2spherical(
        plate.mantle_drag_torque_x.values,
        plate.mantle_drag_torque_y.values,
        plate.mantle_drag_torque_z.values
    )

    # Calculate great circle distance between the pole of rotation and the point
    distance = haversine_distance(
        point.lat.values,
        point.lon.values,
        md_torque_sph[0],
        md_torque_sph[1]
    )

    # Get positions of grid points in Cartesian coordinates
    points_xyz = np.column_stack(geocentric_spherical2cartesian(
        point.lat.values,
        point.lon.values,
        6371e3,
    ))

    # Get mantle drag torque in Cartesian coordinates
    md_torque_xyz = np.array([
        plate.mantle_drag_torque_x.values[0],
        plate.mantle_drag_torque_y.values[0],
        plate.mantle_drag_torque_z.values[0],
    ])

    # Weight individual components of the mantle drag torque
    # weights = point.segment_length_lon.values**2 * point.segment_length_lat.values**2# * 6371e3# / point.segment_length_lon.values.sum()
    # md_torque_xyz_weighted = md_torque_xyz[:, None] / weights

    distance = haversine_distance(
        point.lat.values,
        point.lon.values,
        md_torque_sph[0],
        md_torque_sph[1]
    )

    # plane_normal = np.cross(md_torque_xyz[:, None], points_xyz.T, axis=0)
    # plane_normal /= np.linalg.norm(plane_normal)  # Normalize

    # print(plane_normal.shape, md_torque_xyz.shape, points_xyz.shape)

    # Force magnitude is the dot product of the mantle drag torque and the plane normal
    # md_force_xyz = plane_normal * (md_torque_xyz[0] * plane_normal[0] + md_torque_xyz[1] * plane_normal[1] + md_torque_xyz[2] * plane_normal[2])

    # Magnitude is weighted by the distance from the torque vector
    # md_force_xyz *= plane_normal #/ np.sin(distance) / point.segment_length_lon.values * point.segment_length_lat.values

    if "keels" in case:
        asthenosphere_thicknesses = np.where(
            (point.seafloor_age.isna()) & (point.LAB_depth > 100e3), 
            200e3 - (point.LAB_depth-100e3),
            200e3,
        )
    else:
        asthenosphere_thicknesses = 200e3

    # weights = point.segment_length_lon.values * 6371e3# * asthenosphere_thicknesses

    distance = haversine_distance(
        point.lat.values,
        point.lon.values,
        md_torque_sph[0],
        md_torque_sph[1]
    )

    distances.append(np.median(distance))

    sign = np.where(
        distance < np.pi,
        1,
        -1
    )

    # distance = np.clip(distance, 0, np.pi)

    md_force_xyz = np.cross(md_torque_xyz[:, None], points_xyz.T, axis=0)

    # weights = point.segment_length_lon.values * point.segment_length_lat.values / \
    #     (point.segment_length_lon.values * point.segment_length_lat.values).sum() / \
    #     asthenosphere_thicknesses * asthenosphere_thicknesses.sum()

    # weights = point.segment_length_lon.values * point.segment_length_lat.values / asthenosphere_thicknesses / \
    #     np.sum(point.segment_length_lon.values * point.segment_length_lat.values / asthenosphere_thicknesses) / \
    weights = 4/np.pi * point.segment_length_lon.values * point.segment_length_lat.values / asthenosphere_thicknesses / \
        np.sum(point.segment_length_lon.values**2 * point.segment_length_lat.values**2 / asthenosphere_thicknesses * 6731e3**2) / np.sin(np.pi-distance)
    
    # Magnitude is weighted by the area of the grid cell
    md_force_xyz *= weights
    md_force_mag = np.linalg.norm(md_force_xyz, axis=0)# / (6371e3**2 * point.segment_length_lon.values * point.segment_length_lat.values).sum() * (as)
    
    syn_md_force_mags.extend(md_force_mag)
    ref_md_force_mags.extend(point.mantle_drag_force_mag)

    median_errors.append(np.median(md_force_mag/point.mantle_drag_force_mag))

    # Convert to spherical coordinates
    md_force_lat, md_force_lon, _, _ = tangent_cartesian2spherical(
        md_force_xyz.T,
        point.lat.values,
        point.lon.values,
        False
    )

    # md_force_lon *= sign; md_force_lat *= sign
    # mask = (distance < np.pi/2) & (distance > np.pi/4)
    p = ax.scatter(
    point.lon.values,
    point.lat.values,
    # c=np.rad2deg(distance)[mask],
    c=md_force_lat / 1.25e20 * asthenosphere_thicknesses * 3.1539e9,
    transform=ccrs.PlateCarree(),
    s=1,
    cmap = "viridis",
    vmin=-12,
    vmax=12,
    )

ax.set_global()
ax.coastlines()
fig.colorbar(p, orientation='horizontal', label="Synthetic speed [cm/a]")
plt.show()

# %%
plt.plot(
    np.rad2deg(distances),
)

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
p = ax.scatter(
    points.lon.values,
    points.lat.values,
    c=points.mantle_drag_force_lat / 1.25e20 * asthenosphere_thicknesses * 3.1539e9,
    transform=ccrs.PlateCarree(),
    s=1,
    cmap = "viridis",
    vmin=-12,
    vmax=12,
    )

ax.set_global()
ax.coastlines()
fig.colorbar(p, orientation='horizontal', label="Reconstructed speed [cm/a]")
plt.show()

# %%
plt.scatter(
    point.mantle_drag_force_mag / 1.25e20 * asthenosphere_thicknesses * 3.1539e9 -
    md_force_mag / 1.25e20 * asthenosphere_thicknesses * 3.1539e9,
    point.lat)

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=160)})
p = ax.scatter(
    points.lon.values,
    points.lat.values,
    c=points.mantle_drag_force_mag / 1.25e20 * asthenosphere_thicknesses * 3.1539e9,
    transform=ccrs.PlateCarree(),
    s=1,
    cmap = "viridis",
    vmin = 0,
    vmax = 12,
    # vmax=points.mantle_drag_force_mag.max(),
    )

ax.set_global()
ax.coastlines()
fig.colorbar(p, orientation='horizontal')
plt.show()
# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=160)})
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    c=md_force_mag / 1.25e20 * asthenosphere_thicknesses * 3.1539e9,
    transform=ccrs.PlateCarree(),
    s=1,
    cmap = "viridis",
    vmin = 0,
    vmax = 5,
    # vmax=points.mantle_drag_force_mag.max(),
    )

ax.set_global()
ax.coastlines()
fig.colorbar(p, orientation='horizontal')
plt.show()
# %%
# plt.scatter(
#     distances,
#     errors,
# )
plt.scatter(
    ref_md_force_mags,
    syn_md_force_mags,
)
plt.xlim(0, 2.5e6)
plt.ylim(0, 2.5e6)
plt.plot([0, 2.5e6], [0, 2.5e6], color="red", ls="--")
# plt.plot([0, 2.5e6], np.array([0, 2.5e6]), color="green", ls="--")

# %%
len(all_md_force_mags)
# plt.yscale("log")
# plt.ylim(min(errors), max(errors))
# plt.vlines(90, 0, 10e99, color="red", ls="--")

# %%
print(np.nanmax(np.array(ref_md_force_mags)/np.array(syn_md_force_mags)))
print(np.nanmin(np.array(ref_md_force_mags)/np.array(syn_md_force_mags)))
# plt.xscale("log")
# %%

recovered_md_torque_xyz = np.sum(np.cross(points_xyz.T, md_force_xyz, axis=0), axis=1)

recovered_md_torque_lat, recovered_md_torque_lon, recovered_md_torque_mag, recovered_md_torque_azi = geocentric_cartesian2spherical(
    recovered_md_torque_xyz[0], recovered_md_torque_xyz[1], recovered_md_torque_xyz[2]
)

print(recovered_md_torque_lat, recovered_md_torque_lon, recovered_md_torque_mag)
print(md_torque_sph[0], md_torque_sph[1], md_torque_sph[2])
print(np.rad2deg(haversine_distance(
    recovered_md_torque_lat * np.sign(recovered_md_torque_mag),
    recovered_md_torque_lon * np.sign(recovered_md_torque_mag),
    md_torque_sph[0],
    md_torque_sph[1]
)) % 180)

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=160)})
ax.set_global()
ax.coastlines()

ax.scatter(
    md_torque_sph[1] * np.sign(md_torque_sph[2]),
    md_torque_sph[0] * np.sign(md_torque_sph[2]),
    transform=ccrs.PlateCarree(),
    s=200,
    color='red',
    marker = 'o',
    zorder=100
)
ax.scatter(
    plate.pole_lon.values[0] * np.sign(plate.pole_angle.values[0]),
    plate.pole_lat.values[0] * np.sign(plate.pole_angle.values[0]),
    transform=ccrs.PlateCarree(),
    s=200,
    color='blue',
    marker = 'o',
    zorder=100
)
ax.scatter(
    recovered_md_torque_lon * np.sign(recovered_md_torque_mag),
    recovered_md_torque_lat * np.sign(recovered_md_torque_mag),
    transform=ccrs.PlateCarree(),
    s=400,
    color='green',
    marker = '*',
    zorder=101
)
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=160)})
ax.set_global()
ax.coastlines()
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    c=point.mantle_drag_force_mag,
    transform=ccrs.PlateCarree(),
    s=1
)
fig.colorbar(p, orientation='horizontal')
plt.show()

# %%
plt.plot(md_force_lat)

# %%
print(np.median(point.mantle_drag_force_mag/md_force_mag))

# %%
x = np.abs(.5*np.pi-distance); y = point.mantle_drag_force_mag/md_force_mag
plt.scatter(
        x,
        y,
)
plt.ylabel("Ratio of true to recovered force magnitude")
plt.xlabel("cos(distance)")

# Detrend the data
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(x, y)

plt.plot(
    x,
    slope * x + intercept,
    color="red"
)
plt.show()
# # Remove the trend and set the mean to 1
# y_new = y - (slope * x + intercept) + 1

# plt.scatter(
#     x,
#     y_new
# )

# %%
md_force_mag_new = md_force_mag - (slope * x + intercept) + 1

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=160)})
ax.set_global()
ax.coastlines()
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    c=md_force_mag/point.mantle_drag_force_mag,
    transform=ccrs.PlateCarree(),
    s=1
)
fig.colorbar(p, orientation='horizontal')

# %%
# Convert mantle drag torque to force in Cartesian coordinates
# md_force_xyz = np.cross(points_xyz.T, md_torque_xyz_weighted, axis=0)

# distance = haversine_distance(point.lat.values, point.lon.values, md_torque_sph[0], md_torque_sph[1])

# md_force_xyz = (md_torque_xyz_weighted / np.cos(.5*np.pi-distance)) / 6371e3 * \
#     (np.cross(
#         md_torque_xyz_weighted / np.linalg.norm(md_torque_xyz_weighted/np.cos(.5*np.pi-distance)),
#         points_xyz.T, 
#         axis=0)
#     )

# md_force_xyz = np.cross(np.cross(points_xyz.T, plane_normal, axis=0), points_xyz.T, axis=0)

# md_force_xyz *= md_torque_xyz_weighted / np.cos(distance)

# dot_product = points_xyz.T[0] * md_torque_xyz_weighted[0] + points_xyz.T[1] * md_torque_xyz_weighted[1] + points_xyz.T[2] * md_torque_xyz_weighted[2]

# cos_theta = dot_product / (np.linalg.norm(points_xyz, axis=1) * np.linalg.norm(md_torque_xyz_weighted, axis=0))

# cos_theta = np.clip(cos_theta, -1, 1)

# theta = np.arccos(cos_theta)

# magnitude = np.sin(theta) * np.linalg.norm(md_torque_xyz)


# magnitude = md_torque_xyz_weighted / np.cos(np.pi-distance)

# md_force_xyz *= np.cos(distance)
# md_force_xyz /= magnitude

# plt.scatter(
#     np.log10(np.linalg.norm(md_force_xyz, axis=0)),
#     distance
# )

# md_force_xyz /= np.linalg.norm(md_force_xyz)  # Normalize

# md_force_xyz = np.cross(
#     md_torque_xyz_weighted,
#     points_xyz.T,
#     axis=0,
# )



#%%

plt.scatter(
    recovered_md_torque_lon, recovered_md_torque_lat,
    c="green",
    label="recovered"
)
plt.scatter(
    md_torque_sph[1], md_torque_sph[0],
    c="red",
    label="true"
)
plt.scatter(
    plate.pole_lon.values, plate.pole_lat.values,
    c="blue",
    label="rotation pole"
)
plt.legend()



error = np.linalg.norm(md_force_xyz, axis=0) / point.mantle_drag_force_mag

# %%


# %%
plt.scatter(
    np.rad2deg(distance),
    error,
)
plt.xlabel("great-circle distance")
plt.ylabel("log10(error)")
plt.xlim(0, 180)
# plt.ylim(0, 1)
plt.grid(ls=":")
# %%
plt.scatter(
    np.cos(point.lat),
    error,
)
plt.yscale("log")
# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
ax.set_global()
ax.coastlines()
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    transform=ccrs.PlateCarree(),
    s=2,
    c=np.linalg.norm(md_force_xyz, axis=0),
    zorder=20,
    # vmax = point.mantle_drag_force_mag.max(),
)
fig.colorbar(p, orientation='horizontal')
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
ax.set_global()
ax.coastlines()
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    transform=ccrs.PlateCarree(),
    s=2,
    c=np.log10(np.linalg.norm(md_force_xyz, axis=0)/point.mantle_drag_force_mag),
    zorder=20,
)
fig.colorbar(p, orientation='horizontal')
plt.show()

# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=0)})
ax.set_global()
ax.coastlines()
p = ax.scatter(
    point.lon.values,
    point.lat.values,
    transform=ccrs.PlateCarree(),
    s=2,
    c=point.mantle_drag_force_mag,
    zorder=20,
)
fig.colorbar(p, orientation='horizontal')
plt.show()


# %%


# %%
