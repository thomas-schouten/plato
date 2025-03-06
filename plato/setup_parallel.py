# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Parallel processor for accelerating the retrieval of plate IDs
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import contextlib
import io
import os as _os
import tempfile
import shutil
import warnings

from collections import defaultdict
from typing import Optional
from typing import Union

# Third-party libraries
import geopandas as _geopandas
import matplotlib.pyplot as plt

import numpy as _numpy
import gplately as _gplately
import pandas as _pandas
import pygplates as _pygplates
import xarray as _xarray

from multiprocessing import Pool
from tqdm import tqdm

# Local libraries
from functions_main import set_constants
from functions_main import mag_azi2lat_lon
from functions_main import project_points

def get_plates(
        rotations: _pygplates.RotationModel,
        reconstruction_time: int,
        resolved_topologies: list, 
        options: dict,
    ):
    """
    Function to get data on plates in reconstruction.

    :param rotations:             rotation model
    :type rotations:              _pygplates.RotationModel object
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param resolved_topologies:   resolved topologies
    :type resolved_topologies:    list of resolved topologies
    :param options:               options for the case
    :type options:                dict

    :return:                      plates
    :rtype:                       pandas.DataFrame
    """
    # Set constants
    constants = set_constants()

    # Make _pandas.df with all plates
    # Initialise list
    plates = _numpy.zeros([len(resolved_topologies),10])
    
    # Loop through plates
    for n, topology in enumerate(resolved_topologies):

        # Get plateID
        plates[n,0] = topology.get_resolved_feature().get_reconstruction_plate_id()

        # Get plate area
        plates[n,1] = topology.get_resolved_geometry().get_area() * constants.mean_Earth_radius_m**2

        # Get Euler rotations
        stage_rotation = rotations.get_rotation(
            to_time=reconstruction_time,
            moving_plate_id=int(plates[n,0]),
            from_time=reconstruction_time + options["Velocity time step"],
            anchor_plate_id=options["Anchor plateID"]
        )
        pole_lat, pole_lon, pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()
        plates[n,2] = pole_lat
        plates[n,3] = pole_lon
        plates[n,4] = pole_angle

        # Get plate centroid
        centroid = topology.get_resolved_geometry().get_interior_centroid()
        centroid_lat, centroid_lon = centroid.to_lat_lon_array()[0]
        plates[n,5] = centroid_lon
        plates[n,6] = centroid_lat

        # Get velocity [cm/a] at centroid
        centroid_velocity = get_velocities([centroid_lat], [centroid_lon], (pole_lat, pole_lon, pole_angle))
    
        plates[n,7] = centroid_velocity[1]
        plates[n,8] = centroid_velocity[0]
        plates[n,9] = centroid_velocity[2]

    # Convert to DataFrame    
    plates = _pandas.DataFrame(plates)

    # Initialise columns
    plates.columns = ["plateID", "area", "pole_lat", "pole_lon", "pole_angle", "centroid_lon", "centroid_lat", "centroid_v_lon", "centroid_v_lat", "centroid_v_mag"]

    # Merge topological networks with main plate; this is necessary because the topological networks have the same PlateID as their host plate and this leads to computational issues down the road
    main_plates_indices = plates.groupby("plateID")["area"].idxmax()

    # Create new DataFrame with the main plates
    merged_plates = plates.loc[main_plates_indices]

    # Aggregating the area column by summing the areas of all plates with the same plateID
    merged_plates["area"] = plates.groupby("plateID")["area"].sum().values

    # Get plate names
    merged_plates["name"] = _numpy.nan; merged_plates.name = get_plate_names(merged_plates.plateID)
    merged_plates["name"] = merged_plates["name"].astype(str)

    # Sort and index by plate ID
    merged_plates = merged_plates.sort_values(by="plateID")
    merged_plates = merged_plates.reset_index(drop=True)

    # Initialise columns to store other whole-plate properties
    merged_plates["trench_length"] = 0.; merged_plates["omega"] = 0.
    merged_plates["v_rms_mag"] = 0.; merged_plates["v_rms_azi"] = 0.
    merged_plates["slab_flux"] = 0.; merged_plates["sediment_flux"] = 0.

    # Initialise columns to store whole-plate torques (Cartesian) and force at plate centroid (North-East).
    torques = ["slab_pull", "GPE", "slab_bend", "mantle_drag", "driving", "residual"]
    axes = ["x", "y", "z", "mag"]
    coords = ["lat", "lon", "mag", "azi"]
    
    merged_plates[[torque + "_torque_" + axis for torque in torques for axis in axes]] = [[0.] * len(torques) * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["slab_pull_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["mantle_drag_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["driving_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[["residual_torque_opt_" + axis for axis in axes]] = [[0.] * len(axes) for _ in range(len(merged_plates.plateID))]
    merged_plates[[torque + "_force_" + coord for torque in torques for coord in coords]] = [[0.] * len(torques) * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["slab_pull_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["mantle_drag_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["driving_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]
    merged_plates[["residual_force_opt_" + coord for coord in coords]] = [[0.] * len(coords) for _ in range(len(merged_plates.plateID))]

    return merged_plates

def get_slabs(
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_time: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
    ):
    """
    Function to get data on slabs in reconstruction.

    :param reconstruction:        reconstruction
    :type reconstruction:         _gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param topology_geometries:   topology geometries
    :type topology_geometries:    geopandas.GeoDataFrame
    :param options:               options for the case
    :type options:                dict
    
    :return:                      slabs
    :rtype:                       pandas.DataFrame
    """
    # Set constants
    constants = set_constants()

    # Tesselate subduction zones and get slab pull and bend torques along subduction zones
    slabs = reconstruction.tessellate_subduction_zones(reconstruction_time, ignore_warnings=True, tessellation_threshold_radians=(options["Slab tesselation spacing"]/constants.mean_Earth_radius_km))

    # Convert to _pandas.DataFrame
    slabs = _pandas.DataFrame(slabs)

    # Kick unused columns
    slabs = slabs.drop(columns=[2, 3, 4, 5])

    slabs.columns = ["lon", "lat", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

    # Convert trench segment length from degree to m
    slabs.trench_segment_length *= constants.equatorial_Earth_circumference / 360

    # Get plateIDs of overriding plates
    sampling_lat, sampling_lon = project_points(
        slabs.lat,
        slabs.lon,
        slabs.trench_normal_azimuth,
        100
    )
    slabs["upper_plateID"] = get_plateIDs(
        reconstruction,
        topology_geometries,
        sampling_lat,
        sampling_lon,
        reconstruction_time,
    )

    # Get absolute velocities of upper and lower plates
    for plate in ["upper_plate", "lower_plate", "trench_plate"]:
        # Loop through lower plateIDs to get absolute lower plate velocities
        for plateID in slabs[plate + "ID"].unique():
            # Select all points with the same plateID
            selected_slabs = slabs[slabs[plate + "ID"] == plateID]

            # Get stage rotation for plateID
            selected_plate = plates[plates.plateID == plateID]

            if len(selected_plate) == 0:
                stage_rotation = reconstruction.rotation_model.get_rotation(
                    to_time=reconstruction_time,
                    moving_plate_id=int(plateID),
                    from_time=reconstruction_time + options["Velocity time step"],
                    anchor_plate_id=options["Anchor plateID"]
                ).get_lat_lon_euler_pole_and_angle_degrees()
            else:
                stage_rotation = (
                    selected_plate.pole_lat.values[0],
                    selected_plate.pole_lon.values[0],
                    selected_plate.pole_angle.values[0]
                )

            # Get plate velocities
            selected_velocities = get_velocities(
                selected_slabs.lat,
                selected_slabs.lon,
                stage_rotation
            )

            # Store in array
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lat"] = selected_velocities[0]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_lon"] = selected_velocities[1]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_mag"] = selected_velocities[2]
            slabs.loc[slabs[plate + "ID"] == plateID, "v_" + plate + "_azi"] = selected_velocities[3]

    # Calculate convergence rates
    slabs["v_convergence_lat"] = slabs.v_lower_plate_lat - slabs.v_trench_plate_lat
    slabs["v_convergence_lon"] = slabs.v_lower_plate_lon - slabs.v_trench_plate_lon
    slabs["v_convergence_mag"] = _numpy.sqrt(slabs.v_convergence_lat**2 + slabs.v_convergence_lon**2)

    # Initialise other columns to store seafloor ages and forces
    # Upper plate
    slabs["upper_plate_thickness"] = 0.
    slabs["upper_plate_age"] = 0.
    slabs["continental_arc"] = False
    slabs["erosion_rate"] = 0.

    # Lower plate
    slabs["lower_plate_age"] = 0.
    slabs["lower_plate_thickness"] = 0.
    slabs["sediment_thickness"] = 0.
    slabs["sediment_fraction"] = 0.
    slabs["slab_length"] = options["Slab length"]

    # Forces
    forces = ["slab_pull", "slab_bend", "residual"]
    coords = ["mag", "lat", "lon"]
    slabs[[force + "_force_" + coord for force in forces for coord in coords]] = [[0] * 9 for _ in range(len(slabs))]
    slabs["residual_force_azi"] = 0.
    slabs["residual_alignment"] = 0.

    # Make sure all the columns are floats
    slabs = slabs.apply(lambda x: x.astype(float) if x.name != "continental_arc" else x)

    return slabs

def get_points(
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_time: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
    ):
    """
    Function to get data on regularly spaced grid points in reconstruction.

    :param reconstruction:        reconstruction
    :type reconstruction:         _gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param topology_geometries:   topology geometries
    :type topology_geometries:    geopandas.GeoDataFrame
    :param options:               options for the case
    :type options:                dict

    :return:                      points
    :rtype:                       pandas.DataFrame    
    """
    # Set constants
    constants = set_constants()
    
    # Define grid spacing and 
    lats = _numpy.arange(-90,91,options["Grid spacing"], dtype=float)
    lons = _numpy.arange(-180,180,options["Grid spacing"], dtype=float)

    # Create a meshgrid of latitudes and longitudes
    lon_grid, lat_grid = _numpy.meshgrid(lons, lats)
    lon_grid, lat_grid = lon_grid.flatten(), lat_grid.flatten()

    # Get plateIDs for points
    plateIDs = get_plateIDs(
        reconstruction,
        topology_geometries,
        lat_grid,
        lon_grid,
        reconstruction_time,
    )

    # Initialise empty array to store velocities
    velocity_lat, velocity_lon = _numpy.zeros_like(lat_grid), _numpy.zeros_like(lat_grid)
    velocity_mag, velocity_azi = _numpy.zeros_like(lat_grid), _numpy.zeros_like(lat_grid)

    # Loop through plateIDs to get velocities
    for plateID in _numpy.unique(plateIDs):
        # Your code here
        # Select all points with the same plateID
        selected_lon, selected_lat = lon_grid[plateIDs == plateID], lat_grid[plateIDs == plateID]

        # Get stage rotation for plateID
        selected_plate = plates[plates.plateID == plateID]

        if len(selected_plate) == 0:
            stage_rotation = reconstruction.rotation_model.get_rotation(
                to_time=reconstruction_time,
                moving_plate_id=int(plateID),
                from_time=reconstruction_time + options["Velocity time step"],
                anchor_plate_id=options["Anchor plateID"]
            ).get_lat_lon_euler_pole_and_angle_degrees()
        else:
            stage_rotation = (selected_plate.pole_lat.values[0], selected_plate.pole_lon.values[0], selected_plate.pole_angle.values[0])

        # Get plate velocities
        selected_velocities = get_velocities(selected_lat, selected_lon, stage_rotation)

        # Store in array
        velocity_lat[plateIDs == plateID] = selected_velocities[0]
        velocity_lon[plateIDs == plateID] = selected_velocities[1]
        velocity_mag[plateIDs == plateID] = selected_velocities[2]
        velocity_azi[plateIDs == plateID] = selected_velocities[3]

    # Convert degree spacing to metre spacing
    segment_length_lat = constants.mean_Earth_radius_m * (_numpy.pi/180) * options["Grid spacing"]
    segment_length_lon = constants.mean_Earth_radius_m * (_numpy.pi/180) * _numpy.cos(_numpy.deg2rad(lat_grid)) * options["Grid spacing"]

    # Organise as DataFrame
    points = _pandas.DataFrame({"lat": lat_grid, 
                           "lon": lon_grid, 
                           "plateID": plateIDs, 
                           "segment_length_lat": segment_length_lat,
                           "segment_length_lon": segment_length_lon,
                           "v_lat": velocity_lat, 
                           "v_lon": velocity_lon,
                           "v_mag": velocity_mag,
                           "v_azi": velocity_azi,},
                           dtype=float
                        )

    # Add additional columns to store seafloor ages and forces
    points["seafloor_age"] = 0
    points["lithospheric_thickness"] = 0
    points["crustal_thickness"] = 0
    points["water_depth"] = 0
    points["U"] = 0
    forces = ["GPE", "mantle_drag"]
    coords = ["lat", "lon", "mag"]

    points[[force + "_force_" + coord for force in forces for coord in coords]] = [[0.] * len(forces) * len(coords) for _ in range(len(points))]
    
    return points

def extract_geometry_data(
        topology_geometries
    ) -> list:
    """
    Function to extract geometry data from topology geometries relevant to obtaining plate IDs.

    :param topology_geometries:        topology geometries
    :type topology_geometries:         geopandas.GeoDataFrame

    :return:                           nested tuples containting geometry and plate ID of each topology geometry
    :rtype:                            list
    """
    return [(geom, plateID) for geom, plateID in zip(topology_geometries.geometry, topology_geometries.PLATEID1)]

def process_plateIDs(
        geometries_data: list,
        lats_chunk: _numpy.array,
        lons_chunk: _numpy.array,
    ) -> list:
    """
    Function to process plate IDs for a chunk of latitudes and longitudes.

    :param geometries_data:           geometry data
    :type geometries_data:            list
    :param lats_chunk:                chunk of latitudes
    :type lats_chunk:                 numpy.array
    :param lons_chunk:                chunk of longitudes
    :type lons_chunk:                 numpy.array

    :return:                          plateIDs
    :rtype:                           numpy.array
    """
    plateIDs = _numpy.zeros(len(lats_chunk))
    
    for topology_geometry, topology_plateID in geometries_data:
        mask = topology_geometry.contains(_geopandas.points_from_xy(lons_chunk, lats_chunk))
        plateIDs[mask] = topology_plateID

        # Break if all points have been assigned a plate ID
        if plateIDs.all():
            break

    return plateIDs

def get_plateIDs(
        reconstruction: _gplately.PlateReconstruction,
        topology_geometries: _geopandas.GeoDataFrame,
        lats: Union[list or _numpy.array],
        lons: Union[list or _numpy.array],
        reconstruction_time: int,
        NUM_PROCESSES: Optional[int] = None,
    ):
    """
    Function to get plate IDs for a set of latitudes and longitudes.

    :param reconstruction:             reconstruction
    :type reconstruction:              _gplately.PlateReconstruction
    :param topology_geometries:        topology geometries
    :type topology_geometries:         geopandas.GeoDataFrame
    :param lats:                       latitudes
    :type lats:                        list or _numpy.array
    :param lons:                       longitudes
    :type lons:                        list or _numpy.array
    :param reconstruction_time:        reconstruction time
    :type reconstruction_time:         integer

    :return:                           plateIDs
    :rtype:                            list
    """
    # Extract geometry data from topology geometries
    geometries_data = extract_geometry_data(topology_geometries)

    # Parallel processing of plate IDs
    lats = _numpy.asarray(lats)
    lons = _numpy.asarray(lons)

    # Use all available CPUs if NUM_PROCESSES is not specified
    if NUM_PROCESSES is None:
        NUM_PROCESSES = _os.cpu_count()

    # Split the data into chunks
    chunk_size = len(lats) // NUM_PROCESSES
    chunks = [(geometries_data.copy(), lats[i:i + chunk_size].copy(), lons[i:i + chunk_size].copy()) for i in range(0, len(lats), chunk_size)]

    # Create a Pool of workers
    with Pool(NUM_PROCESSES) as pool:
        # Map the process_chunk function to chunks
        results = pool.starmap(process_plateIDs, chunks)

    # Concatenate results from all chunks
    plateIDs = _numpy.concatenate(results)

    # Use gplately's Point object to find and assign plate IDs for remaining points
    no_plateID_mask = plateIDs == 0
    if no_plateID_mask.any():
        no_plateID_grid = _gplately.Points(
            reconstruction,
            lons[no_plateID_mask],
            lats[no_plateID_mask],
            time=reconstruction_time
        )
        plateIDs[no_plateID_mask] = no_plateID_grid.plate_id

    return plateIDs

def get_velocities(
        lats: Union[list or _numpy.array],
        lons: Union[list or _numpy.array],
        stage_rotation: tuple,
    ):
    """
    Function to get velocities for a set of latitudes and longitudes.
    NOTE: This function is not vectorised yet, but has not been a bottleneck in the code so far.

    :param lats:                     latitudes
    :type lats:                      list or numpy.array
    :param lons:                     longitudes
    :type lons:                      list or numpy.array
    :param stage_rotation:           stage rotation defined by pole latitude, pole longitude and pole angle
    :type stage_rotation:            tuple

    :return:                         velocities_lat, velocities_lon, velocities_mag, velocities_azi
    :rtype:                          numpy.array, numpy.array, numpy.array, numpy.array
    """
    # Convert lats and lons to numpy arrays if they are not already
    lats = _numpy.asarray(lats)
    lons = _numpy.asarray(lons)

    # Initialise empty array to store velocities
    velocities_lat = _numpy.zeros(len(lats))
    velocities_lon = _numpy.zeros(len(lats))
    velocities_mag = _numpy.zeros(len(lats))
    velocities_azi = _numpy.zeros(len(lats))

    # Loop through points to get velocities
    for i, _ in enumerate(lats):
        # Convert to LocalCartesian
        point = _pygplates.PointOnSphere((lats[i], lons[i]))

        # Calculate magnitude and azimuth of velocities at points
        velocity_mag_azi = _numpy.asarray(
            _pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                point,
                _pygplates.calculate_velocities(
                    point, 
                    _pygplates.FiniteRotation((stage_rotation[0], stage_rotation[1]), _numpy.deg2rad(stage_rotation[2])), 
                    1.,
                    velocity_units = _pygplates.VelocityUnits.cms_per_yr
                )
            )
        )

        # Get magnitude and azimuth of velocities
        velocities_mag[i] = velocity_mag_azi[0][0]; velocities_azi[i] = velocity_mag_azi[0][1]

    # Convert to lat and lon components
    velocities_lat, velocities_lon = mag_azi2lat_lon(velocities_mag, _numpy.rad2deg(velocities_azi))

    return velocities_lat, velocities_lon, velocities_mag, velocities_azi

def get_topology_geometries(
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_time: int,
        anchor_plateID: int
    ):
    """
    Function to resolve topologies and get geometries as a GeoDataFrame

    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param anchor_plateID:        anchor plate ID
    :type anchor_plateID:         integer
    :return:                      resolved_topologies
    :rtype:                       geopandas.GeoDataFrame
    """
    # Make temporary directory to hold shapefiles
    temp_dir = tempfile.mkdtemp()

    # Resolve topological networks and load as GeoDataFrame
    topology_file = _os.path.join(temp_dir, "topologies.shp")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Normalized/laundered field name:"
        )
        _pygplates.resolve_topologies(
            reconstruction.topology_features, 
            reconstruction.rotation_model, 
            topology_file, 
            reconstruction_time, 
            anchor_plate_id=anchor_plateID
        )
        if _os.path.exists(topology_file):
            topology_geometries = _geopandas.read_file(topology_file)

    # Remove temporary directory
    shutil.rmtree(temp_dir)

    return topology_geometries

def get_geometric_properties(
        plates: _pandas.DataFrame,
        slabs: _pandas.DataFrame,
    ):
    """
    Function to get geometric properties of plates.

    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param slabs:                 slabs
    :type slabs:                  pandas.DataFrame

    :return:                      plates
    :rtype:                       pandas.DataFrame
    """
    # Calculate trench length and omega
    for plateID in plates.plateID:
        if plateID in slabs.lower_plateID.unique():
            plates.loc[plates.plateID == plateID, "trench_length"] = slabs[slabs.lower_plateID == plateID].trench_segment_length.sum()
            plates.loc[plates.plateID == plateID, "omega"] = plates[plates.plateID == plateID].area.values[0] / plates[plates.plateID == plateID].trench_length.values[0]

    return plates

def get_plate_names(
        plate_id_list: Union[list or _numpy.array],
    ):
    """
    Function to get plate names corresponding to plate ids

    :param plate_id_list:        list of plate ids
    :type plate_id_list:         list or numpy.array

    :return:                     plate_names
    :rtype:                      list
    """
    plate_name_dict = {
        101: "N America",
        201: "S America",
        301: "Eurasia",
        302: "Baltica",
        501: "India",
        503: "Arabia",
        511: "Capricorn",
        701: "S Africa",
        702: "Madagascar",
        709: "Somalia",
        714: "NW Africa",
        715: "NE Africa",
        801: "Australia",
        802: "Antarctica",
        901: "Pacific",
        902: "Farallon",
        904: "Aluk",
        909: "Cocos",
        911: "Nazca",
        918: "Kula",
        919: "Phoenix",
        926: "Izanagi",
        5400: "Burma",
        5599: "Tethyan Himalaya",
        7520: "Argoland",
        9002: "Farallon",
        9006: "Izanami",
        9009: "Izanagi",
        9010: "Pontus"
    } 

    # Create a defaultdict with the default value as the plate ID
    default_plate_name = defaultdict(lambda: "Unknown", plate_name_dict)

    # Retrieve the plate names based on the plate IDs
    plate_names = [default_plate_name[plate_id] for plate_id in plate_id_list]

    return plate_names

def get_options(
        file_name: str,
        sheet_name: Optional[str]
    ):
    """
    Function to get options from excel file.

    :param file_name:            file name
    :type file_name:             string
    :param sheet_name:           sheet name
    :type sheet_name:            string

    :return:                     cases, options
    :rtype:                      list, dict
    """
    # Read file
    case_options = _pandas.read_excel(file_name, sheet_name=sheet_name, comment="#")

    # Initialise list of cases
    cases = []

    # Initialise options dictionary
    options = {}

    # Define all options
    all_options = ["Slab pull torque",
                   "GPE torque",
                   "Mantle drag torque",
                   "Slab bend torque",
                   "Slab bend mechanism",
                   "Reconstructed motions",
                   "Continental crust",
                   "Seafloor age variable",
                   "Seafloor age profile",
                   "Sample sediment grid", 
                   "Active margin sediments",
                   "Sample erosion grid", 
                   "Erosion to sediment ratio",
                   "Sediment subduction",
                   "Shear zone width",
                   "Slab length",
                   "Strain rate",
                   "Slab pull constant",
                   "Mantle viscosity",
                   "Slab tesselation spacing",
                   "Grid spacing",
                   "Minimum plate area",
                   "Anchor plateID",
                   "Velocity time step"
                   ]
    
    # Define default values
    default_values = [True,
                      True,
                      True,
                      False,
                      "viscous",
                      True,
                      False,
                      "z",
                      "half space cooling",
                      False,
                      0,
                      False,
                      2,
                      False,
                      2e3,
                      700e3,
                      1e-12,
                      0.0316,
                      1.22e20,
                      250,
                      1,
                      7.5e12,
                      0,
                      1,
                      ]

    # Adjust TRUE/FALSE values in excel file to boolean
    boolean_options = ["Slab pull torque",
                       "GPE torque",
                       "Mantle drag torque",
                       "Slab bend torque",
                       "Reconstructed motions",
                       "Continental crust",
                       "Randomise trench orientation",
                       "Randomise slab age"]

    # Loop over rows to obtain options from excel file
    for _, row in case_options.iterrows():
        case = row["Name"]
        cases.append(case)
        options[case] = {}
        for i, option in enumerate(all_options):
            if option in case_options:
                if option in boolean_options and row[option] == 1:
                    row[option] = True
                elif option in boolean_options and row[option] == 0:
                    row[option] = False
                options[case][option] = row[option]
            else:
                options[case][option] = default_values[i]

    return cases, options

def get_seafloor_grid(
        reconstruction_name: str,
        reconstruction_time: int,
        DEBUG_MODE: bool = False
    ) -> _xarray.Dataset:
    """
    Function to obtain seafloor grid from GPlately DataServer.
    
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param reconstruction_times:   reconstruction times
    :type reconstruction_times:    list or numpy.array
    :param DEBUG_MODE:             whether to run in debug mode
    :type DEBUG_MODE:              bool

    :return:                       seafloor_grids
    :rtype:                        xarray.Dataset
    """
    # Call _gplately"s DataServer from the download.py module
    gdownload = _gplately.download.DataServer(reconstruction_name)

    if DEBUG_MODE:
        # Let the user know what is happening
        print(f"Downloading age grid for {reconstruction_name} at {reconstruction_time} Ma")
        age_raster = gdownload.get_age_grid(time=reconstruction_time)
    else:
        # Suppress print statements if not in debug mode
        with contextlib.redirect_stdout(io.StringIO()):
            age_raster = gdownload.get_age_grid(time=reconstruction_time)

    # Convert the data to a masked array
    seafloor_ages_ma = _numpy.ma.masked_invalid(age_raster.data)
    
    # Convert the masked array to a regular numpy array with NaN for masked values
    seafloor_ages = seafloor_ages_ma.filled(_numpy.nan)

    lon = age_raster.lons
    lat = age_raster.lats

    # Create a xarray dataset
    age_grid = _xarray.Dataset(
        {
            "seafloor_age": (["latitude", "longitude"], seafloor_ages.astype(_numpy.float64)),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
        },
    )
    
    return age_grid

def get_velocity_grid(
        points: _pandas.DataFrame,
        seafloor_grid: _xarray.DataArray,
    ):
    """
    Function to obtain velocity grid from the velocity sampled at the points interpolated to the resolution of the seafloor grid.

    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param reconstruction_time:    reconstruction time
    :type reconstruction_time:     integer
    :param seafloor_grid:          seafloor ages
    :type seafloor_grid:           xarray.DataArray

    :return:                       velocity_grid
    :rtype:                        xarray.Dataset
    """
    # Make xarray velocity grid
    velocity_grid = _xarray.Dataset(
            {
                "velocity_magnitude": (["latitude", "longitude"], points.v_mag.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_latitude": (["latitude", "longitude"], points.v_lat.values.reshape(points.lat.unique().size, points.lon.unique().size)),
                "velocity_longitude": (["latitude", "longitude"], points.v_lon.values.reshape(points.lat.unique().size, points.lon.unique().size)),
            },
            coords={
                "latitude": points.lat.unique(),
                "longitude": points.lon.unique(),
            },
        )
    
    # Interpolate to resolution of seafloor grid
    velocity_grid = velocity_grid.interp(latitude=seafloor_grid.latitude, longitude=seafloor_grid.longitude, method="linear")

    # Interpolate NaN values along the dateline
    velocity_grid = velocity_grid.interpolate_na()

    return velocity_grid

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PROCESS CASES 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

def process_cases(cases, options, target_options):
    """
    Function to process cases and options to accelerate computation. Each case is assigned a dictionary of identical cases for a given set of target options.
    The goal here is that if these target options are identical, the computation is only peformed once and the results are copied to the other cases.

    :param cases:           cases
    :type cases:            list
    :param options:         options
    :type options:          dict
    :param target_options:  target options
    :type target_options:   list

    :return:                case_dict
    :rtype:                 dict
    """
    # Initialise dictionary to store processed cases
    processed_cases = set()
    case_dict = {}

    # Loop through cases to process
    for case in cases:
        # Ignore processed cases
        if case in processed_cases:
            continue
        
        # Initialise list to store similar cases
        case_dict[case] = [case]

        # Add case to processed cases
        processed_cases.add(case)

        # Loop through other cases to find similar cases
        for other_case in cases:
            # Ignore if it is the same case
            if case == other_case:
                continue
            
            # Add case to processed cases if it is similar
            if all(options[case][opt] == options[other_case][opt] for opt in target_options):
                case_dict[case].append(other_case)
                processed_cases.add(other_case)

    return case_dict

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def DataFrame_to_parquet(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        reconstruction_time: int,
        case: str,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save DataFrame to a Parquet file in a folder efficiently.

    :param data:                  data
    :type data:                   pandas.DataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param case:                  case
    :type case:                   string
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the file path
    target_dir = folder if folder else _os.getcwd()
    file_name = f"{data_name}_{reconstruction_name}_{case}_{reconstruction_time}Ma.parquet"
    file_path = _os.path.join(target_dir, data_name, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Saving {data_name} to {file_path}")

    # Ensure the directory exists
    _os.makedirs(_os.path.dirname(file_path), exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
    except FileNotFoundError:
        pass  # No need to remove if file does not exist

    # Save the data to Parquet
    data.to_parquet(file_path, index=False)

def DataFrame_to_csv(
        data: _pandas.DataFrame,
        data_name: str,
        reconstruction_name: str,
        reconstruction_time: int,
        case: str,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save DataFrame to a folder efficiently.

    :param data:                  data
    :type data:                   pandas.DataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   reconstruction time
    :type reconstruction_time:    integer
    :param case:                  case
    :type case:                   string
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the file path
    target_dir = folder if folder else _os.getcwd()
    file_name = f"{data_name}_{reconstruction_name}_{case}_{reconstruction_time}Ma.csv"
    file_path = _os.path.join(target_dir, data_name, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Saving {data_name} to {file_path}")

    # Ensure the directory exists
    _os.makedirs(_os.path.dirname(file_path), exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
    except FileNotFoundError:
        pass  # No need to remove if file does not exist

    # Save the data to CSV
    data.to_csv(file_path, index=False)
    
def GeoDataFrame_to_geoparquet(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        reconstruction_time: int,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save GeoDataFrame to a GeoParquet file in a folder efficiently.

    :param data:                  data
    :type data:                   geopandas.GeoDataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   age of reconstruction in Ma
    :type reconstruction_time:    int
    :param folder:                folder name
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the target directory and file path
    target_dir = _os.path.join(folder if folder else _os.getcwd(), data_name)
    file_name = f"{data_name}_{reconstruction_name}_{reconstruction_time}Ma.parquet"
    file_path = _os.path.join(target_dir, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {reconstruction_time}: {file_path}")

    # Ensure the directory exists
    _os.makedirs(target_dir, exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")
    except FileNotFoundError:
        pass  # File does not exist, no need to remove

    # Save the data to a GeoParquet file
    data.to_parquet(file_path)

def GeoDataFrame_to_shapefile(
        data: _geopandas.GeoDataFrame,
        data_name: str,
        reconstruction_name: str,
        reconstruction_time: int,
        folder: str,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save GeoDataFrame to a folder efficiently.

    :param data:                  data
    :type data:                   geopandas.GeoDataFrame
    :param data_name:             name of dataset
    :type data_name:              string
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param reconstruction_time:   age of reconstruction in Ma
    :type reconstruction_time:    int
    :param folder:                folder
    :type folder:                 string
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    """
    # Construct the target directory and file path
    target_dir = _os.path.join(folder if folder else _os.getcwd(), data_name)
    file_name = f"{data_name}_{reconstruction_name}_{reconstruction_time}Ma.shp"
    file_path = _os.path.join(target_dir, file_name)
    
    # Debug information
    if DEBUG_MODE:
        print(f"Target directory for {data_name}: {target_dir}")
        print(f"File path for {data_name} at {reconstruction_time}: {file_path}")

    # Ensure the directory exists
    _os.makedirs(target_dir, exist_ok=True)
    
    # Delete old file if it exists
    try:
        _os.remove(file_path)
        if DEBUG_MODE:
            print(f"Deleted old file {file_path}")
    except FileNotFoundError:
        pass  # File does not exist, no need to remove

    # Save the data to a shapefile
    data.to_file(file_path)

def check_dir(target_dir):
    """
    Function to check if a directory exists, and create it if it doesn't
    """
    # Check if a directory exists, and create it if it doesn't
    if not _os.path.exists(target_dir):
        _os.makedirs(target_dir)