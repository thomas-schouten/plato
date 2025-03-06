# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Setup
# Thomas Schouten, 2023
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
from typing import Dict, List, Optional, Tuple, Union

# Third-party libraries
import geopandas as _geopandas

import numpy as _numpy
import gplately as _gplately
import pandas as _pandas
import pygplates as _pygplates
import xarray as _xarray
from tqdm import tqdm

# Local libraries
from functions_main import set_constants
from functions_main import mag_azi2lat_lon
from functions_main import project_points
from settings import Settings
from reconstruction import Reconstruction

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISATION 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_point_data(
        reconstruction: _gplately.PlateReconstruction,
        age: int,
        plates: _pandas.DataFrame,
        topology_geometries: _geopandas.GeoDataFrame,
        options: dict,
        PARALLEL_MODE: Optional[bool] = False,
    ):
    """
    Function to get data on regularly spaced grid points in reconstruction.

    :param reconstruction:      Reconstruction
    :type reconstruction:       gplately.PlateReconstruction
    :param age:                 reconstruction time
    :type age:                  integer
    :param plates:              plates
    :type plates:               pandas.DataFrame
    :param topology_geometries: topology geometries
    :type topology_geometries:  geopandas.GeoDataFrame
    :param options:             options for the case
    :type options:              dict

    :return:                    points
    :rtype:                     pandas.DataFrame    
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
        age,
        PARALLEL_MODE=PARALLEL_MODE
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
                to_time=age,
                moving_plate_id=int(plateID),
                from_time=age + options["Velocity time step"],
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

def get_globe_data(
        _plates: dict,
        _slabs: dict,
        _points: dict,
        _seafloor_grid: dict,
        _ages: _numpy.array,
        _case: str,
    ):
    """
    Function to get relevant geodynamic data for the entire globe.

    :param plates:                plates
    :type plates:                 dict
    :param slabs:                 slabs
    :type slabs:                  dict
    :param points:                points
    :type points:                 dict
    :param seafloor_grid:         seafloor grid
    :type seafloor_grid:          dict

    :return:                      globe
    :rtype:                       pandas.DataFrame
    """
    # Initialise empty arrays
    num_plates = _numpy.zeros_like(_ages)
    slab_length = _numpy.zeros_like(_ages)
    v_rms_mag = _numpy.zeros_like(_ages)
    v_rms_azi = _numpy.zeros_like(_ages)
    mean_seafloor_age = _numpy.zeros_like(_ages)

    for i, _age in enumerate(_ages):
        # Get number of plates
        num_plates[i] = len(_plates[_age][_case].plateID.values)

        # Get slab length
        slab_length[i] = _slabs[_age][_case].trench_segment_length.sum()

        # Get global RMS velocity
        # Get area for each grid point as well as total area
        areas = _points[_age][_case].segment_length_lat.values * _points[_age][_case].segment_length_lon.values
        total_area = _numpy.sum(areas)

        # Calculate RMS speed
        v_rms_mag[i] = _numpy.sum(_points[_age][_case].v_mag * areas) / total_area

        # Calculate RMS azimuth
        v_rms_sin = _numpy.sum(_numpy.sin(_points[_age][_case].v_lat) * areas) / total_area
        v_rms_cos = _numpy.sum(_numpy.cos(_points[_age][_case].v_lat) * areas) / total_area
        v_rms_azi[i] = _numpy.rad2deg(
            -1 * (_numpy.arctan2(v_rms_sin, v_rms_cos) + 0.5 * _numpy.pi)
        )

        # Get mean seafloor age
        mean_seafloor_age[i] = _numpy.nanmean(_seafloor_grid[_age].seafloor_age.values)

    # Organise as pd.DataFrame
    globe = _pandas.DataFrame({
        "number_of_plates": num_plates,
        "total_slab_length": slab_length,
        "v_rms_mag": v_rms_mag,
        "v_rms_azi": v_rms_azi,
        "mean_seafloor_age": mean_seafloor_age,
    })
        
    return globe

def extract_geometry_data(topology_geometries):
    """
    Function to extract only the geometry and plateID from topology geometries.

    :param topology_geometries:        topology geometries
    :type topology_geometries:         geopandas.GeoDataFrame

    :return:                           geometries_data
    :rtype:                            list
    """
    return [(geom, plateID) for geom, plateID in zip(topology_geometries.geometry, topology_geometries.PLATEID1)]

def process_plateIDs(
        geometries_data: list,
        lats_chunk: _numpy.array,
        lons_chunk: _numpy.array,
    ) -> list:
    """
    Function to process plateIDs for a chunk of latitudes and longitudes.

    :param geometries_data:        geometry data
    :type geometries_data:         list
    :param lats_chunk:             chunk of latitudes
    :type lats_chunk:              numpy.array
    :param lons_chunk:             chunk of longitudes
    :type lons_chunk:              numpy.array

    :return:                       plateIDs
    :rtype:                        numpy.array
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
        PARALLEL_MODE: Optional[bool] = False,
        num_workers: Optional[int] = None,
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
    # Convert lats and lons to numpy arrays if they are not already
    lats = _numpy.asarray(lats)
    lons = _numpy.asarray(lons)

    # Extract geometry data
    geometries_data = extract_geometry_data(topology_geometries)

    # Get plateIDs for the entire dataset
    plateIDs = process_plateIDs(geometries_data, lats, lons)

    # Use vectorised operations to find and assign plate IDs for remaining points
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
            _age, 
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
            plates.loc[plates.plateID == plateID, "zeta"] = plates[plates.plateID == plateID].area.values[0] / plates[plates.plateID == plateID].trench_length.values[0]

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
        file_name: Optional[str] = None,
        sheet_name: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Dict[str, Optional[str]]]]:
    """
    Function to get options from excel file. If no arguments are provided,
    returns the default options and assigns 'ref' to the case.

    :param file_name:            file name (optional)
    :type file_name:             str, optional
    :param sheet_name:           sheet name (optional)
    :type sheet_name:            str, optional

    :return:                     cases, options
    :rtype:                      list, dict
    """
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

    # If no file_name is provided, return default values with case "ref"
    if not file_name:
        cases = ["ref"]
        options = {"ref": {option: default_values[i] for i, option in enumerate(all_options)}}
        return cases, options

    # Read file
    case_options = _pandas.read_excel(file_name, sheet_name=sheet_name, comment="#")

    # Initialise list of cases
    cases = []

    # Initialise options dictionary
    options = {}

    # Loop over rows to obtain options from excel file
    for _, row in case_options.iterrows():
        _case = row.get("Name", "ref")  # Assign "ref" if no Name column or no case name
        cases.append(_case)
        options[_case] = {}
        for i, option in enumerate(all_options):
            if option in case_options.columns:
                # Handle boolean conversion
                if option in boolean_options and row[option] == 1:
                    row[option] = True
                elif option in boolean_options and row[option] == 0:
                    row[option] = False
                options[_case][option] = row[option]
            else:
                options[_case][option] = default_values[i]

    # If no cases were found, use the default "ref" case
    if not cases:
        cases = ["ref"]
        options["ref"] = {option: default_values[i] for i, option in enumerate(all_options)}

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
    :param ages:   reconstruction times
    :type ages:    list or numpy.array
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

def get_settings(
        settings: Optional[Settings] = None,
        ages: Optional[Union[int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray]] = None,
        cases: Optional[Union[str, List[str]]] = None,
        reconstruction_name: Optional[str] = None,
    ):
    """
    Function to set settings or initialise a new object.
    """
    if settings:
            _settings = settings
    else:
        if ages and cases:
            if reconstruction_name:
                name = reconstruction_name
            else:
                name = "Unknown Reconstruction"
            _settings = Settings(
                name = name,
                ages = ages,
            )
        else:
            raise ValueError("Settings object or ages and cases should be provided.")
        
    return _settings

def get_ages(
        ages: Union[None, int, float, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
        default_ages: _numpy.ndarray,
    ) -> _numpy.ndarray:
    """
    Function to check and get ages.

    :param ages:            ages
    :type ages:             None, int, float, list, numpy.integer, numpy.floating, numpy.ndarray
    :param default_ages:    settings ages
    :type default_ages:     numpy.ndarray

    :return:                ages
    :rtype:                 numpy.ndarray
    """
    # Define ages
    if ages is None:
        # If no ages are provided, use default ages
        _ages = default_ages

    elif isinstance(ages, (int, float, _numpy.integer, _numpy.floating)):
        # If a single value is provided, convert to numpy array
        _ages = _numpy.array([ages])

    elif isinstance(ages, list):
        # If a list is provided, convert to numpy array
        _ages = _numpy.array(ages)

    elif isinstance(ages, _numpy.ndarray):
        # If a numpy array is provided, use as is
        _ages = ages

    return _ages

def get_cases(
    cases: Union[None, str, List[str]],
    default_cases: List[str],
    ) -> List[str]:
    """
    Function to check and get cases.

    :param cases:           cases (can be None, a single case as a string, or a list of cases)
    :type cases:            None, str, or list of strings
    :param default_cases:   default cases to use if cases is not provided
    :type default_cases:    list of strings

    :return:                 a list of cases
    :rtype:                  list of strings
    """
    # Define cases
    if cases is None:
        # If no cases are provided, use default cases
        _cases = default_cases

    else:
        # Check if cases is a single value (str), convert to list
        if isinstance(cases, str):
            _cases = [cases]

    return _cases

def get_iterable(
        cases: Union[None, str, List[str]],
        default_iterable: List[str],
    ) -> Dict[str, List[str]]:
    """
    Function to check and get iterable.

    :param cases:               cases (can be None, a single case as a string, or a list of cases)
    :type cases:                None, str, or list of strings
    :param default_iterable:    default iterable to use if cases is not provided
    :type default_iterable:     list of strings

    :return:                 iterable
    :rtype:                  dict
    """
    # Define iterable
    if cases is None:
        # If no cases are provided, use the default iterable
        _iterable = default_iterable

    else:
        # Check if cases is a single value (str), convert to list
        if isinstance(cases, str):
            cases = [cases]
        
        # Make dictionary of iterable
        _iterable = {case: [] for case in cases}

    return _iterable

def get_plates(
        plate_IDs: Union[None, int, list, _numpy.integer, _numpy.floating, _numpy.ndarray],
    ) -> _numpy.ndarray:
    """
    Function to check and get plate IDs.

    :param plate_IDs:        plate IDs
    :type plate_IDs:         None, int, list, numpy.integer, numpy.floating, numpy.ndarray

    :return:                 plate IDs
    :rtype:                  numpy.ndarray
    """
    # Define plate IDs
    if isinstance(plates, (int, float, _numpy.floating, _numpy.integer)):
            plates = [plates]

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
    for _case in cases:
        # Ignore processed cases
        if _case in processed_cases:
            continue
        
        # Initialise list to store similar cases
        case_dict[_case] = [_case]

        # Add case to processed cases
        processed_cases.add(_case)

        # Loop through other cases to find similar cases
        for other_case in cases:
            # Ignore if it is the same case
            if _case == other_case:
                continue
            
            # Add case to processed cases if it is similar
            if all(options[_case][opt] == options[other_case][opt] for opt in target_options):
                case_dict[_case].append(other_case)
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

def Dataset_to_netcdf(
        data: _xarray.Dataset,
        data_name: str,
        reconstruction_name: str,
        reconstruction_time: int,
        folder: str,
        case: str = None,
        DEBUG_MODE: bool = False
    ):
    """
    Function to save Dataset to a NetCDF file in a folder efficiently.

    :param data:                  data
    :type data:                   xarray.Dataset
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
    if case:
        file_name = f"{data_name}_{reconstruction_name}_{case}_{reconstruction_time}Ma.nc"
    else:
        file_name = f"{data_name}_{reconstruction_name}_{reconstruction_time}Ma.nc"
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
        pass

    # Save the data to a NetCDF file
    data.to_netcdf(file_path)

def check_dir(target_dir):
    """
    Function to check if a directory exists, and create it if it doesn't
    """
    # Check if a directory exists, and create it if it doesn't
    if not _os.path.exists(target_dir):
        _os.makedirs(target_dir)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(
        data: dict,
        reconstruction: _gplately.PlateReconstruction,
        reconstruction_name: str,
        ages: list,
        type: str,
        all_cases: list,
        all_options: dict,
        matching_case_dict: dict,
        files_dir: Optional[str] = None,
        plates = None,
        resolved_topologies = None,
        resolved_geometries = None,
        DEBUG_MODE: Optional[bool] = False,
        PARALLEL_MODE: Optional[bool] = False,
    ) -> dict:
    """
    Function to load DataFrames from a folder, or initialise new DataFrames
    
    :param data:                  data
    :type data:                   dict
    :param reconstruction:        reconstruction
    :type reconstruction:         gplately.PlateReconstruction
    :param reconstruction_name:   name of reconstruction
    :type reconstruction_name:    string
    :param ages:  reconstruction times
    :type ages:   list or _numpy.array
    :param type:                  type of data
    :type type:                   string
    :param all_cases:             all cases
    :type all_cases:              list
    :param all_options:           all options
    :type all_options:            dict
    :param matching_case_dict:    matching case dictionary
    :type matching_case_dict:     dict
    :param files_dir:             files directory
    :type files_dir:              string
    :param plates:                plates
    :type plates:                 pandas.DataFrame
    :param resolved_topologies:   resolved topologies
    :type resolved_topologies:    geopandas.GeoDataFrame
    :param resolved_geometries:   resolved geometries
    :type resolved_geometries:    geopandas.GeoDataFrame
    :param DEBUG_MODE:            whether to run in debug mode
    :type DEBUG_MODE:             bool
    :param PARALLEL_MODE:         whether to run in parallel mode
    :type PARALLEL_MODE:          bool

    :return:                      data
    :rtype:                       dict
    """
    def load_or_initialise_case(data, _age, _case):
        df = None
        if files_dir:
            df = DataFrame_from_parquet(files_dir, type, reconstruction_name, _case, _age)
            if df is not None:
                return _case, df
    
        # Check for matching case and copy data
        matching_key = next((key for key, cases in matching_case_dict.items() if _case in cases), None)
        if matching_key:
            for matching_case in matching_case_dict[matching_key]:
                if matching_case in data[_age]:
                    return _case, data[_age][matching_case].copy()

        # Initialize new DataFrame if not found
        if df is None:
            if DEBUG_MODE:
                print(f"Initializing new DataFrame for {type} for {reconstruction_name} at {_age} Ma for case {_case}...")
            if type == "Plates":
                df = get_plate_data(
                    reconstruction.rotation_model,
                    _age,
                    resolved_topologies[_age],
                    all_options[_case]
                )

            elif type == "Slabs":
                df = get_slab_data(
                    reconstruction,
                    _age,
                    plates[_age][_case],
                    resolved_geometries[_age],
                    all_options[_case],
                    PARALLEL_MODE=PARALLEL_MODE
                )

            elif type == "Points":
                df = get_point_data(
                    reconstruction,
                    _age,
                    plates[_age][_case],
                    resolved_geometries[_age],
                    all_options[_case],
                    PARALLEL_MODE=PARALLEL_MODE
                )

        return _case, df

        # Sequential processing
        # NOTE: Parallel processing increases computation time by a factor 1.5, so this function is kept sequential
        for _age in tqdm(_ages, desc=f"Loading {type} DataFrames", disable=DEBUG_MODE):
            for case in all_cases:
                if isinstance(data, dict):
                    if _age in data.keys():
                        if _case in data[_age]:
                            df = data[_age][_case].copy()
                            
                _case, df = load_or_initialise_case(data, _age, _case)
                data[_age][_case] = df

        return data 

def load_grid(
        grid: dict,
        reconstruction_name: str,
        ages: list,
        type: str,
        files_dir: str,
        points: Optional[dict] = None,
        seafloor_grid: Optional[_xarray.Dataset] = None,
        cases: Optional[list] = None,
        DEBUG_MODE: Optional[bool] = False
    ) -> dict:
    """
    Function to load grid from a folder.

    :param grids:                  grids
    :type grids:                   dict
    :param reconstruction_name:    name of reconstruction
    :type reconstruction_name:     string
    :param ages:   reconstruction times
    :type ages:    list or numpy.array
    :param type:                   type of grid
    :type type:                    string
    :param files_dir:              files directory
    :type files_dir:               string
    :param points:                 points
    :type points:                  dict
    :param seafloor_grid:          seafloor grid
    :type seafloor_grid:           xarray.Dataset
    :param cases:                  cases
    :type cases:                   list
    :param DEBUG_MODE:             whether or not to run in debug mode
    :type DEBUG_MODE:              bool

    :return:                       grids
    :rtype:                        xarray.Dataset
    """
    # Loop through times
    for reconstruction_time in tqdm(ages, desc=f"Loading {type} grids", disable=DEBUG_MODE):
        # Check if the grid for the reconstruction time is already in the dictionary
        if reconstruction_time in grid:
            # Rename variables and coordinates in seafloor age grid for clarity
            if type == "Seafloor":
                if "z" in grid[_age].data_vars:
                    grid[_age] = grid[_age].rename({"z": "seafloor_age"})
                if "lat" in grid[_age].coords:
                    grid[_age] = grid[_age].rename({"lat": "latitude"})
                if "lon" in grid[_age].coords:
                    grid[_age] = grid[_age].rename({"lon": "longitude"})

            continue

        # Load grid if found
        if type == "Seafloor":
            # Load grid if found
            grid[_age] = Dataset_from_netCDF(files_dir, type, _age, reconstruction_name)

            # Download seafloor age grid from GPlately DataServer
            grid[_age] = get_seafloor_grid(reconstruction_name, reconstruction_time)

        elif type == "Velocity" and cases:
            # Initialise dictionary to store velocity grids for cases
            grid[_age] = {}

            # Loop through cases
            for case in cases:
                # Load grid if found
                grid[_age][_case] = Dataset_from_netCDF(files_dir, type, _age, reconstruction_name, case=case)

                # If not found, initialise a new grid
                if grid[_age][_case] is None:
                
                    # Interpolate velocity grid from points
                    if type == "Velocity":
                        for case in cases:
                            if DEBUG_MODE:
                                print(f"{type} grid for {reconstruction_name} at {reconstruction_time} Ma not found, interpolating from points...")

                            # Get velocity grid
                            grid[_age][_case] = get_velocity_grid(points[_age][_case], seafloor_grid[_age])

    return grid

def DataFrame_from_parquet(
        folder: str,
        type: str,
        reconstruction_name: str,
        case: str,
        reconstruction_time: int
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param type:                 type of data
    :type type:                  str
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   int
    
    :return:                     data
    :rtype:                      pandas.DataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{case}_{reconstruction_time}Ma.parquet"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _pandas.read_parquet(target_file)
    else:
        return None

def DataFrame_from_csv(
        folder: str,
        type: str,
        reconstruction_name: str,
        case: str,
        reconstruction_time: int
    ) -> _pandas.DataFrame:
    """
    Function to load DataFrames from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param type:                 type of data
    :type type:                  str
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 case
    :type case:                  str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   int
    
    :return:                     data
    :rtype:                      pandas.DataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{case}_{reconstruction_time}Ma.csv"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _pandas.read_csv(target_file)
    else:
        return None

def Dataset_from_netCDF(
        folder: str,
        type: str,
        reconstruction_time: int,
        reconstruction_name: str,
        case: Optional[str] = None
    ) -> _xarray.Dataset:
    """
    Function to load xarray Dataset from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str
    :param case:                 optional case
    :type case:                  str, optional

    :return:                     data
    :rtype:                      xarray.Dataset or None
    """
    # Construct the file name based on whether a case is provided
    file_name = f"{type}_{reconstruction_name}_{case + '_' if case else ''}{reconstruction_time}Ma.nc"

    # Construct the full path to the target file
    target_file = _os.path.join(folder if folder else _os.getcwd(), type, file_name)

    # Check if the target file exists and load the dataset
    if _os.path.exists(target_file):
        return _xarray.open_dataset(target_file)
    else:
        return None
    
def GeoDataFrame_from_geoparquet(
        folder: str,
        type: str,
        reconstruction_time: int,
        reconstruction_name: str
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str

    :return:                     data
    :rtype:                      geopandas.GeoDataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{reconstruction_time}Ma.parquet"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _geopandas.read_parquet(target_file)
    else:
        return None
    
def GeoDataFrame_from_shapefile(
        folder: str,
        type: str,
        reconstruction_time: int,
        reconstruction_name: str
    ) -> _geopandas.GeoDataFrame:
    """
    Function to load GeoDataFrame from a folder efficiently.

    :param folder:               folder
    :type folder:                str
    :param reconstruction_time:  reconstruction time
    :type reconstruction_time:   int
    :param reconstruction_name:  name of reconstruction
    :type reconstruction_name:   str

    :return:                     data
    :rtype:                      geopandas.GeoDataFrame or None
    """
    # Construct the target file path
    target_file = _os.path.join(
        folder if folder else _os.getcwd(),
        type,
        f"{type}_{reconstruction_name}_{reconstruction_time}Ma.shp"
    )

    # Check if target file exists and load data
    if _os.path.exists(target_file):
        return _geopandas.read_file(target_file)
    else:
        return None