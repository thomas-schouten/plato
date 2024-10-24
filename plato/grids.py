import logging
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as _numpy
import gplately as _gplately
import xarray as _xarray
from tqdm import tqdm as _tqdm

from . import utils_data, utils_init
from .settings import Settings

class Grids():
    def __init__(
            self,
            settings: Optional[Settings] = None,
            reconstruction: Optional[_gplately.PlateReconstruction]= None,
            rotation_file: Optional[str]= None,
            topology_file: Optional[str]= None,
            polygon_file: Optional[str]= None,
            reconstruction_name: Optional[str] = None,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases_file: Optional[list[str]]= None,
            cases_sheet: Optional[str]= "Sheet1",
            files_dir: Optional[str]= None,
            seafloor_age_grids: Optional[Dict] = None,
            sediment_grids: Optional[Dict] = None,
            continental_grids: Optional[Dict] = None,
            velocity_grids: Optional[Dict] = None,
            DEBUG_MODE: Optional[bool] = False,
            PARALLEL_MODE: Optional[bool] = False,
        ):
        """
        Object to hold gridded data.
        Seafloor grids contain lithospheric age and, optionally, sediment thickness.
        Continental grids contain lithospheric thickness and, optionally, crustal thickness.
        Velocity grids contain plate velocity data.
        """
        # Store settings object
        self.settings = utils_init.get_settings(
            settings, 
            reconstruction_name,
            ages, 
            cases_file,
            cases_sheet,
            files_dir,
            PARALLEL_MODE = PARALLEL_MODE,
            DEBUG_MODE = DEBUG_MODE,
        )
            
        # Store reconstruction object
        self.reconstruction = utils_init.get_reconstruction(
            reconstruction,
            rotation_file,
            topology_file,
            polygon_file,
            reconstruction_name,
        )

        # Initialise dictionary to store grids
        self.data = {_age: None for _age in self.settings.ages}

        # Initialise dictionary to store grids
        self.seafloor_age = {_age: None for _age in self.settings.ages}

        # Load seafloor grids
        for _age in _tqdm(self.settings.ages, desc="Loading grids", disable=self.settings.logger.level==logging.INFO):
            if seafloor_age_grids is not None and _age in seafloor_age_grids.keys() and isinstance(seafloor_age_grids[_age], _xarray.Dataset):
                # If the seafloor is present in the provided dictionary, copy
                logging.info(f"Loading seafloor age grid for {_age} Ma.")
                self.seafloor_age[_age] = seafloor_age_grids[_age]

            else:
                logging.info(f"Downloading seafloor age grid for {_age} Ma.")
                self.seafloor_age[_age] = utils_data.get_seafloor_age_grid(
                    self.settings.name,
                    _age,
                )

            # Make sure that the coordinates and variables are named correctly
            if len(self.seafloor_age[_age].data_vars) == 1 and "z" in self.seafloor_age[_age].data_vars:
                self.seafloor_age[_age] = utils_data.rename_coordinates_and_variables(self.seafloor_age[_age], "z", "seafloor_age")
                
        # Store sediment, continental and velocity grids, if provided, otherwise initialise empty dictionaries to store them at a later stage.
        self.sediment = {_age: None for _age in self.settings.ages}
        if isinstance(sediment_grids, Dict):
            for _age in _tqdm(self.settings.ages, desc="Loading sediment grids", disable=self.settings.logger.level==logging.INFO):
                if _age in sediment_grids.keys() and isinstance(sediment_grids[_age], _xarray.Dataset):
                    # If the sediment is present in the provided dictionary, copy
                    logging.info(f"Loading sediment grid for {_age} Ma.")
                    self.sediment[_age] = sediment_grids[_age]

                    # Make sure that the coordinates and variables are named correctly
                    self.sediment[_age] = utils_data.rename_coordinates_and_variables(self.sediment[_age], "z", "sediment_thickness")

        self.continent = continental_grids if continental_grids else None
        if continental_grids is Dict:
            for _age in _tqdm(self.settings.ages, desc="Loading sediment grids", disable=self.settings.logger.level==logging.INFO):
                if _age in continental_grids.keys() and isinstance(continental_grids[_age], _xarray.Dataset):
                    # If the sediment is present in the provided dictionary, copy
                    logging.info(f"Loading sediment grid for {_age} Ma.")
                    self.continent[_age] = continental_grids[_age]

                    # Make sure that the coordinates and variables are named correctly
                    self.continent[_age] = utils_data.rename_coordinates_and_variables(self.sediment[_age], "z", "continental_thickness")

        self.velocity = {_age: None for _age in self.settings.ages}

    def __str__(self):
        return f"Plato grids object with global grids."
    
    def __repr__(self):
        return self.__str__()

    def add_grid(
            self,
            input_grids: Union[Dict[Union[int, float, _numpy.integer, _numpy.floating], _xarray.Dataset], _xarray.Dataset],
            variable_name: str = "new_grid",
            grid_type: str = "seafloor_age",
            target_variable: str = "z",
            mask_continents: Optional[bool] = False,
            interpolate: Optional[bool] = True,
            prefactor: Optional[float] = 1.,
        ):
        """
        Function to add another grid of a variable to the seafloor grid.
        The grids should be organised in a dictionary with each item being an xarray.Dataset with each key being the corresponding reconstruction age, or a single xarray.Dataset, in which case it will be stored without an age.
        'mask_continents' is a boolean that determines whether or not to cut the grids to the seafloor. It should only be used for grids that only describe the seafloor, e.g. marine sediment distributions, and not e.g. continental erosion rate grids.
        """

        # Check if the attribute exists and is initially None
        if getattr(self, grid_type) is None:
            # Initialize to the type of input_grids
            if isinstance(input_grids, dict):
                setattr(self, grid_type, {})
            else:
                setattr(self, grid_type, _xarray.Dataset())

        # Access the storage dictionary directly (not creating a local copy)
        new_grids = getattr(self, grid_type)

        # If input_grids is a single xarray.Dataset, store it directly
        if isinstance(input_grids, _xarray.Dataset):
            if target_variable in input_grids.variables:
                # Copy the input grid to avoid modifying the original
                _input_grids = deepcopy(input_grids)

                # Rename coordinates and variables
                _input_grids = utils_data.rename_coordinates_and_variables(_input_grids, target_variable, variable_name)

                # Store the grid with the prefactor applied
                _input_grids[variable_name] *= prefactor

                # Update the attribute
                new_grids[variable_name] = _input_grids

            else:
                print(f"Target variable '{target_variable}' does not exist in the input grids.")

        # If input_grids is a dictionary of xarray.Datasets, loop through the ages to store the grids
        elif isinstance(input_grids, Dict):
            # Loop through ages to load, interpolate, and store variables in the specified grid
            for age in self.settings.ages:
                # Check if target_variable exists in input grids
                if age in input_grids and target_variable in input_grids[age].variables:
                    # Copy the dataset to avoid modifying the original
                    _input_grid = deepcopy(input_grids[age])

                    # Rename coordinates and variables
                    _input_grid = utils_data.rename_coordinates_and_variables(_input_grid, target_variable, variable_name)

                    if interpolate:
                        # Interpolate input grids to the resolution of the seafloor age grid
                        _input_grid = _input_grid.interp_like(self.seafloor_age[age]["seafloor_age"], method="nearest")
                        _input_grid[variable_name] *= prefactor

                    # Align grids if mask_continents is True
                    if mask_continents and self.seafloor_age[age] is _xarray.Dataset:
                        mask = {}
                        for variable_1 in self.seafloor_age[age].data_vars:
                            mask[variable_1] = _numpy.isnan(self.seafloor_age[age][variable_1].values)

                            # Apply masks to all grids
                            for variable_2 in self.seafloor_age[age].data_vars:
                                _input_grid[variable_2] = self.seafloor_age[age][variable_2].where(~mask[variable_1])

                    # Update the dictionary stored in the attribute
                    new_grids[age] = _input_grid

                else:
                    print(f"Target variable '{target_variable}' does not exist in the input grids for {age} Ma.")
        
        else:
            raise ValueError("Input grids should be either a single xarray.Dataset or a dictionary of xarray.Datasets.")
        
        # Ensure the modified new_grids is saved back to the object's attribute
        setattr(self, grid_type, new_grids)
        logging.info(f"{grid_type} updated:", getattr(self, grid_type))

    def save_all(
        self,
        ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
        cases: Union[None, str, List[str]] = None,
        file_dir: Optional[str] = None,
        ):
        """
        Function to save all the grids
        """
        # Save seafloor grid
        self.save_seafloor_age(ages, file_dir)

        # Save sediment grid
        self.save_sediment(ages, cases, file_dir)

        # Save continental grid
        self.save_continent(ages, cases, file_dir)

        # Save velocity grid
        self.save_velocity(ages, cases, file_dir)

    def save_seafloor_age(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the seafloor age grid.
        """
        self.save_grid(self.seafloor_age, "Seafloor_age", ages, None, file_dir)

    def save_sediment(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the sediment grid.
        """
        if self.sediment is not None:
            self.save_grid(self.sediment, "Sediment", ages, cases, file_dir)

    def save_continent(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the continental grid.
        """
        # Check if grids exists
        if self.continent is not None:
            self.save_grid(self.continent, "Continent", ages, cases, file_dir)

    def save_velocity(
            self,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save the the velocity grid.
        """
        # Check if grids exists
        if self.velocity is not None:
            self.save_grid(self.velocity, "Velocity", ages, cases, file_dir)
        
    def save_grid(
            self,
            data: Dict,
            type: str,
            ages: Union[None, List[int], List[float], _numpy.ndarray] = None,
            cases: Union[None, str, List[str]] = None,
            file_dir: Optional[str] = None,
        ):
        """
        Function to save a grid
        """
        # Define ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Define cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Get file dir
        _file_dir = self.settings.dir_path if file_dir is None else file_dir
        
        # Loop through ages
        for _age in _tqdm(_ages, desc=f"Saving {type} grids", disable=self.settings.logger.level==logging.INFO):
            if data[_age] is Dict:
                # Loop through cases
                for _case in _cases:
                    if data[_age][_case] is _xarray.Dataset:
                        utils_data.Dataset_to_netcdf(
                            data[_age][_case],
                            type,
                            self.settings.name,
                            _age,
                            _case,
                            _file_dir,
                        )
    
            else:
                if data[_age] is _xarray.Dataset:
                    utils_data.Dataset_to_netcdf(
                            data[_age],
                            type,
                            self.settings.name,
                            _age,
                            None,
                            _file_dir,
                        )