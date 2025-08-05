# Standard libraries
import logging
import warnings
from typing import Dict, List, Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as _pandas
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm as _tqdm

# Plato libraries
from . import utils_data, utils_calc, utils_opt
from .plate_torques import PlateTorques

# For plotting
cm2in = 0.3937008
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"

class Optimisation():
    """
    A class to optimise the slab pull coefficient and mantle viscosity to match plate motions.

    :param plate_torques:   plate torques object
    :type plate_torques:    PlateTorques
    """    
    def __init__(
            self,
            plate_torques: Optional[PlateTorques] = None,
        ):
        """
        Constructor for the `Optimisation` class.
        """
        # Store the input data, if provided
        if isinstance(plate_torques, PlateTorques):
            # Store plate torques object
            self.plate_torques = plate_torques

            # Create shortcuts to the settings, plates, slabs, points, grids, and globe objects, and the ages and cases
            self.settings = plate_torques.settings
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe

            # Set shortcut to ages, cases and options
            self.ages = self.settings.ages
            self.cases = self.settings.cases
            self.options = self.settings.options

    def minimise_residual_torque(
            self,
            ages: Optional[Union[int, float]] = None, 
            cases: Optional[str] = None, 
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grid_size: Optional[int] = 500, 
            viscosity_range: Optional[List[Union[int, float]]] = [5e18, 5e20],
            weight_by_area: Optional[bool] = True,
            minimum_plate_area: Optional[Union[int, float]] = None,
            plot: Optional[bool] = True,
            savefig: Optional[str] = False,
            RECALCULATE_SLAB_PULL: Optional[bool] = True,
            INCLUDE_SLAB_SUCTION: Optional[bool] = False,
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search.
        The optimal coefficients are those that minimise the residual torque normalised by the driving torque.
        If more than one plate is provided, the normalised residual torque is weighted by the plate area.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
        :rtype:                         float, float, numpy.ndarray, tuple

        NOTE: This is a wrapper function that calls either `minimise_residual_torque_v0` or `minimise_residual_torque_v1` depending on the value of `INCLUDE_SLAB_SUCTION`.
        This could be written more elegantly in the future.

        NOTE: Lateral viscosity variations need to be introduced along a third dimension.
        """

        if INCLUDE_SLAB_SUCTION:
            return self.minimise_residual_torque_v1(
                ages=ages, 
                cases=cases, 
                plateIDs=plateIDs,
                grid_size=grid_size, 
                viscosity_range=viscosity_range,
                weight_by_area=weight_by_area,
                minimum_plate_area=minimum_plate_area,
                plot=plot,
                savefig=savefig,
            )
        
        else:
            return self.minimise_residual_torque_v0(
                ages=ages, 
                cases=cases, 
                plateIDs=plateIDs,
                grid_size=grid_size, 
                viscosity_range=viscosity_range,
                weight_by_area=weight_by_area,
                minimum_plate_area=minimum_plate_area,
                plot=plot,
                savefig=savefig,
                RECALCULATE_SLAB_PULL=RECALCULATE_SLAB_PULL,
            )
        
    def minimise_residual_torque_v0(
            self,
            ages: Optional[Union[int, float]] = None, 
            cases: Optional[str] = None, 
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grid_size: Optional[int] = 500, 
            viscosity_range: Optional[List[Union[int, float]]] = [5e18, 5e20],
            weight_by_area: Optional[bool] = True,
            minimum_plate_area: Optional[Union[int, float]] = None,
            plot: Optional[bool] = True,
            savefig: Optional[str] = False,
            RECALCULATE_SLAB_PULL: Optional[bool] = True,
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search.
        The optimal coefficients are those that minimise the residual torque normalised by the driving torque.
        If more than one plate is provided, the normalised residual torque is weighted by the plate area.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
        :rtype:                         float, float, numpy.ndarray, tuple

        NOTE: Lateral viscosity variations need to be introduced along a third dimension.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Set range of viscosities
        viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)

        # Initialise dictionaries to store results
        normalised_residual_torques = {_age: {_case: None for _case in _cases} for _age in _ages}
        optimal_parameters = {_age: {_case: {} for _case in _cases} for _age in _ages}
        optimal_indices = {_age: {_case: None for _case in _cases} for _age in _ages}
        
        # Loop through ages
        for _age in _ages:
            for _case in _cases:
                # Set range of slab pull coefficients
                if self.settings.options[_case]["Sediment subduction"]:
                    # Range is smaller with sediment subduction
                    sp_consts = _numpy.linspace(1e-5, 0.1, grid_size)
                else:
                    sp_consts = _numpy.linspace(1e-5, 1., grid_size)

                # Create grids from ranges of viscosities and slab pull coefficients
                visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
                ones_grid = _numpy.ones_like(visc_grid)

                # Filter plates
                _data = self.plates.data[_age][_case].copy()

                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                if plateIDs is not None:
                    _data = _data[_data["plateID"].isin(_plateIDs)].copy()

                if minimum_plate_area is not None:
                    _data = _data[_data["area"] > minimum_plate_area].copy()

                # Recalculate slab pull torque correctly to prevent issues
                if RECALCULATE_SLAB_PULL:
                    # Get slab data and filter accordingly
                    _slab_data = self.slabs.data[_age][_case].copy()
                    _slab_data = _slab_data[_slab_data.lower_plateID.isin(_data.plateID)]

                    # Calculate slab pull force
                    _slab_pull_force_lat = _slab_data.slab_pull_force_lat / _slab_data.slab_pull_constant * self.settings.options[_case]["Slab pull constant"]
                    _slab_pull_force_lon = _slab_data.slab_pull_force_lon / _slab_data.slab_pull_constant * self.settings.options[_case]["Slab pull constant"]

                    # Calculate slab pull torque with the modified slab pull forces
                    computed_data = utils_calc.compute_torque_on_plates(
                        _data,
                        _slab_data.lat.values,
                        _slab_data.lon.values,
                        _slab_data.lower_plateID.values,
                        _slab_pull_force_lat, 
                        _slab_pull_force_lon,
                        _slab_data.trench_segment_length.values,
                        torque_var = "slab_pull",
                    )
                    _data = computed_data

                # Get total area
                total_area = _data["area"].sum()
                    
                driving_mag = _numpy.zeros_like(sp_const_grid)
                residual_mag = _numpy.zeros_like(sp_const_grid)

                # Get torques
                for k, _ in enumerate(_data.plateID):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                        
                        # Compute magnitude of driving torque
                        if weight_by_area:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]

                        # Compute magnitude of residual
                        if weight_by_area:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
                
                # Divide residual by driving torque
                residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

                # Find the indices of the minimum value
                opt_i, opt_j = _numpy.unravel_index(_numpy.argmin(residual_mag_normalised), residual_mag_normalised.shape)
                opt_visc = visc_grid[opt_i, opt_j]
                opt_sp_const = sp_const_grid[opt_i, opt_j]

                # Plot
                if plot == True:
                    fig, ax = plt.subplots(figsize=(9*cm2in*2, 7.2*cm2in*2))
                    im = ax.imshow(residual_mag_normalised.T, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
                    ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_yticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(viscosity_range[0], viscosity_range[1], 5)])
                    ax.set_xticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                    ax.set_ylabel("Asthenospheric viscosity [Pa s]")
                    ax.set_xlabel("Slab pull coefficient")
                    ax.scatter(opt_i, opt_j, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
                    fig.colorbar(im, label = "Log(residual torque/driving torque)", extend="both", extendfrac=5e-2)
                    if savefig is not False:
                        plt.savefig(savefig, dpi=300, bbox_inches="tight")
                    plt.show()

                # Print results
                print(f"Optimal coefficients for ", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
                print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
                print("Optimum asthenospheric viscosity [Pa s]: {:.2e}".format(opt_visc))
                print("Optimum drag coefficient [Pa s/m]: {:.2e}".format(opt_visc / self.settings.mech.La))
                print("Optimum slab pull coefficient: {:.2%}".format(opt_sp_const))

                # Store results
                normalised_residual_torques[_age][_case] = residual_mag_normalised
                optimal_parameters[_age][_case]["Slab pull constant"] = opt_sp_const
                optimal_parameters[_age][_case]["Mantle viscosity"] = opt_visc
                optimal_indices[_age][_case] = (opt_i, opt_j)
            
        return normalised_residual_torques, optimal_parameters, optimal_indices

    def minimise_residual_torque_v1(
            self,
            ages: Optional[Union[int, float]] = None, 
            cases: Optional[str] = None, 
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            grid_size: Optional[int] = 500, 
            viscosity_range: Optional[List[Union[int, float]]] = [5e18, 5e20],
            weight_by_area: Optional[bool] = True,
            minimum_plate_area: Optional[Union[int, float]] = None,
            plot: Optional[bool] = True,
            savefig: Optional[str] = False,
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search.
        The optimal coefficients are those that minimise the residual torque normalised by the driving torque.
        If more than one plate is provided, the normalised residual torque is weighted by the plate area.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean

        :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
        :rtype:                         float, float, numpy.ndarray, tuple

        NOTE: Lateral viscosity variations need to be introduced along a third dimension.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Set range of viscosities
        viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)

        # Make dictionary to store results
        normalised_residual_torques = {_age: {_case: None for _case in _cases} for _age in _ages}
        optimal_sp_consts = {_age: {_case: None for _case in _cases} for _age in _ages}
        optimal_viscs = {_age: {_case: None for _case in _cases} for _age in _ages}
        optimal_ss_consts = {_age: {_case: None for _case in _cases} for _age in _ages}
        optimal_indices = {_age: {_case: None for _case in _cases} for _age in _ages}
        
        # Loop through ages
        for _age in _ages:
            for _case in _cases:
                # Set range of slab pull and slab suction coefficients
                if self.settings.options[_case]["Sediment subduction"]:
                    # Range is smaller with sediment subduction
                    sp_consts = _numpy.linspace(1e-5, 0.1, grid_size)
                    ss_consts = _numpy.linspace(1e-5, 0.1, grid_size)
                else:
                    sp_consts = _numpy.linspace(1e-5, 1., grid_size)
                    ss_consts = _numpy.linspace(1e-5, 1., grid_size)

                # Create grids from ranges of viscosities and slab pull coefficients
                visc_grid, sp_const_grid, ss_const_grid = _numpy.meshgrid(viscs, sp_consts, ss_consts)
                ones_grid = _numpy.ones_like(visc_grid)

                # Filter plates
                _data = self.plates.data[_age][_case]

                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                if plateIDs is not None:
                    _data = _data[_data["plateID"].isin(_plateIDs)].copy()

                if minimum_plate_area is not None:
                    _data = _data[_data["area"] > minimum_plate_area].copy()

                # Recalculate slab pull and suction torques correctly to prevent issues
                # Get slab data and filter accordingly
                _slab_data = self.slabs.data[_age][_case]
                _slab_data = _slab_data[_slab_data.lower_plateID.isin(_data.plateID)].copy()

                # Calculate slab pull force
                _slab_pull_force_lat = _slab_data.slab_pull_force_lat / \
                    _slab_data.slab_pull_constant * self.settings.options[_case]["Slab pull constant"]
                _slab_pull_force_lon = _slab_data.slab_pull_force_lon / \
                    _slab_data.slab_pull_constant * self.settings.options[_case]["Slab pull constant"]

                # Calculate slab pull torque with the modified slab pull forces
                computed_data = utils_calc.compute_torque_on_plates(
                    _data,
                    _slab_data.lat.values,
                    _slab_data.lon.values,
                    _slab_data.lower_plateID.values,
                    _slab_pull_force_lat, 
                    _slab_pull_force_lon,
                    _slab_data.trench_segment_length.values,
                    torque_var = "slab_pull",
                )
                _data = computed_data

                # Calculate slab suction force
                _slab_suction_force_lat = _slab_data.slab_suction_force_lat / \
                    _slab_data.slab_suction_constant * self.settings.options[_case]["Slab suction constant"]
                _slab_suction_force_lon = _slab_data.slab_suction_force_lon / \
                    _slab_data.slab_suction_constant * self.settings.options[_case]["Slab suction constant"]  
                              
                # Calculate slab pull torque with the modified slab pull forces
                computed_data = utils_calc.compute_torque_on_plates(
                    _data,
                    _slab_data.lat.values,
                    _slab_data.lon.values,
                    _slab_data.upper_plateID.values,
                    _slab_suction_force_lat, 
                    _slab_suction_force_lon,
                    _slab_data.trench_segment_length.values,
                    torque_var = "slab_suction",
                )
                _data = computed_data

                # Get total area
                total_area = _data["area"].sum()
                    
                driving_mag = _numpy.zeros_like(sp_const_grid)
                residual_mag = _numpy.zeros_like(sp_const_grid)

                # Get torques
                for k, _ in enumerate(_data.plateID):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)
                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab pull constant"]

                        if self.settings.options[_case]["Slab suction torque"] and "slab_suction_torque_x" in _data.columns:
                            residual_x -= _data.slab_suction_torque_x.iloc[k] * ss_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_y -= _data.slab_suction_torque_y.iloc[k] * ss_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_z -= _data.slab_suction_torque_z.iloc[k] * ss_const_grid / self.settings.options[_case]["Slab suction constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                        
                        # Compute magnitude of driving torque
                        if weight_by_area:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[_case]["Mantle viscosity"]

                        # Compute magnitude of residual
                        if weight_by_area:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
                
                # Divide residual by driving torque
                residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

                # # Find the index of the global minimum value
                opt_index = _numpy.unravel_index(_numpy.argmin(residual_mag_normalised), residual_mag_normalised.shape)

                # Unpack the indices
                opt_i, opt_j, opt_k = opt_index
                opt_visc = visc_grid[opt_i, opt_j, opt_k]
                opt_sp_const = sp_const_grid[opt_i, opt_j, opt_k]
                opt_ss_const = ss_const_grid[opt_i, opt_j, opt_k]

                # Plot
                if plot == True:
                    fig = plt.figure(figsize=(18*cm2in*2, 8*cm2in*2))
                    gs = gridspec.GridSpec(1, 2)

                    ax1 = plt.subplot(gs[0, 0])
                    im1 = ax1.imshow(residual_mag_normalised[:, :, opt_k].T, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
                    ax1.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax1.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax1.set_yticklabels(["{:.1e}".format(visc) for visc in _numpy.linspace(viscosity_range[0], viscosity_range[1], 5)])
                    ax1.set_xticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                    ax1.set_ylabel("Asthenospheric viscosity [Pa s]")
                    ax1.set_xlabel("Slab pull coefficient")
                    ax1.scatter(opt_i, opt_j, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
                    ax1.annotate("a", xy=(0, 1.03), xycoords="axes fraction", fontsize=18, fontweight="bold")
                    
                    ax2 = plt.subplot(gs[0, 1])
                    im2 = ax2.imshow(residual_mag_normalised[opt_i, :, :], cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
                    ax2.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax2.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax2.set_yticklabels([])
                    ax2.set_xticklabels(["{:.2f}".format(ss_const) for ss_const in _numpy.linspace(ss_consts.min(), ss_consts.max(), 5)])
                    ax2.set_ylabel("")
                    ax2.set_xlabel("Slab suction coefficient")
                    ax2.scatter(opt_k, opt_j, marker="*", facecolor="none", edgecolor="k", s=30)  # Use opt_i and opt_k here
                    ax2.annotate("b", xy=(0, 1.03), xycoords="axes fraction", fontsize=18, fontweight="bold")

                    cax = fig.add_axes([0.362, -.02, 0.3, 0.02*(10.5/8)])
                    cbar = plt.colorbar(im2, cax=cax, orientation="horizontal", extend="both", extendfrac=5e-2)
                    cbar.set_label("Log10(residual torque/driving torque)")
                    if savefig:
                        fig.savefig(savefig, dpi=300, bbox_inches="tight")
                    plt.show()

                # Print results
                print(f"Optimal coefficients for case {_case} at {_age} Ma: for the", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
                print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
                print("Optimum viscosity [Pa s]: {:.2e}".format(opt_visc))
                print("Optimum drag coefficient [Pa s/m]: {:.2e}".format(opt_visc / self.settings.mech.La))
                print("Optimum slab pull constant: {:.2%}".format(opt_sp_const))
                print("Optimum slab suction constant: {:.2%}".format(opt_ss_const))

                # Store results
                normalised_residual_torques[_age][_case] = residual_mag_normalised
                optimal_sp_consts[_age][_case] = opt_sp_const
                optimal_viscs[_age][_case] = opt_visc
                optimal_ss_consts[_age][_case] = opt_ss_const
                optimal_indices[_age][_case] = opt_index

        return normalised_residual_torques, optimal_sp_consts, optimal_viscs, optimal_ss_consts, optimal_indices
    
    def optimise_slab_pull_constant(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            grid_size = 500, 
            viscosity = None,
            max_slab_pull_constant = .9,
            PLOT = False, 
        ):
        """
        Function to find optimised slab pull coefficient for a given (set of) plates using a grid search.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float
        """
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        for _age in _tqdm(_ages, desc="Optimising slab pull coefficient"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _case in _cases:
                    # Select viscosity, if not provided
                    if viscosity is None:
                        viscosity = self.settings.options[_case]["Mantle viscosity"]

                    # Generate range of possible slab pull coefficients
                    sp_consts = _numpy.linspace(1e-5, max_slab_pull_constant, grid_size)
                    ones = _numpy.ones_like(sp_consts)

                    # Filter plates
                    _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                    # First reset all the slab pull forces to their original values to avoid overwriting, as this sometimes results in NaN values
                    self.slabs.data[_age][_case].loc[self.slabs.data[_age][_case]["lower_plateID"].isin(_plateIDs), "slab_pull_constant"] = self.settings.options[_case]["Slab pull constant"]
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateIDs, PROGRESS_BAR=False)

                    # Select data
                    _data = self.plates.data[_age][_case]
                    _data = _data[_data["plateID"].isin(_plateIDs)]

                    # Loop through plates
                    for _plateID in _data.plateID:
                        # Reset the slab pull constant
                        slab_mask = self.slabs.data[_age][_case]["lower_plateID"] == _plateID
                        if slab_mask.sum() == 0:
                            continue

                        # Initialise residual torque arrays
                        residual_x = _numpy.zeros_like(sp_consts)
                        residual_y = _numpy.zeros_like(sp_consts)
                        residual_z = _numpy.zeros_like(sp_consts)

                        # Make plate mask
                        plate_mask = _data["plateID"] == _plateID

                        # Calculate slab pull torque
                        if self.settings.options[_case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
                            residual_x -= _data.loc[plate_mask, "slab_pull_torque_x"].values[0] * sp_consts / self.settings.options[_case]["Slab pull constant"]
                            residual_y -= _data.loc[plate_mask, "slab_pull_torque_y"].values[0] * sp_consts / self.settings.options[_case]["Slab pull constant"]
                            residual_z -= _data.loc[plate_mask, "slab_pull_torque_z"].values[0] * sp_consts / self.settings.options[_case]["Slab pull constant"]

                        # Add GPE torque
                        if self.settings.options[_case]["GPE torque"] and "GPE_torque_x" in _data.columns:
                            residual_x -= _data.loc[plate_mask, "GPE_torque_x"].values[0] * ones
                            residual_y -= _data.loc[plate_mask, "GPE_torque_y"].values[0] * ones
                            residual_z -= _data.loc[plate_mask, "GPE_torque_z"].values[0] * ones

                        # Compute magnitude of driving torque
                        driving_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

                        # Add slab bend torque
                        if self.settings.options[_case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
                            residual_x -= _data.loc[plate_mask, "slab_bend_torque_x"].values[0] * ones
                            residual_y -= _data.loc[plate_mask, "slab_bend_torque_y"].values[0] * ones
                            residual_z -= _data.loc[plate_mask, "slab_bend_torque_z"].values[0] * ones

                        # Add mantle drag torque
                        if self.settings.options[_case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
                            residual_x -= _data.loc[plate_mask, "mantle_drag_torque_x"].values[0] * viscosity / self.settings.options[_case]["Mantle viscosity"]
                            residual_y -= _data.loc[plate_mask, "mantle_drag_torque_y"].values[0] * viscosity / self.settings.options[_case]["Mantle viscosity"]
                            residual_z -= _data.loc[plate_mask, "mantle_drag_torque_z"].values[0] * viscosity / self.settings.options[_case]["Mantle viscosity"]

                        # Compute magnitude of residual
                        residual_mag = _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2)

                        if PLOT:
                            fig, ax = plt.subplots(figsize=(5,5))
                            ax.set_title(f"Normalised residual torque as a function of slab pull constant for plate {_plateID}")
                            p = ax.plot(sp_consts, residual_mag/driving_mag)
                            ax.scatter(
                                sp_consts[_numpy.argmin(residual_mag/driving_mag)],
                                _numpy.min(residual_mag/driving_mag),
                                color="red"
                            )
                            closest_index = (_numpy.abs(sp_consts - self.settings.options[_case]["Slab pull constant"])).argmin()
                            ax.scatter(
                                sp_consts[closest_index],
                                (residual_mag/driving_mag)[closest_index], 
                                color="green"
                            )
                            ax.semilogy()
                            # ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                            # ax.set_xticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(0, 1, 5)])
                            ax.set_ylim([10**-3.5, 10**1.5])
                            ax.set_xlim([0, 1])
                            ax.set_ylabel("Normalised residual torque")
                            ax.set_xlabel("Slab pull coefficient")
                            plt.show()
                        
                        # Find optimal slab pull coefficient
                        opt_sp_const = sp_consts[_numpy.argmin(residual_mag/driving_mag)]

                        # Store optimal slab pull coefficient
                        self.slabs.data[_age][_case].loc[slab_mask, "slab_pull_constant"] = opt_sp_const

                        # Recalculate all the relevant torques
                        self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                        self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                        self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

    def invert_residual_torque(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            max_slab_pull_constant = 0.9,
            min_slab_pull_constant = 0.0,
            vmin = -14.0, 
            vmax = -6.0,
            step = .25, 
            NUM_ITERATIONS = 100,
            PLOT = False, 
        ):
        """
        Function to find optimised slab pull constant by projecting the residual torque onto the subduction zones.
        The function loops through all the unique combinations of ages and cases and optimises the slab pull coefficient for each plate.
        The model space for the slab pull constant is explored using an iterative approach with an adaptive step size.

        :param age:                     reconstruction age to optimise
        :type age:                      int, float
        :param case:                    case to optimise
        :type case:                     str
        :param plateIDs:                plate IDs to include in optimisation
        :type plateIDs:                 list of integers or None
        :param grid_size:               size of the grid to find optimal slab pull coefficient
        :type grid_size:                int
        :param NUM_ITERATIONS:          number of iterations to perform
        :type NUM_ITERATIONS:           int
        :param PLOT:                    whether or not to plot the grid
        :type PLOT:                     boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _plate_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _slab_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # NOTE: This is a temporary fix to avoid some runtime warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Loop through ages
            for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
                
                for _case in _cases:
                    # Obtain plateIDs for all unique combinations of ages and cases
                    _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                    # Initialise entries for each plate ID in dictionaries
                    minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                    opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}                    

                    # Loop through plates
                    for _plateID in _plateIDs[_age][_case]:
                        if _plateID not in self.plates.data[_age][_case]["plateID"].values:
                            continue

                        # Select data
                        _plate_data = self.plates.data[_age][_case].copy()
                        _slab_data = self.slabs.data[_age][_case].copy()

                        # Filter data
                        _plate_data = _plate_data[_plate_data["plateID"] == _plateID]
                        _slab_data = _slab_data[_slab_data["lower_plateID"] == _plateID]

                        # Skip if no data
                        if _slab_data.empty:
                            continue
                        
                        # First reset all the slab pull forces to their original values to avoid overwriting, as this sometimes results in NaN values
                        self.slabs.data[_age][_case].loc[self.slabs.data[_age][_case]["lower_plateID"] == _plateID, "slab_pull_constant"] = self.settings.options[_case]["Slab pull constant"]
                        self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                        # Get the slab pull force along subduction zones
                        slab_pull_force_lat = _slab_data["slab_pull_force_lat"].values
                        slab_pull_force_lon = _slab_data["slab_pull_force_lon"].values
                        slab_pull_force_mag = _slab_data["slab_pull_force_mag"].values

                        # Get the maximum slab pull force magnitude
                        max_slab_pull_force_mag = (slab_pull_force_mag / _slab_data["slab_pull_constant"].values) * max_slab_pull_constant

                        # Get the residual force along subduction zones
                        residual_force_lat = _slab_data["slab_residual_force_lat"].values
                        residual_force_lon = _slab_data["slab_residual_force_lon"].values

                        # Extract the driving and residual torques
                        driving_torque = _plate_data[_plate_data["plateID"] == _plateID]["driving_torque_mag"].values[0]
                        residual_torque = _plate_data[_plate_data["plateID"] == _plateID]["residual_torque_mag"].values[0]

                        # Initialise dictionaries to store all the relevant values values
                        _normalised_residual_torque = [_numpy.log10(residual_torque / driving_torque)]
                        _slab_pull_force_mag, _slab_pull_force_lat, _slab_pull_force_lon = {}, {}, {}
                        _residual_force_lat, _residual_force_lon = {}, {}

                        # Assign initial values
                        _slab_pull_force_mag[0] = slab_pull_force_mag
                        _slab_pull_force_lat[0] = slab_pull_force_lat
                        _slab_pull_force_lon[0] = slab_pull_force_lon
                        _residual_force_lat[0] = residual_force_lat
                        _residual_force_lon[0] = residual_force_lon

                        for k in range(NUM_ITERATIONS):
                            # Define constants for this iteration
                            constants = _numpy.arange(vmin, vmax, step)

                            # Store existing values and scores
                            existing_values = []; existing_scores = []

                            # Loop through constants
                            for i in _numpy.arange(0, len(constants)+10):

                                # Propose initial value
                                if i > len(constants)-1:
                                    # The inversion should start with a more general, grid-based exploration of the parameter space
                                    # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                                    constant = utils_calc.propose_value(existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                                else:
                                    constant = 10**constants[i]
                                                                        
                                existing_values.append(constant)

                                # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                                # This step should be performed in Cartesian coordinates.
                                _iter_slab_pull_force_mag = _slab_data.slab_pull_force_mag - (
                                    residual_force_lat * slab_pull_force_lat + \
                                    residual_force_lon * slab_pull_force_lon
                                ) * constant

                                # Ensure the slab pull force magnitude is positive, not larger than the original slab pull force magnitude and that NaN values are replaced with 0
                                _iter_slab_pull_force_mag = _numpy.where(_iter_slab_pull_force_mag < min_slab_pull_constant, min_slab_pull_constant, _iter_slab_pull_force_mag)
                                _iter_slab_pull_force_mag = _numpy.where(_numpy.isnan(_iter_slab_pull_force_mag), slab_pull_force_mag, _iter_slab_pull_force_mag)
                                _iter_slab_pull_force_mag = _numpy.where(_iter_slab_pull_force_mag > max_slab_pull_force_mag, max_slab_pull_force_mag, _iter_slab_pull_force_mag)

                                # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                                _iter_slab_pull_force_lat = _numpy.cos(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _iter_slab_pull_force_mag
                                _iter_slab_pull_force_lon = _numpy.sin(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _iter_slab_pull_force_mag
                                
                                # Calculate the torques with the modified slab pull forces
                                _iter_torques = utils_calc.compute_torque_on_plates(
                                    _plate_data,
                                    _slab_data.lat.values,
                                    _slab_data.lon.values,
                                    _slab_data.lower_plateID.values,
                                    _iter_slab_pull_force_lat, 
                                    _iter_slab_pull_force_lon,
                                    _slab_data.trench_segment_length.values,
                                    torque_var = "slab_pull",
                                )

                                # Calculate driving and residual torque
                                _iter_torques = utils_calc.sum_torque(_iter_torques, "driving")
                                _iter_torques = utils_calc.sum_torque(_iter_torques, "residual")

                                # Extract the driving and residual torques
                                _iter_driving_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["driving_torque_mag"].values[0]
                                _iter_residual_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["residual_torque_mag"].values[0]

                                # Calculate normalised residual torque and append to list
                                normalised_residual_torque = _numpy.log10(_iter_residual_torque / _iter_driving_torque)
                                existing_scores.append(normalised_residual_torque)

                            # Find the minimum normalised residual torque and the corresponding constant
                            # NOTE: This sometimes threw and error, so it used to be wrapped in a try-except block
                            try:
                                opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                                _normalised_residual_torque.append(_numpy.nanmin(_numpy.asarray(existing_scores)))
                                opt_constant = _numpy.asarray(existing_values)[opt_index]
                            except:
                                opt_index = 0
                                _normalised_residual_torque.append(_normalised_residual_torque[0])
                                opt_constant = 0

                            if PLOT == True or PLOT == _plateID:
                                # Plot the optimisation process
                                fig, axes = plt.subplots(1, 2)
                                axes[0].plot(existing_scores)
                                axes[0].set_xlabel("Iteration")
                                axes[0].set_ylabel("Score")
                                axes[0].set_ylim(-10, 1)
                                axes[0].scatter(opt_index, _numpy.asarray(existing_scores)[opt_index])

                                p = axes[1].scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                                axes[1].scatter(_numpy.log10(opt_constant), _numpy.asarray(existing_scores)[opt_index], c="k", marker="*")
                                axes[1].set_ylim(-10, 1)
                                axes[1].set_xlim(vmin, vmax)
                                fig.colorbar(p, label="iteration", orientation="vertical")
                                axes[1].set_xlabel("Parameter value")
                                axes[1].set_yticklabels([])
                                fig.suptitle(f"Optimisation for plateID {_plateID}")
                                plt.show()
                                
                                print(f"Starting normalised residual torque for plateID {_plateID}: {_numpy.asarray(existing_scores)[0]}")
                                print(f"Optimal constant for plateID {_plateID}: {opt_constant}")
                                print(f"Optimal normalised residual torque for plateID {_plateID}: {_numpy.asarray(existing_scores)[opt_index]}")
                            
                            # Apply the minimum residual torque to the slab data
                            _slab_data["slab_pull_force_mag"] -= (
                                residual_force_lat * slab_pull_force_lat + \
                                residual_force_lon * slab_pull_force_lon
                            ) * opt_constant

                            # Ensure the slab pull force magnitude is positive, not larger than the original slab pull force magnitude and that NaN values are replaced with the original slab pull force magnitude
                            _slab_data["slab_pull_force_mag"] = _numpy.where(_slab_data["slab_pull_force_mag"] < 0, 0, _slab_data["slab_pull_force_mag"])
                            _slab_data["slab_pull_force_mag"] = _numpy.where(_slab_data["slab_pull_force_mag"] > max_slab_pull_force_mag, max_slab_pull_force_mag, _slab_data["slab_pull_force_mag"])
                            _slab_data["slab_pull_force_mag"] = _numpy.where(_numpy.isnan(_slab_data["slab_pull_force_mag"]), _slab_data["slab_pull_force_mag"].values, _slab_data["slab_pull_force_mag"])
                            _slab_data["slab_pull_force_mag"] = _numpy.nan_to_num(_slab_data["slab_pull_force_mag"], nan=0.0)

                            # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                            slab_pull_force_lat = _numpy.cos(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _slab_data["slab_pull_force_mag"]
                            slab_pull_force_lon = _numpy.sin(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _slab_data["slab_pull_force_mag"]
                            
                            # Calculate the torques with the modified slab pull forces
                            _iter_torques = utils_calc.compute_torque_on_plates(
                                _plate_data,
                                _slab_data.lat.values,
                                _slab_data.lon.values,
                                _slab_data.lower_plateID.values,
                                slab_pull_force_lat, 
                                slab_pull_force_lon,
                                _slab_data.trench_segment_length.values,
                                torque_var = "slab_pull",
                            )

                            # Calculate residual torque
                            _residual_torques = utils_calc.sum_torque(_iter_torques, "residual")
                            _residual_torque = _residual_torques[_residual_torques["plateID"] == _plateID]

                            # print("New residual torque: ", _residual_torque["residual_torque_mag"].values[0])

                            # Calculate residual forces at slabs
                            residual_force_lat, residual_force_lon, _, _ = utils_calc.compute_residual_force(
                                _slab_data,
                                _residual_torque,
                                plateID_col = "lower_plateID",
                                weight_col = "trench_normal_azimuth",
                            )

                            # Store residual torques in dictionaries
                            _slab_pull_force_mag[k+1] = _slab_data["slab_pull_force_mag"]
                            
                            # Catch any NaN values
                            residual_force_lat = _numpy.nan_to_num(residual_force_lat, nan=0.0)
                            residual_force_lon = _numpy.nan_to_num(residual_force_lon, nan=0.0)

                        # Find iteration with minimum value
                        try:
                            _normalised_residual_torque = _numpy.nan_to_num(_numpy.asarray(_normalised_residual_torque), nan=_numpy.inf)
                            opt_iter = _numpy.argmin(_normalised_residual_torque)
                        except:
                            opt_iter = 0

                        self.slabs.data[_age][_case].loc[_slab_data.index, "slab_pull_constant"] = _numpy.nan_to_num(
                            _slab_pull_force_mag[opt_iter] / (max_slab_pull_force_mag / max_slab_pull_constant),
                            nan=0.0
                        )

                        if PLOT:
                            fig, axes = plt.subplots(2, 1)
                            axes[0].plot(_normalised_residual_torque)
                            axes[0].set_ylim(-10, 1)
                            axes[0].set_xlabel("Iteration")
                            axes[0].set_ylabel("Normalised residual torque")
                            axes[0].scatter(opt_iter, _normalised_residual_torque[opt_iter], c="k", marker="*")

                            p = axes[1].scatter(
                                _slab_data.lon,
                                _slab_data.lat,
                                c=_slab_pull_force_mag[opt_iter] / max_slab_pull_force_mag,
                                vmin=0, vmax=.5
                            )
                            axes[1].set_xlim(-180, 180)
                            axes[1].set_ylim(-90, 90)
                            fig.colorbar(p, label="Slab pull constant", orientation="horizontal")
                
                        # Recalculate all the relevant torques
                        self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                        self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                        self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

    def optimise_torques(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            optimisation_age = None,
            optimisation_case = None,
            optimisation_plateID = None
        ):
        """
        Function to optimsie the torques in the PlateTorques object.

        :param ages:                    ages to optimise (default: None)
        :type ages:                     list of int, float
        :param cases:                   cases to optimise (default: None)
        :type cases:                    list of str
        :param plateIDs:                plate IDs to optimise (default: None)
        :type plateIDs:                 list of int
        :param optimisation_age:        age to optimise (default: None)
        :type optimisation_age:         int, float
        :param optimisation_case:       case to optimise (default: None)
        :type optimisation_case:        str
        :param optimisation_plateID:    plate ID of plate to optimise (default: None)
        :type optimisation_plateID:     int

        NOTE: This function is not written in the most efficient way and can be improved.
        """
        # Select ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages and cases
        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                for k, _plateID in enumerate(self.plates.data[_age][_case].plateID):
                    if _plateID in _plateIDs:
                        # Filter data
                        _plate_data = self.plates.data[_age][_case][self.plates.data[_age][_case]["plateID"] == _plateID].copy()
                        _slab_data = self.slabs.data[_age][_case][self.slabs.data[_age][_case]["lower_plateID"] == _plateID].copy()
                        _point_data = self.points.data[_age][_case][self.points.data[_age][_case]["plateID"] == _plateID].copy()
                        
                        # Select optimal slab pull coefficient and viscosity
                        if optimisation_age or optimisation_case or optimisation_plateID is None:
                            opt_sp_const = self.opt_sp_const[_age][_case][k]
                            opt_visc = self.opt_visc[_age][_case][k]
                        else:
                            opt_sp_const = self.opt_sp_const[optimisation_age][optimisation_case][optimisation_plateID] 
                            opt_visc = self.opt_visc[optimisation_age][optimisation_case][optimisation_plateID]

                        # Optimise plate torque
                        for coord in ["x", "y", "z", "mag"]:
                            _plate_data.loc[:, f"slab_pull_torque_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]
                            self.plates.data[_age][_case].loc[_plate_data.index, f"slab_pull_torque_{coord}"] = _plate_data[f"slab_pull_torque_{coord}"].values[0]

                            _plate_data.loc[:, f"mantle_drag_torque_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]
                            self.plates.data[_age][_case].loc[_plate_data.index, f"mantle_drag_torque_{coord}"] = _plate_data[f"mantle_drag_torque_{coord}"].values[0]

                        # Optimise slab pull force
                        for coord in ["lat", "lon", "mag"]:
                            _slab_data.loc[:, f"slab_pull_force_{coord}"] *= opt_sp_const / self.settings.options[_case]["Slab pull constant"]
                            self.slabs.data[_age][_case].loc[_slab_data.index, f"slab_pull_force_{coord}"] = _slab_data[f"slab_pull_force_{coord}"].values[0]

                        # Optimise mantle drag torque
                        for coord in ["lat", "lon", "mag"]:
                            _point_data.loc[:, f"mantle_drag_force_{coord}"] *= opt_visc / self.settings.options[_case]["Mantle viscosity"]
                            self.points.data[_age][_case].loc[_point_data.index, f"mantle_drag_force_{coord}"] = _point_data[f"mantle_drag_force_{coord}"].values[0]

        # Recalculate driving and residual torques
        self.plate_torques.calculate_driving_torque(_ages, _cases)
        self.plate_torques.calculate_residual_torque(_ages, _cases)

    def remove_net_rotation(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
        ):
        """
        Remove net rotation 
        
        NOTE: This function is not yet implemented.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through plates
        for _age in _tqdm(_ages, desc="Removing net lithospheric rotation"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                # Select data
                _plate_data = self.plates.data[_age][_case].copy()
                _point_data = self.points.data[_age][_case].copy()

                # Filter plates
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs)]
                    _point_data = _point_data[_point_data["plateID"].isin(_plateIDs)]

                # Remove net rotation
                computed_data = utils_calc.compute_no_net_rotation(_plate_data, _point_data)

                # Feed data back into the object
                self.plates.data[_age][_case] = computed_data

        # Recalculate new velocities
        self.points.calculate_velocities(_ages, _cases, self.plates.data)
        self.slabs.calculate_velocities(_ages, _cases, self.plates.data)
        self.plate_torques.calculate_rms_velocity(_ages, _cases)

    def remove_net_trench_migration(
            self,
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            type: str = None
        ):
        """
        Remove net rotation 
        
        NOTE: This function is not yet implemented.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through plates
        for _age in _tqdm(_ages, desc="Removing net trench migration"):
            for _case in _cases:
                # Select plateIDs
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                # Select data
                _plate_data = self.plates.data[_age][_case].copy()
                _slab_data = self.slabs.data[_age][_case].copy()

                # Filter plates
                if plateIDs is not None:
                    _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs)]
                    _slab_data = _slab_data[_slab_data["plateID"].isin(_plateIDs)]

                # Remove net rotation
                computed_data = utils_calc.compute_no_net_trench_migration(
                    _plate_data,
                    _slab_data,
                    self.options[_case],
                    type
                )

                # Feed data back into the object
                self.plates.data[_age][_case] = computed_data

        # Recalculate new velocities
        self.points.calculate_velocities(_ages, _cases, self.plates.data)
        self.slabs.calculate_velocities(_ages, _cases, self.plates.data)
        self.plate_torques.calculate_rms_velocity(_ages, _cases)

    def extract_data_through_time(
            self,
            ages = None,
            cases = None,
            plateIDs = None,
            type = "plates",
            var = "residual_torque_mag",
        ):
        """
        Function to extract data through time.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        """
        return self.plate_torques.extract_data_through_time(ages, cases, plateIDs, type, var)
    
    def apply_parameters(
            self,
            parameters: Dict[str, Union[float, int]],
            ages: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
            cases: Optional[Union[str, List[str]]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]], _numpy.ndarray]] = None,
        ):
        """
        Function to apply parameters to the data.

        :param ages:        ages of interest (default: None)
        :type ages:         float, int, list, numpy.ndarray
        :param cases:       cases of interest (default: None)
        :type cases:        str, list
        :param plateIDs:    plateIDs of interest (default: None)
        :type plateIDs:     int, float, list, numpy.ndarray
        :param parameters:  parameters to apply (default: None)
        :type parameters:   dict
        """
        # Select ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.cases)

        # Loop through ages
        for _age in _ages:
            for _case in _cases:
                # Select plateIDs, if not provided
                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                plate_data = self.plates.data[_age][_case]
                point_data = self.points.data[_age][_case]
                slab_data = self.slabs.data[_age][_case]

                # Apply filters but keep references to original DataFrames
                plate_mask = plate_data["plateID"].isin(_plateIDs)
                point_mask = point_data["plateID"].isin(_plateIDs)
                slab_mask = slab_data["lower_plateID"].isin(_plateIDs)

                for component in ["x", "y", "z", "mag"]:
                    plate_data.loc[plate_mask, f"slab_pull_torque_{component}"] *= (
                        parameters[_case]["Slab pull constant"] / self.settings.options[_case]["Slab pull constant"]
                    )
                    plate_data.loc[plate_mask, f"mantle_drag_torque_{component}"] *= (
                        parameters[_case]["Mantle viscosity"] / self.settings.options[_case]["Mantle viscosity"]
                    )

                for component in ["lat", "lon", "mag"]:
                    plate_data.loc[plate_mask, f"slab_pull_force_{component}"] *= (
                        parameters[_case]["Slab pull constant"] / self.settings.options[_case]["Slab pull constant"]
                    )
                    slab_data.loc[slab_mask, f"slab_pull_force_{component}"] *= (
                        parameters[_case]["Slab pull constant"] / self.settings.options[_case]["Slab pull constant"]
                    )

                    plate_data.loc[plate_mask, f"mantle_drag_force_{component}"] *= (
                        parameters[_case]["Mantle viscosity"] / self.settings.options[_case]["Mantle viscosity"]
                    )
                    point_data.loc[point_mask, f"mantle_drag_force_{component}"] *= (
                        parameters[_case]["Mantle viscosity"] / self.settings.options[_case]["Mantle viscosity"]
                    )

        for _case in _cases:
            # Set the new parameters
            self.settings.options[_case]["Slab pull constant"] = parameters[_case]["Slab pull constant"]
            self.settings.options[_case]["Mantle viscosity"] = parameters[_case]["Mantle viscosity"]

        # Recalculate torques
        self.plate_torques.calculate_driving_torque(_ages, _cases)
        self.plate_torques.calculate_residual_torque(_ages, _cases)