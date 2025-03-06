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
from . import utils_calc

def minimise_residual_torque_for_plate(
        _plate_data: _pandas.DataFrame,
        _slab_data: _pandas.DataFrame,
        _plateID: int,
        vmin: float = -14.0,
        vmax: float = -6.0,
        step: float = .25,
        constants = dict,
        NUM_ITERATIONS: int = 100,
        PLOT: bool = False,
    ) -> Union[None, Dict]:
    # Select the plate and slab data for the given plateID
    _plate_data = _plate_data[_plate_data["plateID"] == _plateID].copy()
    _slab_data = _slab_data[_slab_data["lower_plateID"] == _plateID].copy()

    if _slab_data.empty:
        return _slab_data.slab_pull_constant
    
    # Get the slab pull force along subduction zones
    slab_pull_force_lat = _slab_data["slab_pull_force_lat"].values
    slab_pull_force_lon = _slab_data["slab_pull_force_lon"].values
    slab_pull_force_mag = _slab_data["slab_pull_force_mag"].values

    # Get the maximum slab pull force magnitude
    max_slab_pull_force_mag = slab_pull_force_mag / (_slab_data["slab_pull_constant"].values * 2)

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
        prefactors = _numpy.arange(vmin, vmax, step)

        # Store existing values and scores
        existing_values = [0]; existing_scores = [_normalised_residual_torque[0]]

        # Loop through constants
        for i in _numpy.arange(0, len(prefactors)+10):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Propose initial value
                if i > len(prefactors)-1:
                    # The inversion should start with a more general, grid-based exploration of the parameter space
                    # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                    prefactor = utils_calc.propose_value(existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                else:
                    prefactor = 10**prefactors[i]
                                                        
                existing_values.append(prefactor)

                # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                # This step should be performed in Cartesian coordinates.
                _iter_slab_pull_force_mag = _slab_data.slab_pull_force_mag - (
                    residual_force_lat * slab_pull_force_lat + \
                    residual_force_lon * slab_pull_force_lon
                ) * prefactor

                # Ensure the slab pull force magnitude is positive, not larger than the original slab pull force magnitude and that NaN values are replaced with 0
                _iter_slab_pull_force_mag = _numpy.where(_iter_slab_pull_force_mag < 0, 0, _iter_slab_pull_force_mag)
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
                    constants,
                    torque_var = "slab_pull",
                )

                # Calculate driving and residual torque
                _iter_torques = utils_calc.sum_torque(_iter_torques, "driving", constants)
                _iter_torques = utils_calc.sum_torque(_iter_torques, "residual", constants)

                # Extract the driving and residual torques
                _iter_driving_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["driving_torque_mag"].values[0]
                _iter_residual_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["residual_torque_mag"].values[0]

                # Calculate normalised residual torque and append to list
                normalised_residual_torque = _numpy.log10(_iter_residual_torque / _iter_driving_torque)
                existing_scores.append(normalised_residual_torque)

        assert len(existing_values) == len(existing_scores)

        # Find the minimum normalised residual torque and the corresponding constant
        # NOTE: This sometimes threw and error, so it used to be wrapped in a try-except block
        try:
            existing_scores = _numpy.nan_to_num(existing_scores, nan=_numpy.inf)
            opt_index = _numpy.argmin(_numpy.asarray(existing_scores))
            _normalised_residual_torque.append(_numpy.asarray(existing_scores)[opt_index])
            opt_constant = _numpy.asarray(existing_values)[opt_index]
        except:
            opt_index = 0
            _normalised_residual_torque.append(_numpy.nan)
            opt_constant = _numpy.nan

        if PLOT == True or PLOT == _plateID:
            # Plot the optimisation process
            fig, axes = plt.subplots(1, 2)
            axes[0].plot(existing_scores)
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Score")
            axes[0].set_ylim(-10, 1)
            axes[0].scatter(opt_index, _numpy.asarray(existing_scores)[opt_index])

            p = axes[1].scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+2))
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
            constants,
            torque_var = "slab_pull",
        )

        # Calculate residual torque
        _residual_torques = utils_calc.sum_torque(_iter_torques, "residual", constants)
        _residual_torque = _residual_torques[_residual_torques["plateID"] == _plateID]

        # Calculate residual forces at slabs
        residual_force_lat, residual_force_lon, _, _ = utils_calc.compute_residual_force(
            _slab_data,
            _residual_torque,
            plateID_col = "lower_plateID",
            weight_col = "trench_segment_length",
        )

        # Store residual torques in dictionaries
        _slab_pull_force_mag[k+1] = _slab_data["slab_pull_force_mag"]

        # Catch any NaN values
        residual_force_lat = _numpy.where(_numpy.isnan(residual_force_lat), 0, residual_force_lat)
        residual_force_lon = _numpy.where(_numpy.isnan(residual_force_lon), 0, residual_force_lon)

        # if k == NUM_ITERATIONS:
    # Find iteration with minimum value
    try:
        opt_iter = _numpy.nanargmin(_numpy.asarray(_normalised_residual_torque))
    except:
        opt_iter = 0

    if PLOT:
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(_normalised_residual_torque)
        axes[0].set_ylim(-10, 1)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Normalised residual torque")
        axes[0].scatter(opt_iter, _normalised_residual_torque[opt_iter], c="k", marker="*")

        axes[1].scatter(
            _slab_data.lon,
            _slab_data.lat,
            c=_slab_pull_force_mag[opt_iter] / (max_slab_pull_force_mag * 2),
        )

    return _slab_pull_force_mag[opt_iter] / (max_slab_pull_force_mag * 2)