# IMPORT MODULES
# Standard library imports
import os
import logging
import sys
import traceback

# Third-party imports
import cmcrameri as cmc
import matplotlib.pyplot as plt
import numpy as np
import gplately
import pandas as pd

# Local application imports
# Add the path to the plato directory
new_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Check if the path is successfully added
if new_path not in sys.path:
    sys.path.append(new_path)
    if new_path not in sys.path:
        raise RuntimeError("Error: Path not added")

from plato.settings import Settings
from plato.plates import Plates
from plato.points import Points
from plato.slabs import Slabs
from plato.grids import Grids
from plato.plate_torques import PlateTorques
from plato.globe import Globe
from plato.optimisation import Optimisation
from plato.plot import Plot

def test_settings(
        settings_file=None,
        test_ages=[0]
    ):
    """Test the settings module of the plato package."""
    logging.info("Testing settings module...")
    
    # Test initialisation of the Settings object
    try:
        settings_test = Settings(
            name="test",
            ages=test_ages,
            cases_file=settings_file,
            files_dir="output",
            PARALLEL_MODE=False,
            DEBUG_MODE=False,
        )
        logging.info("Successfully initialised 'Settings' object.")

    except Exception as e:
        logging.error(f"Settings test failed: {e}")
        traceback.print_exc()

    logging.info("Successfully tested settings module.")

    return settings_test

def test_plates(
        settings=None, 
        settings_file=None, 
        test_ages=[0], 
        reconstruction_files=None, 
        reconstruction_name="Muller2016", 
        test_functions=True
    ):
    """Test the plates module of the plato package."""
    logging.info("Testing plates module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])
    else:
        reconstruction = None

    # Test initialisation of Plates object
    try:
        plates_test = Plates(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            files_dir="output",
        )
        logging.info("Successfully initialised 'Plates' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Plates' object: {e}")
        traceback.print_exc()

        # Set plates_test to None if an error occurs
        plates_test = None

    # Test various functions of the Plates class
    if plates_test is not None and test_functions:
        # # Test saving
        # try:
        #     plates_test.save(
        #         ages=test_ages,
        #     )
        #     logging.info(f"Successfully saved 'Plates' object!")

        # except Exception as e:
        #     logging.error(f"An error occurred during saving of the 'Plates' object: {e}")
        #     traceback.print_exc()

        # # Test exporting
        # try:
        #     plates_test.export(
        #         ages=test_ages,
        #     )
        #     logging.info(f"Successfully exported 'Plates' object!")

        # except Exception as e:
        #     logging.error(f"An error occurred during exporting of the 'Plates' object: {e}")
        #     traceback.print_exc()

        # # Test calculation of RMS plate velocities
        # try:
        #     plates_test.calculate_rms_velocity()
        #     logging.info("Successfully calculated RMS velocities.")

        # except Exception as e:
        #     logging.error(f"An error occurred during RMS velocity calculation: {e}")
        #     traceback.print_exc()

        # # Test calculation of plate torques
        # try:
        #     plates_test.calculate_torque_on_plates()
        #     logging.info("Successfully calculated plate torques.")

        # except Exception as e:
        #     logging.error(f"An error occurred during plate torque calculation: {e}")
        #     traceback.print_exc()

        # # Test calculation of plate driving torques
        # try:
        #     plates_test.calculate_driving_torques()
        #     logging.info("Successfully calculated driving torques.")

        # except Exception as e:
        #     logging.error(f"An error occurred during driving torque calculation: {e}")
        #     traceback.print_exc()

        # # Test calculation of plate residual torques
        # try:
        #     plates_test.calculate_residual_torques()
        #     logging.info("Successfully calculated residual torques.")

        # except Exception as e:
        #     print(f"An error occurred during residual torque calculation: {e}")
        #     traceback.print_exc()
        
        # Test optimisation of plate torques (this should probably moved to the optimisation module)
        # try:
        #     plates_test.optimise_torques()
        #     if print_results:
        #         print(f"Optimised torques: {plates_test.optimised_torques}")
        # except Exception as e:
        #     print(f"An error occurred during torque optimisation: {e}")
        #     traceback.print_exc()

        # Test extraction of data through time
        try:
            data = plates_test.extract_data_through_time()
            logging.info("Successfully extracted data through time.")

            print(data[list(data.keys())[0]])

        except Exception as e:
            logging.error(f"An error occurred during testing of the 'extract_data' function: {e}")
            traceback.print_exc()

        logging.info("Successfully completed plates test.")
        
        return plates_test

def test_points(
        settings=None,
        settings_file=None,
        test_ages=[0],
        reconstruction_files=None,
        reconstruction_name="Muller2016",
        seafloor_grid=None,
        test_functions=False,
    ):
    """Test the points module of the plato package."""
    logging.info("Testing 'points' module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])
    else:
        reconstruction = None

    # Test initialisation of Points object
    try:
        points_test = Points(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            files_dir="output",
        )
        logging.info("Successfully initialised 'Points' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Points' object: {e}")
        traceback.print_exc()

        # Set plates_test to None if an error occurs
        points_test = None

    # Test functions of the Points class
    if points_test is not None and test_functions:
        # # Test saving
        # try:
        #     points_test.save(
        #         ages=test_ages,
        #     )
        #     logging.info(f"Successfully saved 'Points' object!")

        # except Exception as e:
        #     logging.error(f"An error occurred during saving of the 'Points' object: {e}")
        #     traceback.print_exc()

        # # Test exporting
        # try:
        #     points_test.export(
        #         ages=test_ages,
        #     )
        #     logging.info(f"Successfully exported 'Points' object!")

        # except Exception as e:
        #     logging.error(f"An error occurred during exporting of the 'Points' object: {e}")
        #     traceback.print_exc()

        # # Test sampling of seafloor age grid at points
        # if seafloor_grid:
        #     try:
        #         points_test.sample_points(seafloor_grid = seafloor_grid)

        #     except Exception as e:
        #         logging.error(f"An error occurred during testing of the 'sample_points' function: {e}")
        #         traceback.print_exc()
        # else:
        #     logging.info("No seafloor grid provided for sampling. Testing of 'sample_points()' function skipped.")

        # # Test computation of GPE force
        # try:
        #     points_test.compute_gpe_force()

        # except Exception as e:
        #     logging.error(f"An error occurred during testing of the 'compute_gpe_force' function: {e}")
        #     traceback.print_exc()

        # # Test computation of mantle drag force
        # try:
        #     points_test.compute_mantle_drag_force()
        #     logging.info("Successfully computed mantle drag force.")

        # except Exception as e:
        #     print(f"An error occurred during testing of the 'compute_mantle_drag_force' function: {e}")
        #     traceback.print_exc()

        # Test extraction of data through time
        try:
            points_test.extract_data()
            logging.info("Successfully extracted data through time.")

        except Exception as e:
            logging.error(f"An error occurred during testing of the 'extract_data' function: {e}")
            traceback.print_exc()

        logging.info("Testing of the 'points' module complete.")

    return points_test

def test_slabs(
        settings=None,
        settings_file=None,
        test_ages=[0],
        reconstruction_files=None,
        reconstruction_name="Muller2016",
        seafloor_grid=None,
        test_functions=False,
    ):
    """Test the slabs module of the plato package."""
    logging.info("Testing 'slabs' module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])
    else:
        reconstruction = None

    # Test initialisation of Points object
    try:
        slabs_test = Slabs(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            files_dir="output",
        )
        logging.info("Successfully initialised 'Slabs' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Slabs' object: {e}")
        traceback.print_exc()

        # Set slabs_test to None if an error occurs
        slabs_test = None

    # Test functions of the Slabs object
    # if slabs_test is not None and test_functions:
    #     # Test sampling of seafloor age grid at slabs and upper plate
    #     if seafloor_grid:
    #         try:
    #             slabs_test.sample_slabs(seafloor_grid = seafloor_grid)
    #         except Exception as e:
    #             print(f"An error occurred during 'sample_slabs' function: {e}")
    #             traceback.print_exc()
    #         try:
    #             slabs_test.sample_upper_plates(seafloor_grid = seafloor_grid)
    #         except Exception as e:
    #             print(f"An error occurred during 'sample_upper_plates' function: {e}")
    #             traceback.print_exc()
    #     else:
    #         print("No seafloor grid provided for sampling.")

    #     # Test computation of slab pull force
    #     try:
    #         slabs_test.compute_slab_pull_force()

    #     except Exception as e:
    #         print(f"An error occurred during testing the 'compute_slab_pull_force' function: {e}")
    #         traceback.print_exc()

    #     # Test computation of slab bend force
    #     try:
    #         slabs_test.compute_slab_bend_force()

    #     except Exception as e:
    #         print(f"An error occurred during testing the 'compute_slab_bend_force' function: {e}")
    #         traceback.print_exc()

    # if print_results:
    #     print("Testing of the 'slabs' module complete.")

    return slabs_test

def test_grids(
        settings=None,
        settings_file=None,
        reconstruction_name="Muller2016",
        reconstruction_files=None,
        test_ages=[0],
        seafloor_age_grids=None,
        sediment_grids=None,
        continental_grids=None,
        velocity_grids=None,
        point_data_var=None,
        test_functions=False,
    ):
    """Test the grids module of the plato package."""
    logging.info("Testing 'grids' module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1], reconstruction_files[2])
    else:
        reconstruction = None

    # Test initialisation of Grids object
    try:
        grids_test = Grids(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            seafloor_age_grids=seafloor_age_grids,
            sediment_grids=sediment_grids,
            files_dir="output",
        )
        logging.info("Successfully initialised 'Grids' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Slabs' object: {e}")
        traceback.print_exc()

        # Set grids_test to None if an error occurs
        grids_test = None

    # Test functions of the Grids object
    # TODO: Actually implement these functions in such a way that they can be tested
    if grids_test is not None and test_functions:
        # Test saving
        try:
            grids_test.save_all(
                ages=test_ages,
            )
        
        except Exception as e:
            logging.error(f"An error occurred during saving of the 'Grids' object: {e}")
            traceback.print_exc()

    #     # Test making an xarray dataset from an a series of xarray data arrays
    #     try:
    #         grids_test.data_arrays2dataset(point_data_var=point_data_var)

    #     except Exception as e:
    #         print(f"An error occurred during testing of the 'data_arrays2dataset' function: {e}")
    #         traceback.print_exc()

    #     # Test interpolation of data to the resolution of the seafloor grid
    #     try:
    #         grids_test.array2data_array()

    #     except Exception as e:
    #         print(f"An error occurred during testing of the 'array2data_array' function: {e}")
    #         traceback.print_exc()

    logging.info("Testing of the 'grids' module complete.")

    return grids_test

def test_globe(
        settings=None,
        settings_file=None,
        reconstruction_name="Muller2016",
        reconstruction_files=None,
        test_ages=[0],
        test_functions=False,
    ):
    """Test the globe module of the plato package."""
    logging.info("Testing 'globe' module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])
    else:
        reconstruction = None

    # Test initialisation of Globe object
    try:
        globe_test = Globe(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            files_dir="output",
        )
        logging.info("Successfully initialised 'Globe' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'Globe' object: {e}")
        traceback.print_exc()

        # Set grids_test to None if an error occurs
        globe_test = None

    if globe_test is not None and test_functions:
        # Test saving
        try:
            globe_test.save()
            logging.info(f"Successfully saved 'Globe' object!")

        except Exception as e:
            logging.error(f"An error occurred during saving of the 'Globe' object: {e}")
            traceback.print_exc()

        # Test exporting
        try:
            globe_test.export()
            logging.info(f"Successfully exported 'Globe' object!")

        except Exception as e:
            logging.error(f"An error occurred during exporting of the 'Globe' object: {e}")
            traceback.print_exc()

        # Print some results to check
        print(globe_test.data["test"])

    logging.info("Testing of the 'globe' module complete.")

    return globe_test

def test_plate_torques(
        settings=None,
        settings_file=None,
        reconstruction_name="Muller2016",
        reconstruction_files=None,
        test_ages=[0],
        seafloor_age_grids=None,
        sediment_grids=None,
        continental_grids=None,
        velocity_grids=None,
        test_functions=False,
    ):
    """Test the globe module of the plato package."""
    logging.info("Testing 'plate_torques' module...")

    # Make a PlateReconstruction object if files are provided
    if reconstruction_files:
        reconstruction = gplately.PlateReconstruction(reconstruction_files[0], reconstruction_files[1])
    else:
        reconstruction = None

    # Test initialisation of Globe object
    try:
        plate_torques_test = PlateTorques(
            settings=settings,
            ages=test_ages, 
            cases_file=settings_file,
            reconstruction=reconstruction,
            reconstruction_name=reconstruction_name,
            files_dir="output",
            seafloor_age_grids=seafloor_age_grids,
            sediment_grids=sediment_grids,
            continental_grids=continental_grids,
            velocity_grids=velocity_grids,
        )
        logging.info("Successfully initialised 'PlateTorques' object.")

    except Exception as e:
        logging.error(f"An error occurred during initialisation of the 'PlateTorques' object: {e}")
        traceback.print_exc()

        # Set grids_test to None if an error occurs
        plate_torques_test = None

    if plate_torques_test is not None and test_functions:
        # Test calculation of RMS velocity
        try:
            plate_torques_test.calculate_rms_velocity(
                ages=test_ages
            )
            logging.info(f"Successfully calculated RMS velocities!")

        except Exception as e:
            logging.error(f"An error occurred during calculation of the RMS velocity in the 'PlateTorques' object: {e}")
            traceback.print_exc()

        # Test calculation of net lithospheric rotation
        try:
            plate_torques_test.calculate_net_rotation(
                ages=test_ages
            )
            logging.info(f"Successfully calculated net lithospheric rotation!")

        except Exception as e:
            logging.error(f"An error occurred during calculation of the net lithospheric rotation in the 'PlateTorques' object: {e}")
            traceback.print_exc()

        # Test sampling of seafloor age grid at points
        try:
            plate_torques_test.sample_point_seafloor_ages()
            logging.info("Successfully sampled seafloor age grid at points.")

        except Exception as e:
            logging.error(f"An error occurred during sampling of the seafloor age grid at points: {e}")
            traceback.print_exc()

        # Test sampling of seafloor age grid at slabs
        try:
            plate_torques_test.sample_slab_seafloor_ages()
            logging.info("Successfully sampled seafloor age grid at slabs.")

        except Exception as e:
            logging.error(f"An error occurred during sampling of the seafloor age grid at slabs: {e}")
            traceback.print_exc()

        # Test sampling of seafloor age grid at slabs
        try:
            plate_torques_test.sample_arc_seafloor_ages()
            logging.info("Successfully sampled seafloor age grid at slabs.")

        except Exception as e:
            logging.error(f"An error occurred during sampling of the seafloor age grid at slabs: {e}")
            traceback.print_exc()

        # Test sampling of sediment thickness grid at slabs
        try:
            plate_torques_test.sample_slab_sediment_thicknesses()
            logging.info("Successfully sampled sediment thickness grid at slabs.")

        except Exception as e:
            logging.error(f"An error occurred during sampling of the sediment thickness grid at slabs: {e}")
            traceback.print_exc()

        # Test computation of GPE torque
        try:
            plate_torques_test.calculate_gpe_torque()
            logging.info("Successfully computed GPE torque.")

        except Exception as e:
            logging.error(f"An error occurred during computation of the GPE torque: {e}")
            traceback.print_exc()

        # Test computation of slab pull torque
        try:
            plate_torques_test.calculate_slab_pull_torque()
            logging.info("Successfully computed slab pull torque.")

        except Exception as e:
            logging.error(f"An error occurred during computation of the slab pull torque: {e}")
            traceback.print_exc()

        # Test computation of mantle drag torque
        try:
            plate_torques_test.calculate_mantle_drag_torque()
            logging.info("Successfully computed mantle drag torque.")

        except Exception as e:
            logging.error(f"An error occurred during computation of the mantle drag torque: {e}")
            traceback.print_exc()

        # Test computation of the driving torque
        try:
            plate_torques_test.calculate_driving_torque()
            logging.info("Successfully computed driving torque.")

        except Exception as e:
            logging.error(f"An error occurred during computation of the driving torque: {e}")
            traceback.print_exc()
        
        # Test computation of the residual torque
        try:
            plate_torques_test.calculate_residual_torque()
            logging.info("Successfully computed residual torque.")

        except Exception as e:
            logging.error(f"An error occurred during computation of the residual torque: {e}")
            traceback.print_exc()

        # Test computation of synthetic velocity
        try:
            plate_torques_test.calculate_synthetic_velocity()
            logging.info("Successfully computed synthetic velocity.")

            print("plateID", "slab_pull_torque", "slab_bend_torque", "GPE_torque", "mantle_drag_torque", "driving_torque")
            for i in range(len(plate_torques_test.plates.data[0]["test"].plateID.values)):
                print(
                    plate_torques_test.plates.data[0]["test"].plateID.values[i],
                    plate_torques_test.plates.data[0]["syn"].slab_pull_torque_x.values[i]/plate_torques_test.plates.data[0]["test"].slab_pull_torque_x.values[i],
                    plate_torques_test.plates.data[0]["syn"].slab_bend_torque_x.values[i]/plate_torques_test.plates.data[0]["test"].slab_bend_torque_x.values[i],
                    plate_torques_test.plates.data[0]["syn"].GPE_torque_x.values[i]/plate_torques_test.plates.data[0]["test"].GPE_torque_x.values[i],
                    plate_torques_test.plates.data[0]["syn"].mantle_drag_torque_x.values[i]/plate_torques_test.plates.data[0]["test"].mantle_drag_torque_x.values[i],
                    plate_torques_test.plates.data[0]["syn"].driving_torque_x.values[i]/plate_torques_test.plates.data[0]["test"].driving_torque_x.values[i],
                )
            # print(plate_torques_test.points.data[0]["test"].velocity_mag.median())
            plt.scatter(
                plate_torques_test.points.data[0]["test"].lon,
                plate_torques_test.points.data[0]["test"].lat,
                c=plate_torques_test.points.data[0]["test"].velocity_mag,
                marker = "o",
                cmap = "cmc.bilbao_r",
                vmin = 0,
                vmax = 25,
            )
            plt.colorbar()
            plt.show()

            plt.scatter(
                plate_torques_test.points.data[0]["syn"].lon,
                plate_torques_test.points.data[0]["syn"].lat,
                c=plate_torques_test.points.data[0]["syn"].velocity_mag,
                marker = "o",
                cmap = "cmc.bilbao_r",
                vmin = 0,
                vmax = 25,
            )
            plt.colorbar()
            plt.show()

            plt.scatter(
                plate_torques_test.points.data[0]["test"].lon,
                plate_torques_test.points.data[0]["test"].lat,
                c=plate_torques_test.points.data[0]["test"].spin_rate_mag,
                marker = "o",
                cmap = "cmc.vik",
                vmin = -1e-4,
                vmax = 1e-4,
            )
            plt.colorbar()
            plt.show()

            plt.scatter(
                plate_torques_test.points.data[0]["syn"].lon,
                plate_torques_test.points.data[0]["syn"].lat,
                c=plate_torques_test.points.data[0]["syn"].spin_rate_mag,
                marker = "o",
                cmap = "cmc.vik",
                vmin = -1e-4,
                vmax = 1e-4,
            )
            plt.colorbar()
            plt.show()

            # for i in range(len(plate_torques_test.plates.data[0]["syn"].plateID.values)):
            #     print(
            #         plate_torques_test.plates.data[0]["syn"].plateID.values[i], 
            #         plate_torques_test.plates.data[0]["syn"].centroid_velocity_mag.values[i]/plate_torques_test.plates.data[0]["test"].centroid_velocity_mag.values[i], 
            #     )

            # plt.scatter(
            #     plate_torques_test.plates.data[0]["syn"].area.values,
            #     plate_torques_test.plates.data[0]["syn"].centroid_velocity_mag.values/plate_torques_test.plates.data[0]["test"].centroid_velocity_mag.values, 
            # )
            # plt.xlabel("Area")
            # plt.ylabel("Centroid velocity magnitude ratio")
            # plt.show()

            # plt.scatter(
            #     plate_torques_test.plates.data[0]["test"].pole_lon.values,
            #     plate_torques_test.plates.data[0]["test"].pole_lat.values,
            #     c=plate_torques_test.plates.data[0]["test"].pole_angle,
            #     marker = "o",
            # )
            # plt.scatter(
            #     plate_torques_test.plates.data[0]["syn"].pole_lon.values,
            #     plate_torques_test.plates.data[0]["syn"].pole_lat.values,
            #     c=plate_torques_test.plates.data[0]["syn"].pole_angle,
            #     marker = "d",
            # )
            # plt.colorbar()
            # plt.show()

        except Exception as e:
            logging.error(f"An error occurred during computation of synthetic velocity: {e}")
            traceback.print_exc()

        # Test saving
        try:
            plate_torques_test.save_all(
                ages=test_ages,
            )
            logging.info(f"Successfully saved 'PlateTorques' object!")

        except Exception as e:
            logging.error(f"An error occurred during saving of the 'PlateTorques' object: {e}")
            traceback.print_exc()

        # Test exporting
        try:
            plate_torques_test.export_all(
                ages=test_ages,
            )
            logging.info(f"Successfully exported 'Points' object!")

        except Exception as e:
            logging.error(f"An error occurred during exporting of the 'PlateTorques' object: {e}")
            traceback.print_exc()
            
    logging.info("Testing of the 'plate_torques' module complete.")

def plot_test(plate_torques=None, print_results=False):
    """Test the plot module of the plato package."""
    if print_results:
        print("Testing plot module...")
    
    # Test initialisation of Plot object
    try:
        if plate_torques:
            plot_test = Plot(plate_torques)
            print(f"Plot test complete.")
        else:
            print("No PlateTorques object provided for plotting.")

    except Exception as e:
        print(f"An error occurred during plot testing: {e}")
        traceback.print_exc()

    # Test several plotting functions of the Plot object
    try:
        pass
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        traceback.print_exc()

    return plot_test