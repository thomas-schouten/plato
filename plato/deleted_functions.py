# def minimise_residual_torque_v2(
    #         self,
    #         age = None, 
    #         case = None, 
    #         plateIDs = None, 
    #         grid_size = 500, 
    #         lvv_range = [1, 100],
    #         plot = True,
    #         weight_by_area = True,
    #         minimum_plate_area = None
    #     ):
    #     """
    #     Function to find optimised coefficients to match plate motions using a grid search

    #     :param age:                     reconstruction age to optimise
    #     :type age:                      int, float
    #     :param case:                    case to optimise
    #     :type case:                     str
    #     :param plateIDs:                plate IDs to include in optimisation
    #     :type plateIDs:                 list of integers or None
    #     :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
    #     :type grid_size:                int
    #     :param plot:                    whether or not to plot the grid
    #     :type plot:                     boolean
    #     :param weight_by_area:          whether or not to weight the residual torque by plate area
    #     :type weight_by_area:           boolean

    #     :return:                        The optimal slab pull coefficient, optimal viscosity, normalised residual torque, and indices of the optimal coefficients
    #     :rtype:                         float, float, numpy.ndarray, tuple
    #     """
    #     # Define age and case if not provided
    #     if age is None:
    #         age = self.settings.ages[0]

    #     if case is None:
    #         case = self.settings.cases[0]

    #     # Set range of viscosities
    #     viscs = _numpy.linspace(viscosity_range[0], viscosity_range[1], grid_size)

    #     # Set range of slab pull constants
    #     if self.settings.options[case]["Sediment subduction"]:
    #         # Range is smaller with sediment subduction
    #         sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
    #     else:
    #         sp_consts = _numpy.linspace(1e-5, 1., grid_size)

    #     # Set range of slab suction constants
    #     ss_consts = _numpy.linspace(1e-5, 1., grid_size)

    #     # Set range of lateral viscosity variations
    #     lvv_range = _numpy.linspace(1e-5, 100., grid_size)

    #     # Create grids from ranges of viscosities and slab pull coefficients
    #     visc_grid, lvv_grid, sp_const_grid, ss_cont_grid = _numpy.meshgrid(viscs, lvv_range, sp_consts, ss_consts)
    #     ones_grid = _numpy.ones_like(visc_grid)

    #     # Select plateIDs
    #     _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[age][case].plateID)

    #     # Filter plates
    #     _data = self.plates.data[age][case]


    #     if plateIDs is not None:
    #         _data = _data[_data["plateID"].isin(_plateIDs)]

    #     if minimum_plate_area is not None:
    #         _data = _data[_data["area"] > minimum_plate_area]

    #     # Get total area
    #     total_area = _data["area"].sum()
            
    #     driving_mag = _numpy.zeros_like(sp_const_grid); 
    #     residual_mag = _numpy.zeros_like(sp_const_grid); 

    #     # Get torques
    #     for k, _ in enumerate(_data.plateID):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")

    #             residual_x = _numpy.zeros_like(sp_const_grid); residual_y = _numpy.zeros_like(sp_const_grid); residual_z = _numpy.zeros_like(sp_const_grid)

    #             # Add slab pull torque
    #             if self.settings.options[case]["Slab pull torque"] and "slab_pull_torque_x" in _data.columns:
    #                 residual_x -= _data.slab_pull_torque_x.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
    #                 residual_y -= _data.slab_pull_torque_y.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]
    #                 residual_z -= _data.slab_pull_torque_z.iloc[k] * sp_const_grid / self.settings.options[case]["Slab pull constant"]

    #             # Add slab suction torque
    #             if self.settings.options[case]["Slab suction torque"] and "slab_suction_torque_x" in _data.columns:
    #                 # print("Adding slab suction torque")
    #                 residual_x -= _data.slab_suction_torque_x.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])
    #                 residual_y -= _data.slab_suction_torque_y.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])
    #                 residual_z -= _data.slab_suction_torque_z.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"])

    #                 # print(_data.slab_suction_torque_x.iloc[k] * ss_cont_grid * sp_const_grid / (self.settings.options[case]["Slab suction constant"] * self.settings.options[case]["Slab pull constant"]))

    #             # Add GPE torque
    #             if self.settings.options[case]["GPE torque"] and "GPE_torque_x" in _data.columns:
    #                 residual_x -= _data.GPE_torque_x.iloc[k] * ones_grid
    #                 residual_y -= _data.GPE_torque_y.iloc[k] * ones_grid
    #                 residual_z -= _data.GPE_torque_z.iloc[k] * ones_grid
                
    #             # Compute magnitude of driving torque
    #             if weight_by_area:
    #                 driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
    #             else:
    #                 driving_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]

    #             # Add slab bend torque
    #             if self.settings.options[case]["Slab bend torque"] and "slab_bend_torque_x" in _data.columns:
    #                 residual_x -= _data.slab_bend_torque_x.iloc[k] * ones_grid
    #                 residual_y -= _data.slab_bend_torque_y.iloc[k] * ones_grid
    #                 residual_z -= _data.slab_bend_torque_z.iloc[k] * ones_grid

    #             # Add mantle drag torque
    #             if self.settings.options[case]["Mantle drag torque"] and "mantle_drag_torque_x" in _data.columns:
    #                 if self.settings.options[case]["Continental keels"]:
    #                     # print("Adding mantle drag torque with continental keels")
    #                     # print(1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1))
    #                     residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                     residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                     residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)) / \
    #                         (self.settings.options[case]["Mantle viscosity"] * (1 + lvv_grid * (_data.continental_fraction.iloc[k] - 1)))
    #                 else:
    #                     residual_x -= _data.mantle_drag_torque_x.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
    #                     residual_y -= _data.mantle_drag_torque_y.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]
    #                     residual_z -= _data.mantle_drag_torque_z.iloc[k] * visc_grid / self.settings.options[case]["Mantle viscosity"]

    #             # Compute magnitude of residual
    #             if weight_by_area:
    #                 residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
    #             else:
    #                 residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]
        
    #     # Divide residual by driving torque
    #     residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

    #     # Find the indices of the minimum value directly using _numpy.argmin
    #     opt_indices = _numpy.unravel_index(_numpy.argmin(residual_mag), residual_mag.shape)
    #     opt_visc = visc_grid[opt_indices]; opt_lvv = lvv_grid[opt_indices]
    #     opt_sp_const = sp_const_grid[opt_indices]; opt_ss_cont = ss_cont_grid[opt_indices]

    #     # Assign optimal values to last entry in arrays
    #     # self.opt_sp_const[age][case][-1] = opt_sp_const
    #     # self.opt_visc[age][case][-1] = opt_visc

    #     # Plot
    #     if plot == True:
    #         fig, ax = plt.subplots(figsize=(15, 12))
    #         im = ax.imshow(residual_mag_normalised, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
    #         ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
    #         ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
    #         ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(viscosity_range[0], viscosity_range[1], 5)])
    #         ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
    #         ax.set_xlabel("Mantle viscosity [Pa s]")
    #         ax.set_ylabel("Slab pull reduction factor")
    #         ax.scatter(opt_j, opt_i, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
    #         fig.colorbar(im, label = "Log(residual torque/driving torque)")
    #         plt.show()

    #     # Print results
    #     print(f"Optimal coefficients for ", ", ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
    #     print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
    #     print("Optimum viscosity [Pa s]: {:.2e}".format(opt_visc))
    #     print("Optimum lateral viscosity Variation: {:.2f}".format(opt_lvv))
    #     print("Optimum drag coefficient [Pa s/m]: {:.2e}".format(opt_visc / self.settings.mech.La))
    #     print("Optimum slab pull constant: {:.2%}".format(opt_sp_const))
    #     print("Optimum slab suction constant: {:.2%}".format(opt_ss_cont))

    #     return residual_mag_normalised, (opt_visc, opt_lvv, opt_sp_const, opt_ss_cont), opt_indices

def invert_residual_torque_v6(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -14.0, 
            vmax = -6.0,
            step = .25, 
            NUM_ITERATIONS = 100,
            PLOT = False, 
            PARALLEL_MODE = False,
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

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Obtain plateIDs for all unique combinations of ages and cases
                    _plateIDs = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                    # First reset all the slab pull forces to their original values to avoid overwriting, as this sometimes results in NaN values
                    for _plateID in _plateIDs:
                        self.slabs.data[_age][_case].loc[self.slabs.data[_age][_case]["lower_plateID"] == _plateID, "slab_pull_constant"] = self.settings.options[_case]["Slab pull constant"]

                    # Recalculate slab pull
                    self.slabs.calculate_slab_pull_force(ages=_age, cases=_case, plateIDs=_plateIDs, PROGRESS_BAR=False)

                    # Get data
                    _plate_data = self.plates.data[_age][_case].copy()
                    _slab_data = self.slabs.data[_age][_case].copy()

                    # Initialise entries for each plate ID in dictionaries
                    minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs}
                    opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs}
                    _slab_pull_constants = {_plateID: None for _plateID in _plateIDs}
                    
                    if PARALLEL_MODE and __name__ == "__main__":
                        # Parallelise the loop
                        # NOTE: This doesn't work in Jupyter notebooks
                        with ProcessPoolExecutor() as executor:
                            futures = {
                                executor.submit(
                                    utils_opt.minimise_residual_torque_for_plate, 
                                    _plate_data, 
                                    _slab_data, 
                                    _plateID,
                                    vmin, 
                                    vmax, 
                                    step, 
                                    self.settings.constants, 
                                    NUM_ITERATIONS, 
                                    PLOT
                                ): _plateID for _plateID in _plateIDs
                            }
                            
                            # Collect results in a dictionary to avoid overwriting
                            for future in as_completed(futures):
                                plateID = futures[future]  # Get the corresponding plateID
                                try:
                                    _slab_pull_constants[plateID] = future.result()
                                except Exception as e:
                                    print(f"Error processing plate {plateID}: {e}")

                    else:
                        for _plateID in _plateIDs:
                            _slab_pull_constants[_plateID] = utils_opt.minimise_residual_torque_for_plate(
                                _plate_data, 
                                _slab_data, 
                                _plateID,
                                vmin, 
                                vmax, 
                                step, 
                                self.settings.constants,
                                NUM_ITERATIONS, 
                                PLOT
                            )

                # Assign optimal slab pull constants
                for _plateID in _plateIDs:
                    self.slabs.data[_age][_case].loc[self.slabs.data[_age][_case]["lower_plateID"] == _plateID, "slab_pull_constant"] = _slab_pull_constants[_plateID]

                # Recalculate all the relevant torques
                self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateIDs, PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs, PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs, PROGRESS_BAR=False)            
                
    def invert_residual_torque_v5(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -18.0, 
            vmax = -2.0,
            step = .5, 
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
        # # Raise error if parameter is not slab pull coefficient or mantle viscosity
        # if parameter not in ["Sl", "Mantle viscosity"]:
        #     raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_pull_constants = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_suction_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _plate_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _slab_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                # Make sure that the slab pull and suction constants are non-zero, otherwise the inversion will not work
                # If the constants are zero, set them to an initial guess of .2
                if self.slabs.data[_age][_case]["slab_pull_constant"].mean() == 0:
                    self.slabs.data[_age][_case]["slab_pull_constant"] = .2
                    self.slabs.calculate_slab_pull_force(ages=_age, cases=_case, PROGRESS_BAR=False)
                    
                if self.slabs.data[_age][_case]["slab_suction_constant"].mean() == 0:
                    self.slabs.data[_age][_case]["slab_suction_constant"] = .2
                    self.slabs.calculate_slab_suction_force(ages=_age, cases=_case, PROGRESS_BAR=False)

                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case]["plateID"].values)

                # Initialise entries for each plate ID in dictionaries
                minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_pull_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_suction_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Loop through plates
                for _plateID in _plateIDs[_age][_case]:
                    if _plateID not in self.plates.data[_age][_case]["plateID"].values:
                        continue

                    # Select data
                    _plate_data = self.plates.data[_age][_case].copy()
                    _slab_data = self.slabs.data[_age][_case].copy()
                    _arc_data = self.slabs.data[_age][_case].copy()

                    # Filter data
                    _plate_data = _plate_data[_plate_data["plateID"] == _plateID]
                    _slab_data = _slab_data[_slab_data["lower_plateID"] == _plateID]
                    _arc_data = _arc_data[_arc_data["upper_plateID"] == _plateID]

                    # Skip if no data
                    if _slab_data.empty and _arc_data.empty:
                        continue

                    # Get the slab pull force along subduction zones
                    slab_pull_force_lat = _slab_data["slab_pull_force_lat"].values
                    slab_pull_force_lon = _slab_data["slab_pull_force_lon"].values
                    slab_pull_force_mag = _slab_data["slab_pull_force_mag"].values

                    # Get the slab suction force along subduction zones
                    slab_suction_force_lat = _arc_data["slab_suction_force_lat"].values
                    slab_suction_force_lon = _arc_data["slab_suction_force_lon"].values
                    slab_suction_force_mag = _arc_data["slab_suction_force_mag"].values

                    # Get the maximum slab pull force magnitude
                    max_slab_pull_force_mag = slab_pull_force_mag / (_slab_data["slab_pull_constant"].values * 2)
                    max_slab_suction_force_mag = slab_suction_force_mag / (_arc_data["slab_suction_constant"].values * 2)

                    # Get the residual force for the upper and lower plates along subduction zones
                    slab_residual_force_lat = _slab_data["slab_residual_force_lat"].values; slab_residual_force_lon = _slab_data["slab_residual_force_lon"].values
                    arc_residual_force_lat = _arc_data["arc_residual_force_lat"].values; arc_residual_force_lon = _arc_data["arc_residual_force_lon"].values

                    # Extract the driving and residual torques
                    driving_torque = _plate_data[_plate_data["plateID"] == _plateID]["driving_torque_mag"].values[0]
                    residual_torque = _plate_data[_plate_data["plateID"] == _plateID]["residual_torque_mag"].values[0]

                    # Initialise dictionaries to store all the relevant values values
                    _normalised_residual_torque = [_numpy.log10(residual_torque / driving_torque)]
                    _slab_pull_force_mag, _slab_pull_force_lat, _slab_pull_force_lon = {}, {}, {}
                    _slab_residual_force_lat, _slab_residual_force_lon = {}, {}
                    _slab_suction_force_mag, _slab_suction_force_lat, _slab_suction_force_lon = {}, {}, {}
                    _arc_residual_force_lat, _arc_residual_force_lon = {}, {}

                    # Assign initial values
                    _slab_pull_force_mag[0] = slab_pull_force_mag
                    _slab_pull_force_lat[0] = slab_pull_force_lat
                    _slab_pull_force_lon[0] = slab_pull_force_lon
                    _slab_residual_force_lat[0] = slab_residual_force_lat
                    _slab_residual_force_lon[0] = slab_residual_force_lon

                    _slab_suction_force_mag[0] = slab_suction_force_mag
                    _slab_suction_force_lat[0] = slab_suction_force_lat
                    _slab_suction_force_lon[0] = slab_suction_force_lon
                    _arc_residual_force_lat[0] = arc_residual_force_lat
                    _arc_residual_force_lon[0] = arc_residual_force_lon

                    for k in range(NUM_ITERATIONS):
                        # Define constants for this iteration
                        constants = _numpy.arange(vmin, vmax, step)

                        # Store existing values and scores
                        pull_existing_values = []; suction_existing_values = []
                        existing_scores = []

                        # plt.scatter(
                        #     _slab_data.lon,
                        #     _slab_data.lat,
                        #     c=_slab_data.slab_pull_force_mag,
                        #     # vmin=.1, vmax=.5,
                        # )
                        # plt.xlim(-180, 180); plt.ylim(-90, 90)
                        # plt.suptitle(f"Maximum slab pull force magnitude at iteration {k}")
                        # plt.show()

                        # Loop through constants
                        for i in _numpy.arange(0, len(constants)+10):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")

                                # Propose initial value
                                if i > len(constants)-1:
                                    # The inversion should start with a more general, grid-based exploration of the parameter space
                                    # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                                    pull_constant = utils_calc.propose_value(pull_existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                                    suction_constant = utils_calc.propose_value(suction_existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                                else:
                                    pull_constant = 10**constants[i]
                                    suction_constant = 10**constants[i]
                                                                        
                                pull_existing_values.append(pull_constant); suction_existing_values.append(suction_constant)

                                # Modify the magnitude of the slab pull and suction forces using the 2D dot product of the residual force and the slab pull force and the constant
                                _iter_slab_pull_force_mag = _slab_data.slab_pull_force_mag - (
                                    slab_residual_force_lat * slab_pull_force_lat + \
                                    slab_residual_force_lon * slab_pull_force_lon
                                ) * pull_constant

                                _iter_slab_suction_force_mag = _arc_data.slab_suction_force_mag - (
                                    arc_residual_force_lat * slab_suction_force_lat + \
                                    arc_residual_force_lon * slab_suction_force_lon
                                ) * suction_constant

                                # Ensure the slab pull force magnitude is positive, not larger than the original slab pull force magnitude and that NaN values are replaced with 0
                                _iter_slab_pull_force_mag = _numpy.where(_iter_slab_pull_force_mag < 0, 0, _iter_slab_pull_force_mag)
                                _iter_slab_pull_force_mag = _numpy.where(_numpy.isnan(_iter_slab_pull_force_mag), slab_pull_force_mag, _iter_slab_pull_force_mag)
                                _iter_slab_pull_force_mag = _numpy.where(_iter_slab_pull_force_mag > max_slab_pull_force_mag, max_slab_pull_force_mag, _iter_slab_pull_force_mag)

                                # Ensure the slab suction force magnitude is positive, not larger than the original slab suction force magnitude and that NaN values are replaced with 0
                                _iter_slab_suction_force_mag = _numpy.where(_iter_slab_suction_force_mag < 0, 0, _iter_slab_suction_force_mag)
                                _iter_slab_suction_force_mag = _numpy.where(_numpy.isnan(_iter_slab_suction_force_mag), slab_suction_force_mag, _iter_slab_suction_force_mag)
                                _iter_slab_suction_force_mag = _numpy.where(_iter_slab_suction_force_mag > max_slab_suction_force_mag, max_slab_suction_force_mag, _iter_slab_suction_force_mag)

                                # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                                _iter_slab_pull_force_lat = _numpy.cos(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _iter_slab_pull_force_mag
                                _iter_slab_pull_force_lon = _numpy.sin(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _iter_slab_pull_force_mag

                                # Decompose the slab suction force into latitudinal and longitudinal components using the trench normal azimuth
                                _iter_slab_suction_force_lat = _numpy.cos(_numpy.deg2rad((_arc_data["trench_normal_azimuth"] + 180) % 360)) * _iter_slab_suction_force_mag
                                _iter_slab_suction_force_lon = _numpy.sin(_numpy.deg2rad((_arc_data["trench_normal_azimuth"] + 180) % 360)) * _iter_slab_suction_force_mag
                                
                                # Calculate the torques with the modified slab pull forces
                                _iter_torques = utils_calc.compute_torque_on_plates(
                                    _plate_data,
                                    _slab_data.lat.values,
                                    _slab_data.lon.values,
                                    _slab_data.lower_plateID.values,
                                    _iter_slab_pull_force_lat, 
                                    _iter_slab_pull_force_lon,
                                    _slab_data.trench_segment_length.values,
                                    self.settings.constants,
                                    torque_var = "slab_pull",
                                )

                                # Calculate the torques with the modified slab suction forces
                                _iter_torques = utils_calc.compute_torque_on_plates(
                                    _plate_data,
                                    _arc_data.lat.values,
                                    _arc_data.lon.values,
                                    _arc_data.lower_plateID.values,
                                    _iter_slab_suction_force_lat,
                                    _iter_slab_suction_force_lon,
                                    _arc_data.trench_segment_length.values,
                                    self.settings.constants,
                                    torque_var = "slab_suction",
                                )

                                # Calculate driving and residual torque
                                _iter_torques = utils_calc.sum_torque(_iter_torques, "driving", self.settings.constants)
                                _iter_torques = utils_calc.sum_torque(_iter_torques, "residual", self.settings.constants)

                                # Extract the driving and residual torques
                                _iter_driving_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["driving_torque_mag"].values[0]
                                _iter_residual_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["residual_torque_mag"].values[0]

                                # Calculate normalised residual torque and append to list
                                normalised_residual_torque = _numpy.log10(_iter_residual_torque / _iter_driving_torque)
                                existing_scores.append(normalised_residual_torque)

                        if PLOT == True or PLOT == _plateID:
                            # Plot the optimisation process
                            fig, axes = plt.subplots(1, 2)
                            axes[0].plot(existing_scores)
                            axes[0].set_xlabel("Iteration")
                            axes[0].set_ylabel("Score")
                            axes[0].set_ylim(-20, 1)

                            p = axes[1].scatter(_numpy.log10(_numpy.asarray(pull_existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                            # p2 = axes[1].scatter(_numpy.log10(_numpy.asarray(suction_existing_values)), existing_scores, c=_numpy.arange(0, i+1), marker="D")
                            axes[1].set_ylim(-20, 1)
                            axes[1].set_xlim(vmin, vmax)
                            fig.colorbar(p, label="iteration", orientation="horizontal")
                            axes[1].set_xlabel("Parameter value")
                            axes[1].set_yticklabels([])
                            fig.suptitle(f"Optimisation for plateID {_plateID}")
                            plt.show()
                        
                        # Find the minimum normalised residual torque and the corresponding constant
                        # NOTE: This sometimes threw and error, so it used to be wrapped in a try-except block
                        opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                        _normalised_residual_torque.append(_numpy.nanmin(_numpy.asarray(existing_scores)))
                        opt_pull_constant = _numpy.asarray(pull_existing_values)[opt_index]
                        opt_suction_constant = _numpy.asarray(suction_existing_values)[opt_index]

                        # print(f"Iteration {k+1}: minimum normalised residual torque: {_normalised_residual_torque[k+1]}")
                        
                        # Apply the minimum residual torque to the slab data
                        _slab_data["slab_pull_force_mag"] -= (
                            slab_residual_force_lat * slab_pull_force_lat + \
                            slab_residual_force_lon * slab_pull_force_lon
                        ) * opt_pull_constant

                        _arc_data["slab_suction_force_mag"] -= (
                            arc_residual_force_lat * slab_suction_force_lat + \
                            arc_residual_force_lon * slab_suction_force_lon
                        ) * opt_suction_constant

                        # Ensure the slab pull force magnitude is positive, not larger than the original slab pull force magnitude and that NaN values are replaced with the original slab pull force magnitude
                        _slab_data["slab_pull_force_mag"] = _numpy.where(_slab_data["slab_pull_force_mag"] < 0, 0, _slab_data["slab_pull_force_mag"])
                        _slab_data["slab_pull_force_mag"] = _numpy.where(_slab_data["slab_pull_force_mag"] > max_slab_pull_force_mag, max_slab_pull_force_mag, _slab_data["slab_pull_force_mag"])
                        _slab_data["slab_pull_force_mag"] = _numpy.where(_numpy.isnan(_slab_data["slab_pull_force_mag"]), _slab_data["slab_pull_force_mag"].values, _slab_data["slab_pull_force_mag"])

                        # Ensure the slab suction force magnitude is positive, not larger than the original slab suction force magnitude and that NaN values are replaced with the original slab suction force magnitude
                        _arc_data["slab_suction_force_mag"] = _numpy.where(_arc_data["slab_suction_force_mag"] < 0, 0, _arc_data["slab_suction_force_mag"])
                        _arc_data["slab_suction_force_mag"] = _numpy.where(_arc_data["slab_suction_force_mag"] > max_slab_suction_force_mag, max_slab_suction_force_mag, _arc_data["slab_suction_force_mag"])
                        _arc_data["slab_suction_force_mag"] = _numpy.where(_numpy.isnan(_arc_data["slab_suction_force_mag"]), _arc_data["slab_suction_force_mag"].values, _arc_data["slab_suction_force_mag"])

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        slab_pull_force_lat = _numpy.cos(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _slab_data["slab_pull_force_mag"]
                        slab_pull_force_lon = _numpy.sin(_numpy.deg2rad(_slab_data["trench_normal_azimuth"])) * _slab_data["slab_pull_force_mag"]

                        # Decompose the slab suction force into latitudinal and longitudinal components using the trench normal azimuth
                        slab_suction_force_lat = _numpy.cos(_numpy.deg2rad((_arc_data["trench_normal_azimuth"] + 180) % 360)) * _arc_data["slab_suction_force_mag"]
                        slab_suction_force_lon = _numpy.sin(_numpy.deg2rad((_arc_data["trench_normal_azimuth"] + 180) % 360)) * _arc_data["slab_suction_force_mag"]
                        
                        # Calculate the torques with the modified slab pull forces
                        _iter_torques = utils_calc.compute_torque_on_plates(
                            _plate_data,
                            _slab_data.lat.values,
                            _slab_data.lon.values,
                            _slab_data.lower_plateID.values,
                            slab_pull_force_lat, 
                            slab_pull_force_lon,
                            _slab_data.trench_segment_length.values,
                            self.settings.constants,
                            torque_var = "slab_pull",
                        )

                        _iter_torques = utils_calc.compute_torque_on_plates(
                            _plate_data,
                            _arc_data.lat.values,
                            _arc_data.lon.values,
                            _arc_data.lower_plateID.values,
                            slab_suction_force_lat, 
                            slab_suction_force_lon,
                            _arc_data.trench_segment_length.values,
                            self.settings.constants,
                            torque_var = "slab_suction",
                        )

                        # Calculate residual torque
                        _residual_torques = utils_calc.sum_torque(_iter_torques, "residual", self.settings.constants)
                        _residual_torque = _residual_torques[_residual_torques["plateID"] == _plateID]

                        print("New slab suction torque: ", _residual_torque["slab_suction_torque_mag"].values[0])
                        print("New residual torque: ", _residual_torque["residual_torque_mag"].values[0])

                        # print("New residual torque: ", _residual_torque["residual_torque_mag"].values[0])

                        # Calculate residual forces at slabs
                        slab_residual_force_lat, slab_residual_force_lon, _, _ = utils_calc.compute_residual_force(
                            _slab_data,
                            _residual_torque,
                            plateID_col = "lower_plateID",
                            weight_col = "trench_segment_length",
                        )

                        arc_residual_force_lat, arc_residual_force_lon, _, _ = utils_calc.compute_residual_force(
                            _arc_data,
                            _residual_torque,
                            plateID_col = "upper_plateID",
                            weight_col = "trench_segment_length",
                        )

                        # print(arc_residual_force_lat, arc_residual_force_lon)

                        # Store residual torques in dictionaries
                        _slab_pull_force_mag[k+1] = _slab_data["slab_pull_force_mag"]
                        _slab_suction_force_mag[k+1] = _arc_data["slab_suction_force_mag"]
                        # _slab_pull_force_lat[k+1] = slab_pull_force_lat
                        # _slab_pull_force_lon[k+1] = slab_pull_force_lon
                        # _residual_force_lat[k+1] = residual_force_lat
                        # _residual_force_lon[k+1] = residual_force_lon

                        # Catch any NaN values
                        slab_residual_force_lat = _numpy.where(_numpy.isnan(slab_residual_force_lat), 0, slab_residual_force_lat)
                        slab_residual_force_lon = _numpy.where(_numpy.isnan(slab_residual_force_lon), 0, slab_residual_force_lon)
                        arc_residual_force_lat = _numpy.where(_numpy.isnan(arc_residual_force_lat), 0, arc_residual_force_lat)
                        arc_residual_force_lon = _numpy.where(_numpy.isnan(arc_residual_force_lon), 0, arc_residual_force_lon)

                        # Check if there are NaN values in the slab pull force magnitude
                        # if _numpy.isnan(slab_pull_force_mag).any():
                            # print(f"{len(_numpy.isnan(slab_pull_force_mag))} NaN values in slab pull force magnitude")

                        # Assign the optimal values to the slab data
                        # if k < NUM_ITERATIONS-1:
                            # print("Starting new iteration...")

                        if k == NUM_ITERATIONS-1:
                            # Find iteration with minimum value
                            opt_iter = _numpy.nanargmin(_numpy.asarray(_normalised_residual_torque))

                            # print(f"Optimal iteration: {opt_iter}")
                            # print(f"Minimum normalised residual torque: {_normalised_residual_torque[opt_iter]}")
                            # print(f"Minimum residual, driving torque: {_residual_torque['residual_torque_mag'].values[0]}, {_iter_torques['driving_torque_mag'].values[0]}")

                            # Feed optimal values back into slab data
                            # self.slabs.data[_age][_case].loc[_slab_data.index, "slab_pull_force_mag"] = _slab_pull_force_mag[opt_iter]
                            # self.slabs.data[_age][_case].loc[_slab_data.index, "slab_pull_force_lat"] = _slab_pull_force_lat[opt_iter]
                            # self.slabs.data[_age][_case].loc[_slab_data.index, "slab_pull_force_lon"] = _slab_pull_force_lon[opt_iter]

                            # self.slabs.data[_age][_case].loc[_slab_data.index, "residual_force_lat"] = _residual_force_lat[opt_iter]
                            # self.slabs.data[_age][_case].loc[_slab_data.index, "residual_force_lon"] = _residual_force_lon[opt_iter]

                            self.slabs.data[_age][_case].loc[_slab_data.index, "slab_pull_constant"] = (
                                _slab_pull_force_mag[opt_iter] / (max_slab_pull_force_mag * 2)
                            )

                            self.slabs.data[_age][_case].loc[_arc_data.index, "slab_suction_constant"] = (
                                _slab_suction_force_mag[opt_iter] / (max_slab_suction_force_mag * 2)
                            )
                            
                            # plt.plot(_normalised_residual_torque)
                            # plt.scatter(opt_iter, _normalised_residual_torque[opt_iter], color="red")
                            # plt.show()

                            # plt.scatter(
                            #     _slab_data.lon,
                            #     _slab_data.lat,
                            #     c=_slab_pull_force_mag[opt_iter] / (max_slab_pull_force_mag * 2),
                            # )
                            # plt.xlim(-180, 180); plt.ylim(-90, 90)
                            # plt.colorbar()
                            # plt.show()

                            # Recalculate all the relevant torques
                            # self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateID, torque_var="slab_pull", PROGRESS_BAR=False)
                            self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_slab_suction_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

def invert_residual_torque_v0(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -13.0, 
            vmax = -7.0,
            step = .1, 
            plot = False, 
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
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        
        :return:                        The optimal slab pull coefficient
        :rtype:                         float

        NOTE: This function should be optimised in several places.
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        driving_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        residual_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                driving_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                residual_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Loop through constants
                # for i, constant in enumerate(constants):
                for _plateID in _plateIDs[_age][_case]:
                    # Extract rotation pole
                    rotation_pole_lat = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lat")
                    rotation_pole_lon = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lon")
                    rotation_pole_mag = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_angle")
                    
                    # Convert rotation pole to Cartesian coordinates
                    rotation_pole_x, rotation_pole_y, rotation_pole_z = utils_calc.geocentric_spherical2cartesian(
                        rotation_pole_lat[_plateID].values[0], 
                        rotation_pole_lon[_plateID].values[0],
                        rotation_pole_mag[_plateID].values[0]
                    )

                    existing_values = []; existing_scores = []; dot_products = []
                    # for i in _numpy.arange(0, 1e2):
                    for i, constant in enumerate(constants):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # # Propose initial value
                            # if i == 0:
                            #     # The inversion should start with a more general, grid-based exploration of the parameter space
                            #     # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                            #     if j !=0 and _plateID in opt_constants[_ages[j-1]][_case]:
                            #         constant = opt_constants[_ages[j-1]][_case][_plateID]
                            #     else:
                            #         constant = 1e-10

                            # else:
                            #     constant = utils_calc.propose_value(existing_values, existing_scores, 0.3)


                            constant = 10**constant
                                                                    
                            existing_values.append(constant)

                            # Inform the user which constant is being optimised
                            # logging.info(f"Optimising for {constant}")

                            # Calculate the torques the normal way
                            self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                            # Select data
                            _data[_age][_case] = self.slabs.data[_age][_case].copy()

                            # Filter data, if necessary
                            if plateIDs is not None:
                                _data[_age][_case] = _data[_age][_case][_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                            # Skip if no data
                            if _data[_age][_case].empty:
                                continue

                            # Get the slab pull force magnitude
                            max_slab_pull_force_mag = _data[_age][_case]["slab_pull_force_mag"] / _data[_age][_case]["slab_pull_constant"]

                            # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                            # This step should be performed in Cartesian coordinates.
                            _data[_age][_case]["slab_pull_force_mag"] -= (
                                _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                                _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                            ) * constant

                            # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                            _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                            _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                            # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                            _data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                            _data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                        
                            # Calculate the torques with the modified slab pull forces
                            self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateID, torque_var="slab_pull", PROGRESS_BAR=False)

                            # This should be rewritten, as calclculating the residual vecter in spherical coordinates is time-consuming
                            # Instead, the dot product should be calculated in Cartesian coordinates
                            self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                            # Extract the driving and residual torques
                            _iter_driving_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_mag")
                            _iter_residual_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var= "residual_torque_mag")

                            # Extract driving torque
                            _iter_driving_torque_x = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_x")
                            _iter_driving_torque_y = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_y")
                            _iter_driving_torque_z = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_z")

                            # Calculate dot product 
                            normalised_dot_product = _numpy.abs(
                                    _iter_driving_torque_x[_plateID].values[0] * rotation_pole_x + \
                                    _iter_driving_torque_y[_plateID].values[0] * rotation_pole_y + \
                                    _iter_driving_torque_z[_plateID].values[0] * rotation_pole_z
                                )
                            normalised_dot_product /= _numpy.abs(_iter_driving_torque[_plateID].values[0] * rotation_pole_mag[_plateID].values[0])

                            dot_products.append(1-normalised_dot_product)
                            # Calculate normalised residual torque
                            # Why is this only the magnitude? Another approach would be to calculate the difference of each component
                            normalised_residual_torque = _numpy.log10(_iter_residual_torque[_plateID].values[0] / _iter_driving_torque[_plateID].values[0])
                            
                            score = normalised_residual_torque
                            existing_scores.append(score)
                            # for _plateID in _plateIDs[_age][_case]:
                            #     driving_torque_opt_stack[_age][_case][_plateID][i] = _iter_driving_torque[_plateID].values[0]
                            #     residual_torque_opt_stack[_age][_case][_plateID][i] = _iter_residual_torque[_plateID].values[0]
 
                            # After 20 iterations, check the difference between the minimum and maximum values
                            # if i > 10:
                                # if (existing_scores[int(i)] - existing_scores[-1]) < 0.01:
                                #     break
                                    # if _numpy.nanmin(_numpy.asarray(existing_scores)) == existing_scores[-10]:
                                # if _numpy.nanmin(_numpy.asarray(existing_scores)) == existing_scores[-10]:
                                #     break
                                # elif _numpy.abs(existing_scores[-1] - existing_scores[-2]) < 0.01:
                                #     break

                        # Calculate the residual torque
                    # plt.scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                    # plt.ylim([-2, 1])
                    # plt.xlim([vmin, vmax])
                    # plt.scatter(_numpy.log10(_numpy.asarray(existing_values)), _numpy.log10(_numpy.asarray(dot_products)), c="r")
                    # plt.colorbar(label="Iteration")
                    # plt.title(f"{_plateID} ({_numpy.log10(_numpy.sum(max_slab_pull_force_mag))} N")
                    # plt.show()
                    # plt.plot(existing_scores)
                    # plt.ylim([-2, 1])
                    # plt.title(f"{_plateID} ({_numpy.log10(_numpy.sum(max_slab_pull_force_mag))} N")
                    # plt.show()

                    # This sometimes throws and error, so it is wrapped in a try-except block
                    try:        
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(_numpy.asarray(existing_scores))
                        opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                        opt_constants[_age][_case][_plateID] = _numpy.asarray(existing_values)[opt_index]
                    except:
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nan
                        opt_constants[_age][_case][_plateID] = _numpy.nan

                    # if not _data[_age][_case].empty:
                    #     minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(residual_torque_opt_stack[_age][_case][_plateID]/driving_torque_opt_stack[_age][_case][_plateID])
                    #     opt_index = _numpy.nanargmin(residual_torque_opt_stack[_age][_case][_plateID]/driving_torque_opt_stack[_age][_case][_plateID])
                    #     opt_constants[_age][_case][_plateID] = constants[opt_index]

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
    
    def invert_residual_torque_v2(
            self,
            age = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = 8.5, 
            vmax = 13.5,
            step = .1, 
            plot = False, 
        ):
        """
        Function to find optimised slab pull coefficient or mantle viscosity by projecting the residual torque onto the subduction zones and grid points.

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

        NOTE: This function is still under development and may not work as intended.
        """
        # Raise error if parameter is not slab pull coefficient or mantle viscosity
        if parameter not in ["Slab pull constant", "Mantle viscosity"]:
            raise ValueError("Free parameter must be 'Slab pull constant' or 'Mantle viscosity")
        
        # Select ages, if not provided
        _ages = utils_data.select_ages(age, self.settings.ages)

        # Select cases, if not provided
        _cases = utils_data.select_cases(cases, self.settings.reconstructed_cases)

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        pole_lat_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        pole_lon_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        pole_angle_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for _age in _tqdm(_ages, desc="Inverting residual torque"):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                pole_lat_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                pole_lon_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                pole_angle_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Store reconstructed pole of rotation
                reconstructed_pole_lat = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lat")
                reconstructed_pole_lon = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lon")
                reconstructed_pole_angle = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_angle")

                for i, constant in enumerate(constants):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Inform the user which constant is being optimised
                        logging.info(f"Optimising for {constant}")

                        # Calculate the torques the normal way
                        self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                        self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                        self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

                        # Select data
                        _data[_age][_case] = self.slabs.data[_age][_case].copy()

                        # Filter data, if necessary
                        if plateIDs is not None:
                            _data[_age][_case] = _data[_age][_case][_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                        # Skip if no data
                        if _data[_age][_case].empty:
                            continue

                        # Get the slab pull force magnitude
                        max_slab_pull_force_mag = _data[_age][_case]["slab_pull_force_mag"] / _data[_age][_case]["slab_pull_constant"]

                        # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                        _data[_age][_case]["slab_pull_force_mag"] -= (
                            _data[_age][_case]["residual_force_lat"] * _data[_age][_case]["slab_pull_force_lat"] + \
                            _data[_age][_case]["residual_force_lon"] * _data[_age][_case]["slab_pull_force_lon"]
                        ) * 10**-constant

                        # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data[_age][_case].loc[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                        _data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data[_age][_case]["trench_normal_azimuth"])) * _data[_age][_case]["slab_pull_force_mag"]
                    
                        # Calculate the torques with the modified slab pull forces
                        self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                        self.plates.calculate_synthetic_velocity(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False, RECONSTRUCTED_CASES=True)

                        # Extract the pole of rotation
                        _iter_pole_lat = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lon")
                        _iter_pole_lon = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_lat")
                        _iter_pole_angle = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], var="pole_angle")

                        for _plateID in _plateIDs[_age][_case]:
                            # print(_age, _case, _plateID, constant, _iter_pole_lat[_plateID].values[0], _iter_pole_lon[_plateID].values[0], _iter_pole_angle[_plateID].values[0])
                            pole_lon_opt_stack[_age][_case][_plateID][i] = _iter_pole_lat[_plateID].values[0]
                            pole_lat_opt_stack[_age][_case][_plateID][i] = _iter_pole_lon[_plateID].values[0]
                            pole_angle_opt_stack[_age][_case][_plateID][i] = _iter_pole_angle[_plateID].values[0]
                            
                for _plateID in _plateIDs[_age][_case]:
                    # Calculate distance to the pole of rotation
                    distances = utils_calc.haversine(
                        pole_lon_opt_stack[_age][_case][_plateID],
                        pole_lat_opt_stack[_age][_case][_plateID],
                        _numpy.repeat(reconstructed_pole_lon[_plateID], len(constants)),
                        _numpy.repeat(reconstructed_pole_lon[_plateID], len(constants)),
                    )
                    plt.plot(constants, distances, label="Distance to pole of rotation")
                    plt.plot(constants, pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants)), label="Ratio between synthetic and reconstructed pole angle")
                    plt.plot(constants, distances + _numpy.abs(pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants))), label="Equally weighted components")
                    plt.legend()
                    plt.show()

                    opt_index = _numpy.nanargmin(distances + _numpy.abs(pole_angle_opt_stack[_age][_case][_plateID]-_numpy.repeat(reconstructed_pole_angle[_plateID], len(constants))))
                    opt_constants[_age][_case][_plateID] = constants[opt_index]

                    # print("Optimal constant for plate", _plateID, f"for case {_case} at age {_age}", opt_constants[_age][_case][_plateID])

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * 10**-opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

    def invert_residual_torque_v3(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            parameter = "Slab pull constant",
            vmin = -14.0, 
            vmax = -6.0,
            step = .25, 
            plot = False, 
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
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        
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

        # Define constants
        constants = _numpy.arange(vmin, vmax, step)

        # Initialise dictionaries to store minimum residual torque and optimal constants
        driving_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        residual_torque_opt_stack = {_age: {_case: None for _case in _cases} for _age in _ages}
        minimum_residual_torque = {_age: {_case: None for _case in _cases} for _age in _ages}
        opt_constants = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Initialise data and plateID dictionary
        _plate_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _slab_data = {_age: {_case: None for _case in _cases} for _age in _ages}
        _plateIDs = {_age: {_case: None for _case in _cases} for _age in _ages}

        # Loop through ages
        for j, _age in enumerate(_tqdm(_ages, desc="Inverting residual torque")):
            for _case in _cases:
                # Obtain plateIDs for all unique combinations of ages and cases
                _plateIDs[_age][_case] = utils_data.select_plateIDs(plateIDs, self.slabs.data[_age][_case]["lower_plateID"].unique())

                # Initialise entries for each plate ID in dictionaries
                driving_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                residual_torque_opt_stack[_age][_case] = {_plateID: _numpy.zeros((len(constants))) for _plateID in _plateIDs[_age][_case]}
                minimum_residual_torque[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}
                opt_constants[_age][_case] = {_plateID: _numpy.nan for _plateID in _plateIDs[_age][_case]}

                # Loop through plates
                for _plateID in _plateIDs[_age][_case]:
                    if _plateID not in self.plates.data[_age][_case]["plateID"].values:
                        continue
                    # # Extract rotation pole
                    # rotation_pole_lat = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lat")
                    # rotation_pole_lon = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_lon")
                    # rotation_pole_mag = self.plates.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="pole_angle")
                    
                    # # Convert rotation pole to Cartesian coordinates
                    # rotation_pole_x, rotation_pole_y, rotation_pole_z = utils_calc.geocentric_spherical2cartesian(
                    #     rotation_pole_lat[_plateID].values[0], 
                    #     rotation_pole_lon[_plateID].values[0],
                    #     rotation_pole_mag[_plateID].values[0]
                    # )

                    existing_values = []; existing_scores = []; dot_products = []
                    for i in _numpy.arange(0, len(constants)+10):
                    # for i, constant in enumerate(constants):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            # Propose initial value
                            if i > len(constants)-1:
                                # The inversion should start with a more general, grid-based exploration of the parameter space
                                # Only after ~20 iterations or so, the algorithm should start to adapt the step size
                                constant = utils_calc.propose_value(existing_values, existing_scores, lower_bound=vmin, upper_bound=vmax)
                            else:
                                constant = 10**constants[i]

                            # else:
                            #     constant = utils_calc.propose_value(existing_values, existing_scores, 0.3)

                            # constant = 10**constant
                                                                    
                            existing_values.append(constant)

                            # Inform the user which constant is being optimised
                            # logging.info(f"Optimising for {constant}")

                            # Calculate the torques the normal way
                            # self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False, CALCULATE_AT_POINTS=False)

                            # Select data
                            _plate_data = self.plates.data[_age][_case].copy()
                            _slab_data[_age][_case] = self.slabs.data[_age][_case].copy()

                            # Filter data, if necessary
                            if plateIDs is not None:
                                _plate_data = _plate_data[_plate_data["plateID"].isin(_plateIDs[_age][_case])]
                                _slab_data[_age][_case] = _slab_data[_age][_case][_slab_data[_age][_case]["lower_plateID"].isin(_plateIDs[_age][_case])]

                            # Skip if no data
                            if _slab_data[_age][_case].empty:
                                continue

                            # print(_plate_data)

                            # Get the slab pull force magnitude
                            max_slab_pull_force_mag = _slab_data[_age][_case]["slab_pull_force_mag"] / _slab_data[_age][_case]["slab_pull_constant"]

                            # Modify the magnitude of the slab pull force using the 2D dot product of the residual force and the slab pull force and the constant
                            # This step should be performed in Cartesian coordinates.
                            _slab_data[_age][_case]["slab_pull_force_mag"] -= (
                                _slab_data[_age][_case]["residual_force_lat"] * _slab_data[_age][_case]["slab_pull_force_lat"] + \
                                _slab_data[_age][_case]["residual_force_lon"] * _slab_data[_age][_case]["slab_pull_force_lon"]
                            ) * constant

                            # Ensure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                            _slab_data[_age][_case].loc[_slab_data[_age][_case]["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                            _slab_data[_age][_case].loc[_slab_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_slab_data[_age][_case]["slab_pull_force_mag"] > max_slab_pull_force_mag]

                            # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                            _slab_data[_age][_case]["slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_slab_data[_age][_case]["trench_normal_azimuth"])) * _slab_data[_age][_case]["slab_pull_force_mag"]
                            _slab_data[_age][_case]["slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_slab_data[_age][_case]["trench_normal_azimuth"])) * _slab_data[_age][_case]["slab_pull_force_mag"]
                        
                            # Calculate the torques with the modified slab pull forces
                            # Calculate torques
                            _iter_torques = utils_calc.compute_torque_on_plates(
                                _plate_data,
                                _slab_data[_age][_case].lat.values,
                                _slab_data[_age][_case].lon.values,
                                _slab_data[_age][_case].lower_plateID.values,
                                _slab_data[_age][_case].slab_pull_force_lat.values, 
                                _slab_data[_age][_case].slab_pull_force_lon.values,
                                _slab_data[_age][_case].trench_segment_length.values,
                                torque_var = "slab_pull",
                            )
                            # self.plates.calculate_torque_on_plates(_data, ages=_age, cases=_case, plateIDs=_plateID, torque_var="slab_pull", PROGRESS_BAR=False)

                            # This should be rewritten, as calculating the residual vecter in spherical coordinates is time-consuming
                            # Instead, the dot product should be calculated in Cartesian coordinates
                            # driving_torques = utils_calc.compute_driving_torque(

                            # Calculate driving and residual torque
                            _iter_torques = utils_calc.sum_torque(_iter_torques, "driving")
                            _iter_torques = utils_calc.sum_torque(_iter_torques, "residual")

                            # print(_iter_torques)
                            # self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                            # self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False, CALCULATE_AT_POINTS=False)

                            # Extract the driving and residual torques
                            # _iter_driving_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_mag")
                            # _iter_residual_torque = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var= "residual_torque_mag")

                            _iter_driving_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["driving_torque_mag"].values
                            _iter_residual_torque = _iter_torques[_iter_torques["plateID"] == _plateID]["residual_torque_mag"].values

                            # # Extract driving torque
                            # _iter_driving_torque_x = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_x")
                            # _iter_driving_torque_y = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_y")
                            # _iter_driving_torque_z = self.plate_torques.extract_data_through_time(ages=_age, cases=_case, plateIDs=_plateID, var="driving_torque_z")

                            # # Calculate dot product 
                            # normalised_dot_product = _numpy.abs(
                            #         _iter_driving_torque_x[_plateID].values[0] * rotation_pole_x + \
                            #         _iter_driving_torque_y[_plateID].values[0] * rotation_pole_y + \
                            #         _iter_driving_torque_z[_plateID].values[0] * rotation_pole_z
                            #     )
                            # normalised_dot_product /= _numpy.abs(_iter_driving_torque[_plateID].values[0] * rotation_pole_mag[_plateID].values[0])

                            # dot_products.append(1-normalised_dot_product)
                            # Calculate normalised residual torque
                            # Why is this only the magnitude? Another approach would be to calculate the difference of each component
                            normalised_residual_torque = _numpy.log10(_iter_residual_torque / _iter_driving_torque)
                            
                            score = normalised_residual_torque
                            existing_scores.append(score)

                    if plot == True or plot == _plateID:
                        # Plot the optimisation process
                        fig, axes = plt.subplots(1, 2)
                        axes[0].plot(existing_scores)
                        axes[0].set_xlabel("Iteration")
                        axes[0].set_ylabel("Score")
                        axes[0].set_ylim(-2, 1)

                        p = axes[1].scatter(_numpy.log10(_numpy.asarray(existing_values)), existing_scores, c=_numpy.arange(0, i+1))
                        axes[1].set_ylim(-2, 1)
                        axes[1].set_xlim(vmin, vmax)
                        fig.colorbar(p, label="iteration", orientation="horizontal")
                        axes[1].set_xlabel("Parameter value")
                        axes[1].set_yticklabels([])
                        fig.suptitle(f"Optimisation for plateID {_plateID}")
                    
                    # Find the minimum value
                    # NOTE: This sometimes throws and error, so it is wrapped in a try-except block
                    try:        
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nanmin(_numpy.asarray(existing_scores))
                        opt_index = _numpy.nanargmin(_numpy.asarray(existing_scores))
                        opt_constants[_age][_case][_plateID] = _numpy.asarray(existing_values)[opt_index]
                    except:
                        minimum_residual_torque[_age][_case][_plateID] = _numpy.nan
                        opt_constants[_age][_case][_plateID] = _numpy.nan

        for _age in _tqdm(_ages, desc="Optimising torques"):
            for _case in _cases:
                for _plateID in _plateIDs[_age][_case]:
                    # Recalculate all the relevant torques
                    self.plate_torques.calculate_slab_pull_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)
                    self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateID, PROGRESS_BAR=False)

                    # Select data
                    _data = self.slabs.data[_age][_case].copy()
                    _data = _data[_data["lower_plateID"] == _plateID]

                    if not _data.empty:
                        # Get the old slab pull force magnitude
                        max_slab_pull_force_mag = _data["slab_pull_force_mag"].values / _data["slab_pull_constant"].values

                        # Calculate the slab pull force
                        _data.loc[_data.index, "slab_pull_force_mag"] -= (
                                _data["residual_force_lat"] * _data["slab_pull_force_lat"] + \
                                _data["residual_force_lon"] * _data["slab_pull_force_lon"]
                            ) * opt_constants[_age][_case][_plateID]
                        
                        # Make sure the slab pull force magnitude is positive and not larger than the original slab pull force magnitude
                        _data.loc[_data["slab_pull_force_mag"] < 0, "slab_pull_force_mag"] = 0
                        _data.loc[_data["slab_pull_force_mag"] > max_slab_pull_force_mag, "slab_pull_force_mag"] = max_slab_pull_force_mag[_data["slab_pull_force_mag"] > max_slab_pull_force_mag]

                        # Decompose the slab pull force into latitudinal and longitudinal components using the trench normal azimuth
                        _data.loc[_data.index, "slab_pull_force_lat"] = _numpy.cos(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]
                        _data.loc[_data.index, "slab_pull_force_lon"] = _numpy.sin(_numpy.deg2rad(_data["trench_normal_azimuth"])) * _data["slab_pull_force_mag"]

                        # Calculate the slab pull constant
                        _data.loc[_data.index, "slab_pull_constant"] = _data.loc[_data.index, "slab_pull_force_mag"] / max_slab_pull_force_mag

                        # Make sure the slab pull constant is between 0 and 1
                        _data.loc[_data["slab_pull_constant"] < 0, "slab_pull_constant"] = 0
                        _data.loc[_data["slab_pull_constant"] > 1, "slab_pull_constant"] = 1

                        # Feed optimal values back into slab data
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_mag"] = _data["slab_pull_force_mag"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lat"] = _data["slab_pull_force_lat"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_force_lon"] = _data["slab_pull_force_lon"].values
                        self.slabs.data[_age][_case].loc[_data.index, "slab_pull_constant"] = _data["slab_pull_constant"].values

                # Recalculate all the relevant torques
                self.plates.calculate_torque_on_plates(self.slabs.data, ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], torque_var="slab_pull", PROGRESS_BAR=False)
                self.plate_torques.calculate_driving_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)
                self.plate_torques.calculate_residual_torque(ages=_age, cases=_case, plateIDs=_plateIDs[_age][_case], PROGRESS_BAR=False)

def minimise_residual_torque_v2(
            self,
            ages = None, 
            cases = None, 
            plateIDs = None, 
            grid_size = 500, 
            lateral_viscosity_variation_range = [1, 100],
            plot = True,
            weight_by_area = True,
            minimum_plate_area = None
        ):
        """
        Function to find optimised coefficients to match plate motions using a grid search

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

        NOTE: Slab suction will be removed shortly and optimisation of lateral viscosity variation will be moved to the original `minimise_residual_torque` function.
        """
        # Define ages if not provided
        _ages = utils_data.select_ages(ages, self.settings.ages)
        
        # Define cases if not provided
        _cases = utils_data.select_cases(cases, self.settings.point_cases)

        # Set range of viscosities
        viscs = _numpy.linspace(lateral_viscosity_variation_range[0], lateral_viscosity_variation_range[1], grid_size)
        
        # Loop through ages
        for _age in _ages:
            for _case in _cases:

                # Set range of slab pull coefficients
                if self.settings.options[_case]["Sediment subduction"]:
                    # Range is smaller with sediment subduction
                    sp_consts = _numpy.linspace(1e-5, 0.25, grid_size)
                else:
                    sp_consts = _numpy.linspace(1e-5, 1., grid_size)

                # Create grids from ranges of viscosities and slab pull coefficients
                visc_grid, sp_const_grid = _numpy.meshgrid(viscs, sp_consts)
                ones_grid = _numpy.ones_like(visc_grid)

                # Filter plates
                _data = self.plates.data[_age][_case]

                _plateIDs = utils_data.select_plateIDs(plateIDs, self.plates.data[_age][_case].plateID)

                if plateIDs is not None:
                    _data = _data[_data["plateID"].isin(_plateIDs)]

                if minimum_plate_area is not None:
                    _data = _data[_data["area"] > minimum_plate_area]

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
                            residual_x -= _data.slab_pull_torque_x.iloc[k] * ones_grid
                            residual_y -= _data.slab_pull_torque_y.iloc[k] * ones_grid
                            residual_z -= _data.slab_pull_torque_z.iloc[k] * ones_grid

                        # Add slab suction torque
                        if self.settings.options[_case]["Slab suction torque"] and "slab_suction_torque_x" in _data.columns:
                            residual_x -= _data.slab_suction_torque_x.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_y -= _data.slab_suction_torque_y.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]
                            residual_z -= _data.slab_suction_torque_z.iloc[k] * sp_const_grid / self.settings.options[_case]["Slab suction constant"]

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
                            residual_x -= _data.mantle_drag_torque_x.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_x.iloc[k] * (1-_data.continental_fraction.iloc[k])
                            residual_y -= _data.mantle_drag_torque_y.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_y.iloc[k] * (1-_data.continental_fraction.iloc[k])
                            residual_z -= _data.mantle_drag_torque_z.iloc[k] * _data.continental_fraction.iloc[k] * visc_grid / self.settings.options[_case]["Lateral viscosity variation"] + \
                                _data.mantle_drag_torque_z.iloc[k] * (1-_data.continental_fraction.iloc[k])
                        
                        # Compute magnitude of residual
                        if weight_by_area:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) * _data.area.iloc[k] / total_area
                        else:
                            residual_mag += _numpy.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / _data.area.iloc[k]                        
                
                # Divide residual by driving torque
                residual_mag_normalised = _numpy.log10(residual_mag / driving_mag)

                # Find the indices of the minimum value directly using _numpy.argmin
                opt_i = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=1))
                opt_j = _numpy.argmin(_numpy.min(residual_mag_normalised, axis=0))
                # opt_i, opt_j = _numpy.unravel_index(_numpy.argmin(residual_mag), residual_mag.shape)
                opt_visc = visc_grid[opt_i, opt_j]
                opt_sp_const = sp_const_grid[opt_i, opt_j]

                # Plot
                if plot == True:
                    fig, ax = plt.subplots(figsize=(9*cm2in*2, 7.2*cm2in*2))
                    im = ax.imshow(residual_mag_normalised, cmap="cmc.lapaz_r", vmin=-1.5, vmax=1.5)
                    ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                    ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(lateral_viscosity_variation_range[0], lateral_viscosity_variation_range[1], 5)])
                    ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                    ax.set_xlabel("Lateral viscosity variation")
                    ax.set_ylabel("Slab suction constant")
                    ax.scatter(opt_j, opt_i, marker="*", facecolor="none", edgecolor="k", s=30)  # Adjust the marker style and size as needed
                    fig.colorbar(im, label = "Log(residual torque/driving torque)")
                    plt.show()

                # Print results
                print(f"Optimal coefficients for ", ", for ".join(_data.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(_data.plateID.astype(str)), ")")
                print("Minimum residual torque: {:.2%} of driving torque".format(10**(_numpy.amin(residual_mag_normalised))))
                print("Optimum lateral viscosity variation: {:.2e}".format(opt_visc))
                print("Optimum slab suction constant: {:.2%}".format(opt_sp_const))

        return residual_mag_normalised, (opt_visc, opt_sp_const), (opt_i, opt_j)

def plot_velocity_map(
            self,
            ax,
            fig,
            reconstruction_time,
            case,
            plotting_options
        ):
        """
        Function to create subplot with plate velocities
            ax:                     axes object
            fig:                    figure
            reconstruction_time:    the time for which to display the map
            case:                   case for which to plot the sediments
            plotting_options:       dictionary with options for plotting
        """
        # Check if reconstruction time is in valid times
        if reconstruction_time not in self.times:
            return print("Invalid reconstruction time")
        
        # Set basemap
        ax, gl = self.plot_basemap(ax)

        # Plot plates and coastlines
        self.plot_reconstruction(ax, reconstruction_time, plotting_options, plates=True, trenches=False)

        # Get data
        plate_vectors = self.plates[reconstruction_time][case].loc[self.plates[reconstruction_time][case].area >= self.options[case]["Minimum plate area"]]
        slab_data = self.slabs[reconstruction_time][case].loc[self.slabs[reconstruction_time][case].lower_plateID.isin(plate_vectors.plateID)]
        slab_vectors = slab_data.iloc[::5]

        # Plot velocity magnitude at trenches
        vels = ax.scatter(
            slab_data.lon,
            slab_data.lat,
            c=slab_data.v_lower_plate_mag,
            s=plotting_options["marker size"],
            transform=ccrs.PlateCarree(),
            cmap=plotting_options["velocity magnitude cmap"],
            vmin=0,
            vmax=plotting_options["velocity max"]
        )

        # Plot velocity at subduction zones
        slab_vectors = ax.quiver(
            x=slab_vectors.lon,
            y=slab_vectors.lat,
            u=slab_vectors.v_lower_plate_lon,
            v=slab_vectors.v_lower_plate_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=2e-3,
            scale=3e2,
            zorder=4,
            color='black'
        )

        # Plot velocity at centroid
        centroid_vectors = ax.quiver(
            x=plate_vectors.centroid_lon,
            y=plate_vectors.centroid_lat,
            u=plate_vectors.centroid_v_lon,
            v=plate_vectors.centroid_v_lat,
            transform=ccrs.PlateCarree(),
            # label=vector.capitalize(),
            width=5e-3,
            scale=3e2,
            zorder=4,
            color='white',
            edgecolor='black',
            linewidth=1
        )

        # Colourbar
        if plotting_options["cbar"] is True:
            fig.colorbar(vels, ax=ax, label="Velocity [cm/a]", orientation=plotting_options["orientation cbar"], shrink=0.75, aspect=20)
    
        return ax, vels, centroid_vectors, slab_vectors


def optimise_torques(self, sediments=True):
        """
        Function to apply optimised parameters to torques
        Arguments:
            opt_visc
            opt_sp_const
        """
        # Apply to each torque in DataFrame
        axes = ["_x", "_y", "_z", "_mag"]
        for case in self.cases:
            for axis in axes:
                self.plates[case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[case]["slab_pull_torque" + axis]
                if self.options[case]["Reconstructed motions"]:
                    self.plates[case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[case]["mantle_drag_torque" + axis]
                
                for reconstruction_time in self.times:
                    if sediments == True:
                        self.plates[reconstruction_time][case]["slab_pull_torque_opt" + axis] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_torque" + axis]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_torque_opt" + axis] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["mantle_drag_torque" + axis]

        # Apply to forces at centroid
        coords = ["lon", "lat"]
        for reconstruction_time in self.times:
            for case in self.cases:
                for coord in coords:
                    self.plates[reconstruction_time][case]["slab_pull_force_opt" + coord] = self.options[case]["Slab pull constant"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]
                    if self.options[case]["Reconstructed motions"]:
                        self.plates[reconstruction_time][case]["mantle_drag_force_opt" + coord] = self.options[case]["Mantle viscosity"] * self.plates[reconstruction_time][case]["slab_pull_force" + coord]

        self.optimised_torques = True

def minimise_residual_velocity(self, opt_time, opt_case, plates_of_interest=None, grid_size=10, visc_range=[1e19, 5e20], plot=True, weight_by_area=True, ref_case=None):
        """
        Function to find optimised coefficients to match plate motions using a grid search.

        :param opt_time:                reconstruction time to optimise
        :type opt_time:                 int
        :param opt_case:                case to optimise
        :type opt_case:                 str
        :param plates_of_interest:      plate IDs to include in optimisation
        :type plates_of_interest:       list of integers or None
        :param grid_size:               size of the grid to find optimal viscosity and slab pull coefficient
        :type grid_size:                int
        :param plot:                    whether or not to plot the grid
        :type plot:                     boolean
        :param weight_by_area:          whether or not to weight the residual torque by plate area
        :type weight_by_area:           boolean
        
        :return:                        The optimal slab pull coefficient, the optimal viscosity, the residual plate velocity, and the residual slab velocity.
        :rtype:                         float, float, float, float
        """
        if self.options[opt_case]["Reconstructed motions"]:
            print("Optimisation method designed for synthetic plate velocities only!")
            return
        
        # Get "true" plate velocities
        true_slabs = self.slabs[opt_time][ref_case].copy()

        # Generate grid
        viscs = _numpy.linspace(visc_range[0],visc_range[1],grid_size)
        sp_consts = _numpy.linspace(1e-4,1,grid_size)
        v_upper_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_lower_plate_residual = _numpy.zeros((grid_size, grid_size))
        v_convergence_residual = _numpy.zeros((grid_size, grid_size))

        # Filter plates and slabs
        selected_plates = self.plates[opt_time][opt_case].copy()
        selected_slabs = self.slabs[opt_time][opt_case].copy()
        selected_points = self.points[opt_time][opt_case].copy()

        if plates_of_interest:
            selected_plates = selected_plates[selected_plates["plateID"].isin(plates_of_interest)]
            selected_plates = selected_plates.reset_index(drop=True)
            selected_slabs = selected_slabs[selected_slabs["lower_plateID"].isin(plates_of_interest)]
            selected_slabs = selected_slabs.reset_index(drop=True)
            selected_points = selected_points[selected_points["plateID"].isin(plates_of_interest)]
            selected_points = selected_points.reset_index(drop=True)
            selected_options = self.options[opt_case].copy()
        else:
            plates_of_interest = selected_plates["plateID"]

        # Initialise starting old_plates, old_points, old_slabs by copying self.plates[reconstruction_time][key], self.points[reconstruction_time][key], self.slabs[reconstruction_time][key]
        old_plates = selected_plates.copy(); old_points = selected_points.copy(); old_slabs = selected_slabs.copy()

        # Delete self.slabs[reconstruction_time][key], self.points[reconstruction_time][key], self.plates[reconstruction_time][key]
        del selected_plates, selected_points, selected_slabs
        
        # Loop through plates and slabs and calculate residual velocity
        for i, visc in enumerate(viscs):
            for j, sp_const in enumerate(sp_consts):
                print(i, j)
                # Assign current visc and sp_const to options
                selected_options["Mantle viscosity"] = visc
                selected_options["Slab pull constant"] = sp_const

                # Optimise slab pull force
                [old_plates.update({"slab_pull_torque_opt_" + axis: old_plates["slab_pull_torque_" + axis] * selected_options["Slab pull constant"]}) for axis in ["x", "y", "z"]]

                for k in range(100):
                    # Delete new DataFrames
                    if k != 0:
                        del new_slabs, new_points, new_plates
                    else:
                        old_slabs["v_convergence_mag"] = 0

                    print(_numpy.mean(old_slabs["v_convergence_mag"].values))
                    # Compute interface shear force
                    if self.options[opt_case]["Interface shear torque"]:
                        new_slabs = utils_calc.compute_interface_shear_force(old_slabs, self.options[opt_case], self.mech, self.constants)
                    else:
                        new_slabs = old_slabs.copy()

                    # Compute interface shear torque
                    new_plates = utils_calc.compute_torque_on_plates(
                        old_plates,
                        new_slabs.lat,
                        new_slabs.lon,
                        new_slabs.lower_plateID,
                        new_slabs.interface_shear_force_lat,
                        new_slabs.interface_shear_force_lon,
                        new_slabs.trench_segment_length,
                        1,
                        self.constants,
                        torque_variable="interface_shear_torque"
                    )

                    # Compute mantle drag force
                    new_plates, new_points, new_slabs = utils_calc.compute_mantle_drag_force(old_plates, old_points, new_slabs, self.options[opt_case], self.mech, self.constants)

                    # Compute mantle drag torque
                    new_plates = utils_calc.compute_torque_on_plates(
                        new_plates, 
                        new_points.lat, 
                        new_points.lon, 
                        new_points.plateID, 
                        new_points.mantle_drag_force_lat, 
                        new_points.mantle_drag_force_lon,
                        new_points.segment_length_lat,
                        new_points.segment_length_lon,
                        self.constants,
                        torque_variable="mantle_drag_torque"
                    )

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Calculate convergence rates
                    v_convergence_lat = new_slabs["v_lower_plate_lat"].values - new_slabs["v_upper_plate_lat"].values
                    v_convergence_lon = new_slabs["v_lower_plate_lon"].values - new_slabs["v_upper_plate_lon"].values
                    v_convergence_mag = _numpy.sqrt(v_convergence_lat**2 + v_convergence_lon**2)

                    # Check convergence rates
                    if _numpy.max(abs(v_convergence_mag - old_slabs["v_convergence_mag"].values)) < 1e-2: # and _numpy.max(v_convergence_mag) < 25:
                        print(f"Convergence rates converged after {k} iterations")
                        break
                    else:
                        # Assign new values to latest slabs DataFrame
                        new_slabs["v_convergence_lat"], new_slabs["v_convergence_lon"] = utils_calc.mag_azi2lat_lon(v_convergence_mag, new_slabs.trench_normal_azimuth); new_slabs["v_convergence_mag"] = v_convergence_mag
                        
                        # Delecte old DataFrames
                        del old_plates, old_points, old_slabs
                        
                        # Overwrite DataFrames
                        old_plates = new_plates.copy(); old_points = new_points.copy(); old_slabs = new_slabs.copy()

                # Calculate residual of plate velocities
                v_upper_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_upper_plate_mag - true_slabs.v_upper_plate_mag))
                print("upper_plate_residual: ", v_upper_plate_residual[i,j])
                v_lower_plate_residual[i,j] = _numpy.max(abs(new_slabs.v_lower_plate_mag - true_slabs.v_lower_plate_mag))
                print("lower_plate_residual: ", v_lower_plate_residual[i,j])
                v_convergence_residual[i,j] = _numpy.max(abs(new_slabs.v_convergence_mag - true_slabs.v_convergence_mag))
                print("convergence_rate_residual: ", v_convergence_residual[i,j])

        # Find the indices of the minimum value directly using _numpy.argmin
        opt_upper_plate_i, opt_upper_plate_j = _numpy.unravel_index(_numpy.argmin(v_upper_plate_residual), v_upper_plate_residual.shape)
        opt_upper_plate_visc = viscs[opt_upper_plate_i]
        opt_upper_plate_sp_const = sp_consts[opt_upper_plate_j]

        opt_lower_plate_i, opt_lower_plate_j = _numpy.unravel_index(_numpy.argmin(v_lower_plate_residual), v_lower_plate_residual.shape)
        opt_lower_plate_visc = viscs[opt_lower_plate_i]
        opt_lower_plate_sp_const = sp_consts[opt_lower_plate_j]

        opt_convergence_i, opt_convergence_j = _numpy.unravel_index(_numpy.argmin(v_convergence_residual), v_convergence_residual.shape)
        opt_convergence_visc = viscs[opt_convergence_i]
        opt_convergence_sp_const = sp_consts[opt_convergence_j]

        # Plot
        for i, j, visc, sp_const, residual in zip([opt_upper_plate_i, opt_lower_plate_i, opt_convergence_i], [opt_upper_plate_j, opt_lower_plate_j, opt_convergence_j], [opt_upper_plate_visc, opt_lower_plate_visc, opt_convergence_visc], [opt_upper_plate_sp_const, opt_lower_plate_sp_const, opt_convergence_sp_const], [v_upper_plate_residual, v_lower_plate_residual, v_convergence_residual]):
            if plot == True:
                fig, ax = plt.subplots(figsize=(15*self.constants.cm2in, 12*self.constants.cm2in))
                im = ax.imshow(residual, cmap="cmc.davos_r")#, vmin=-1.5, vmax=1.5)
                ax.set_yticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticks(_numpy.linspace(0, grid_size - 1, 5))
                ax.set_xticklabels(["{:.2e}".format(visc) for visc in _numpy.linspace(visc_range[0], visc_range[1], 5)])
                ax.set_yticklabels(["{:.2f}".format(sp_const) for sp_const in _numpy.linspace(sp_consts.min(), sp_consts.max(), 5)])
                ax.set_xlabel("Mantle viscosity [Pa s]")
                ax.set_ylabel("Slab pull reduction factor")
                ax.scatter(j, i, marker="*", facecolor="none", edgecolor="k", s=30)
                fig.colorbar(im, label = "Residual velocity magnitude [cm/a]")
                plt.show()

            print(f"Optimal coefficients for ", ", ".join(new_plates.name.astype(str)), " plate(s), (PlateIDs: ", ", ".join(new_plates.plateID.astype(str)), ")")
            print("Minimum residual torque: {:.2e} cm/a".format(_numpy.amin(residual)))
            print("Optimum viscosity [Pa s]: {:.2e}".format(visc))
            print("Optimum Drag Coefficient [Pa s/m]: {:.2e}".format(visc / self.mech.La))
            print("Optimum Slab Pull constant: {:.2%}".format(sp_const))
            
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATO
# Algorithm to calculate plate forces from tectonic reconstructions
# Initialiser for accelerated parallel initialisation of plate, point and slab data
# Thomas Schouten, 2024
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import libraries
# Standard libraries
import os as _os
import warnings
from typing import List, Optional, Union

# Third-party libraries
import numpy as _numpy
import matplotlib.pyplot as plt
import geopandas as _gpd
import gplately
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import cmcrameri as cmc
from tqdm import tqdm
import xarray as _xarray
from time import time

# Local libraries
import setup
import setup_parallel
import functions_main

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INITIALISER
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialise_plato(
        reconstruction_name: str, 
        reconstruction_times: List[int] or _numpy.array, 
        cases_file: str, 
        cases_sheet: Optional[str] = "Sheet1", 
        files_dir: Optional[str] = None,
        rotation_file: Optional[List[str]] = None,
        topology_file: Optional[List[str]] = None,
        polygon_file: Optional[List[str]] = None,
    ):

    # Store cases and case options
    cases, options = setup.get_options(cases_file, cases_sheet)

    # Group cases for initialisation of slabs and points
    slab_options = ["Slab tesselation spacing"]
    slab_cases = setup.process_cases(cases, options, slab_options)
    point_options = ["Grid spacing"]
    point_cases = setup.process_cases(cases, options, point_options)

    # Convert GPlates file inputs to GPlates objects
    rotations = _pygplates.RotationModel(rotation_file)
    topologies = _pygplates.FeatureCollection(topology_file)
    polygons = _pygplates.FeatureCollection(polygon_file)

    reconstruction = gplately.PlateReconstruction(rotations, topologies, polygons)
    
    for reconstruction_time in tqdm(reconstruction_times, desc="Initialising and saving files"):
        # Get geometries of plates
        resolved_geometries = setup.get_topology_geometries(
            reconstruction, reconstruction_time, anchor_plateID=0
        )

        # Resolve topologies to use to get plates
        # NOTE: This is done because some information is retrieved from the resolved topologies and some from the resolved geometries
        #       This step could be sped up by extracting all information from the resolved geometries, but so far this has not been the main bottleneck
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="Normalized/laundered field name:"
            )
            resolved_topologies = []
            _pygplates.resolve_topologies(
                topologies,
                rotations, 
                resolved_topologies, 
                reconstruction_time, 
                anchor_plate_id=0
            )

        # Get plates
        plates = setup_parallel.get_plates(
                    reconstruction.rotation_model,
                    reconstruction_time,
                    resolved_topologies,
                    options[cases[0]],
                )
        
        # Get slabs
        slabs = {}
        for key, entries in slab_cases.items():
            slabs[key] = setup_parallel.get_slabs(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )
            
            # Copy DataFrames to other cases
            for entry in entries[1:]:
                slabs[entry] = slabs[key].copy()

        # Get points
        points = {}
        for key, entries in point_cases.items():
            points[key] = setup_parallel.get_points(
                        reconstruction,
                        reconstruction_time,
                        plates,
                        resolved_geometries,
                        options[key],
                    )

            # Copy DataFrames to other cases
            for entry in entries[1:]:
                points[entry] = points[key].copy()

        # Save all data to files
        for case in cases:
            # Save plates
            setup.DataFrame_to_parquet(
                plates,
                "Plates",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save slabs
            setup.DataFrame_to_parquet(
                slabs[case],
                "Slabs",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

            # Save points
            setup.DataFrame_to_parquet(
                points[case],
                "Points",
                reconstruction_name,
                reconstruction_time,
                case,
                files_dir,
            )

# if __name__ == "__main__":
#     # Set reconstruction times
#     reconstruction_times = _numpy.arange(31, 41, 1)

#     # Define path to directory containing reconstruction files
#     main_dir = _os.path.join("", "Users", "thomas", "Documents", "_Plato", "Reconstruction_analysis")

#     # Define the parameter settings for the model
#     settings_file = _os.path.join(main_dir, "settings.xlsx")

#     settings_sheet = "Sheet2"
    
#     M2016_model_dir = _os.path.join(main_dir, "M2016")

#     # Get start time
#     start_time = time()

#     # Set up PlateForces object
#     initialise_plato(
#         "Muller2016",
#         reconstruction_times,
#         settings_file,
#         cases_sheet = settings_sheet,
#         files_dir = _os.path.join("Output", M2016_model_dir, "Lr-Hb"),
#         rotation_file = _os.path.join("GPlates_files", "M2016", f"M2016_rotations_Lr-Hb.rot"),
#         topology_file = _os.path.join("GPlates_files", "M2016", "M2016_topologies.gpml"),
#         polygon_file = _os.path.join("GPlates_files", "M2016", "M2016_polygons.gpml"),
#     )

#     # Print the time taken to set up the model
#     set_up_time = time() - start_time
#     print(f"Time taken to set up the model: {set_up_time:.2e} seconds")


def load_data(
        _data: dict,
        _age: Union[float, int, _numpy.floating, _numpy.integer], 
        cases: List[str], 
        matching_cases: dict, 
        type: str,
        rotations: _pygplates.RotationModel,
        resolved_topologies: list,
        options: dict,
        data: Optional[Union[dict, str]] = None,
    ):
    """
    Function to load or initialise data for the current simulation.
    """
    # Load the data if it is available
    if data is not None:
        # Check if data is a dictionary
        if isinstance(data, dict):
            if _age in data.keys():
                # Initialise _age only if there's a matching case
                matching_cases_list = [_case for _case in cases if _case in data[_age].keys()]

                # Only initialise _data[_age] if there are matching cases
                if matching_cases_list:
                    _data[_age] = {}  # Initialise _age for matching cases

                for _case in matching_cases_list:
                    _data[_age][_case] = data[_age][_case]

                # Get missing cases
                existing_cases = set(data[_age].keys())
                missing_cases = _numpy.array([_case for _case in cases if _case not in existing_cases])

                # Loop through missing cases
                for _case in missing_cases:
                    # Check for matching case and copy data
                    matching_key = next((key for key, cases in matching_cases.items() if _case in cases), None)
                    if matching_key:
                        for matching_case in matching_cases[matching_key]:  # Access the cases correctly
                            if matching_case in _data[_age]:
                                # Copy data from the matching case
                                _data[_age][_case] = _data[_age][matching_case].copy()
                                break  # Exit after copying to avoid overwriting

                    else:
                        # Use the provided data retrieval function if no matching case is found
                        if type == "Plates":
                            if type == "Plates":
                                _data[_age][_case] = get_plate_data(
                                    rotations,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )
                            if type == "Slabs":
                                _data[_age][_case] = get_slab_data(
                                    rotations,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )
                            if type == "Points":
                                _data[_age][_case] = get_point_data(
                                    _data,
                                    _age,
                                    resolved_topologies,
                                    options
                                    )

        # TODO: Implement loading data from a file if needed
        if isinstance(data, str):
            pass

        # Initialise data if it is not available
        if _age not in _data.keys():
            _data[_age] = {}
            # Loop through cases
            for _case in cases:
                

    return _data

def get_data(
        rotations: _pygplates.RotationModel,
        resolved_topologies: list,
        options: dict,
        type: str,
    ):
    """
    Function to get data for the current simulation.

    :param rotations:           rotation model
    :type rotations:            _pygplates.RotationModel object
    :param resolved_topologies: resolved topologies
    :type resolved_topologies:  list of resolved topologies
    :param options:             options for the case
    :type options:              dict
    :param type:                type of data
    :type type:                 str

    :return:                    data
    :rtype:                     dict
    """
    if type == "Plates":
        data = get_plate_data(
            rotations,
            _age,
            resolved_topologies,
            options
            )
    if type == "Slabs":
        data = get_slab_data(
            rotations,
            _age,
            resolved_topologies,
            options
            )
    if type == "Points":
        data = get_point_data(
            _data,
            _age,
            resolved_topologies,
            options
            )
    
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
    for _age in _tqdm(ages, desc=f"Loading {type} grids", disable=(DEBUG_MODE, logging.getLogger().getEffectiveLevel() > logging.INFO)):
        # Check if the grid for the reconstruction time is already in the dictionary
        if _age in grid:
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
            grid[_age] = get_seafloor_grid(reconstruction_name, _age)

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
                                print(f"{type} grid for {reconstruction_name} at {_age} Ma not found, interpolating from points...")

                            # Get velocity grid
                            grid[_age][_case] = get_velocity_grid(points[_age][_case], seafloor_grid[_age])

    return grid
    
def load_data(
        data: dict,
        reconstruction_name: str,
        age: Union[list, _numpy.array],
        type: str,
        all_cases: list,
        matching_case_dict: dict,
        files_dir: Optional[str] = None,
        PARALLEL_MODE: Optional[bool] = False,
    ):
    """
    Function to load DataFrames from a folder, or initialise new DataFrames

    :return:                      data
    :rtype:                       dict
    """
    # Initialise list to store available and unavailable cases
    unavailable_cases = all_cases.copy()
    available_cases = []

    # If a file directory is provided, check for the existence of files
    if files_dir:
        for case in all_cases:
            # Load DataFrame if found
            data[case] = DataFrame_from_parquet(files_dir, type, reconstruction_name, case, age)

            if data[case] is not None:
                unavailable_cases.remove(case)
                available_cases.append(case)
            else:
                logging.info(f"DataFrame for {type} for {reconstruction_name} at {age} Ma for case {case} not found, checking for similar cases...")

    # Get available cases
    for unavailable_case in unavailable_cases:
        data[unavailable_case] = get_available_cases(data, unavailable_case, available_cases, matching_case_dict)

        if data[unavailable_case] is not None:
            available_cases.append(unavailable_case)

    return data

def get_available_cases(data, unavailable_case, available_cases, matching_case_dict):
    # Copy dataframes for unavailable cases
    matching_key = None

    # Find dictionary key of list in which unavailable case is located
    for key, matching_cases in matching_case_dict.items():
        for matching_case in matching_cases:
            if matching_case == unavailable_case:
                matching_key = key
                break
        if matching_key:
            break

    # Check if there is an available case in the corresponding list
    for matching_case in matching_case_dict[matching_key]:
        # Copy DataFrame if found
        if matching_case in available_cases:
            # Ensure that matching_case data is not None
            if data[matching_case] is not None:
                data[unavailable_case] = data[matching_case].copy()
                return data[unavailable_case]
            
            else:
                logging.info(f"Data for matching case '{matching_case}' is None; cannot copy to '{unavailable_case}'.")

    # If no matching case is found, return None
    data[unavailable_case] = None

    return data[unavailable_case]
    
    
def sample_slabs(
            self,
            ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            seafloor_grid: Optional[_xarray.Dataset] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age (and optionally, sediment thickness) the lower plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `lower_plate_age`, `sediment_thickness`, and `lower_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample slabs for
        :type _ages:     list
        :param cases:                   cases to sample slabs for (defaults to slab pull cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if ages is None:
            ages = self.settings.ages
        else:
            if isinstance(ages, str):
                ages = [ages]

        # Make iterable
        if cases is None:
            iterable = self.settings.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {_case: [] for _case in cases}

        # Check options for slabs
        for _age in _tqdm(ages, desc="Sampling slabs", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling slabs at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")
                    
                if self.options[key]["Slab pull torque"] or self.options[key]["Slab bend torque"]:
                    # Sample age and sediment thickness of lower plate from seafloor
                    self.data[_age][key]["lower_plate_age"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,
                        self.seafloor[_age], 
                        self.options[key],
                        "lower plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                        continental_arc=self.data[_age][key].continental_arc,
                    )

                    # Calculate lower plate thickness
                    self.data[_age][key]["lower_plate_thickness"], _, _ = utils_calc.compute_thicknesses(
                        self.data[_age][key].lower_plate_age,
                        self.options[key],
                        crust = False, 
                        water = False
                    )

                    # Calculate slab flux
                    self.plates[_age][key] = utils_calc.compute_subduction_flux(
                        self.plates[_age][key],
                        self.data[_age][key],
                        type="slab"
                    )

                    if self.options[key]["Sediment subduction"]:
                        # Calculate sediment subduction
                        self.plates[_age][key] = utils_calc.compute_subduction_flux(
                            self.plates[_age][key],
                            self.data[_age][key],
                            type="sediment"
                        )

                    if len(entries) > 1:
                        for entry in entries[1:]:
                            self.data[_age][entry]["lower_plate_age"] = self.data[_age][key]["lower_plate_age"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
                            self.data[_age][entry]["lower_plate_thickness"] = self.data[_age][key]["lower_plate_thickness"]

        # Set flag to True
        self.sampled_slabs = True

    def sample_upper_plates(
            self,
            _ages: Optional[Union[_numpy.ndarray, List, float, int]] = None,
            cases: Optional[Union[List[str], str]] = None,
            PROGRESS_BAR: Optional[bool] = True,    
        ):
        """
        Samples seafloor age the upper plate along subduction zones
        The results are stored in the `slabs` DataFrame, specifically in the `upper_plate_age`, `upper_plate_thickness` fields for each case and reconstruction time.

        :param _ages:    reconstruction times to sample upper plates for
        :type _ages:     list
        :param cases:                   cases to sample upper plates for (defaults to slab pull cases if not specified).
        :type cases:                    list
        :param PROGRESS_BAR:            whether or not to display a progress bar
        :type PROGRESS_BAR:             bool
        """
        # Define reconstruction times if not provided
        if _ages is None:
            _ages = self.settings.ages
        else:
            # Check if reconstruction times is a single value
            if isinstance(_ages, (int, float, _numpy.integer, _numpy.floating)):
                _ages = [_ages]
        
        # Make iterable
        if cases is None:
            iterable = self.slab_pull_cases
        else:
            if isinstance(cases, str):
                cases = [cases]
            iterable = {case: [] for case in cases}

        # Loop through valid times    
        for _age in tqdm(_ages, desc="Sampling upper plates", disable=(self.DEBUG_MODE or not PROGRESS_BAR)):
            if self.DEBUG_MODE:
                print(f"Sampling overriding plate at {_age} Ma")

            # Select cases
            for key, entries in iterable.items():
                if self.DEBUG_MODE:
                    print(f"Sampling overriding plate for case {key} and entries {entries}...")

                # Check whether to output erosion rate and sediment thickness
                if self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0 and self.options[key]["Sample erosion grid"] in self.seafloor[_age].data_vars:
                    # Sample age and arc type, erosion rate and sediment thickness of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"], self.data[_age][key]["erosion_rate"], self.data[_age][key]["sediment_thickness"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                        sediment_thickness=self.data[_age][key].sediment_thickness,
                    )

                elif self.options[key]["Sediment subduction"] and self.options[key]["Active margin sediments"] != 0:
                    # Sample age and arc type of upper plate from seafloor
                    self.data[_age][key]["upper_plate_age"], self.data[_age][key]["continental_arc"] = utils_calc.sample_slabs_from_seafloor(
                        self.data[_age][key].lat, 
                        self.data[_age][key].lon,
                        self.data[_age][key].trench_normal_azimuth,  
                        self.seafloor[_age],
                        self.options[key],
                        "upper plate",
                    )
                
                # Copy DataFrames to other cases
                if len(entries) > 1 and cases is None:
                    for entry in entries[1:]:
                        self.data[_age][entry]["upper_plate_age"] = self.data[_age][key]["upper_plate_age"]
                        self.data[_age][entry]["continental_arc"] = self.data[_age][key]["continental_arc"]
                        if self.options[key]["Sample erosion grid"]:
                            self.data[_age][entry]["erosion_rate"] = self.data[_age][key]["erosion_rate"]
                            self.data[_age][entry]["sediment_thickness"] = self.data[_age][key]["sediment_thickness"]
        
        # Set flag to True
        self.sampled_upper_plates = True

def get_globe_data(
        plates: dict,
        slabs: dict,
        points: dict,
        seafloor_grid: dict,
        ages: _numpy.array,
        case: str,
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
    num_plates = _numpy.zeros_like(ages)
    slab_length = _numpy.zeros_like(ages)
    v_rms_mag = _numpy.zeros_like(ages)
    v_rms_azi = _numpy.zeros_like(ages)
    mean_seafloor_age = _numpy.zeros_like(ages)

    for i, age in enumerate(ages):
        # Get number of plates
        num_plates[i] = len(plates[age][case].plateID.values)

        # Get slab length
        slab_length[i] = slabs[age][case].trench_segment_length.sum()

        # Get global RMS velocity
        # Get area for each grid point as well as total area
        areas = points[age][case].segment_length_lat.values * points[age][case].segment_length_lon.values
        total_area = _numpy.sum(areas)

        # Calculate RMS speed
        v_rms_mag[i] = _numpy.sum(points[age][case].v_mag * areas) / total_area

        # Calculate RMS azimuth
        v_rms_sin = _numpy.sum(_numpy.sin(points[age][case]["velocity_lat"]) * areas) / total_area
        v_rms_cos = _numpy.sum(_numpy.cos(points[age][case]["velocity_lat"]) * areas) / total_area
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