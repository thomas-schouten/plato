# Standard libraries
import warnings
from typing import List, Optional, Tuple, Union

# Third-party libraries
import gplately as _gplately
from gplately import pygplates as _pygplates
import cartopy.crs as ccrs
import numpy as _numpy
import matplotlib.pyplot as _plt
import xarray as _xarray
import cmcrameri

# Plato libraries
from . import utils_init
from .settings import Settings
from .plates import Plates
from .slabs import Slabs
from .points import Points
from .grids import Grids
from .globe import Globe
from .plate_torques import PlateTorques

class PlotReconstruction():
    """
    A class to make standardised plots of reconstructions.

    :param settings:        settings object
    :type settings:         Settings
    :param plates:          plates object
    :type plates:           Plates
    :param slabs:           slabs object
    :type slabs:            Slabs
    :param points:          points object
    :type points:           Points
    :param grids:           grids object
    :type grids:            Grids
    :param globe:           globe object
    :type globe:            Globe
    :param plate_torques:   plate torques object
    """
    def __init__(
            self,
            plate_torques: Optional[PlateTorques] = None,
            settings: Optional[Settings] = None,
            reconstruction: Optional[_gplately.PlateReconstruction] = None,
            plates: Optional[Plates] = None,
            slabs: Optional[Slabs] = None,
            points: Optional[Points] = None,
            grids: Optional[Grids] = None,
            globe: Optional[Globe] = None,
            coastline_file: Optional[str] = None,
        ):
        """
        Constructor for the Plot class.
        """
        # Store the input data, if provided
        if isinstance(settings, Settings):
            self.settings = settings
        else:
            self.settings = None

        if isinstance(reconstruction, _gplately.PlateReconstruction):
            self.reconstruction = reconstruction
        else:
            self.reconstruction = None

        if isinstance(plates, Plates):
            self.plates = plates
            if self.settings is None:
                self.settings = self.plates.settings
        else:
            self.plates = None

        if isinstance(slabs, Slabs):
            self.slabs = slabs
            if self.settings is None:
                self.settings = self.slabs.settings
        else:
            self.slabs = None

        if isinstance(points, Points):
            self.points = points
            if self.settings is None:
                self.settings = self.points.settings
        else:
            self.points = None

        if isinstance(grids, Grids):
            self.grids = grids
            if self.settings is None:
                self.settings = self.grids.settings
        else:
            self.grids = None

        if isinstance(globe, Globe):
            self.globe = globe
            if self.settings is None:
                self.settings = self.globe.settings
        else:
            self.globe = None

        if isinstance(plate_torques, PlateTorques):
            self.settings = plate_torques.settings
            self.reconstruction = plate_torques.reconstruction
            self.plates = plate_torques.plates
            self.slabs = plate_torques.slabs
            self.points = plate_torques.points
            self.grids = plate_torques.grids
            self.globe = plate_torques.globe

        # Set shortcut to ages, cases and options
        self.ages = self.settings.ages
        self.cases = self.settings.cases
        self.options = self.settings.options

        # Get coastlines if not provided
        self.coastlines = utils_init.get_coastlines(coastline_file, self.settings)

    def plot_seafloor_age_map(
            self,
            ax: object,
            age: int,
            cmap: str = "cmc.lajolla_r",
            vmin: Union[int, float] = 0,
            vmax: Union[int, float] = 250,
            alpha: Union[int, float] = 1,
            log_scale: bool = False,
            coastlines_facecolour: str = "lightgrey",
            coastlines_edgecolour: str = "lightgrey",
            coastlines_linewidth: Union[int, float] = 0,
            plate_boundaries_linewidth: Union[int, float] = 1,
        ) -> object:
        """
        Function to create subplot of the reconstruction with global seafloor age.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float

        :return:                        image object
        :rtype:                         matplotlib.image.AxesImage

        NOTE: This function does need a "case" argument because all cases use the same seafloor age grid.
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        if self.grids.seafloor_age[age] is not None and "seafloor_age" in self.grids.seafloor_age[age].data_vars:
            # ax.imshow(self.grids.seafloor_age[age].seafloor_age,)
            
            # Plot seafloor age grid
            im = self.plot_grid(
                ax,
                self.grids.seafloor_age[age].seafloor_age,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                log_scale = log_scale,
                alpha = alpha,
            )

        else:
            warnings.warn("No seafloor age grid available, only plotting the reconstruction")
            im = None

        # Plot plates and coastlines
        self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im

    def plot_sediment_map(
            self,
            ax: object,
            age: int,
            case: str,
            cmap: str = "cmc.imola",
            vmin: Union[int, float] = 1e0,
            vmax: Union[int, float] = 1e4,
            log_scale: bool = True,
            coastlines_facecolour: str = "lightgrey",
            coastlines_edgecolour: str = "lightgrey",
            coastlines_linewidth: Union[int, float] = 0,
            plate_boundaries_linewidth: Union[int, float] = 1,
            marker_size: Union[int, float] = 20,
        ):
        """
        Function to create subplot of the reconstruction with global sediment thicknesses.
        Seafloor sediments are plotted as the grid stored in the sediment grid with the key corresponding to the 'Sample sediment grid' option for the given case.
        Active margin sediments are plotted as scatter points if the 'Active margin sediments' option is activated.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case:                    case for which to plot the sediments
        :type case:                     str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float
        :param marker_size:             size of the markers
        :type marker_size:              int, float

        :return:                        image object and scatter object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.collections.PathCollection
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Get sediment thickness grid
        if self.grids.sediment[age] is not None and self.settings.options[case]["Sample sediment grid"] in self.grids.sediment[age].data_vars:           
            grid = self.grids.sediment[age][self.settings.options[case]["Sample sediment grid"]].values
        else:
            grid = _numpy.where(_numpy.isnan(self.grids.seafloor_age[age].seafloor_age.values), _numpy.nan, vmin)

        # Plot sediment thickness grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
            )

        # Plot active margin sediments, if activated
        if self.settings.options[case]["Active margin sediments"] != 0 or self.settings.options[case]["Sample erosion grid"]:
            lat = self.slabs.data[age][case].lat.values
            lon = self.slabs.data[age][case].lon.values
            data = self.slabs.data[age][case].sediment_thickness.values
            
            if log_scale is True:
                if vmin == 0:
                    vmin = 1e-3
                if vmax == 0:
                    vmax = 1e3
                vmin = _numpy.log10(vmin)
                vmax = _numpy.log10(vmax)

                data = _numpy.where(
                    data == 0,
                    vmin,
                    _numpy.log10(data),
                )

            sc = ax.scatter(
                lon,
                lat,
                c=data,
                s=marker_size,
                transform=ccrs.PlateCarree(),
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
            )
        
        else:  
            sc = None

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )
            
        return im, sc
    
    def plot_erosion_rate_map(
            self,
            ax: object,
            age: int = None,
            case: str = None,
            cmap = "cmc.davos_r",
            vmin: Union[int, float] = 0,
            vmax: Union[int, float] = 1e6,
            log_scale: bool = True,
            coastlines_facecolour: str = "none",
            coastlines_edgecolour: str = "none",
            coastlines_linewidth: Union[int, float] = 0,
            plate_boundaries_linewidth: Union[int, float] = 1,
        ):
        """
        Function to create subplot of the reconstruction with global erosion rates.
        Erosion rates are plotted as the grid stored in the continent grid with the key corresponding to the 'Sample erosion rate' option.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case:                    case for which to plot the sediments
        :type case:                     str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float

        :return:                        image object
        :rtype:                         matplotlib.image.AxesImage
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Get erosion rate grid
        if age in self.grids.continent and self.settings.options[case]["Sample erosion rate"] in self.grids.continent[age].data_vars:           
            grid = self.grids.continent[age].erosion_rate.values
            # Get erosion rate grid
            im = self.plot_grid(
                ax,
                grid,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                log_scale = log_scale
            )

        else:
            warnings.warn("No erosion rate grid available, only plotting the reconstruction")
            im = None
        
        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )
            
        return im
    
    def plot_LAB_depth_map(
            self,
            ax: object,
            age: int = None,
            case: str = None,
            cmap = "cmc.davos_r",
            vmin: Union[int, float] = 0,
            vmax: Union[int, float] = 2.5e2,
            log_scale: bool = False,
            coastlines_facecolour: str = "none",
            coastlines_edgecolour: str = "none",
            coastlines_linewidth: Union[int, float] = 0,
            plate_boundaries_linewidth: Union[int, float] = 1,
        ):
        """
        Function to create subplot of the reconstruction with continental lithosphere-asthenosphere boundary (LAB) depths in km.
        Erosion rates are plotted as the grid stored in the continent grid with the key corresponding to the 'Sample erosion rate' option.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case:                    case for which to plot the sediments
        :type case:                     str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param log_scale:               whether or not to use log scale (default is False)
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float

        :return:                        image object
        :rtype:                         matplotlib.image.AxesImage
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Get LAB depth grid
        if age in self.grids.continent: 

            data = self.points.data[age][case]

            grid = _xarray.Dataset(
                {
                    "LAB_depth": _xarray.DataArray(
                        data=data.LAB_depth.values.reshape(
                            data.lat.unique().size, data.lon.unique().size
                        ),
                        coords={
                            "lat": data.lat.unique(),
                            "lon": data.lon.unique(),
                        },
                        dims=["lat", "lon"],
                    )
                },
                coords={
                    "lat": (["lat"], data.lat.unique()),
                    "lon": (["lon"], data.lon.unique()),
                },
            )
            grid = grid.interp_like(self.grids.seafloor_age[age], method="spline")

            im = self.plot_grid(
                ax,
                grid.LAB_depth.values/1e3,
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
                log_scale = log_scale
            )

        else:
            warnings.warn("No continental grid available, only plotting the reconstruction")
            im = None
        
        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )
            
        return im
    
    def plot_velocity_map(
            self,
            ax,
            age,
            case = None,
            velocity_component = "velocity_mag",
            plateIDs = None,
            cmap = "cmc.bilbao_r",
            vmin = 0,
            vmax = 25,
            normalise_vectors = False,
            log_scale = False,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "black",
            coastlines_linewidth = 0.1,
            plate_boundaries_linewidth = 1,
            vector_width = 4e-3,
            vector_scale = 3e2,
            vector_scale_units = "width",
            vector_color = "k",
            vector_alpha = 0.5,
            NET_ROTATION_POLE = False,
        ):
        """
        Function to create subplot of the reconstruction with global plate velocities.
        The plate velocities are plotted by the combination of a grid showing the magnitude of a velocity component and a set of vectors showing the direction of the velocity.
        The velocity component can be either the magnitude of the translational component (i.e. the velocity magnitude) or 
        one of the two components of the velocity vector (i.e. the latitude or longitude component) or the rotational component (i.e. the spin rate).

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case:                    case for which to use the velocities
        :type case:                     str
        :param velocity_component:      velocity component to plot
        :type velocity_component:       str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param normalise_vectors:       whether or not to normalise the vectors
        :type normalise_vectors:        bool
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float

        :return:                        image object and quiver object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Extract data for the given age and case
        point_data = {age: {case: self.points.data[age][case].copy()}}

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = point_data[age][case]["plateID"].isin(plateIDs)
            point_data[age][case].loc[~mask, velocity_component] = _numpy.nan

        # Interpolate data from points to grid if not available
        self.grids.generate_velocity_grid(
            age,
            case,
            point_data,
            velocity_component,
            PROGRESS_BAR=False,
        )
        
        # Get velocity grid
        grid = self.grids.velocity[age][case][velocity_component].values
        
        # Plot velocity difference grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Get velocity vectors
        velocity_vectors = self.points.data[age][case].iloc[::208].copy()

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = velocity_vectors["plateID"].isin(plateIDs)
            velocity_vectors.loc[~mask, "velocity_lat"] = _numpy.nan
            velocity_vectors.loc[~mask, "velocity_lon"] = _numpy.nan
            velocity_vectors.loc[~mask, "velocity_mag"] = _numpy.nan

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors.lat.values,
            velocity_vectors.lon.values,
            velocity_vectors.velocity_lat.values,
            velocity_vectors.velocity_lon.values,
            velocity_vectors.velocity_mag.values,
            normalise_vectors = normalise_vectors,
            width = vector_width,
            scale = vector_scale,
            scale_units = vector_scale_units,
            facecolour = vector_color,
            alpha = vector_alpha
        )
        
        # Plot Euler pole of net lithospheric rotation, if necessary
        if NET_ROTATION_POLE:
            for i in range(2):
                _globe_data = self.globe.data[case]
                _index = _numpy.where(_globe_data.age.values == age)[0][0]
                ax.scatter(
                    _globe_data.net_rotation_pole_lon.values[_index] if i == 0 else (_globe_data.net_rotation_pole_lon.values[_index] + 180) % 360,
                    _globe_data.net_rotation_pole_lat.values[_index] if i == 0 else -_globe_data.net_rotation_pole_lat.values[_index],
                    transform=ccrs.PlateCarree(),
                    s=200,
                    marker="*",
                    c="k",
                )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu

    def plot_velocity_difference_map(
            self,
            ax,
            age,
            case1,
            case2,
            velocity_component = "velocity_mag",
            plateIDs = None,
            cmap = "cmc.vik",
            vmin = -10,
            vmax = 10,
            normalise_vectors = False,
            log_scale = False,
            coastlines_facecolour = "none",
            coastlines_edgecolour = "black",
            coastlines_linewidth = 0.1,
            plate_boundaries_linewidth = 1,
            vector_width = 4e-3,
            vector_scale = 3e2,
            vector_color = "k",
            vector_alpha = 0.5,
        ):
        """
        Function to create subplot of the reconstruction with the difference in global plate velocities for two cases.
        The difference in plate velocities is plotted by subtracting the velocity vectors of case 2 from the velocity vectors of case 1.
        The plate velocities are plotted by the combination of a grid showing the magnitude of a velocity component and a set of vectors showing the direction of the velocity.
        The velocity component can be either the magnitude of the translational component (i.e. the velocity magnitude) or
        one of the two components of the velocity vector (i.e. the latitude or longitude component) or the rotational component (i.e. the spin rate).
    
        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case1:                   case for which to use the velocities
        :type case1:                    str
        :param case2:                   case for which to use the velocities
        :type case2:                    str
        :param velocity_component:      velocity component to plot
        :type velocity_component:       str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param normalise_vectors:       whether or not to normalise the vectors
        :type normalise_vectors:        bool
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float
        :param vector_width:            width of the vectors
        :type vector_width:             int, float
        :param vector_scale:            scale of the vectors
        :type vector_scale:             int, float
        :param vector_color:            color of the vectors
        :type vector_color:             str
        :param vector_alpha:            alpha of the vectors
        :type vector_alpha:             int, float

        :return:                        image object and quiver object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case1 is None or case1 not in self.settings.cases:
            warnings.warn("Invalid case a, using first case.")
            case1 = self.settings.cases[0]
        
        if case2 is None or case2 not in self.settings.cases:
            if self.settings.cases[1] is not None:
                warnings.warn("Invalid case b, using second case.")
                case2 = self.settings.cases[1]
            else:
                raise ValueError("No second case provided.")
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Extract data for the given age and case
        point_data1 = {age: {case1: self.points.data[age][case1].copy()}}
        point_data2 = {age: {case2: self.points.data[age][case2].copy()}}

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = point_data1[age][case1]["plateID"].isin(plateIDs)
            point_data1[age][case1].loc[~mask, velocity_component] = _numpy.nan
            point_data2[age][case2].loc[~mask, velocity_component] = _numpy.nan

        # Interpolate data from points to grid if not available
        self.grids.generate_velocity_grid(
            age,
            case1,
            point_data1,
            velocity_component,
        )
        
        self.grids.generate_velocity_grid(
            age,
            case2,
            point_data2,
            velocity_component,
        )

        # Get velocity difference grid
        grid = self.grids.velocity[age][case1][velocity_component].values-self.grids.velocity[age][case2][velocity_component].values

        # Plot velocity difference grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Subsample velocity vectors
        velocity_vectors1 = self.points.data[age][case1].iloc[::208].copy()
        velocity_vectors2 = self.points.data[age][case2].iloc[::208].copy()

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = velocity_vectors1["plateID"].isin(plateIDs)
            velocity_vectors1.loc[~mask, "velocity_lat"] = _numpy.nan
            velocity_vectors1.loc[~mask, "velocity_lon"] = _numpy.nan
            velocity_vectors1.loc[~mask, "velocity_mag"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_lat"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_lon"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_mag"] = _numpy.nan

        # Remove NaN and zero values
        velocity_vectors1 = velocity_vectors1.dropna(subset = ["velocity_lat", "velocity_lon", velocity_component])
        velocity_vectors2 = velocity_vectors2.dropna(subset = ["velocity_lat", "velocity_lon", velocity_component]) 

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors1.lat.values,
            velocity_vectors1.lon.values,
            velocity_vectors1.velocity_lat.values - velocity_vectors2.velocity_lat.values,
            velocity_vectors1.velocity_lon.values - velocity_vectors2.velocity_lon.values,
            velocity_vectors1[velocity_component].values - velocity_vectors2[velocity_component].values,
            normalise_vectors = normalise_vectors,
            width = vector_width,
            scale = vector_scale,
            facecolour = vector_color,
            alpha = vector_alpha
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu
    
    def plot_relative_velocity_difference_map(
            self,
            ax,
            age,
            case1,
            case2,
            velocity_component: str = "velocity_mag",
            plateIDs: Union[list, None] = None,
            cmap: str = "cmc.cork",
            vmin: Union[int, float] = 1e-1,
            vmax: Union[int, float] = 1e1,
            log_scale: bool = True,
            coastlines_facecolour: str = "none",
            coastlines_edgecolour: str = "black",
            coastlines_linewidth: Union[int, float] = 0.1,
            plate_boundaries_linewidth: Union[int, float] = 1,
            vector_width: Union[int, float] = 4e-3,
            vector_scale: Union[int, float] = 3e2,
            vector_color: str = "k",
            vector_alpha: Union[int, float] = 0.5,
        ):
        """
        Function to create subplot of the reconstruction with the relative difference in global plate velocities for two cases.
        The difference in plate velocities is plotted by dividing the velocity vectors of case 2 from the velocity vectors of case 1.
        The plate velocities are plotted by the combination of a grid showing the magnitude of a velocity component and a set of vectors showing the direction of the velocity.
        The velocity component can be either the magnitude of the translational component (i.e. the velocity magnitude) or
        one of the two components of the velocity vector (i.e. the latitude or longitude component) or the rotational component (i.e. the spin rate).
    
        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case1:                   case for which to use the velocities
        :type case1:                    str
        :param case2:                   case for which to use the velocities
        :type case2:                    str
        :param velocity_component:      velocity component to plot
        :type velocity_component:       str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param normalise_vectors:       whether or not to normalise the vectors
        :type normalise_vectors:        bool
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float
        :vector_width:                 width of the vectors
        :type vector_width:            int, float
        :vector_scale:                 scale of the vectors
        :type vector_scale:            int, float
        :vector_color:                 color of the vectors
        :type vector_color:            str
        :vector_alpha:                 transparency of the vectors
        :type vector_alpha:            int, float

        :return:                        image object and quiver object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case1 is None or case1 not in self.settings.cases:
            warnings.warn("Invalid case a, using first case.")
            case1 = self.settings.cases[0]
        
        if case2 is None or case2 not in self.settings.cases:
            if self.settings.cases[1] is not None:
                warnings.warn("Invalid case b, using second case.")
                case2 = self.settings.cases[1]
            else:
                raise ValueError("No second case provided.")
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Extract data for the given age and case
        point_data1 = {age: {case1: self.points.data[age][case1].copy()}}
        point_data2 = {age: {case2: self.points.data[age][case2].copy()}}

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = point_data1[age][case1]["plateID"].isin(plateIDs)
            point_data1[age][case1].loc[~mask, velocity_component] = _numpy.nan
            point_data2[age][case2].loc[~mask, velocity_component] = _numpy.nan

        # Interpolate data from points to grid if not available
        self.grids.generate_velocity_grid(
            age,
            case1,
            point_data1,
            velocity_component,
        )
    
        self.grids.generate_velocity_grid(
            age,
            case2,
            point_data2,
            velocity_component,
        )

        # Get relative velocity difference grid
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = _numpy.where(
                (self.grids.velocity[age][case1][velocity_component].values == 0) | 
                (self.grids.velocity[age][case2][velocity_component].values == 0) | 
                (_numpy.isnan(self.grids.velocity[age][case1][velocity_component].values)) | 
                (_numpy.isnan(self.grids.velocity[age][case2][velocity_component].values)),
                _numpy.nan,
                (self.grids.velocity[age][case1][velocity_component].values / 
                _numpy.where(
                    self.grids.velocity[age][case2][velocity_component].values == 0,
                    1e-10,
                    self.grids.velocity[age][case2][velocity_component].values)
                )
            )

        # Set velocity grid
        im = self.plot_grid(
            ax,
            grid,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            log_scale = log_scale
        )

        # Get velocity vectors
        velocity_vectors1 = self.points.data[age][case1].iloc[::208].copy()
        velocity_vectors2 = self.points.data[age][case2].iloc[::208].copy()

        # Filter velocity vectors by plateIDs
        if plateIDs is not None:
            mask = velocity_vectors1["plateID"].isin(plateIDs)
            velocity_vectors1.loc[~mask, "velocity_lat"] = _numpy.nan
            velocity_vectors1.loc[~mask, "velocity_lon"] = _numpy.nan
            velocity_vectors1.loc[~mask, "velocity_mag"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_lat"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_lon"] = _numpy.nan
            velocity_vectors2.loc[~mask, "velocity_mag"] = _numpy.nan


        vector_lat = velocity_vectors1.velocity_lat.values - velocity_vectors2.velocity_lat.values
        vector_lon = velocity_vectors1.velocity_lon.values - velocity_vectors2.velocity_lon.values
        vector_mag = _numpy.sqrt(vector_lat**2 + vector_lon**2)

        # Plot velocity vectors
        qu = self.plot_vectors(
            ax,
            velocity_vectors1.lat.values,
            velocity_vectors1.lon.values,
            vector_lat,
            vector_lon,
            vector_mag,
            normalise_vectors = True,
            width = vector_width,
            scale = vector_scale,
            facecolour = vector_color,
            alpha = vector_alpha
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return im, qu
    
    def plot_torques_map(
            self,
            ax: object,
            age: int,
            case: str = None,
            cmap: str = "cmc.lajolla_r",
            vmin: Union[int, float] = 0,
            vmax: Union[int, float] = 250,
            alpha: Union[int, float] = 0.5,
            log_scale: bool = False,
            normalise_vectors: bool = False,
            coastlines_facecolour: str = "lightgrey",
            coastlines_edgecolour: str = "lightgrey",
            coastlines_linewidth: Union[int, float] = 0,
            plate_boundaries_linewidth: Union[int, float] = 1,
            vector_width: Union[int, float] = 8e-3,
            vector_scale: Union[int, float] = 4.5e27,
            vector_scale_units: str = "width",
            vector_cmap: str = "cmc.glasgow_r",
            vector_linewidth: Union[int, float] = .5,
            vector_edgecolour: str = "k",
            minimum_plate_area: Optional[Union[int, float]] = None,
            plateIDs: Optional[Union[int, float, List[Union[int, float]]]] = None,
        ) -> Tuple[object, object]:
        """
        Function to create subplot of the reconstruction with the torques acting on the plates.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case:                    case for which to use the velocities
        :type case:                     str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param alpha:                   transparency of the colormap
        :type alpha:                    int, float
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param normalise_vectors:       whether or not to normalise magnitude of torque vector by the magnitude of the driving torque
        :type normalise_vectors:        bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float
        :param vector_width:            width of the vectors
        :type vector_width:             int, float
        :param vector_scale:            scale of the vectors
        :type vector_scale:             int, float
        :param vector_scale_units:      units of the vector scale
        :type vector_scale_units:       str
        :param vector_cmap:             colormap for the vectors
        :type vector_cmap:              str
        :param vector_linewidth:        linewidth of the vectors
        :type vector_linewidth:         int, float
        :param vector_edgecolour:       edgecolour of the vectors
        :type vector_edgecolour:        str
        :param minimum_plate_area:      minimum area for the plates
        :type minimum_plate_area:       int, float
        :param plateIDs:                list of plate IDs to plot
        :type plateIDs:                 int, float, list

        :return:                        image object and quiver object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]

        # Plot seafloor age grid
        im = self.plot_seafloor_age_map(
            ax,
            age,
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            alpha = alpha,
            log_scale = log_scale,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        # Get colours for vectors
        colours = _plt.get_cmap(vector_cmap)(_numpy.linspace(0, 1, 7))

        # Get plate data
        plate_data = self.plates.data[age][case].copy()

        # Subselect plates by plateID
        if plateIDs is not None:
            if isinstance(plateIDs, (int, float)):
                plateIDs = [plateIDs]
            plate_data = plate_data[plate_data.plateID.isin(plateIDs)]

        # Subselect plates with a minimum area
        if minimum_plate_area is not None:
            plate_data = plate_data[plate_data.area >= minimum_plate_area]

        elif self.settings.options["Minimum plate area"] > 0:
            plate_data = plate_data[plate_data.area >= self.settings.options["Minimum plate area"]]

        # Initialise dictionary to store quiver objects
        qu = {}

        # Loop over the different torque components
        for i, torque in enumerate(_numpy.flip(["residual", "slab_pull", "slab_suction", "GPE", "slab_bend", "mantle_drag"])):
            # Only plot vectors if the mean value is larger than 0 (i.e. the torque has been calculated)
            if plate_data[f"{torque}_force_mag"].mean() > 0:
                qu[torque] = self.plot_vectors(
                    ax,
                    plate_data.centroid_lat.values,
                    plate_data.centroid_lon.values,
                    plate_data[f"{torque}_force_lat"].values,
                    plate_data[f"{torque}_force_lon"].values,
                    plate_data[f"driving_force_mag"].values,
                    normalise_vectors = normalise_vectors,
                    width = vector_width,
                    scale = vector_scale,
                    scale_units = vector_scale_units,
                    facecolour = colours[i+2],
                    alpha = 1.,
                    linewidth = vector_linewidth,
                    edgecolour = vector_edgecolour,
                )

        return im, qu

    def plot_residual_force_map(
            self,
            ax,
            age,
            case = None,
            type = "slabs",
            trench_means = False,
            cmap = "cmc.lipari_r",
            vmin = 1e-2,
            vmax = 1e-1,
            normalise_data = True,
            normalise_vectors = True,
            log_scale = True,
            marker_size = 30,
            coastlines_facecolour = "lightgrey",
            coastlines_edgecolour = "lightgrey",
            coastlines_linewidth = 0,
            plate_boundaries_linewidth = 1,
            slab_vector_width = 2e-3,
            slab_vector_scale = 3e2,
            slab_vector_colour = "k",
            slab_vector_alpha = 1,
            plate_vector_width = 5e-3,
            plate_vector_scale = 3e2,
            plate_vector_facecolour = "white",
            plate_vector_edgecolour = "k",
            plate_vector_linewidth = 1,
            plate_vector_alpha = 1,
        ) -> Tuple[object, object]:
        """
        Function to create subplot of the reconstruction with the residual force acting on the plates.
        The residual force is displayed as a vector component (latitudinal, longitudinal, or magnitude), which can be plotted as a grid of the force acting on the points or as scatter points showing the force acting on the slabs.
            
        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param age:                     the age for which to display the map
        :type age:                      int
        :param case1:                   case for which to use the velocities
        :type case1:                    str
        :param case2:                   case for which to use the velocities
        :type case2:                    str
        :param velocity_component:      velocity component to plot
        :type velocity_component:       str
        :param cmap:                    colormap to use
        :type cmap:                     str
        :param vmin:                    minimum value for colormap
        :type vmin:                     int, float
        :param vmax:                    maximum value for colormap
        :type vmax:                     int, float
        :param normalise_vectors:       whether or not to normalise the vectors
        :type normalise_vectors:        bool
        :param log_scale:               whether or not to use log scale
        :type log_scale:                bool
        :param coastlines_facecolour:   facecolour for coastlines
        :type coastlines_facecolour:    str
        :param coastlines_edgecolour:   edgecolour for coastlines
        :type coastlines_edgecolour:    str
        :param coastlines_linewidth:    linewidth for coastlines
        :type coastlines_linewidth:     int, float
        :param plate_boundaries_linewidth: linewidth for plate boundaries
        :type plate_boundaries_linewidth: int, float
        :param slab_vector_width:       width of the slab vectors
        :type slab_vector_width:        int, float
        :param slab_vector_scale:       scale of the slab vectors
        :type slab_vector_scale:        int, float
        :param slab_vector_colour:      colour of the slab vectors
        :type slab_vector_colour:       str
        :param slab_vector_alpha:       transparency of the slab vectors
        :type slab_vector_alpha:        int, float
        :param plate_vector_width:      width of the plate vectors
        :type plate_vector_width:       int, float
        :param plate_vector_scale:      scale of the plate vectors
        :type plate_vector_scale:       int, float
        :param plate_vector_facecolour: facecolour of the plate vectors
        :type plate_vector_facecolour:  str
        :param plate_vector_edgecolour: edgecolour of the plate vectors
        :type plate_vector_edgecolour:  str
        :param plate_vector_linewidth:  linewidth of the plate vectors
        :type plate_vector_linewidth:   int, float
        :param plate_vector_alpha:      transparency of the plate vectors
        :type plate_vector_alpha:       int, float

        :return:                        image object and quiver object
        :rtype:                         matplotlib.image.AxesImage and matplotlib.quiver.Quiver
        """
        # Set age to first in list if not provided
        if age is None or age not in self.settings.ages:
            warnings.warn("Invalid reconstruction age, using youngest age.")
            age = self.settings.ages[0]
        
        # Set case to first in list if not provided
        if case is None or case not in self.settings.cases:
            warnings.warn("Invalid case, using first case.")
            case = self.settings.cases[0]
        
        # Set basemap
        gl = self.plot_basemap(ax)

        # NOTE: We need to explicitly turn of top and right labels here, otherwise they will still show up sometimes
        gl.top_labels = False
        gl.right_labels = False

        # Copy dataframe
        if type == "slabs":
            plot_data = self.slabs.data[age][case].copy()
        elif type == "points":
            plot_data = self.points.data[age][case].copy()
        
        plot_plates = self.plates.data[age][case].copy()

        # Calculate means at trenches for the "residual_force_mag" column
        if type == "slabs" and trench_means is True:
            # Calculate the mean of "residual_force_mag" for each trench_plateID
            mean_values = plot_data.groupby("trench_plateID")["residual_force_mag"].transform("mean")
            
            # Assign the mean values back to the original "residual_force_mag" column
            plot_data["residual_force_mag"] = mean_values

        # Reorder entries to make sure the largest values are plotted on top
        plot_data = plot_data.sort_values("residual_force_mag", ascending=True)

        # Normalise data by dividing by the slab pull force magnitude
        slab_data = plot_data.residual_force_mag.values
        plate_data_lat = plot_plates.residual_force_lat.values
        plate_data_lon = plot_plates.residual_force_lon.values
        plate_data_mag = plot_plates.residual_force_mag.values

        if normalise_data is True:
            normalise_col = "slab_pull_force_mag" if type == "slabs" else "mantle_drag_force_mag"
            slab_data = _numpy.where(
                plot_data[normalise_col].values == 0 | _numpy.isnan(plot_data[normalise_col].values),
                0,
                slab_data / plot_data[normalise_col].values
            )

            plate_data_lat = _numpy.where(
                plot_plates[normalise_col].values == 0 | _numpy.isnan(plot_plates[normalise_col].values),
                0,
                plate_data_lat / plot_plates[normalise_col].values
            )

            plate_data_lon = _numpy.where(
                plot_plates[normalise_col].values == 0 | _numpy.isnan(plot_plates[normalise_col].values),
                0,
                plate_data_lon / plot_plates[normalise_col].values
            )

            plate_data_mag = _numpy.where(
                plot_plates[normalise_col].values == 0 | _numpy.isnan(plot_plates[normalise_col].values),
                0,
                plate_data_mag / plot_plates[normalise_col].values
            )
            
        # Convert to log scale, if needed
        if log_scale is True:
            if vmin == 0:
                vmin = 1e-3
            if vmax == 0:
                vmax = 1e3
            vmin = _numpy.log10(vmin)
            vmax = _numpy.log10(vmax)

            slab_data = _numpy.where(
                slab_data == 0 | _numpy.isnan(slab_data),
                vmin,
                _numpy.log10(slab_data),
            )

            # plate_data_lat = _numpy.where(
            #     plate_data_lat == 0 | _numpy.isnan(plate_data_lat),
            #     0,
            #     _numpy.log10(plate_data_lat),
            # )

            # plate_data_lon = _numpy.where(
            #     plate_data_lon == 0 | _numpy.isnan(plate_data_lon),
            #     0,
            #     _numpy.log10(plate_data_lon),
            # )

        # Plot velocity difference grid
        sc = ax.scatter(
                plot_data.lon.values,
                plot_data.lat.values,
                c = slab_data,
                s = marker_size,
                transform = ccrs.PlateCarree(),
                cmap = cmap,
                vmin = vmin,
                vmax = vmax,
            )

        # Get velocity vectors
        force_vectors = self.slabs.data[age][case].iloc[::5].copy()

        # Plot velocity vectors
        slab_qu = self.plot_vectors(
            ax,
            force_vectors.lat.values,
            force_vectors.lon.values,
            force_vectors.residual_force_lat.values,
            force_vectors.residual_force_lon.values,
            force_vectors.residual_force_mag.values,
            normalise_vectors = normalise_vectors,
            width = slab_vector_width,
            scale = slab_vector_scale,
            facecolour = slab_vector_colour,
            alpha = slab_vector_alpha
        )

        # Plot residual torque vectors at plate centroids
        plate_qu = self.plot_vectors(
            ax,
            plot_plates.centroid_lat.values,
            plot_plates.centroid_lon.values,
            plate_data_lat,
            plate_data_lon,
            plate_data_mag,
            normalise_vectors = normalise_vectors,
            width = plate_vector_width,
            scale = plate_vector_scale,
            facecolour = plate_vector_facecolour,
            edgecolour = plate_vector_edgecolour,
            linewidth = plate_vector_linewidth,
            alpha = plate_vector_alpha,
            zorder = 10
        )

        # Plot plates and coastlines
        ax = self.plot_reconstruction(
            ax,
            age,
            coastlines_facecolour = coastlines_facecolour,
            coastlines_edgecolour = coastlines_edgecolour,
            coastlines_linewidth = coastlines_linewidth,
            plate_boundaries_linewidth = plate_boundaries_linewidth,
        )

        return sc, slab_qu, plate_qu
    
    def plot_basemap(
            self,
            ax: object,
        ) -> object:
        """
        Function to plot a basemap on an axes object.

        :param ax:          axes object
        :type ax:           matplotlib.axes.Axes

        :return:        gridlines object
        :rtype:         cartopy.mpl.gridliner.Gridliner
        """
        # Set labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Set gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), 
            draw_labels=True, 
            linewidth=0.5, 
            color="gray", 
            alpha=0.5, 
            linestyle=":", 
            zorder=5
        )

        # Turn off gridlabels for top and right
        gl.top_labels = False
        gl.right_labels = False  

        return gl
    
    def plot_grid(
            self,
            ax: object,
            grid: Union[_numpy.ndarray, _xarray.DataArray],
            log_scale: bool = False,
            vmin: Union[int, float] = 0,
            vmax: Union[int, float] = 1e3,
            cmap: str = "viridis",
            alpha: Union[int, float] = 1,
        ) -> object:
        """
        Function to plot a grid.

        :param ax:          axes object
        :type ax:           matplotlib.axes.Axes
        :param data:        data to plot
        :type data:         numpy.ndarray
        :param log_scale:   whether or not to use log scale
        :type log_scale:    bool
        :param vmin:        minimum value for colormap
        :type vmin:         float
        :param vmax:        maximum value for colormap
        :type vmax:         float
        :param cmap:        colormap to use
        :type cmap:         str

        :return:            image object
        :rtype:             matplotlib.image.AxesImage
        """
        # Set log scale
        if log_scale:
            if vmin == 0:
                vmin = 1e-3
            if vmax == 0:
                vmax = 1e3
            vmin = _numpy.log10(vmin)
            vmax = _numpy.log10(vmax)

            # Ignore annoying warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid = _numpy.where(
                    (_numpy.isnan(grid)) | (grid <= 0),
                    _numpy.nan,
                    _numpy.log10(grid),
                )

        # Transform data from PlateCarree if necessary
        transform = None if ax.projection == ccrs.PlateCarree() else ccrs.PlateCarree()

        # Plot grid    
        im = ax.imshow(
            grid,
            cmap = cmap,
            transform = transform, 
            zorder = 1, 
            vmin = vmin, 
            vmax = vmax, 
            origin = "lower",
            extent = [-180, 180, -90, 90],
            alpha = alpha,
        )

        return im
    
    def plot_vectors(
            self,
            ax,
            lat,
            lon,
            vector_lat,
            vector_lon,
            vector_mag = None,
            normalise_vectors = False,
            width = 4e-3,
            scale = 3e2,
            scale_units = "width",
            facecolour = "k",
            edgecolour = "none",
            linewidth = 1,
            alpha = 0.5,
            zorder = 4,
        ):
        """
        Function to plot vectors on an axes object.

        :param ax:                  axes object
        :type ax:                   matplotlib.axes.Axes
        :param lat:                 latitude of the vectors
        :type lat:                  numpy.ndarray
        :param lon:                 longitude of the vectors
        :type lon:                  numpy.ndarray
        :param vector_lat:          latitude component of the vectors
        :type vector_lat:           numpy.ndarray
        :param vector_lon:          longitude component of the vectors
        :type vector_lon:           numpy.ndarray
        :param vector_mag:          magnitude of the vectors
        :type vector_mag:           numpy.ndarray
        :param normalise_vectors:   whether or not to normalise the vectors
        :type normalise_vectors:    bool
        :param width:               width of the vectors
        :type width:                float
        :param scale:               scale of the vectors
        :type scale:                float
        :param zorder:              zorder of the vectors
        :type zorder:               int
        :param color:               color of the vectors
        :type color:                str
        :param alpha:               transparency of the vectors
        :type alpha:                float

        :return:                    quiver object
        :rtype:                     matplotlib.quiver.Quiver
        """
        # Normalise vectors, if necessary
        if normalise_vectors and vector_mag is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Normalise by dividing by the magnitude of the vectors
                # Multiply by 10 to make the vectors more visible
                vector_lon = vector_lon / vector_mag * 10
                vector_lat = vector_lat / vector_mag * 10

        # Plot vectors
        # Ignore annoying warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qu = ax.quiver(
                    x = lon,
                    y = lat,
                    u = vector_lon,
                    v = vector_lat,
                    transform = ccrs.PlateCarree(),
                    width = width,
                    scale = scale,
                    scale_units = scale_units,
                    zorder = zorder,
                    color = facecolour,
                    edgecolor = edgecolour,
                    alpha = alpha,
                    linewidth = linewidth,
                )
        
        return qu
        
    def plot_reconstruction(
            self,
            ax: object,
            age: int, 
            coastlines_facecolour: str = "none",
            coastlines_edgecolour: str = "none",
            coastlines_linewidth: str = "none",
            plate_boundaries_linewidth: str = "none",
        ):
        """
        Function to plot reconstructed features: coastlines, plates and trenches.

        :param ax:                      axes object
        :type ax:                       matplotlib.axes.Axes
        :param _age:     the time for which to display the map
        :type _age:      int
        :param plotting_options:        options for plotting
        :type plotting_options:         dict
        :param coastlines:              whether or not to plot coastlines
        :type coastlines:               boolean
        :param plates:                  whether or not to plot plates
        :type plates:                   boolean
        :param trenches:                whether or not to plot trenches
        :type trenches:                 boolean
        :param default_frame:           whether or not to use the default reconstruction
        :type default_frame:            boolean

        :return:                        axes object with plotted features
        :rtype:                         matplotlib.axes.Axes
        """
        # Set gplot object
        gplot = _gplately.PlotTopologies(self.reconstruction, time=age, coastlines=self.coastlines)

        # Set zorder for coastlines. They should be plotted under seafloor grids but above velocity grids.
        if coastlines_facecolour == "none":
            zorder_coastlines = 2
        else:
            zorder_coastlines = -5

        # Plot coastlines
        # NOTE: Some reconstructions on the GPlately DataServer do not have polygons for coastlines, that's why we need to catch the exception.
        try:
            gplot.plot_coastlines(
                ax,
                facecolor = coastlines_facecolour,
                edgecolor = coastlines_edgecolour,
                zorder = zorder_coastlines,
                lw = coastlines_linewidth
            )
        except:
            pass
        
        # Plot plates 
        if plate_boundaries_linewidth:
            gplot.plot_all_topologies(ax, lw=plate_boundaries_linewidth, zorder=4)
            gplot.plot_subduction_teeth(ax, zorder=4)
            
        return ax
