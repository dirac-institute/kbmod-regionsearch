"""
lincc-frameworks provides implementations of ``abstractions.ObserverationIndexer`` that may be composed with an implementation of ``abstractions.Backend`` to provide a complete region search.
"""
import math

import numpy
from astropy import coordinates  # type: ignore
from astropy import units as units
from astropy.coordinates import (  # type: ignore
    EarthLocation,
    SkyCoord,
    solar_system_ephemeris,
)
from astropy.time import Time  # type: ignore

from kbmod.regionsearch.abstractions import ObservationIndexer


class PartitionIndexer(ObservationIndexer):
    """
    Partitions the observations into those that intersect a configured cone and those that do not.
    The cone is defined by an ra, dec and field of view angle with an origin at the solar system barycenter.
    The observation cones are similiarly defined by an ra, dec, and field of view angle with an origin given by the observation location and time.
    The observations are assigned an index of is_in_index if they intersect the cone and an index of is_out_index if they do not.
    """

    def __init__(
        self,
        search_ra: coordinates.Longitude,
        search_dec: coordinates.Latitude,
        search_distance: coordinates.Distance,
        search_fov: coordinates.Angle,
        is_in_index: int,
        is_out_index: int,
        **kwargs
    ):
        """Initializes the PartitionIndexer object.

        Parameters
        ----------

        search_ra : astropy.coordinates.Longitude
            The right ascension of the center of the sphere.
        search_dec : astropy.coordinates.Latitude
            The declination of the center of the sphere.
        search_distance : astropy.coordinates.Distance
            The distance from the barycenter to the center of the sphere.
        search_fov : astropy.coordinates.Angle
            The angle subtended by the sphere at the barycenter.
        is_in_index : int
            The index to assign to observations that intersect the sphere.
        is_out_index : int
            The index to assign to observations that do not intersect the sphere.

        Notes
        -----
        The partition sphere a center specifed by an ra, dec and distance from the barycenter.
        The sphere radius is specified by an angle subtended by the sphere at the barycenter.
        """
        super().__init__(**kwargs)
        self.obssky = coordinates.SkyCoord(search_ra, search_dec, distance=search_distance, frame="icrs")
        self.fov = search_fov
        self.is_in_index = is_in_index
        self.is_out_index = is_out_index

    def observations_to_indices(
        self, pointing: SkyCoord, time: Time, fov: coordinates.Angle, location: EarthLocation
    ) -> numpy.ndarray:
        """
        Returns a numpy array of indices for each observation specified by pointing, time, fov and location.

        Parameters
        ----------
        pointing : astropy.coordinates.SkyCoord
            The pointing of the observations.
        time : astropy.time.Time
            The time of the observations.
        fov : astropy.coordinates.Angle
            The field of view of the observations.
        location : astropy.coordinates.EarthLocation
            The location of the observations. This should be the location of the telescope.

        Returns
        -------
        numpy.ndarray
            A numpy array of indices for each observation specified by pointing, time, fov and location.
            An index of is_in_index is assigned to observations with a cone that intersect the sphere.
            An index of is_out_index is assigned to observations with a cone that do not intersect the sphere.

        Notes
        -----
        The observation cone has an apex at the location of the telescope at the time of the observation with an axis that points in the direction of the pointing and an angle of fov.
        """
        if time is None:
            separation = self.obssky.separation(pointing)
        else:
            if location is None:
                location = time.location

            with solar_system_ephemeris.set("de432s"):
                obs_pos_itrs = location.get_itrs(obstime=time)
                # the next line is slow
                observer_to_target = self.obssky.transform_to(obs_pos_itrs).gcrs
                separation = observer_to_target.separation(pointing)
        indices = numpy.where(separation < (self.fov + fov) * 0.5, self.is_in_index, self.is_out_index)
        return indices
