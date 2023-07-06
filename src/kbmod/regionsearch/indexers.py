"""
lincc-frameworks provides implementations of ``abstractions.ObserverationIndexer`` that may be composed with an implementation of ``abstractions.Backend`` to provide a complete region search.
"""
import math

import numpy
from astropy import coordinates
from astropy import units as units
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time

from kbmod.regionsearch.abstractions import ObservationIndexer


class PartitionIndexer(ObservationIndexer):
    """Assigns a provided index to observation volumes that intersect the configured space and a second index to those that do not.
    ."""

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
        """
        super().__init__(**kwargs)
        self.obssky = coordinates.SkyCoord(search_ra, search_dec, distance=search_distance, frame="icrs")
        self.fov = search_fov
        self.radius = math.sin(search_fov.to(units.rad).value / 2) * search_distance
        self.is_in_index = is_in_index
        self.is_out_index = is_out_index

    def observations_to_indices(
        self, pointing: SkyCoord, time: Time, fov: coordinates.Angle, location: EarthLocation
    ) -> numpy.ndarray:
        """Returns a list of indices"""
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
        indices = numpy.where(separation < (self.fov + fov), self.is_in_index, self.is_out_index)
        return indices
