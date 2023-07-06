"""
lincc-frameworks provides implementations of ``abstractions.Backend`` that may be composed with an implementation of ``abstractions.ObserverationIndexer`` to provide a complete region search.
"""

from dataclasses import dataclass

import numpy as np
from astropy import coordinates as coord
from astropy import time as time
from astropy import units as units

from kbmod.regionsearch.abstractions import Backend
from kbmod.regionsearch.region_search import Filter


@dataclass(init=False)
class List(Backend):
    """A backend for in memory lists of observation pointings, locations and times.

    Attributes
    ----------
    observation_ra : coord.Angle
        The right ascension of the observations.
    observation_dec : coord.Angle
        The declination of the observations.
    observation_time : time.Time
        The time of the observations.
    observation_location : coord.EarthLocation
        The location of the observations.

    The number of elements in each of the above attributes must be the same. This is assumed and not explicitly checked.
    """

    observation_ra: coord.Angle
    observation_dec: coord.Angle
    observation_time: time.Time
    observation_location: coord.EarthLocation
    observation_fov: coord.Angle

    def __init__(
        self,
        observation_ra: coord.Angle,
        observation_dec: coord.Angle,
        observation_time: time.Time,
        observation_location: coord.EarthLocation,
        observation_fov: coord.Angle,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.observation_ra = observation_ra
        self.observation_dec = observation_dec
        self.observation_time = observation_time
        self.observation_location = observation_location
        self.observation_fov = observation_fov

    def region_search(self, filter: Filter) -> np.ndarray:
        """Returns a SearchSet of pointings that match the given filter.

        Parameters
        ----------
        filter : Filter
            The filter to use for the search.

        Returns
        -------
        numpy.ndarray[int]
            A list of matching indices.
        """
        if hasattr(self, "observations_to_indices"):
            pointing = coord.SkyCoord(filter.search_ra, filter.search_dec)
            matching_index = self.observations_to_indices(pointing, None, filter.search_fov, None)  # type: ignore
            pointing = coord.SkyCoord(self.observation_ra, self.observation_dec)
            self.observation_index = self.observations_to_indices(  # type: ignore
                pointing, self.observation_time, self.observation_fov, self.observation_location
            )
            self.index_list = np.nonzero(self.observation_index == matching_index)[0]
        return self.index_list
