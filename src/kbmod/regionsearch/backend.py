"""
Provides implementations of ``abstractions.Backend`` that may be composed with an implementation of ``abstractions.ObserverationIndexer`` to provide a complete region search.
"""

from dataclasses import dataclass

import numpy as np
from astropy import coordinates as coord  # type: ignore
from astropy import time as time
from astropy import units as units

from kbmod.regionsearch.abstractions import Backend
from kbmod.regionsearch.region_search import Filter


@dataclass(init=False)
class ObservationList(Backend):
    """A backend for sources provided explicitily to instance in lists of ra, dec, time, observation location and field of view.

    Attributes
    ----------
    observation_ra : coord.Angle
        The right ascension of the observations.
    observation_dec : coord.Angle
        The declination of the observations.
    observation_time : time.Time
        The time of the observations.
    observation_location : coord.EarthLocation
        The location of the observations. This should be the location of the telescope.
    observation_radius : coord.Angle
        The field of view of the observations. This should enclose any imagery that is associated with the observation.
    observation_identifier : np.ndarray
        The observation identifier. This is an array of values that uniquely identifies each observation.

    Notes
    -----
    The attributes observation_ra, observation_dec, observation_time, observation_location and observation_radius store lists of values comprising each observation.
    They must all have the same shape.
    """

    observation_ra: coord.Angle
    observation_dec: coord.Angle
    observation_time: time.Time
    observation_location: coord.EarthLocation
    observation_radius: coord.Angle
    observation_identifier: np.ndarray

    def __init__(
        self,
        observation_ra: coord.Angle,
        observation_dec: coord.Angle,
        observation_time: time.Time,
        observation_location: coord.EarthLocation,
        observation_radius: coord.Angle,
        observation_identifier: np.ndarray,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.observation_ra = observation_ra
        self.observation_dec = observation_dec
        self.observation_time = observation_time
        self.observation_location = observation_location
        self.observation_radius = observation_radius
        self.observation_identifier = observation_identifier
        if (
            observation_dec.shape != observation_ra.shape
            or observation_time.shape != observation_ra.shape
            or observation_location.shape != observation_ra.shape
            or observation_radius.shape != observation_ra.shape
            or observation_identifier.shape != observation_ra.shape
        ):
            raise ValueError(
                "observation_ra, observation_dec, observation_time, observation_location, observation_radius and observation_identifier must have the same shape"
            )

    def region_search(self, filter: Filter) -> np.ndarray:
        """
        Returns a numpy.ndarray of observation identifiers that match the filter.
        The filter must have attributes search_ra, search_dec, search_time, search_location, and search_radius.

        Parameters
        ----------
        filter : Filter
            The filter to use for the search.
            The filter must have attributes search_ra, search_dec, search_time, search_location, and search_radius.

        Returns
        -------
        numpy.ndarray[int]
            A list of matching indices.
        """
        matching_observation_identifier = np.array([], dtype=self.observation_identifier.dtype)
        if not hasattr(self, "observations_to_indices"):
            raise NotImplementedError("region_search requires an implementation of observations_to_indices")
        pointing = coord.SkyCoord(filter.search_ra, filter.search_dec)
        matching_index = self.observations_to_indices(pointing, None, filter.search_radius, None)  # type: ignore
        pointing = coord.SkyCoord(self.observation_ra, self.observation_dec)
        self.observation_index = self.observations_to_indices(  # type: ignore
            pointing, self.observation_time, self.observation_radius, self.observation_location
        )
        index_list = np.nonzero(self.observation_index == matching_index)[0]
        matching_observation_identifier = self.observation_identifier[index_list]
        return matching_observation_identifier
