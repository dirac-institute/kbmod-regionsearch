from abc import ABC, abstractmethod
from typing import Type

import numpy
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.time import Time

from kbmod.regionsearch.region_search import Filter


class ObservationIndexer(ABC):
    """
    A mixin abstract class with a method to assign pointings to cluster indices.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def observations_to_indices(
        self, pointing: SkyCoord, time: Time, fov: Angle, location: EarthLocation
    ) -> numpy.ndarray:
        """Returns a list of indices

        Returns
        -------
        numpy.ndarray
            Indices for each of the observations.
        """
        pass


class Backend(ABC):
    """An abstract base class for backend connectors."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def region_search(self, filter: Filter) -> numpy.ndarray:
        """Returns indices of pointings that match the filter.

        Parameters
        ----------
        filter : Filter
            The filter to use for the search.

        Returns
        -------
        numpy.ndarray
            Indices of pointings that match the given filter.
        """
        pass
