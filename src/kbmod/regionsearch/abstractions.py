"""
Abstract classes declaring the methods required for composing a backend and an observation indexer implementation into a region search implementation
that can be used to perform region searches on a source of observations. The observation sources could be a database, a file, or a simulation. The observation sources
must provide at least a unique observation identifier, sky position (for example: right ascension, declination), time, observation location, and field of view.
"""

from abc import ABC, abstractmethod

import numpy
from astropy.coordinates import Angle, EarthLocation, SkyCoord  # type: ignore
from astropy.time import Time  # type: ignore

from kbmod.regionsearch.region_search import Filter


class Backend(ABC):
    """
    An abstract mixin class with a method that returns observation identifiers that satisfy constraints provided in a filter.

    Developers will subclass ObservationIndexer for accessing specific observation sources.
    The backend will implement the `region_search`
    method, and then compose the subclass with an implementation of ObservationIndexer to create a concrete class that can be used to
    perform region searches.

    A backend has access to a set of pointings that have at least a unique observation identifier, position, time, observation location, and field of view.
    A backend may use an ObservationIndexer implementation to assign cluster indices to the pointings.
    A backend will implement the region_search method which returns observation identifiers that match a filter.
    The filter is a set of constraints on the pointings known to the backend.

    Parameters
    ----------
    kwargs
        Keyword arguments to pass to the super class. The implementation should
        extract any keyword arguments it needs and pass the rest to the super class.
        It is important for composition that the implementation use `**kwargs` in the signature
        and pass `**kwargs` to the super class. It is also important that the implementation
        and the ObservationIndexer implementation do not use the same keyword arguments or there will
        be a conflict and only one of them will have access to the keyword argument. Which one
        depends on the order of the superclasses in the most derived class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def region_search(self, filter: Filter) -> numpy.ndarray:
        """Returns observation identifiers of pointings that match the filter.

        Parameters
        ----------
        filter : Filter
            The filter to use for the search.
            The filter is a set of constraints on the pointings known to the backend.
            The filter meaning is defined by the backend and typically specifies a region
            of the sky given as a right ascension, declination, a field of view and a distance.

        Returns
        -------
        numpy.ndarray
            Observation identifiers of pointings that match the given filter.
        """
        if not hasattr(self, "observations_to_indices"):
            raise NotImplementedError("region_search requires an implementation of observations_to_indices")
        return numpy.array([])


class ObservationIndexer(ABC):
    """
    Abstract mixin class with a method that assigns cluster indices to pointings. A complete
    implementation may assign cluster indices to source observations to improve region search
    performance.

    Developers should subclass ObservationIndexer, implement the `observations_to_indices`
    method to assign grouping indexes to all the pointings, and then compose the subclass
    with an implementation of Backend to create a concrete class that can be used to
    perform region searches.

    Parameters
    ----------
    kwargs
        Keyword arguments to pass to the super class. The implementation should
        extract any keyword arguments it needs and pass the rest to the super class.
        It is important for composition that the implementation use `**kwargs` in the signature
        and pass `**kwargs` to the super class. It is also important that the implementation
        and the Backend implementation do not use the same keyword arguments or there will
        be a conflict and only one of them will have access to the keyword argument. Which one
        depends on the order of the superclasses in the most derived class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def observations_to_indices(
        self, pointing: SkyCoord, time: Time, radius: Angle, location: EarthLocation
    ) -> numpy.ndarray:
        """Returns a numpy.ndarray of cluster indices for each of the observations in the arguments.
        The interpretation of the cluster indices is up to the implementation. The indices should
        be integers and should be unique for each cluster. The indices should be the same for
        observations that are in the same cluster. An observation could be in multiple clusters.
        Each argument should be of equal length or must be broadcastable to equal length.

        Parameters
        ----------
        pointing : astropy.coordinates.SkyCoord
            The pointing of each observation.
        time : astropy.time.Time
            The time of each observation.
        radius : astropy.coordinates.Angle
            The field of view of each observation. The field of view is the radius of the
            circular region centered on the pointing and contains all the data in the observation.
        location : astropy.coordinates.EarthLocation
            The location of each observation.

        Returns
        -------
        numpy.ndarray
            Array of cluster indices for each of the observations.
        """
        pass
