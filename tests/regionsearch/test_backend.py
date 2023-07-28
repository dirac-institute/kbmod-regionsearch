"""Tests for the Backend classes."""

import numpy as np
import pytest
from astropy import units as u  # type: ignore
from astropy.coordinates import (  # type: ignore
    Angle,
    Distance,
    EarthLocation,
    Latitude,
    Longitude,
)
from astropy.time import Time  # type: ignore

from kbmod.regionsearch import abstractions, backend, indexers, utilities
from kbmod.regionsearch.region_search import Filter


def test_backend_abstract():
    """Tests that the backend raises an exception when region_search is called without an indexer."""

    class TestBackend(abstractions.Backend):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def region_search(self, filter: Filter) -> np.ndarray:
            return super().region_search(filter)

    b = TestBackend()
    assert b is not None
    with pytest.raises(NotImplementedError):
        b.region_search(Filter())
        assert False, "expected NotImplementedError when calling region_search without an indexer"


def test_observationlist_init():
    """Tests the ObservationList backend's constructor."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    radius = np.ones([data.rowcnt]) * Angle(1, "deg")
    observation_identifier = data.cluster_id
    b = backend.ObservationList(ra, dec, time, location, radius, observation_identifier)
    assert all(b.observation_ra == ra)
    assert all(b.observation_dec == dec)
    assert all(b.observation_time == time)
    assert all(b.observation_location == location)
    assert all(b.observation_radius == radius)
    assert all(b.observation_identifier == observation_identifier)


def test_observationlist_consistency():
    """Tests the ObservationList backend's constructor raise error if all elements are not equal in length."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    radius = Angle(1, "deg")
    observation_identifier = data.cluster_id
    with pytest.raises(ValueError):
        backend.ObservationList(ra, dec, time, location, radius, observation_identifier)


def test_observationlist_missing_observation_to_indices():
    """Tests that the backend raises an exception when region_search is called without an indexer."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    radius = np.ones([data.rowcnt]) * Angle(1, "deg")
    observation_identifier = data.cluster_id
    b = backend.ObservationList(ra, dec, time, location, radius, observation_identifier)
    with pytest.raises(NotImplementedError):
        b.region_search(Filter())
        assert (
            False
        ), "Expect NotImplementedError when region_search is called without an observation_to_indices method (missing ObservationIdexer)."
    assert True


def test_observationlist_partition():
    """Tests the ObservationList backend with partition indexer."""

    # Compose a region search
    class TestRegionSearchList(backend.ObservationList, indexers.PartitionIndexer):
        """A test class for the ObservationList Backend with Partition indexer."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)
    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    radius = np.ones([data.rowcnt]) * Angle(2, "deg")
    observation_identifier = np.array([f"file:epyc/observations/{i:04d}" for i in range(data.rowcnt)])
    clusteri = 0
    search_ra = (data.clusters[clusteri][0]) * u.deg
    search_dec = (data.clusters[clusteri][1]) * u.deg
    search_distance = data.clusterdistances[clusteri] * u.au
    search_radius = Angle(4.0, "deg")
    regionsearch = TestRegionSearchList(
        observation_ra=ra,
        observation_dec=dec,
        observation_time=time,
        observation_location=location,
        observation_radius=radius,
        observation_identifier=observation_identifier,
        search_ra=search_ra,
        search_dec=search_dec,
        search_distance=search_distance,
        search_radius=search_radius,
        is_in_index=clusteri,
        is_out_index=~clusteri,
    )
    assert regionsearch is not None
    searchresults = regionsearch.region_search(
        Filter(
            search_ra=search_ra,
            search_dec=search_dec,
            search_distance=search_distance,
            search_radius=search_radius,
        )
    )
    assert searchresults is not None
    checkresults = observation_identifier[np.nonzero(data.cluster_id == clusteri)[0]]
    assert np.all(searchresults == checkresults)
