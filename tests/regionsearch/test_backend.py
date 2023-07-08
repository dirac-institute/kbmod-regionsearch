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

from kbmod.regionsearch import backend, indexers, utilities
from kbmod.regionsearch.region_search import Filter


def test_observationlist_init():
    """Tests the ObservationList backend's constructor."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    fov = np.ones([data.rowcnt]) * Angle(1, "deg")
    observation_identifier = data.cluster_id
    b = backend.ObservationList(ra, dec, time, location, fov, observation_identifier)
    assert all(b.observation_ra == ra)
    assert all(b.observation_dec == dec)
    assert all(b.observation_time == time)
    assert all(b.observation_location == location)
    assert all(b.observation_fov == fov)
    assert all(b.observation_identifier == observation_identifier)


def test_observationlist_consistency():
    """Tests the ObservationList backend's constructor raise error if all elements are not equal in length."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    fov = Angle(1, "deg")
    observation_identifier = data.cluster_id
    with pytest.raises(ValueError):
        backend.ObservationList(ra, dec, time, location, fov, observation_identifier)


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
    fov = np.ones([data.rowcnt]) * Angle(2, "deg")
    observation_identifier = np.array([f"file:epyc/observations/{i:04d}" for i in range(data.rowcnt)])
    clusteri = 0
    search_ra = (data.clusters[clusteri][0]) * u.deg
    search_dec = (data.clusters[clusteri][1]) * u.deg
    search_distance = data.clusterdistances[clusteri] * u.au
    search_fov = Angle(4.0, "deg")
    regionsearch = TestRegionSearchList(
        observation_ra=ra,
        observation_dec=dec,
        observation_time=time,
        observation_location=location,
        observation_fov=fov,
        observation_identifier=observation_identifier,
        search_ra=search_ra,
        search_dec=search_dec,
        search_distance=search_distance,
        search_fov=search_fov,
        is_in_index=clusteri,
        is_out_index=~clusteri,
    )
    assert regionsearch is not None
    searchresults = regionsearch.region_search(
        Filter(
            search_ra=search_ra, search_dec=search_dec, search_distance=search_distance, search_fov=search_fov
        )
    )
    assert searchresults is not None
    checkresults = observation_identifier[np.nonzero(data.cluster_id == clusteri)[0]]
    assert np.all(searchresults == checkresults)