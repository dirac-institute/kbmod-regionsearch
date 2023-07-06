"""Tests for the Backend classes."""

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Distance, EarthLocation, Latitude, Longitude
from astropy.time import Time

from kbmod.regionsearch import backend, indexers, utilities
from kbmod.regionsearch.region_search import Filter


def test_list_init():
    """Tests the List backend's constructor."""
    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)

    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    fov = Angle(1, "deg")
    b = backend.List(ra, dec, time, location, fov)
    assert all(b.observation_ra == ra)
    assert all(b.observation_dec == dec)
    assert all(b.observation_time == time)
    assert all(b.observation_location == location)
    assert b.observation_fov == fov


def test_list_partition():
    """Tests the List backend with partition indexer."""

    # Compose a region search
    class TestRegionSearchList(backend.List, indexers.PartitionIndexer):
        """A test class for the List Backend with Partition indexer."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

    data = utilities.RegionSearchClusterData(clustercnt=5, samplespercluster=10, removecache=True)
    ra = data.observation_pointing.ra
    dec = data.observation_pointing.dec
    time = data.observation_time
    location = data.observation_geolocation
    fov = Angle(2, "deg")
    clusteri = 0
    search_ra = (data.clusters[clusteri][0]) * u.deg
    search_dec = (data.clusters[clusteri][1]) * u.deg
    search_distance = data.clusterdistances[clusteri] * u.au
    search_fov = Angle(2.0, "deg")
    regionsearch = TestRegionSearchList(
        observation_ra=ra,
        observation_dec=dec,
        observation_time=time,
        observation_location=location,
        observation_fov=fov,
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
    checkresults = np.nonzero(np.array(data.cluster_id) == clusteri)[0]
    assert np.all(searchresults == checkresults)
