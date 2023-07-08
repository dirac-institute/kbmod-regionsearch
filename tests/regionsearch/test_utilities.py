"""Tests for utilities."""

import numpy as np

from kbmod.regionsearch.utilities import RegionSearchClusterData


def test_cache_write_read():
    """Test for idempotency of the cache write/read."""
    data1 = RegionSearchClusterData(samplespercluster=5, removecache=True)
    assert data1 is not None
    assert data1.data_loaded
    assert data1.data_generated == True
    # check that written files can be read
    data2 = RegionSearchClusterData(samplespercluster=5, removecache=False)
    assert data2 is not None
    assert data2.data_loaded
    assert data2.data_generated == False
    assert all(data1.observation_pointing == data2.observation_pointing)
    # time.mjd2 likely will differ by a few nanoseconds so exact equality is not expected
    assert all(data1.observation_time - data2.observation_time < 1e-8)
    assert all(data1.observation_geolocation == data2.observation_geolocation)
    assert all(data1.cluster_id == data2.cluster_id)
    assert all(data1.clusterdistances == data2.clusterdistances)
    assert all(data1.bary_to_target == data2.bary_to_target)
    assert all(data1.observer_to_target.ra == data2.observer_to_target.ra)
    assert all(data1.observer_to_target.dec == data2.observer_to_target.dec)
    assert all(data1.observer_to_target.distance == data2.observer_to_target.distance)
    assert all(data1.observer_to_target.obstime.location == data2.observer_to_target.obstime.location)
    assert all(data1.observer_to_target.obstime == data2.observer_to_target.obstime)
    # Check frame equality since special care must be taken to get the units in frame._data to match and so it could regress.
    assert all(data1.observer_to_target.frame == data2.observer_to_target.frame)
    assert all(data1.observer_to_target.cartesian == data2.observer_to_target.cartesian)
    assert all(data1.observer_to_target == data2.observer_to_target)


def test_clusterdata_stats():
    """Verify that the cluster data groups by cluster as expected."""
    clustercnt = 100
    samplespercluster = 5
    data = RegionSearchClusterData(clustercnt=clustercnt, samplespercluster=samplespercluster)
    assert data is not None
    for clusteri in range(data.clustercnt):
        clustersamples = len(np.nonzero(clusteri == data.cluster_id)[0])
        assert clustersamples == samplespercluster
