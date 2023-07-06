"""Tests for indexer classes."""

import astropy.units as u
import numpy

from kbmod.regionsearch.indexers import PartitionIndexer
from kbmod.regionsearch.utilities import RegionSearchClusterData


def test_partition_indexer():
    """Test the PartitionIndexer"""
    data = RegionSearchClusterData(clustercnt=2, samplespercluster=5, removecache=True)
    clusteri = 0
    indexer = PartitionIndexer(
        search_ra=(data.clusters[clusteri][0]) * u.deg,
        search_dec=(data.clusters[clusteri][1]) * u.deg,
        search_distance=data.clusterdistances[clusteri] * u.au,
        search_fov=2.0 * u.deg,
        is_in_index=clusteri,
        is_out_index=~clusteri,
    )
    assert indexer is not None
    indices = indexer.observations_to_indices(
        pointing=data.observation_pointing,
        time=data.observation_time,
        fov=2.0 * u.deg,
        location=data.observation_geolocation,
    )
    assert indices is not None
    indexref = numpy.where(numpy.array(data.cluster_id) == clusteri, clusteri, ~clusteri)
    assert numpy.all(indexref == indices)


def test_cache_write_read():
    """Test the PartitionIndexer"""
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
