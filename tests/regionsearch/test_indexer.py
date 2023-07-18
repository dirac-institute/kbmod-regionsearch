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
    indexref = numpy.where(data.cluster_id == clusteri, clusteri, ~clusteri)
    assert numpy.all(indexref == indices)
