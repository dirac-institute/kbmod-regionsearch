import random

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Column, Table
from astropy.time import Time

from kbmod.regionsearch.region_search import RowSet

rowcnt = 10
random.seed(0)

pointing = SkyCoord(
    ra=[(42.0 + random.uniform(0, 2)) for i in range(rowcnt)] * u.deg,
    dec=[(-28.0 + random.uniform(0, 2)) for i in range(rowcnt)] * u.deg,
    frame="icrs",
)
# time = [ Time(f"2023-06-{(1+3*i):02d}T00:00:00.000", format="isot", scale="utc") for i in range(rowcnt) ]
time = Time([f"2023-06-{(1+3*i):02d}T00:00:00.000" for i in range(rowcnt)], format="isot", scale="utc")
fov = [0.2 for i in range(rowcnt)] * u.deg
distance_far = [random.randrange(7, 10000) for i in range(rowcnt)] * u.au
distance_near = [i - 5 for i in distance_far.value] * u.au
# distance_near = [ distance_far[i] - 5 ] for i in range(rowcnt) ] * u.au
# https://www.lsst.org/scientists/keynumbers
# Site coordinates:  latitude -30:14:40.68  longitude -70:44:57.90  altitude 2647m
# -70.749417
# -30.244633
data_source = ["LSST" for i in range(rowcnt)]
observation_id = [i + 100 for i in range(rowcnt)]
observatory_location = EarthLocation.from_geodetic(
    list(-70.749417 for i in range(rowcnt)) * u.deg,
    list(-30.244633 for i in range(rowcnt)) * u.deg,
    list(2647 + i for i in range(rowcnt)) * u.m,
)


def test_rowset_from_data():
    rowset = RowSet()
    assert rowset is not None
    rowset.add(
        pointing=pointing,
        time=time,
        fov=fov,
        distance_far=distance_far,
        distance_near=distance_near,
        observatory_location=observatory_location,
        data_source=data_source,
        observation_id=observation_id,
    )
    table = rowset.get_table()
    assert table is not None


def test_rowset_from_parts():
    rowset = RowSet()
    assert rowset is not None
    rowset.add(pointing=pointing)
    rowset.add(time=time)
    rowset.add(fov=fov)
    rowset.add(distance_far=distance_far)
    rowset.add(distance_near=distance_near)
    rowset.add(observatory_location=observatory_location)
    rowset.add(data_source=data_source)
    rowset.add(observation_id=observation_id)
    table = rowset.get_table()
    assert table is not None


def test_rowset_missing_parts():
    # the catch with missing columns is that they are typed as dtype object and class Column. No nice accomodations for SkyCoord, Time, EarthLocation.
    rowset = RowSet()
    assert rowset is not None
    # rowset.add(pointing = pointing)
    # rowset.add(time = time)
    # rowset.add(fov = fov)
    # rowset.add(distance_far = distance_far)
    # rowset.add(distance_near = distance_near)
    rowset.add(observatory_location=observatory_location)
    rowset.add(observatory_location=observatory_location)
    # rowset.add(data_source = data_source)
    # rowset.add(observation_id = observation_id)
    try:
        table = rowset.get_table()
    except Exception as e:
        assert False, f"unexpected exception: {e}"
    assert table is not None


def test_rowset_short_parts():
    rowset = RowSet()
    assert rowset is not None
    rowset.add(pointing=pointing[0:5])
    rowset.add(time=time[0:5])
    rowset.add(fov=fov[0:5])
    rowset.add(distance_far=distance_far[0:5])
    rowset.add(distance_near=distance_near[0:5])
    rowset.add(observatory_location=observatory_location[0:5])
    rowset.add(data_source=data_source[0:5])
    rowset.add(observation_id=observation_id)
    try:
        table = rowset.get_table()
    except Exception as e:
        assert False, f"unexpected exception: {e}"
    assert table is not None
