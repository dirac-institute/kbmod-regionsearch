"""Region search facility for kbmod."""

from dataclasses import dataclass
from typing import Union

import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import (
    Angle,
    Distance,
    EarthLocation,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.table import Column, Table
from astropy.time import Time
from astropy.units import Quantity


class RowSet(object):
    """Container for region_search rows.

    A region search row contains the columns used in the region search. Not all columns are needed at all times and the access pattern is unclear. Hence the RowSet, which abstracts the collection from the
    details of the row. The columns of the collection are as follows:

    - ``pointing``: Right ascension abd declination of the observation as a SkyCoord.
    - ``time``: Time of the observation.
    - ``fov``: The field of view of the observation in degrees.
    - ``distance_far``: The farthest distance from the barycenter in which the observation appears with the index.
    - ``distance_near``: The nearest distance from the barycenter in which the observation appears with the index.
    - ``observatory_location``: The location of the observatory in geocentric coordinates. Together with the time this locates the origin of the observation.
    - ``data_source``: The data source of the observation. This is a string that identifies the data source and may be None
    - ``observation_id``: A unique identifier for the observation. Together with data_source this should provide enough information to trace back to the observation source.
    - ``index``: Index of the observation.

    All rows occur uniquely in the collection. The columns pointing, time, and fov are required. The columns index and distance_interval may either both be None or both not None.
    The columns observatory_location, data_source and observation_id may be None.
    If observatory_location is None then the observation is assumed to be geocentric.

    The table is initialized in the __init__ or by calling the add method one or more times. The get_table method returns the data as an astropy.table and freezes the RowSet so that no more rows can be added.
    Completely empty columns are left in the table with all None entries in a Column instance. This is done so that indexing columns and rows always returns data (even if just None).

    """

    names_type1 = ["pointing", "time", "fov", "distance_near", "distance_far", "observatory_location"]
    names_type2 = ["data_source", "observation_id", "index"]
    names = names_type1 + names_type2
    dtype = [SkyCoord, Time, Quantity, Quantity, Quantity, EarthLocation, "str", "uint32", "uint32"]
    units = [None, None, u.deg, u.au, u.au, None, None, None, None]

    def __init__(self):
        """Initializes the RowSet object."""
        self.table = None
        self.pointing = None
        self.time = None
        self.fov = None
        self.distance_far = None
        self.distance_near = None
        self.observatory_location = None
        self.data_source = None
        self.observation_id = None
        self.index = None
        self.rowcnt = 0
        self.dtype = RowSet.dtype.copy()

    def add(self, **kwargs):
        """
        Adds a row to the RowSet.

        The parameters are optional but if present must be the parameter type.

        Parameters
        ----------
        pointing : SkyCoord
            The pointing of the observation.
        time : Time
            The time of the observation.
        fov : Quantity
            The field of view of the observation.
        distance_far : Quantity
            The far distance of the observation.
        distance_near : Quantity
            The near distance of the observation.
        observatory_location : EarthLocation
            The location of the observatory.
        data_source : str
            The data source of the observation.
        observation_id : int
            The observation id of the observation.
        """
        if self.table is not None:
            raise RuntimeError("Cannot add rows to a RowSet that has been converted to a table")

        for k in self.names_type1:
            if k in kwargs:
                if getattr(self, k) is None:
                    setattr(self, k, kwargs[k])
                else:
                    setattr(self, k, kwargs[k].insert(0, getattr(self, k)))
                self.rowcnt = max(self.rowcnt, len(getattr(self, k)))
                del kwargs[k]

        for i in self.names_type2:
            if getattr(self, i) is None:
                if i in kwargs:
                    setattr(self, i, kwargs[i])
                    self.rowcnt = max(self.rowcnt, len(getattr(self, i)))
            else:
                if i in kwargs:
                    getattr(self, i).extend(kwargs[i])
                    self.rowcnt = max(self.rowcnt, len(getattr(self, i)))

    def get_table(self) -> Table:
        """
        Return the current table.

        This instantiates the table on the first call. RowSet does not instatiate an empty table both to improve performance
        and to avoid inconsistent column interpretations.
        """
        if self.table is None:
            coli = 0
            if self.pointing is None:
                rc = self.rowcnt
                self.add(pointing=SkyCoord(ra=[0] * rc * u.deg, dec=[0] * rc * u.deg))
            elif len(self.pointing) < self.rowcnt:
                rc = self.rowcnt - len(self.pointing)
                self.add(pointing=SkyCoord(ra=[0] * rc * u.deg, dec=[0] * rc * u.deg))
            if self.pointing is not None:
                if len(self.pointing) < self.rowcnt:
                    self.pointing.extend([None] * (self.rowcnt - len(self.pointing)))
            else:
                self.pointing = np.array([None] * self.rowcnt, dtype=self.dtype[coli])
            coli = 1
            if self.time is None:
                rc = self.rowcnt
                self.add(time=Time([0] * rc, format="mjd", scale="utc"))
            elif len(self.time) < self.rowcnt:
                rc = self.rowcnt - len(self.time)
                self.add(time=Time([0] * rc, format="mjd", scale="utc"))
            coli = 2
            if self.fov is None:
                rc = self.rowcnt
                self.add(fov=[0] * rc * u.deg)
            elif len(self.fov) < self.rowcnt:
                rc = self.rowcnt - len(self.fov)
                self.add(fov=[0] * rc * u.deg)
            coli = 3
            if self.distance_far is None:
                rc = self.rowcnt
                self.add(distance_far=[0] * rc * u.au)
            elif len(self.distance_far) < self.rowcnt:
                rc = self.rowcnt - len(self.distance_far)
                self.add(distance_far=[0] * rc * u.au)
            coli = 4
            if self.distance_near is None:
                rc = self.rowcnt
                self.add(distance_near=[0] * rc * u.au)
            elif len(self.distance_near) < self.rowcnt:
                rc = self.rowcnt - len(self.distance_near)
                self.add(distance_near=[0] * rc * u.au)
            coli = 5
            if self.observatory_location is None:
                rc = self.rowcnt
                self.add(
                    observatory_location=EarthLocation.from_geocentric(
                        [0] * rc * u.au, [0] * rc * u.au, [0] * rc * u.au
                    )
                )
            elif len(self.observatory_location) < self.rowcnt:
                rc = self.rowcnt - len(self.observatory_location)
                self.add(
                    observatory_location=EarthLocation.from_geocentric(
                        [0] * rc * u.au, [0] * rc * u.au, [0] * rc * u.au
                    )
                )

            coli = 6
            if self.data_source is not None:
                if len(self.data_source) < self.rowcnt:
                    self.data_source.extend([None] * (self.rowcnt - len(self.data_source)))
            else:
                self.data_source = np.array([None] * self.rowcnt, dtype=self.dtype[coli])
            coli = 7
            if self.observation_id is not None:
                if len(self.observation_id) < self.rowcnt:
                    self.observation_id.extend([0] * (self.rowcnt - len(self.observation_id)))
            else:
                # this is a int column and so needs a special case for None values
                self.dtype[coli] = None
                self.observation_id = np.array([None] * self.rowcnt)
            coli = 8
            if self.index is None:
                self.index = []
            if len(self.index) < self.rowcnt:
                self.index.extend([i for i in range(len(self.index), self.rowcnt)])
            _data = [
                self.pointing,
                self.time,
                self.fov,
                self.distance_far,
                self.distance_near,
                self.observatory_location,
                self.data_source,
                self.observation_id,
                self.index,
            ]
            self.table = Table(_data, names=self.names, dtype=self.dtype, units=self.units)
        return self.table


SearchSet = RowSet


@dataclass(init=False)
class Filter:
    """A class for specifying region search filters."""

    search_ra: Angle = None
    search_dec: Angle = None
    search_distance: Distance = None
    search_fov: Angle = None

    def __init__(
        self,
        search_ra: Angle = None,
        search_dec: Angle = None,
        search_distance: Distance = None,
        search_fov: Angle = None,
    ):
        self.with_ra(search_ra)
        self.with_dec(search_dec)
        self.with_distance(search_distance)
        self.with_fov(search_fov)

    def __repr__(self):
        return f"Filter(ra={self.search_ra}, dec={self.search_dec}, search_distance={self.search_distance})"

    def with_ra(self, search_ra: Angle):
        """Sets the right ascension of the center of the search region."""
        if search_ra is not None:
            self.search_ra = Angle(search_ra, unit=u.deg)
        return self

    def with_dec(self, search_dec: Angle):
        """Sets the declination of the center of the search region."""
        if search_dec is not None:
            self.search_dec = Angle(search_dec, unit=u.deg)
        return self

    def with_distance(self, search_distance: Distance):
        """Sets the minimum distance from the barycenter to the search region."""
        if search_distance is not None:
            self.search_distance = Distance(search_distance, unit=u.au)
        return self

    def with_fov(self, search_fov: Angle):
        if search_fov is not None:
            self.search_fov = Angle(search_fov, unit=u.deg)
        return self
