"""Region search facility for kbmod."""

from dataclasses import dataclass

import astropy.units as u  # type: ignore
from astropy import units as u
from astropy.coordinates import Angle, Distance  # type: ignore


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
