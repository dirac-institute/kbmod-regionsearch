import os
import random
import typing

import astropy.table  # type: ignore
import numpy as np
from astropy import time
from astropy import units as u
from astropy.coordinates import (  # type: ignore
    EarthLocation,
    SkyCoord,
    solar_system_ephemeris,
    uniform_spherical_random_surface,
)


class RegionSearchClusterData(object):
    """Randomly generated cluster data for testing the region search.

    Each instance of this class contains lists of pointings, observation times, reference pointings and distances arranged so
    pointings grouped by reference pointing and distance view a position given by the reference pointing and distance from the
    barycenter.

    Attributes
    ----------
    version : int
        The version of the data. This is manually incremented whenever a material change is made to the data generation. It appears in the filename and the
        metadata and is used to check that data in a file is compatible with the code.
    seed : int
        The seed for the random number generator. This appears in the filename. It is used to generate the data idempotently.
    clustercnt : int
        The number of clusters to generate. This appears in the filename.
    samplespercluster : int
        The number of samples per cluster to generate. This appears in the filename.
    data_loaded : bool
        True if the data have been loaded from a file.
    data_generated : bool
        True if the data have been generated.
    """

    def __init__(
        self,
        basename: str = "clustered-data",
        suffix: str = ".ecsv",
        format: str = "ascii.ecsv",
        seed: int = 0,
        clustercnt: int = 10,
        samplespercluster: int = 1000,
        removecache: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        basename : str
            The base name of the file to read or write. This is the first part of the file name.
            The default value is "clustered-data".
            The full name includes the directory, basename, seed, clustercnt, samplespercluster, version, section name, and suffix.
        suffix : str
            The suffix of the file to read or write. Thisis the last part of the file name.
            The default value is ".ecsv" and seems the best choice for astropy tables.
        format : str
            The format of the file to read or write. This is coordinated with the suffix.
            The default value is "ascii.ecsv" and seems the best choice for astropy tables.
        seed : int
            The seed for the random number generator. This is included in the filename.
        clustercnt : int
            The number of clusters to generate. This is included in the filename.
        samplespercluster : int
            The number of samples per cluster to generate. This is included in the filename.
        removecache : bool
            If True, remove the cache files and regenerate the data.
            If False, use the cache files if they exist.

        Notes
        -----
        The data are generated if they do not exist.
        If the data are generated, they are saved in the tmp directory if that exists.
        If the filename is found in the directory "data" then that will be used.
        The files in "tmp" are ignored by git. The files in "data" are controlled by git.
        The full name includes the directory, basename, seed, clustercnt, samplespercluster, version, section name, and suffix.
        The version is checked when reading the file.
        """
        # See https://docs.astropy.org/en/stable/io/unified.html#table-serialization-methods
        self.version = "2"
        # seed for random number generator
        self.seed = seed
        # number of clusters in test
        self.clustercnt = clustercnt
        # number of samples per cluster
        self.samplespercluster = samplespercluster
        # data are not loaded
        self.data_loaded = False
        # data are not generated
        self.data_generated = False

        sections = [
            ["samples", ["bary_to_target", "observer_to_target", "observation_geolocation", "cluster_id"]],
            ["clusters", ["clusters", "clusterdistances"]],
        ]

        # tmp data overrides data. Only tmp gets written. Only data is controlled in git
        # for dirname in ["tmp", "data"]:
        # Only remove cache in tmp. This also implies generating data in tmp.
        if removecache:
            for section in sections:
                filename = self.__filename("tmp", basename, section[0], suffix)
                if os.path.exists(filename):
                    os.remove(filename)
            self.generate_and_save(basename, suffix, format, sections)
        else:
            for dirname in ["tmp", "data"]:
                # sections must be grouped together
                if not self.data_loaded:
                    data_loaded_count = 0
                    for section in sections:
                        filename = self.__filename(dirname, basename, section[0], suffix)
                        if os.path.exists(filename) and self.read_table(filename, format, list(section[1])):
                            data_loaded_count += 1
                        else:
                            # early out since this is not a complete set of data
                            break
                    if data_loaded_count == len(sections):
                        self.data_loaded = True
                        # early out since this is a complete set of data
                        break

            # if data are not loaded, generate them
            if not self.data_loaded:
                self.generate_and_save(basename, suffix, format, sections)

        if not self.data_loaded:
            raise Exception(f"Could not read or generate data")

    def generate_and_save(self, basename, suffix, format, sections):
        """
        Generate the data and save them in the tmp directory.

        Parameters
        ----------
        basename : str
        """
        if self._generate_data():
            if os.path.isdir("tmp"):
                for section in sections:
                    filename = self.__filename("tmp", basename, section[0], suffix)
                    self.write_table(filename, format, section[1])
            self.data_loaded = True
            self.data_generated = True

    def __filename(self, dirname, basename, sectionname, suffix):
        """
        Return the full filename.

        Parameters
        ----------
        dirname : str
            The directory name. This is "tmp" or "data".
        basename : str
            The base name of the file to read or write.
        sectionname : str
            The section name.
        suffix : str
            The suffix of the file to read or write. Thisis the last part of the file name.

        Returns
        -------
        str
            The full filename.
        """
        return f"{dirname}/{basename}-{self.seed}-{self.clustercnt}-{self.samplespercluster}-{self.version}-{sectionname}{suffix}"

    def read_table(self, filename: str, format: str, colnames: typing.List[str]):
        """
        Read the table from the file and validate the version.

        Parameters
        ----------
        filename : str
            The full filename.
        format : str
            The format of the file to read or write.
        colnames : typing.List[str]
            The list of column names to read from the file. The file may contain other columns but it must have these columns.

        Returns
        -------
        bool
            True if the table was read and the version is valid and all of the columns are present.
            False if the table was not read or the version is invalid or any of the columns are missing.
        """
        hold_version = self.version
        try:
            table = astropy.table.Table.read(filename, format=format)
            for metakey in table.meta.keys():
                setattr(self, metakey, table.meta[metakey])
            if self.version != hold_version:
                self.version = hold_version
                raise Exception(f"Version mismatch: {self.version} != {table.meta['version']}")
            for column in colnames:
                if column not in table.colnames:
                    raise Exception(f"Column {column} not found in {filename}")
                setattr(self, column, table[column])
        finally:
            self.version = hold_version
        return True

    def write_table(self, filename: str, format: str, colnames: typing.List[str]):
        """
        Write the table to the file including all the columns in colnames.

        Parameters
        ----------
        filename : str
            The full filename.
        format : str
            The format of the file to read or write.
        colnames : typing.List[str]
            The list of column names to write to the file.
        """
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "w") as f:
            coldata = [getattr(self, column) for column in colnames]
            table = astropy.table.Table(
                data=coldata,
                names=colnames,
            )
            table.meta["version"] = self.version
            table.write(f, format=format, overwrite=True)
        return True

    def _generate_data(self):
        """
        Generate the data for the test.
        """
        if self.data_loaded:
            return

        random.seed(self.seed)
        # timerange to assign to samples. The time of a sample is not related to the cluster or sample in cluster.
        self.timerange = time.Time(
            ["2022-06-01T00:00:00.000", "2023-06-01T00:00:00.000"], format="isot", scale="utc"
        )
        self.timerange.format = "mjd"
        # the nominal distances for each cluster.
        if self.clustercnt <= 1:
            self.clusterdistances = [random.randrange(2, 10000)]
        else:
            self.clusterdistances = [2]
            self.clusterdistances.extend([random.randrange(2, 10000) for _ in range(self.clustercnt - 1)])

        _clusters = uniform_spherical_random_surface(self.clustercnt)
        self.clusters = [
            [i.value - 180.0, j.value] for i, j in zip(_clusters.lon.to(u.deg), _clusters.lat.to(u.deg))
        ]
        clusters = np.stack([_clusters.lon.to(u.deg).value, _clusters.lat.to(u.deg).value + 90.0], axis=1)

        # the number of rows in the test dataset
        self.rowcnt = self.clustercnt * self.samplespercluster
        self.cluster_id = np.array([i for i in range(self.clustercnt) for _ in range(self.samplespercluster)])

        baryra = [
            (i[0] + random.uniform(-1, 1)) % 360.0 - 180.0
            for i in clusters
            for _ in range(self.samplespercluster)
        ] * u.deg
        barydec = [
            (i[1] + random.uniform(-1, 1)) % 180.0 - 90.0
            for i in clusters
            for _ in range(self.samplespercluster)
        ] * u.deg
        bary_distance = [
            self.clusterdistances[i] for i in range(self.clustercnt) for _ in range(self.samplespercluster)
        ] * u.au

        # Vera Rubin Observatory for all samples
        # see https://www.lsst.org/scientists/keynumbers
        self.observation_geolocation = EarthLocation.from_geodetic(
            [-70.749417] * self.rowcnt * u.deg, [-30.244633] * self.rowcnt * u.deg, [2647] * self.rowcnt * u.m
        )

        valmjd = np.array([random.uniform(*self.timerange.value) for _ in range(self.rowcnt)])
        valobservation_time = time.Time(
            # round to millisecond in lieu of modifying file format for column.
            val=valmjd,
            format="mjd",
            scale="utc",
            # location=self.observation_geolocation,
        )
        valobservation_time.format = "isot"
        # convert to exact isot representation so read after write serialization is exact
        valisot = valobservation_time.value
        observation_time = time.Time(
            val=valisot,
            format="isot",
            scale="utc",
            location=self.observation_geolocation,
        )

        with solar_system_ephemeris.set("de432s"):
            self.bary_to_target = SkyCoord(ra=baryra, dec=barydec, distance=bary_distance)
            obs_pos_itrs = observation_time.location.get_itrs(obstime=observation_time)
            # the next line is slow
            # the transform_to changes the angular units in the frame from degrees to radians. Change them back for cache read after write consistency.
            self.observer_to_target = SkyCoord(
                self.bary_to_target.transform_to(obs_pos_itrs).gcrs.spherical, obstime=observation_time
            )
        return True

    @property
    def observation_pointing(self):
        return SkyCoord(ra=self.observer_to_target.ra, dec=self.observer_to_target.dec)

    @property
    def observation_time(self):
        return self.observer_to_target.obstime

    @property
    def observation_distance(self):
        return self.observer_to_target.distance
