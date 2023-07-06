"""Test the `Filter` class for region search"""
from astropy import units as u

from kbmod.regionsearch.region_search import Filter


def test_filter_bareargs() -> None:
    """Verify the output of the `Filter` class using bare quantities"""
    output = Filter(42, -28, 30)
    assert output.search_ra == 42 * u.deg
    assert output.search_dec == -28 * u.deg
    assert output.search_distance == 30 * u.au


def test_filter_namedargs() -> None:
    """Verify the output of the `Filter` class using bare quantities"""
    output = Filter(search_ra=42, search_dec=-28, search_distance=30)
    assert output.search_ra == 42 * u.deg
    assert output.search_dec == -28 * u.deg
    assert output.search_distance == 30 * u.au


def test_filter_unitargs() -> None:
    """Verify the output of the `Filter` class using quantities with units"""
    output = Filter(42 * u.deg, -28 * u.deg, 30 * u.au, 1 * u.deg)
    assert output.search_ra == 42 * u.deg
    assert output.search_dec == -28 * u.deg
    assert output.search_distance == 30 * u.au
    assert output.search_fov == 1 * u.deg


def test_filter_none() -> None:
    """Verify the output of the `Filter` class with no arguments"""
    output = Filter()
    assert output.search_ra is None
    assert output.search_dec is None
    assert output.search_distance is None
    assert output.search_fov is None


def test_filter_literate() -> None:
    """Verify the output of the `Filter` class set using literate methods"""
    output = Filter()
    output.with_ra(0 * u.deg)
    output.with_dec(0 * u.deg)
    output.with_distance(30 * u.au)
    assert output.search_ra == 0 * u.deg
    assert output.search_dec == 0 * u.deg
    assert output.search_distance == 30 * u.au


def test_filter_literate2() -> None:
    """Verify the output of the `Filter` class set using literate methods"""
    # fmt: off
    output = Filter()\
        .with_ra(0 * u.deg)\
        .with_dec(0 * u.deg)\
        .with_distance(30 * u.au)
    # fmt: on
    assert output.search_ra == 0 * u.deg
    assert output.search_dec == 0 * u.deg
    assert output.search_distance == 30 * u.au
