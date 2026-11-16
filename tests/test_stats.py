import math

import pytest

from askrank.stats import share_of_voice, two_proportion_z_test, wilson_interval


def test_wilson_known_value():
    # 4/10 at 95%: textbook Wilson interval is about (0.168, 0.687)
    lo, hi = wilson_interval(4, 10)
    assert lo == pytest.approx(0.1682, abs=1e-3)
    assert hi == pytest.approx(0.6873, abs=1e-3)


def test_wilson_stays_in_bounds_at_extremes():
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == pytest.approx(1.0)
    lo, hi = wilson_interval(0, 5)
    assert hi > 0  # unlike the naive interval, not zero width


def test_wilson_narrows_with_more_data():
    w10 = wilson_interval(4, 10)
    w100 = wilson_interval(40, 100)
    assert (w100[1] - w100[0]) < (w10[1] - w10[0])


def test_wilson_bad_input():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_z_test_known_value():
    # 20/100 vs 35/100: z about 2.37, p about 0.018
    r = two_proportion_z_test(20, 100, 35, 100)
    assert r.z == pytest.approx(2.37, abs=0.01)
    assert r.p_value == pytest.approx(0.018, abs=0.002)
    assert r.significant()


def test_z_test_small_change_not_significant():
    r = two_proportion_z_test(4, 10, 5, 10)
    assert not r.significant()


def test_z_test_identical_all_zero():
    r = two_proportion_z_test(0, 10, 0, 10)
    assert r.p_value == 1.0


def test_share_of_voice():
    sov = share_of_voice({"a": 3, "b": 1, "c": 0})
    assert sov == {"a": 0.75, "b": 0.25, "c": 0.0}
    assert math.isclose(sum(sov.values()), 1.0)
    assert share_of_voice({"a": 0}) == {"a": 0.0}
