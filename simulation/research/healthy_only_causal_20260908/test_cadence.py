import numpy as np
import pytest
from cadence import dense_flags, phase_vote, stride_from_history


def test_stride_uses_history_only():
    tr = np.arange(30); cal = np.arange(30, 50)
    assert stride_from_history(tr, cal) == 1
    assert stride_from_history(np.arange(100), np.arange(100, 200)) == 2


def test_small_stride_has_no_fake_phase_votes():
    r = phase_vote(np.ones(40), np.arange(20), np.arange(20, 40), stride=2, phases=5, votes=4)
    assert r["metadata"]["offsets"] == [0, 1]
    assert r["metadata"]["votes"] == 2


def test_fallback_and_finite_validation():
    r = phase_vote(np.ones(30), np.arange(10), np.arange(10, 30), stride=100)
    assert r["metadata"]["reference_per_phase"]
    with pytest.raises(ValueError):
        phase_vote(np.array([1., np.nan]), [0], [1], 1)


def test_dense_route_is_exact_rolling_history_max():
    x = np.arange(80, dtype=float)
    got = dense_flags(x, np.arange(30), np.arange(30, 60))
    hist = x[np.arange(60)]
    ref = np.max([np.partition(hist[i-20:i+1], 8)[8] for i in range(20, 60)])
    expected = np.zeros(80, bool)
    for i in range(20, 80):
        expected[i] = np.partition(x[i-20:i+1], 8)[8] > 2 * ref
    expected[:60] = False
    np.testing.assert_array_equal(got, expected)


def test_prefix_invariance_and_hold_until_retained_update():
    x = np.r_[np.ones(60), np.array([10., 10., 10.]), np.ones(10)]
    a = phase_vote(x, np.arange(30), np.arange(30, 60), 3)
    b = phase_vote(np.r_[x, 1e9], np.arange(30), np.arange(30, 60), 3)
    np.testing.assert_array_equal(a["flags"], b["flags"][:len(x)])


def test_phase_decision_holds_then_refreshes_at_retained_acquisition():
    # stride 2, offset 0 retains 0,2,4,...; a decision at 62 is held at 63
    x = np.ones(80)
    x[40:63] = 10.0
    r = phase_vote(x, np.arange(30), np.arange(30, 60), stride=2, phases=1, votes=1)
    assert r["flags"][62] == r["flags"][63]
    assert r["metadata"]["phase_decisioncounts"] == [10]
    assert len(r["metadata"]["phase_alarmcounts"]) == 1


def test_long_stream_completes_without_recomputing_history():
    x = np.ones(10_000)
    r = phase_vote(x, np.arange(100), np.arange(100, 500), stride=5)
    assert len(r["flags"]) == len(x)
