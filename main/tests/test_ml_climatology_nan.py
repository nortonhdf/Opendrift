"""The climatology must survive targets that do not exist.

NOT YET EXECUTED — written on the machine being handed over, after the
defect was found in a live log but before the suite could be re-run. Run
`python -m pytest main/tests -o addopts=""` before trusting anything here.

The defect: a fully beached slick has no drifting centroid, so its target is
NaN at that horizon. The climatology averaged its block with plain `mean`,
so ONE such scenario made the whole block NaN and the baseline emitted no
prediction at all — at which point every model "beats" it by default. It
never fired while beaching was ~0 (the six fields) and fires immediately on
the seed grid, where coastal locations are deliberately included.
"""

import numpy as np
import pytest

from main.ml.forecast import (
    block_mean, fit_climatology, predict_climatology,
)


def test_one_missing_target_does_not_destroy_the_block():
    """The whole point: a single NaN must not silence the baseline."""
    Y = np.array([[10.0, 1.0], [20.0, np.nan], [30.0, 3.0]])
    got = block_mean(Y)
    assert got[0] == pytest.approx(20.0)
    assert got[1] == pytest.approx(2.0)      # would be NaN with plain mean


def test_a_block_with_no_outcome_at_all_stays_nan():
    """Honest answer when there is genuinely nothing to average."""
    got = block_mean(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert got[0] == pytest.approx(2.0)
    assert np.isnan(got[1])


def test_climatology_predicts_where_a_block_has_partial_data():
    Y = np.array([[10.0, 1.0], [20.0, np.nan], [100.0, 100.0]])
    blocks = np.array(["a_jan", "a_jan", "b_jul"])
    table, glob = fit_climatology(Y, blocks)
    p = predict_climatology((table, glob), np.array(["a_jan"]))
    assert p[0][0] == pytest.approx(15.0)
    assert p[0][1] == pytest.approx(1.0)     # the NaN row is skipped, not fatal


def test_climatology_fallback_also_ignores_missing_targets():
    Y = np.array([[10.0, np.nan], [20.0, 4.0]])
    blocks = np.array(["a_jan", "a_jan"])
    _, glob = fit_climatology(Y, blocks)
    assert glob[1] == pytest.approx(4.0)
