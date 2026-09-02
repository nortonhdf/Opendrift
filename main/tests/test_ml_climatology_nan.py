"""The climatology must survive targets that do not exist.

VERIFIED 2026-09-02 on the new machine: 151 passed. (The handover block in
main/CLAUDE.md predicted 133 = 129 + 4, but the 129 was itself stale — the
suite was already at 147 before these tests landed.)

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
    HORIZONS_D, block_mean, fit_climatology, leave_one_field_out,
    predict_climatology,
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


def test_paired_sample_is_not_shrunk_by_a_beached_training_run():
    """The failure as it actually presented: missing sample, not an error.

    The unit tests above pin block_mean. This one pins the consequence, which
    is what went unnoticed for a whole 57-minute run: the evaluation still
    finished, still printed a table, and simply had fewer scenarios in it —
    480 paired at D+1, 241 at D+3, 3 at D+5, 0 at D+7. Nothing raised, so
    only the shrinking denominator gave it away.
    """
    rng = np.random.default_rng(0)
    fields = np.repeat(["a", "b", "c"], 12)
    seasons = np.tile(np.repeat(["jan", "jul"], 6), 3)
    blocks = np.array([f"{f}_{s}" for f, s in zip(fields, seasons)])
    feat_names = np.array(["f0", "f1", "f2"])

    X = rng.normal(size=(36, 3))
    Y = rng.normal(size=(36, len(HORIZONS_D) * 4)) * 10.0

    clean = leave_one_field_out(X, Y.copy(), blocks, fields, feat_names)

    # Beach one run of field "a", season "jan". It is TRAINING data for the
    # folds holding out "b" and "c", so with plain mean it voided january's
    # baseline in both, taking every january test scenario with it.
    Y_beached = Y.copy()
    Y_beached[0, :] = np.nan
    beached = leave_one_field_out(X, Y_beached, blocks, fields, feat_names)

    for h in HORIZONS_D:
        n_clean = clean["paired"][f"D+{h}"]["n"]
        n_beached = beached["paired"][f"D+{h}"]["n"]
        assert n_beached >= n_clean - 1, (
            f"D+{h}: {n_clean} -> {n_beached}; one beached run cost more than "
            "itself, so a block mean was poisoned")
