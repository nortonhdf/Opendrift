"""Footprint layer: relative grid, swept targets, metrics and the plume kernel.

The load-bearing properties here are (a) the release-relative grid must be an
exact, reversible mapping — everything downstream is expressed in it, (b) the
target must be the SWEPT area and must exclude particles that are no longer
active, and (c) the negative subsampling must preserve the base rate, because
Brier and the capture area are read straight off the fitted probabilities.
"""

import numpy as np
import pytest
import xarray as xr

from main.ml import footprint as fp
from main.ml import footprint_forecast as ff
from main.ml.metrics import iou as metrics_iou
from main.ml.scenario import HORIZONS_D


# ── relative grid ────────────────────────────────────────────────────────────

def test_cell_offsets_invert_cell_of():
    dx = np.array([0.0, 25.0, -140.0, 333.3])
    dy = np.array([0.0, -60.0, 12.5, -401.0])
    ids, inside = fp.cell_of(dx, dy)
    assert inside.all()
    cx, cy = fp.cell_offsets_km(ids)
    # A cell centre is within half a cell of any point inside that cell.
    assert np.all(np.abs(cx - dx) <= fp.CELL_KM / 2 + 1e-9)
    assert np.all(np.abs(cy - dy) <= fp.CELL_KM / 2 + 1e-9)


def test_out_of_extent_is_flagged_not_wrapped():
    """A particle beyond the frame must be reported, never folded back in."""
    ids, inside = fp.cell_of(np.array([900.0, 0.0]), np.array([0.0, -700.0]))
    assert not inside.any()
    assert (ids == -1).all()


def test_cells_map_back_to_geography():
    lon0, lat0 = -40.0, -22.0
    ids, _ = fp.cell_of(np.array([100.0]), np.array([-50.0]))
    lon, lat = fp.cells_to_lonlat(ids, lon0, lat0)
    dx = (lon[0] - lon0) * fp.KM_PER_DEG * np.cos(np.radians(lat0))
    dy = (lat[0] - lat0) * fp.KM_PER_DEG
    assert dx == pytest.approx(100.0, abs=fp.CELL_KM)
    assert dy == pytest.approx(-50.0, abs=fp.CELL_KM)


# ── swept targets ────────────────────────────────────────────────────────────

def _run(tmp_path, hours=168, deg_per_day=0.5, status=None, name="run.nc"):
    """Patch drifting due east at a fixed rate, 30-min output."""
    n_t = hours * 2 + 1
    times = np.array([np.datetime64("2025-01-01") + np.timedelta64(30 * i, "m")
                      for i in range(n_t)])
    days = np.arange(n_t) * 0.5 / 24.0
    lon = np.tile(-40.0 + deg_per_day * days, (3, 1))
    lat = np.full((3, n_t), -22.0)
    st = np.zeros((3, n_t), int) if status is None else status
    ds = xr.Dataset(
        {"lon": (("trajectory", "time"), lon),
         "lat": (("trajectory", "time"), lat),
         "status": (("trajectory", "time"), st)},
        coords={"time": times},
    )
    ds["status"].attrs = {"flag_values": np.array([0, 1], np.int32),
                          "flag_meanings": "active stranded"}
    p = tmp_path / name
    ds.to_netcdf(p)
    return p


def test_swept_grows_with_horizon_and_contains_the_snapshot(tmp_path):
    m = fp.masks_from_run(_run(tmp_path))
    sizes = [len(m["swept"][h]) for h in HORIZONS_D]
    assert sizes == sorted(sizes), "a área varrida não pode encolher"
    assert sizes[-1] > sizes[0]
    for h in HORIZONS_D:
        assert set(m["snap"][h]).issubset(set(m["swept"][h]))
    # 0.5 deg/day east at lat -22 is ~51.6 km/day; by D+7 the ribbon spans
    # ~360 km, i.e. tens of 11-km cells, while the snapshot is one.
    assert len(m["snap"][7]) == 1
    assert len(m["swept"][7]) >= 25


def test_deactivated_particles_stop_contributing_cells(tmp_path):
    """A stranded element must not keep painting cells along a ghost track."""
    n_t = 168 * 2 + 1
    st = np.zeros((3, n_t), int)
    st[:, 96:] = 1                       # all stranded after 48 h (D+2)
    m = fp.masks_from_run(_run(tmp_path, status=st, name="stranded.nc"))
    assert len(m["swept"][2]) == len(m["swept"][7]), \
        "células apareceram depois de todas as partículas encalharem"


def test_horizon_beyond_the_run_is_none(tmp_path):
    m = fp.masks_from_run(_run(tmp_path, hours=120, name="short.nc"))
    assert m["swept"][5] is not None
    assert m["swept"][7] is None and m["snap"][7] is None


def test_release_cell_is_always_oiled(tmp_path):
    m = fp.masks_from_run(_run(tmp_path))
    home, _ = fp.cell_of(np.array([0.0]), np.array([0.0]))
    assert home[0] in set(m["swept"][1])


# ── metrics ──────────────────────────────────────────────────────────────────

def test_iou_rows_agrees_with_the_reference_implementation():
    rng = np.random.default_rng(0)
    P = rng.random((5, 40))
    T = rng.random((5, 40)) > 0.7
    got = ff.iou_rows(P, T, 0.5)
    for r in range(5):
        assert got[r] == pytest.approx(metrics_iou(P[r] >= 0.5, T[r]))


def test_capture_area_of_a_perfect_forecast_is_the_footprint_itself():
    """Ranking the true cells first, 80 % capture costs exactly ceil(0.8n)."""
    T = np.zeros((1, 100), bool)
    T[0, :10] = True
    P = T.astype(float)                       # perfect, confident
    area = ff.capture_area_rows(P, T)[0]
    assert area == pytest.approx(8 * fp.CELL_KM ** 2)


def test_capture_area_punishes_a_wrong_ranking():
    T = np.zeros((1, 100), bool)
    T[0, :10] = True
    P = np.linspace(0, 1, 100)[None, :]       # ranks the truth last
    good = ff.capture_area_rows(T.astype(float), T)[0]
    bad = ff.capture_area_rows(P, T)[0]
    assert bad > good


def test_capture_area_is_nan_when_nothing_was_oiled():
    T = np.zeros((1, 20), bool)
    assert np.isnan(ff.capture_area_rows(np.zeros((1, 20)), T)[0])


# ── model internals ──────────────────────────────────────────────────────────

def test_negative_sampling_preserves_the_base_rate():
    """Weighted counts must reconstruct the full candidate set, not the sample."""
    rng = np.random.default_rng(1)
    n_cand, pos = 5000, np.arange(40)
    sel, lab, w = ff.sample_cells(pos, n_cand, rng, n_neg=200)
    assert len(sel) == 240
    assert w[lab == 1].sum() == pytest.approx(40)
    assert w[lab == 0].sum() == pytest.approx(n_cand - 40)
    base = w[lab == 1].sum() / w.sum()
    assert base == pytest.approx(40 / n_cand, rel=1e-9)


def test_sampling_never_labels_a_positive_as_negative():
    rng = np.random.default_rng(2)
    pos = np.array([3, 7, 11])
    sel, lab, _ = ff.sample_cells(pos, 50, rng, n_neg=40)
    assert set(sel[lab == 0]).isdisjoint(set(pos))


def test_occupancy_columns_match_their_declared_names():
    """The exported product feeds rows in this order — pin it."""
    x = np.arange(8, dtype=np.float32)
    cdx = np.array([10.0, -10.0])
    cdy = np.array([0.0, 0.0])
    iu, iv = 2, 3                       # x[2] = u = 2 m/s, x[3] = v = 3 m/s
    F = ff.occupancy_features(x, cdx, cdy, iu, iv)
    assert F.shape == (2, len(x) + len(ff.GEOM_NAMES))
    names = list(ff.GEOM_NAMES)
    speed = np.hypot(2.0, 3.0)
    down = names.index("cell_down_km") + len(x)
    cross = names.index("cell_cross_km") + len(x)
    assert F[0, down] == pytest.approx(10.0 * 2.0 / speed)
    assert F[0, cross] == pytest.approx(10.0 * -3.0 / speed)
    assert F[1, down] == pytest.approx(-F[0, down])


def test_occupancy_flow_columns_are_nan_without_an_antecedent_window():
    x = np.zeros(8, np.float32)
    x[2] = x[3] = np.nan             # the 7-day window fell outside the year
    F = ff.occupancy_features(x, np.array([10.0]), np.array([0.0]), 2, 3)
    assert np.isnan(F[0, -1]) and np.isnan(F[0, -2])


def test_dist_to_path_measures_the_path_not_the_endpoint():
    """The footprint is a ribbon along the track — an L-shaped path proves it."""
    path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    cx = np.array([50.0, 100.0])          # mid-leg, and near the corner
    cy = np.array([0.0, 50.0])
    d = ff.dist_to_path(path, cx, cy)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.0)
    # A cell straight past the endpoint is far from every segment.
    far = ff.dist_to_path(path, np.array([100.0]), np.array([300.0]))
    assert far[0] == pytest.approx(200.0)


def test_path_frame_measures_arc_length_and_offset_in_km():
    """Along = distance travelled on the path; across = signed offset."""
    path = np.array([[0.0, 0.0], [100.0, 0.0]])
    s, c, total = ff.path_frame(path, np.array([50.0, 150.0, -30.0]),
                                np.array([25.0, 0.0, 0.0]))
    assert total == pytest.approx(100.0)
    assert s[0] == pytest.approx(50.0)
    assert abs(c[0]) == pytest.approx(25.0)
    # Past the end and before the start the axis keeps going, so an
    # overshooting footprint does not pile onto the last bin.
    assert s[1] == pytest.approx(150.0)
    assert s[2] == pytest.approx(-30.0)


def test_path_frame_follows_a_bend():
    """Arc length accumulates along the polyline, not along the chord."""
    path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    s, c, total = ff.path_frame(path, np.array([100.0]), np.array([50.0]))
    assert total == pytest.approx(200.0)
    assert s[0] == pytest.approx(150.0)
    assert c[0] == pytest.approx(0.0, abs=1e-9)


def test_path_frame_undefined_without_displacement():
    assert ff.path_frame(np.array([[0.0, 0.0], [0.0, 0.0]]),
                         np.array([1.0]), np.array([1.0])) is None


def test_plume_kernel_recovers_a_planted_shape():
    """Plant a ribbon along the predicted track; the kernel must find it."""
    cdx, cdy = np.meshgrid(np.arange(-100, 301, 10.0),
                           np.arange(-150, 151, 10.0), indexing="xy")
    cdx, cdy = cdx.ravel(), cdy.ravel()
    paths, truths = [], []
    for L in (100.0, 200.0, 300.0):
        p = np.array([[0.0, 0.0], [L, 0.0]])
        a, c, _ = ff.path_frame(p, cdx, cdy)
        # In km now, so the planted ribbon is the same physical strip for
        # every scale — which is the point of dropping the normalisation.
        truths.append((a > 20.0) & (a < L - 20.0) & (np.abs(c) < 30.0))
        paths.append(p)
    T = np.array(truths)
    ok = np.ones(len(paths), bool)
    kernel = ff.fit_plume_kernel(paths, T, ok, cdx, cdy)
    P = ff.predict_plume(kernel, paths, cdx, cdy, np.zeros_like(T, float))
    assert P[T].mean() > 0.6
    assert P[~T].mean() < 0.15
    assert P[T].mean() > 4 * P[~T].mean()


def test_plume_falls_back_when_the_frame_is_undefined():
    cdx = np.array([0.0, 10.0])
    cdy = np.array([0.0, 0.0])
    ea, ec = ff.plume_bins(300.0)
    kernel = {"P": np.zeros((len(ea) - 1, len(ec) - 1)), "base": 0.0,
              "outside": 0.0, "edges_a": ea, "edges_c": ec}
    fallback = np.array([[0.42, 0.42]])
    P = ff.predict_plume(kernel, [np.array([[0.0, 0.0], [0.0, 0.0]])],
                         cdx, cdy, fallback)
    assert np.allclose(P, 0.42)


# ── the exported product (what the app will call) ────────────────────────────

class _Const:
    """Stand-in for one fitted regressor: always predicts the same value."""

    def __init__(self, v):
        self.v = float(v)

    def predict(self, X):
        return np.full(len(X), self.v)


class _Corridor:
    """Stand-in for the fitted isotonic distance -> probability mapping."""

    def predict(self, d):
        return np.exp(-np.asarray(d, float) / 50.0)


def _payload(cand_ids, P, edges_a, edges_c, horizon=1, dx=100.0, dy=0.0,
             default="plume"):
    """Shapes payload + the v4 payload that places them — they ship apart."""
    from main.ml.forecast import Q, tcol
    models = [_Const(0.0) for _ in range(len(HORIZONS_D) * len(Q))]
    hi = HORIZONS_D.index(horizon)
    models[tcol(hi, "dx_km")] = _Const(dx)
    models[tcol(hi, "dy_km")] = _Const(dy)
    shapes = {
        "feature_names": ["a", "b", "c"], "centroid_from": "test",
        "horizons_d": HORIZONS_D, "cell_km": fp.CELL_KM,
        "default_model": default,
        "kernels": {horizon: {
            "P": P, "outside": 0.0, "base": 0.0, "radius_km": 400.0,
            "edges_a": edges_a, "edges_c": edges_c,
            "cand_ids": np.asarray(cand_ids, np.int32),
            "isotonic": _Corridor(),
            "climatology": {"jan": np.full(len(cand_ids), 0.11)},
            "threshold": {"plume": 0.3, "centroid": 0.25}}},
    }
    return shapes, {"point_models": models, "horizons_d": HORIZONS_D}


def test_predict_footprint_draws_the_corridor_by_default():
    """The product default is the shape that won leave-one-field-out."""
    ids, _ = fp.cell_of(np.array([50.0, -200.0]), np.array([0.0, 0.0]))
    ea, ec = ff.plume_bins(300.0)
    P = np.zeros((len(ea) - 1, len(ec) - 1))
    shapes, fc = _payload(ids, P, ea, ec, default="centroid")
    out = ff.predict_footprint(shapes, fc, np.zeros(3), 1, lon0=-40.0,
                               lat0=-22.0, season="jan")
    assert out["model"] == "centroid"
    assert out["threshold"] == pytest.approx(0.25)
    # On the track it is near-certain; 300 km upstream it is not.
    assert out["prob"][0] > out["prob"][1]
    assert out["prob"][0] == pytest.approx(np.exp(-5.566 / 50.0), rel=0.05)


def test_predict_footprint_places_the_plume_on_the_predicted_track():
    ids, _ = fp.cell_of(np.array([50.0, -200.0]), np.array([0.0, 0.0]))
    ea, ec = ff.plume_bins(300.0)
    P = np.zeros((len(ea) - 1, len(ec) - 1))
    # Coordinates are km along and across the 100 km predicted track, and a
    # cell centre sits up to half a cell off it.
    w = ea[1] - ea[0]
    on_track = (ea[:-1] >= 0.0) & (ea[:-1] < 100.0)
    centred = np.abs(ec[:-1] + w / 2) < 15.0
    P[np.ix_(on_track, centred)] = 1.0
    shapes, fc = _payload(ids, P, ea, ec)
    out = ff.predict_footprint(shapes, fc, np.zeros(3), 1, lon0=-40.0,
                               lat0=-22.0, season="jan", model="plume")
    assert out["prob"][0] == pytest.approx(1.0)     # halfway along the track
    assert out["prob"][1] == pytest.approx(0.0)     # upstream of the release
    assert out["path_dx_km"][-1] == pytest.approx(100.0)
    lon, lat = fp.cells_to_lonlat(ids, -40.0, -22.0)
    assert out["lon"] == pytest.approx(lon)
    assert out["lat"] == pytest.approx(lat)


def test_predict_footprint_falls_back_to_the_season_climatology():
    """No predicted displacement means no frame — the app still gets a field."""
    ids, _ = fp.cell_of(np.array([50.0]), np.array([0.0]))
    ea, ec = ff.plume_bins(300.0)
    P = np.ones((len(ea) - 1, len(ec) - 1))
    shapes, fc = _payload(ids, P, ea, ec, dx=0.0, dy=0.0)
    out = ff.predict_footprint(shapes, fc, np.zeros(3), 1, lon0=-40.0,
                               lat0=-22.0, season="jan", model="plume")
    assert out["prob"][0] == pytest.approx(0.11)


# ── the join between the two datasets ────────────────────────────────────────

def _write_fp(path, uids):
    n = len(uids)
    arrays = {}
    for kind in fp.KINDS:
        for h in HORIZONS_D:
            arrays[f"{kind}_cells_d{h}"] = np.arange(n, dtype=np.int32)
            arrays[f"{kind}_ptr_d{h}"] = np.arange(n + 1, dtype=np.int64)
    np.savez_compressed(
        path, uid=np.asarray(uids),
        field=np.array(["Marlim"] * n), season=np.array(["jan"] * n),
        year=np.array([int(u.split(":")[0]) for u in uids], np.int32),
        lon0=np.zeros(n), lat0=np.zeros(n),
        horizons_d=np.asarray(HORIZONS_D, np.int32),
        cell_km=np.float64(fp.CELL_KM), n_side=np.int32(fp.N_SIDE),
        half_cells=np.int32(fp.HALF_CELLS), n_out_of_extent=np.int64(0),
        **arrays)


def _write_sc(path, uids, n_feat=3):
    years = [int(u.split(":")[0]) for u in uids]
    keys = [u.split(":", 1)[1] for u in uids]
    np.savez_compressed(
        path, X=np.arange(len(uids) * n_feat, dtype=np.float32).reshape(
            len(uids), n_feat),
        Y=np.zeros((len(uids), 4), np.float32),
        year=np.asarray(years, np.int32), run_key=np.asarray(keys),
        feature_names=np.asarray(["a", "b", "c"]),
        block=np.asarray(["Marlim_jan"] * len(uids)),
        field=np.asarray(["Marlim"] * len(uids)),
        season=np.asarray(["jan"] * len(uids)))


def test_join_is_by_run_identity_not_position(tmp_path):
    """The two builders skip runs independently — position joins would lie."""
    fp_path, sc_path = tmp_path / "fp.npz", tmp_path / "sc.npz"
    _write_fp(fp_path, ["2022:a_jan_d01", "2022:b_jan_d02", "2023:c_jan_d03"])
    # Same runs, different order, plus one the footprint builder dropped.
    _write_sc(sc_path, ["2023:c_jan_d03", "2022:zz_jan_d99", "2022:a_jan_d01",
                        "2022:b_jan_d02"])
    ctx = ff.load_pair(fp_path, sc_path)
    assert list(ctx["uid"]) == ["2022:a_jan_d01", "2022:b_jan_d02",
                                "2023:c_jan_d03"]
    # Features must follow the identity, not the row number.
    assert ctx["X"][0][0] == pytest.approx(6.0)     # third row of the sc file
    assert ctx["X"][2][0] == pytest.approx(0.0)     # first row of the sc file


def test_join_keeps_the_footprint_index_for_lookups(tmp_path):
    """cells_of must reindex through ctx['row'] — the join drops rows."""
    fp_path, sc_path = tmp_path / "fp2.npz", tmp_path / "sc2.npz"
    _write_fp(fp_path, ["2022:a_jan_d01", "2022:b_jan_d02", "2022:c_jan_d03"])
    _write_sc(sc_path, ["2022:b_jan_d02", "2022:c_jan_d03"])
    ctx = ff.load_pair(fp_path, sc_path)
    assert list(ctx["row"]) == [1, 2]
    assert ff.cells_of(ctx, 0, HORIZONS_D[0])[0] == 1
