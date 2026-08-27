"""The deployment bundle: what ships must match what the app opens.

A bundle that silently omits a file produces an app that starts and then
fails on the tab nobody tested, so the plan is pinned against the paths the
code actually reads.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "deploy_bundle", ROOT / "main" / "scripts" / "deploy_bundle.py")
db = importlib.util.module_from_spec(_spec)
sys.modules["deploy_bundle"] = db
_spec.loader.exec_module(db)


def test_every_tab_contributes_something():
    for tab in db.TAB_FILES:
        items = db.plan([tab], years=[2025])
        assert items, f"aba {tab} não pede nenhum arquivo"


def test_forecast_tab_ships_the_products_and_a_currents_file():
    """Without the currents field there are no antecedent features at all."""
    rels = [str(r) for r, _, _ in db.plan(["forecast"], years=[2025])]
    assert any("forecast_product.joblib" in r for r in rels)
    assert any("footprint_plume.joblib" in r for r in rels)
    assert any("currents" in r for r in rels)


def test_one_year_bundle_is_smaller_than_four():
    one = sum(s for _, s, _ in db.plan(["forecast"], years=[2025]))
    four = sum(s for _, s, _ in db.plan(["forecast"],
                                        years=[2022, 2023, 2024, 2025]))
    assert one < four


def test_plan_has_no_duplicates():
    items = db.plan(list(db.TAB_FILES), years=sorted(db.YEAR_FILES))
    rels = [str(r) for r, _, _ in items]
    assert len(rels) == len(set(rels))


def test_raw_inputs_are_never_bundled():
    """They feed prep, not the app — 410 MB that must not travel."""
    items = db.plan(list(db.TAB_FILES), years=sorted(db.YEAR_FILES))
    assert not [r for r, _, _ in items if "_raw" in str(r)]


def test_run_archives_are_never_bundled():
    """Evidence behind the published numbers, not data the app opens."""
    items = db.plan(list(db.TAB_FILES), years=sorted(db.YEAR_FILES))
    bad = [str(r) for r, _, _ in items
           if any(k in str(r) for k in ("training", "ensemble", "holdout"))]
    assert not bad, bad


def test_everything_the_plan_names_actually_exists():
    """A missing file here means the bundle would deploy a broken app."""
    missing = [str(r) for r, _, present in
               db.plan(list(db.TAB_FILES), years=sorted(db.YEAR_FILES))
               if not present]
    assert not missing, missing


def test_custom_run_tab_is_what_needs_the_wind_field():
    """It is the only tab that runs a simulation, and the wind is 86 MB."""
    without = [str(r) for r, _, _ in db.plan(["forecast"], years=[2025])]
    with_custom = [str(r) for r, _, _ in db.plan(["custom"], years=[2025])]
    assert not any("wind_cf" in r for r in without)
    assert any("wind_cf" in r for r in with_custom)
