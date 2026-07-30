"""--fresh must wipe ALL selected stage manifests up front (2026-07-30 bug).

The old per-stage deletion left later manifests stale after a crash in an
earlier stage, so a --resume silently mixed output generations.
"""

from main.scripts import rebuild_all


def test_wipe_manifests_removes_all_selected(tmp_path, monkeypatch):
    m_scen = tmp_path / "scen_manifest.json"
    m_ens = tmp_path / "ens_manifest.json"
    m_scen.write_text("{}")
    m_ens.write_text("{}")
    monkeypatch.setattr(rebuild_all, "MANIFESTS",
                        {"scenarios": m_scen, "ensemble": m_ens})
    monkeypatch.setattr(rebuild_all, "ROOT", tmp_path)

    # Wiping ONLY the later stage's selection must not require reaching it.
    rebuild_all.wipe_manifests(["scenarios", "ensemble", "risk", "beaching"])
    assert not m_scen.exists()
    assert not m_ens.exists()


def test_wipe_manifests_respects_stage_subset(tmp_path, monkeypatch):
    m_scen = tmp_path / "scen_manifest.json"
    m_ens = tmp_path / "ens_manifest.json"
    m_scen.write_text("{}")
    m_ens.write_text("{}")
    monkeypatch.setattr(rebuild_all, "MANIFESTS",
                        {"scenarios": m_scen, "ensemble": m_ens})
    monkeypatch.setattr(rebuild_all, "ROOT", tmp_path)

    rebuild_all.wipe_manifests(["ensemble"])
    assert m_scen.exists()          # not selected — untouched
    assert not m_ens.exists()
