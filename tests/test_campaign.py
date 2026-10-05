"""Campagne automatique : analyse après chaque test, rapport et fichier de campagne."""
import json

from tests.conftest import FakeMUPool, make_timeline, run_fake_session
from tests.test_meaning_units import SENT

LONG = SENT + " Die kontinuierliche Erfassung von Prozessdaten bildet die Grundlage, " \
              "damit Optimierungsverfahren automatisch arbeiten können."


def test_analysis_runs_automatically_after_the_test(patch_online_asr, tmp_path):
    msgs = run_fake_session(make_timeline([(LONG, "de")], word_s=0.4, gap_s=0.2),
                            {"source": "de", "targets": ["en"], "voice": "none"}, tmp_path,
                            label="de_test_campagne", pool=FakeMUPool(), analyze=True,
                            timeout_s=40)
    saved = [m for m in msgs if m["type"] == "run_saved"][0]
    assert saved["analysis"] is True
    progress = [m for m in msgs if m["type"] == "analysis_progress"]
    assert progress and progress[-1]["done"] == progress[-1]["total"] == 10   # 2 cibles x 5
    done = [m for m in msgs if m["type"] == "analysis_done"][0]
    assert "error" not in done and set(done["resume"]) == {"en", "fr"}

    run = saved["path"]
    result = json.load(open(f"{run}/analyse.json", encoding="utf-8"))
    en = result["cibles"]["en"]["politiques"]
    assert set(en) == {"B0", "P0", "MU_k2", "MU_k3", "MU_k4"}
    assert en["MU_k3"]["unites_validees"] >= 1
    assert en["MU_k3"]["delai_mot_moyen"] < en["P0"]["delai_mot_moyen"]
    assert en["MU_k3"]["gain_delai_pct"] > 0
    assert en["MU_k3"]["ecart_traduction_vs_P0_chrf"] == 100.0     # même traduction finale
    assert result["cibles"]["en"]["verdict_auto"].startswith("gain")
    assert "## de → en" in open(f"{run}/analyse.md", encoding="utf-8").read()

    lines = open(tmp_path / "runs" / "campagne.jsonl", encoding="utf-8").read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["test"] == "de_test_campagne"


def test_without_campaign_no_analysis(patch_online_asr, tmp_path):
    msgs = run_fake_session(make_timeline([(SENT, "de")]), {"source": "de", "targets": ["en"],
                            "voice": "none"}, tmp_path, pool=FakeMUPool())
    assert not any(m["type"].startswith("analysis") for m in msgs)
    assert not (tmp_path / "runs" / "campagne.jsonl").exists()
