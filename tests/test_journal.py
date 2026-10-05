"""Journal unique : chaque test y est ajouté, avec tous ses détails."""
import json

from tests.conftest import FakeMUPool, make_timeline, run_fake_session
from tests.test_campaign import LONG

TL = [("Die Entwicklung wird durch Nachhaltigkeit geprägt. "
       "Digitale Zwillinge begleiten dabei sämtliche Prozessschritte.", "de")]


def read_journal(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "runs" / "journal_des_tests.jsonl",
                                        encoding="utf-8")]


def test_every_test_is_appended_with_all_details(patch_online_asr, tmp_path):
    for i in range(2):
        msgs = run_fake_session(make_timeline(TL), {"source": "de", "targets": ["en"],
                                "voice": "none"}, tmp_path, label=f"essai_{i + 1}")
        assert [m for m in msgs if m["type"] == "run_saved"][0]["journal"].endswith(
            "journal_des_tests.jsonl")
    tests = read_journal(tmp_path)
    assert [t["test"] for t in tests] == ["essai_1", "essai_2"]
    t = tests[0]
    assert t["reglages"]["politique_de_coupe"]["th_high"] and t["reglages"]["reglages_langues"]
    assert t["resultats_direct"]["phrases"] == len(t["phrases"]) == 2
    p = t["phrases"][0]
    assert p["texte"].startswith("Die Entwicklung") and p["raison"] and "traduction_en" in p
    assert isinstance(p["id"], int) and t["analyse"] is None


def test_campaign_test_is_written_once_with_its_analysis(patch_online_asr, tmp_path):
    run_fake_session(make_timeline([(LONG, "de")], word_s=0.4, gap_s=0.2),
                     {"source": "de", "targets": ["en"], "voice": "none"}, tmp_path,
                     label="camp_01_vortrag", pool=FakeMUPool(), analyze=True, timeout_s=40)
    (t,) = read_journal(tmp_path)                      # une seule ligne, analyse comprise
    assert t["test"] == "camp_01_vortrag" and t["phrases"]
    assert set(t["analyse"]["cibles"]) == {"en", "fr"}
    assert "MU_k3" in t["analyse"]["cibles"]["en"]["politiques"]


def test_settings_changed_during_the_test_are_recorded(patch_online_asr, tmp_path):
    import asyncio
    import time

    import numpy as np

    from app.config import load_config
    from app.session import Session
    from tests.conftest import FakeRegistry

    cfg = load_config()
    cfg.asr.min_chunk_s = 0.5
    cfg.logs.runs_dir = str(tmp_path / "runs")
    tl = make_timeline(TL)

    async def main():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        s = Session(FakeRegistry(tl), cfg, loop, q)
        s.start({"source": "de", "targets": ["en"], "voice": "none"}, "reglages_changes")
        chunk = (np.ones(1600) * 3000).astype(np.int16).tobytes()
        for i in range(int((tl[-1][1] + 1.5) * 10)):
            s.push_audio(chunk)
            if i == 10:
                s.update_policy({"th_high": 0.7})
            await asyncio.sleep(0.004)
        s.stop()
        end = time.time() + 20
        while time.time() < end:
            try:
                m = await asyncio.wait_for(q.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            if m.get("type") == "run_saved":
                break
        s.close()

    asyncio.run(main())
    (t,) = read_journal(tmp_path)
    assert t["changements_pendant_le_test"][0]["valeurs"] == {"th_high": 0.7}
