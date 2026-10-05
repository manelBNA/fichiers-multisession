"""Le rejeu d'un test enregistré redonne exactement les mêmes phrases.

C'est la garantie que scripts/replay_run.py est fiable pour vérifier la
non-régression sur de vrais tests (avec le vrai SaT, sur ta machine).
"""
import csv
import json

import pytest

from app.segmentation.boundary_policy import PolicyParams
from scripts.replay_run import replay
from tests.conftest import FakeSat, make_timeline, run_fake_session

CASES = {
    "fixe": ([("Die zukünftige Entwicklung wird durch Nachhaltigkeit geprägt. "
               "Digitale Zwillinge begleiten dabei sämtliche Prozessschritte.", "de")],
             {"source": "de", "targets": ["en"], "voice": "none"}),
    "changement_de_langue": ([("Hello everyone. Today I will explain how attention works.", "en"),
                              ("Bonjour à tous. Je vais maintenant répondre à votre question.", "fr")],
                             {"source": "auto", "targets": ["en", "fr"], "voice": "none"}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_replay_reproduces_recorded_segments(name, patch_online_asr, tmp_path):
    parts, settings = CASES[name]
    msgs = run_fake_session(make_timeline(parts), settings, tmp_path, label=name)
    (run,) = [m["path"] for m in msgs if m["type"] == "run_saved"]
    events = [json.loads(l) for l in open(f"{run}/events.jsonl", encoding="utf-8")]
    meta = json.load(open(f"{run}/meta.json", encoding="utf-8"))
    with open(f"{run}/segments.csv", encoding="utf-8-sig", newline="") as f:
        recorded = [r["texte"] for r in csv.DictReader(f, delimiter=";")]
    replayed = [s["text"] for s in replay(events, FakeSat(), PolicyParams(**meta["politique_de_coupe"]))]
    assert replayed == recorded and recorded
