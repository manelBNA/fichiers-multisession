"""Étape 1 : micros et orateurs présents partout, sans changement de comportement."""
import csv

from app.config import SourcesCfg
from app.sources import SpeakerRegistry, resolve_sources
from tests.conftest import make_timeline, run_fake_session
from tests.test_translator import finals, make


def test_default_profile_is_single_without_warning():
    profile, sources, warnings = resolve_sources(SourcesCfg())
    assert profile == "single" and [s.id for s in sources] == ["main"] and warnings == []


def test_future_profiles_fall_back_to_single_with_warning():
    for p in ("interview", "panel", "mixed", "custom", "inconnu"):
        profile, sources, warnings = resolve_sources(SourcesCfg(profile=p))
        assert profile == "single" and len(sources) == 1 and warnings


def test_speaker_registry():
    reg = SpeakerRegistry()
    a, b = reg.add("m1"), reg.add("m2")
    assert (a.id, b.id) == ("S1", "S2") and a.color != b.color
    reg.rename("S2", "Question du public")
    assert reg.get("S2").name == "Question du public"


def test_context_is_kept_per_speaker():
    tr, msgs, _ = make(["fr"])
    tr._final(1, "Hello everyone.", "en", "S1")
    tr._final(2, "Thank you for the talk.", "en", "S2")   # autre orateur : pas de contexte
    tr._final(3, "Today I will explain.", "en", "S1")     # même orateur : contexte
    assert finals(msgs) == [(1, "fr", False), (2, "fr", False), (3, "fr", True)]


def test_messages_and_logs_carry_source_and_speaker(patch_online_asr, tmp_path):
    tl = make_timeline([("Die Entwicklung wird durch Nachhaltigkeit geprägt. "
                         "Digitale Zwillinge begleiten dabei sämtliche Prozessschritte.", "de")])
    msgs = run_fake_session(tl, {"source": "de", "targets": ["en"], "voice": "none"}, tmp_path)
    segs = [m for m in msgs if m["type"] == "segment"]
    assert segs and all(m["source"] == "main" and m["speaker"] == "S1" for m in segs)
    assert all(m.get("speaker") == "S1" for m in msgs if m["type"] == "translation")
    (run,) = [m["path"] for m in msgs if m["type"] == "run_saved"]
    with open(f"{run}/segments.csv", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        cols, rows = reader.fieldnames, list(reader)
    # Les colonnes historiques restent en tête ; les nouvelles sont ajoutées après
    assert cols[:3] == ["id", "langue", "raison"]
    assert cols.index("texte") < cols.index("source") < cols.index("orateur")
    assert {r["orateur"] for r in rows} == {"S1"}
