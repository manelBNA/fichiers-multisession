"""Étape 2 : plusieurs micros, un orateur par micro (profil « panel »)."""
import time

import numpy as np
import pytest

from app.audio_framing import make_frame, parse_frame
from app.config import SourcesCfg
from app.coordinator import Automixer, DuplicateFilter
from app.sources import SourceSettings, resolve_sources
from app.types import SessionSettings
from tests.conftest import make_timeline, run_fake_multi_session

SOURCES = [{"id": "m1", "name": "Intervenante", "language": "en"},
           {"id": "m2", "name": "Public", "language": "fr"}]
A = "Hello everyone. Today I will explain how attention works in transformers."
B = "Bonjour à tous. Je vais maintenant répondre à votre question."


def shift(tl, dt):
    return [(s + dt, e + dt, w, l) for s, e, w, l in tl]


def span(tl):
    return tl[0][0], tl[-1][1]


def segs(msgs):
    return [(m["source"], m["speaker"], m["text"]) for m in msgs if m["type"] == "segment"]


# ---------- composants ----------

def test_frames_legacy_and_tagged():
    pcm = np.arange(8, dtype=np.int16).tobytes()
    assert parse_frame(pcm, tagged=False) == (None, pcm)
    assert parse_frame(make_frame(3, pcm), tagged=True) == (3, pcm)
    with pytest.raises(ValueError):
        parse_frame(b"\x07\x00" + pcm, tagged=True)


def test_automixer_silences_bleed_but_keeps_simultaneous_speech():
    mix = Automixer(level_margin_db=6)
    loud, bleed, other = np.full(1600, 0.3), np.full(1600, 0.03), np.full(1600, 0.25)
    # bloc 0 : m2 (diaphonie, 20 dB plus bas) arrive AVANT m1 ; bloc 1 : parole simultanée
    assert mix.process("m2", bleed) is None and mix.process("m1", loud) is None
    assert not mix.process("m2", other).any()          # bloc 0 de m2 décidé : silence
    assert mix.process("m1", loud).any()               # bloc 0 de m1 gardé
    assert mix.flush("m2").any()                       # bloc 1 de m2 : gardé (voix réelle)
    assert mix.flush("m1").any()
    assert mix.floor_holder() in ("m1", "m2")


def test_duplicate_filter():
    f = DuplicateFilter(similarity=0.8)
    assert f.accept("m1", 0.0, 3.0, "Hello everyone, today I explain attention.")
    assert not f.accept("m2", 0.1, 3.1, "HELLO EVERYONE, TODAY I EXPLAIN ATTENTION.")
    assert f.accept("m2", 0.2, 3.0, "Bonjour à tous, je réponds à votre question.")


def test_resolve_panel_from_interface():
    profile, sources, warnings = resolve_sources(SourcesCfg(), {
        "profile": "panel",
        "sources": [{"id": "Micro 1!", "name": "Intervenante", "language": "de"},
                    {"id": "m2", "name": "Public", "mode": "auto", "language": "xx"}]})
    assert profile == "panel"
    assert [(s.id, s.index, s.language) for s in sources] == [("micro1", 0, "de"), ("m2", 1, None)]
    assert len(warnings) == 2   # mode « auto » (étape 3) et langue inconnue


def test_resolve_interview_still_waits_for_step3():
    profile, sources, warnings = resolve_sources(SourcesCfg(profile="interview"))
    assert profile == "single" and len(sources) == 1 and warnings


def test_source_language_overrides_session():
    settings = SessionSettings("auto", ["fr"], "auto")
    src = resolve_sources(SourcesCfg(), {"profile": "panel", "sources": [
        {"id": "m1", "language": "de"}, {"id": "m2"}]})[1]
    assert SourceSettings(settings, src[0]).source == "de"
    assert SourceSettings(settings, src[1]).source == "auto"
    assert SourceSettings(settings, src[1]).targets_for("en") == ["fr"]


# ---------- sessions complètes ----------

def _turns():
    tl_a = make_timeline([(A, "en")])
    tl_b = shift(make_timeline([(B, "fr")]), span(tl_a)[1] + 1.5)
    (a0, a1), (b0, b1) = span(tl_a), span(tl_b)
    # Chaque micro entend aussi l'autre orateur, 20 dB plus bas (diaphonie)
    timelines = {"m1": tl_a + tl_b, "m2": tl_a + tl_b}

    def amplitude(src, t):
        in_a, in_b = a0 - 0.1 <= t <= a1 + 0.1, b0 - 0.1 <= t <= b1 + 0.1
        if src == "m1":
            return 1.0 if in_a else (0.1 if in_b else 0.0)
        return 1.0 if in_b else (0.1 if in_a else 0.0)
    return timelines, amplitude, b1 + 1.5


def test_turn_taking_with_crosstalk_protection(patch_online_asr, tmp_path):
    timelines, amplitude, total = _turns()
    msgs = run_fake_multi_session(SOURCES, timelines, amplitude,
                                  {"source": "auto", "targets": ["en", "fr"], "voice": "auto"},
                                  tmp_path, total)
    got = segs(msgs)
    assert [(s, spk) for s, spk, _ in got] == [("m1", "S1"), ("m2", "S2"), ("m2", "S2")]
    assert got[0][2] == A and " ".join(t for *_, t in got[1:]) == B
    # traductions : l'anglais de m1 vers le français, le français de m2 vers l'anglais
    tr = {(m["id"], m["lang"], m["speaker"]) for m in msgs
          if m["type"] == "translation" and m["final"]}
    assert tr == {(1, "fr", "S1"), (2, "en", "S2"), (3, "en", "S2")}
    assert any(m["type"] == "run_saved" and m["complete"] for m in msgs)


def test_without_crosstalk_protection_bleed_is_transcribed_twice(patch_online_asr, tmp_path):
    timelines, amplitude, total = _turns()
    msgs = run_fake_multi_session(SOURCES, timelines, amplitude,
                                  {"source": "auto", "targets": ["en", "fr"], "voice": "none"},
                                  tmp_path, total, crosstalk=False)
    texts = [t.lower() for *_, t in segs(msgs)]
    assert sum("hello everyone" in t for t in texts) >= 2   # la diaphonie crée des doublons


def test_simultaneous_speech_on_two_microphones(patch_online_asr, tmp_path):
    tl_a = make_timeline([(A, "en")])
    tl_b = shift(make_timeline([(B, "fr")]), 0.5)          # B parle en même temps que A
    end = max(span(tl_a)[1], span(tl_b)[1])

    def amplitude(src, t):
        return 1.0 if t <= end + 0.1 else 0.0

    msgs = run_fake_multi_session(SOURCES, {"m1": tl_a, "m2": tl_b}, amplitude,
                                  {"source": "auto", "targets": ["en", "fr"], "voice": "auto"},
                                  tmp_path, end + 1.5)
    by_src = {}
    for s, spk, text in segs(msgs):
        by_src.setdefault((s, spk), []).append(text)
    assert " ".join(by_src[("m1", "S1")]) == A
    assert " ".join(by_src[("m2", "S2")]) == B
    # une seule voix à la fois : les phrases qui chevauchent l'autre orateur ne sont pas lues
    spoken = [m for m in msgs if m["type"] == "speak"]
    skipped = [m for m in msgs if m["type"] == "speak_skipped"]
    assert spoken and skipped
    assert len({m["speaker"] for m in spoken}) >= 1
