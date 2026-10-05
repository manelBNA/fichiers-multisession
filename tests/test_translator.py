"""Traduction : cibles, contexte par paire, voix, sans lancer le thread."""
from app.config import MtCfg
from app.types import SessionSettings
from app.workers.translator_worker import TranslatorWorker
from tests.conftest import FakePool


def make(targets, voice="auto"):
    msgs, spoken = [], []
    s = SessionSettings("auto", targets, voice)
    tr = TranslatorWorker(FakePool(), MtCfg(), s, msgs.append,
                          lambda i, lang, text: spoken.append((i, lang)))
    return tr, msgs, spoken


def finals(msgs):
    return [(m["id"], m["lang"], m["context"]) for m in msgs
            if m["type"] == "translation" and m["final"]]


def test_each_target_except_own_language():
    tr, msgs, spoken = make(["en", "fr"])
    tr._final(1, "Hallo zusammen.", "de")
    assert sorted(l for _, l, _ in finals(msgs)) == ["en", "fr"]
    assert spoken == [(1, "en")]


def test_context_is_kept_per_pair():
    tr, msgs, _ = make(["en", "fr"])
    tr._final(1, "Hello everyone.", "en")
    tr._final(2, "Bonjour à tous.", "fr")
    tr._final(3, "Today I will explain.", "en")
    assert finals(msgs) == [(1, "fr", False), (2, "en", False), (3, "fr", True)]


def test_provisional_masks_last_words():
    tr, msgs, _ = make(["fr"])
    tr._provisional(5, "one two three four five six", "en")
    (m,) = [m for m in msgs if m["type"] == "translation"]
    assert m["final"] is False and m["hidden"] == 3
