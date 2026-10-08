"""MarianPool : chargement sérialisé dans tout le processus, nouvel essai, erreurs signalées.

Sans vrai modèle : le chargement (_load_translator) est remplacé par un faux traducteur.
"""
import threading
import time

import pytest

from app.config import MtCfg
from app.translation import marian_translator as mt
from app.types import SessionSettings
from app.workers.translator_worker import TranslatorWorker
from tests.conftest import FakePool


class FakeModel:
    def __init__(self, name):
        self.name = name

    def warmup(self):
        pass

    def translate(self, text):
        return f"[{self.name}] {text}"


@pytest.fixture
def loader(monkeypatch):
    """Faux chargement : `fail[name]` échecs avant succès ; `not_found` = modèle inexistant."""
    state = {"fail": {}, "not_found": set(), "calls": [], "active": 0, "max_active": 0}
    guard = threading.Lock()
    real_lock = mt._LOAD_LOCK

    def fake(name, cfg):
        with real_lock:                       # le vrai verrou global, comme _load_translator
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["calls"].append(name)
            try:
                time.sleep(0.02)
                if name in state["not_found"]:
                    raise OSError(f"{name} is not a valid model identifier listed on the Hub")
                if state["fail"].get(name, 0) > 0:
                    state["fail"][name] -= 1
                    raise NotImplementedError("Cannot copy out of meta tensor; no data!")
                return FakeModel(name)
            finally:
                with guard:
                    state["active"] -= 1

    monkeypatch.setattr(mt, "_load_translator", fake)
    monkeypatch.setattr(mt, "RETRY_DELAYS_S", (0.0, 0.0))
    return state


def name(src, tgt):
    return MtCfg().model_pattern.format(src=src, tgt=tgt)


def test_concurrent_loads_in_several_pools_are_serialized(loader):
    pools = [mt.MarianPool(MtCfg()) for _ in range(3)]
    pairs = [("de", "en"), ("en", "fr"), ("fr", "de")]
    barrier = threading.Barrier(6)

    def load(pool, pair):
        barrier.wait()
        pool.get(*pair)

    threads = [threading.Thread(target=load, args=(p, pair)) for p in pools[:2] for pair in pairs]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert loader["max_active"] == 1
    assert all(p.loaded() == ["de-en", "en-fr", "fr-de"] for p in pools[:2])
    assert all(not p.missing and not p.failures for p in pools)


def test_one_failure_is_retried_and_pair_stays_available(loader):
    loader["fail"][name("de", "fr")] = 1
    pool = mt.MarianPool(MtCfg())
    assert pool.get("de", "fr") is not None
    assert not pool.missing and not pool.failure_messages()


def test_repeated_failures_raise_without_missing_or_silent_pivot(loader, monkeypatch):
    monkeypatch.setattr(mt, "RETRY_AFTER_S", 60.0)
    loader["fail"][name("de", "fr")] = mt.LOAD_ATTEMPTS
    pool = mt.MarianPool(MtCfg())
    with pytest.raises(mt.MarianLoadError, match="de → fr indisponible"):
        pool.translate("de", "fr", "Guten Morgen.")
    assert not pool.missing
    assert pool.loaded() == []                         # le pivot n'a pas été utilisé
    assert len(pool.failure_messages()) == 1
    calls = len(loader["calls"])
    with pytest.raises(mt.MarianLoadError):            # pendant le délai : pas de rechargement
        pool.get("de", "fr")
    assert len(loader["calls"]) == calls


def test_failed_pair_is_retried_after_delay_and_error_cleared(loader, monkeypatch):
    monkeypatch.setattr(mt, "RETRY_AFTER_S", 0.05)
    loader["fail"][name("de", "fr")] = mt.LOAD_ATTEMPTS
    pool = mt.MarianPool(MtCfg())
    with pytest.raises(mt.MarianLoadError):
        pool.get("de", "fr")
    time.sleep(0.1)
    assert pool.translate("de", "fr", "Hallo.") == (f"[{name('de', 'fr')}] Hallo.", False)
    assert not pool.failure_messages()


def test_clear_failures_allows_immediate_retry(loader, monkeypatch):
    monkeypatch.setattr(mt, "RETRY_AFTER_S", 60.0)
    loader["fail"][name("de", "fr")] = mt.LOAD_ATTEMPTS
    pool = mt.MarianPool(MtCfg())
    with pytest.raises(mt.MarianLoadError):
        pool.get("de", "fr")
    pool.clear_failures()                              # nouvelle session dans la salle
    assert pool.get("de", "fr") is not None


def test_model_that_does_not_exist_goes_through_pivot(loader):
    loader["not_found"].add(name("de", "fr"))
    pool = mt.MarianPool(MtCfg())
    text, pivot = pool.translate("de", "fr", "Hallo.")
    assert pivot is True and ("de", "fr") in pool.missing
    assert pool.loaded() == ["de-en", "en-fr"] and not pool.failure_messages()


class OnePairBroken(FakePool):
    def translate_with_context(self, src, tgt, prev_src, prev_tgt, cur):
        if tgt == "fr":
            raise mt.MarianLoadError("Traduction de → fr indisponible")
        return super().translate_with_context(src, tgt, prev_src, prev_tgt, cur)


def test_failing_pair_does_not_drop_other_targets_and_is_reported():
    msgs = []
    tr = TranslatorWorker(OnePairBroken(), MtCfg(), SessionSettings("auto", ["en", "fr"], "auto"),
                          msgs.append, lambda *a: None)
    tr._final(1, "Hallo zusammen.", "de")
    assert [(m["lang"], m["final"]) for m in msgs if m["type"] == "translation"] == [("en", True)]
    (err,) = [m for m in msgs if m["type"] == "error"]
    assert err["where"] == "translation" and err["lang"] == "fr" and err["id"] == 1
