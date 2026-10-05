"""Validation des mots stables (LocalAgreement) et changement de langue."""
import numpy as np

from app.asr.local_agreement import SR, HypothesisBuffer, OnlineASR
from app.config import AsrCfg
from app.types import Word


def test_only_common_prefix_is_committed():
    hb = HypothesisBuffer(1)
    hb.insert([Word(0, .3, " Hello"), Word(.4, .7, " every")], 0)
    assert hb.flush() == []
    hb.insert([Word(0, .3, " Hello"), Word(.4, .7, " everyone."), Word(.8, 1.0, " Today")], 0)
    assert [w.text for w in hb.flush()] == [" Hello"]
    hb.insert([Word(0, .3, " Hello"), Word(.4, .7, " everyone."), Word(.8, 1.0, " Today"),
               Word(1.1, 1.3, " I")], 0)
    assert [w.text for w in hb.flush()] == [" everyone.", " Today"]


def test_small_spelling_difference_is_tolerated():
    hb = HypothesisBuffer(1)
    hb.insert([Word(0, .5, " Qualitätsprüfung")], 0)
    hb.flush()
    hb.insert([Word(0, .5, " Qualitatsprüfung")], 0)
    assert len(hb.flush()) == 1


def test_language_keeps_on_words():
    hb = HypothesisBuffer(1)
    hb.insert([Word(0, .3, " Hallo", "de")], 2.0)
    hb.flush()
    hb.insert([Word(0, .3, " Hallo", "de")], 2.0)
    (w,) = hb.flush()
    assert (w.lang, w.start) == ("de", 2.0)


def test_switch_language_retracts_words_in_window():
    online = OnlineASR(engine=None, cfg=AsrCfg())
    online.insert_audio(np.zeros(10 * SR, dtype=np.float32))
    online.language = "en"
    online.committed = [Word(1, 1.3, " hello", "en"), Word(8.5, 8.8, " BONJOUR", "en")]
    online.hyp.committed_in_buffer = list(online.committed)
    kept, retract_from = online.switch_language("fr", window_s=3.0)
    assert retract_from == 8.5
    assert [w.text for w in online.committed] == [" hello"]
    assert online.language == "fr"
    assert abs(online.offset - 7.0) < 1e-6  # audio coupé au début de la fenêtre
