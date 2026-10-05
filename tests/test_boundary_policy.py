"""Politique de coupe : les cas réels rencontrés pendant les tests."""
import pytest

from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.types import Word


def words(text, lang="en", step=0.4, dur=0.3):
    return [Word(i * step, i * step + dur, " " + w, lang) for i, w in enumerate(text.split())]


@pytest.fixture
def policy():
    return BoundaryPolicy(PolicyParams())  # seuils par défaut : 0,5 / 0,25 / pause 0,5 s


def test_hesitation_after_article_is_never_cut(policy):
    ws = words("we look at the")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.6], audio_end=ws[-1].end + 1.5) is None


def test_sat_with_pause(policy):
    ws = words("we look at results")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.3], audio_end=ws[-1].end + 0.8) == (3, "sat+pause")


def test_sat_sure_needs_one_word_after(policy):
    ws = words("we look at results now")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.9, 0.02], audio_end=ws[-1].end) == (3, "sat")


def test_last_word_without_following_word_is_not_cut_by_sat_alone(policy):
    ws = words("we look at results")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.9], audio_end=ws[-1].end + 0.1) is None


def test_min_words(policy):
    ws = words("Hello everyone. Today")
    assert policy.decide(ws, [0.02, 0.9, 0.02], audio_end=ws[-1].end) is None


def test_german_verb_final_can_be_cut(policy):
    ws = words("weil das Modell trainiert wurde", "de")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.02, 0.4], audio_end=ws[-1].end + 0.8) == (4, "sat+pause")


def test_german_hesitation_after_die(policy):
    ws = words("wir schauen auf die", "de")
    assert policy.decide(ws, [0.02, 0.02, 0.02, 0.6], audio_end=ws[-1].end + 1.5) is None


def test_german_separable_particle_an(policy):
    t = "Mithilfe künstlicher Intelligenz können Prozessparameter automatisch an. Was ich"
    ws = words(t, "de")
    probs = [0.99 if w.endswith((".", "?")) else 0.01 for w in t.split()]
    assert policy.decide(ws, probs, audio_end=ws[-1].end + 0.1) == (6, "sat")


def test_blind_last_word_needs_long_pause(policy):
    t = "Während in der Vergangenheit die Maximierung mechanischer"
    ws = words(t, "de")
    probs = [0.0] * 6 + [1.0]
    assert policy.decide(ws, probs, audio_end=ws[-1].end + 0.8, tail_context=False) is None
    assert policy.decide(ws, probs, audio_end=ws[-1].end + 1.5, tail_context=False) == (6, "pause")


def test_blind_last_word_with_period_is_trusted(policy):
    ws = words("Das ist eine gute Methode.", "de")
    assert policy.decide(ws, [0, 0, 0, 0, 0.99], audio_end=ws[-1].end + 0.8,
                         tail_context=False) == (4, "sat+pause")


def test_max_words_fallback_prefers_comma(policy):
    long = [Word(i * 0.4, i * 0.4 + 0.3, " w%d%s" % (i, "," if i == 12 else "")) for i in range(30)]
    assert policy.decide(long, [0.05] * 30, audio_end=long[-1].end) == (12, "max_words")


def test_flush(policy):
    ws = words("an unfinished thought")
    assert policy.decide(ws, [0.0, 0.0, 0.0], audio_end=ws[-1].end, force=True) == (2, "flush")


def test_update_accepts_all_sat_modes(policy):
    for mode in ("raw", "nopunct", "normalized"):
        assert policy.update({"sat_input": mode}).sat_input == mode
    policy.update({"sat_input": "n'importe quoi"})
    assert policy.params.sat_input == "normalized"
    assert policy.update({"th_high": "0.7"}).th_high == 0.7
