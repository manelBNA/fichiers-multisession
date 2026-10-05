"""SaT : passage des probabilités par caractère aux probabilités par mot, et modes d'entrée."""
import threading

import numpy as np
import pytest

from app.segmentation.sat_segmenter import SatSegmenter


class EndOfTextSat:
    """Reproduit l'effet observé : probabilité 1 sur le dernier caractère du texte."""

    def predict_proba(self, text):
        p = np.zeros(len(text))
        p[-1] = 1.0
        return p


@pytest.fixture
def seg():
    s = SatSegmenter.__new__(SatSegmenter)  # sans charger de vrai modèle
    s.lock = threading.Lock()
    s.cfg = type("C", (), {"sat_input": "raw"})()
    s.sat = EndOfTextSat()
    return s


def test_context_words_remove_end_of_text_effect(seg):
    assert seg.word_probs(["die", "Maximierung", "mechanischer"])[0] == [0.0, 0.0, 1.0]
    probs, _, text = seg.word_probs(["die", "Maximierung", "mechanischer"],
                                    context=["Kennwerte", "im"])
    assert probs == [0.0, 0.0, 0.0]          # le contexte n'a pas de probabilité
    assert text.endswith("Kennwerte im")     # mais il est bien envoyé à SaT


def test_input_modes(seg):
    assert seg.prepare("Qualitätsprüfung.", "raw") == "Qualitätsprüfung."
    assert seg.prepare("Qualitätsprüfung.", "nopunct") == "Qualitätsprüfung"
    assert seg.prepare("Qualitätsprüfung.", "normalized") == "qualitätsprüfung"
    assert seg.prepare("CO2-Fußabdruck,", "nopunct") == "CO2-Fußabdruck"


def test_punctuation_only_word_gets_zero(seg):
    probs, _, _ = seg.word_probs(["Hallo", "—", "Welt"], mode="normalized")
    assert len(probs) == 3 and probs[1] == 0.0
