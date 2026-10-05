"""Test de référence (« golden ») : une session complète avec faux modèles.

La première exécution avec UPDATE_GOLDEN=1 enregistre le comportement actuel dans
tests/golden/. Ensuite, toute modification du projet doit reproduire exactement
les mêmes phrases, traductions et voix : sinon, c'est une régression.

    UPDATE_GOLDEN=1 python -m pytest tests/test_session_golden.py   (enregistrer)
    python -m pytest tests                                          (vérifier)
"""
import json
import os

import pytest

from tests.conftest import GOLDEN, make_timeline, run_fake_session, stable_view

SCENARIOS = {
    "conversation_en_fr": (
        [("Hello everyone. Today I will explain how attention works in transformers.", "en"),
         ("Bonjour à tous. Je vais maintenant répondre à votre question.", "fr")],
        {"source": "auto", "targets": ["en", "fr"], "voice": "auto"},
    ),
    "lecture_de_fixe": (
        [("Die zukünftige Entwicklung textiler Werkstoffe wird durch Nachhaltigkeit geprägt. "
          "Digitale Zwillinge begleiten dabei sämtliche Prozessschritte. "
          "Mit Hilfe künstlicher Intelligenz können Prozessparameter automatisch angepasst werden.", "de")],
        {"source": "de", "targets": ["en", "fr"], "voice": "en"},
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_session_matches_golden(name, patch_online_asr, tmp_path):
    parts, settings = SCENARIOS[name]
    msgs = run_fake_session(make_timeline(parts), settings, tmp_path, label=name)
    view = stable_view(msgs)
    path = GOLDEN / f"session_{name}.json"
    if os.environ.get("UPDATE_GOLDEN") == "1" or not path.exists():
        path.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")
        pytest.skip(f"référence enregistrée : {path.name}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert view == expected
    # le test est aussi enregistré sur disque, sans erreur
    assert any(m["type"] == "run_saved" and m["complete"] for m in msgs)
