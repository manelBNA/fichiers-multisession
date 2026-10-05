"""Configuration : l'ancien fichier reste valable, les clés manquantes ont des défauts."""
from app.config import ROOT, load_config


def test_project_config_loads():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.asr.model and cfg.segmentation.model and cfg.translation.model_pattern


def test_empty_config_uses_defaults(tmp_path):
    p = tmp_path / "vide.yaml"
    p.write_text("", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.server.port == 8000
    assert cfg.segmentation.th_high == 0.5
    assert cfg.logs.enabled is True


def test_unknown_keys_are_ignored(tmp_path):
    p = tmp_path / "ancien.yaml"
    p.write_text("asr:\n  model: small\n  cle_inconnue: 3\nsection_inconnue: {a: 1}\n",
                 encoding="utf-8")
    assert load_config(p).asr.model == "small"
