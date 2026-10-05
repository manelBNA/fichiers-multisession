"""Journal unique de tous les tests : runs/journal_des_tests.jsonl

Une ligne JSON par test, ajoutée à la fin du test (ou à la fin de son analyse
automatique, en campagne). Chaque ligne contient TOUT le test :

  test, dossier, debut, termine_normalement
  reglages            langues, politique de coupe, modèles, configuration, micros
  changements_pendant_le_test   réglages modifiés en direct (heure, valeurs)
  resultats_direct    le résumé chiffré (summary.json)
  phrases             chaque phrase : raison de coupe, probabilité, horodatages,
                      traductions, latences, unités de sens, délai par mot
  analyse             l'analyse automatique (5 politiques × langues cibles), si demandée

Les fichiers détaillés restent dans runs/<test>/ (events.jsonl contient chaque
événement, trop volumineux pour le journal).
"""
from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path

JOURNAL_NAME = "journal_des_tests.jsonl"
_lock = threading.Lock()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    for row in rows:                                   # nombres lisibles, cases vides retirées
        for k, v in list(row.items()):
            if v in ("", None):
                row.pop(k)
                continue
            try:
                row[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                row[k] = {"True": True, "False": False}.get(v, v)
    return rows


def _changes(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"dir": "in"' in line:
                e = json.loads(line)
                out.append({"t": e.get("t"), "type": e.get("type"), "valeurs": e.get("values")})
    return out


def build_record(run_dir: Path, analysis: dict | None = None) -> dict:
    meta = _read_json(run_dir / "meta.json")
    summary = _read_json(run_dir / "summary.json")
    return {
        "test": meta.get("label") or run_dir.name,
        "dossier": run_dir.name,
        "debut": meta.get("debut"),
        "ecrit_le": datetime.now().isoformat(timespec="seconds"),
        "termine_normalement": summary.get("termine_normalement"),
        "reglages": {k: v for k, v in meta.items() if k not in ("label", "debut")},
        "changements_pendant_le_test": _changes(run_dir),
        "resultats_direct": summary,
        "phrases": _read_rows(run_dir / "segments.csv"),
        "analyse": analysis,
    }


def append(run_dir: str | Path, analysis: dict | None = None) -> Path:
    """Ajoute le test au journal (thread-safe) ; retourne le chemin du journal."""
    run_dir = Path(run_dir)
    record = build_record(run_dir, analysis)
    journal = run_dir.parent / JOURNAL_NAME
    line = json.dumps(record, ensure_ascii=False)
    with _lock, open(journal, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return journal
