"""Rejoue la segmentation d'un test enregistré, sans audio ni Whisper.

À partir de events.jsonl (mots validés, mots en attente, horodatages, changements
de langue), refait exactement les décisions de coupe avec le VRAI SaT et la
politique du test, puis compare aux phrases enregistrées dans segments.csv.

Deux usages :
  1. Non-régression : après une mise à jour du projet, le rejeu d'un ancien test
     doit redonner les mêmes phrases.
         python -m scripts.replay_run runs/20260924-095148_EN_fichier
  2. Expérimenter hors ligne : rejouer le MÊME flux avec d'autres réglages, sans
     repasser la vidéo (le résultat est rigoureusement comparable).
         python -m scripts.replay_run runs/... --set th_high=0.7 --set sat_input=nopunct

Seuls les tests enregistrés avec cette version (champ « words » dans les messages
asr) peuvent être rejoués.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.types import Word
from app.workers.segmenter_worker import SegmenterWorker


class _CollectTranslator:
    """Remplace le traducteur : garde seulement les phrases coupées."""

    def submit_final(self, *a, **k):
        pass

    def submit_provisional(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


def _words(items, lang) -> list[Word]:
    return [Word(float(s), float(e), t, lang or "en") for s, e, t in items]


def feed(events: list[dict], seg: SegmenterWorker, on_time=None) -> None:
    """Rejoue dans `seg` les messages ASR enregistrés, dans l'ordre.

    `on_time(t)` est appelé avant chaque message avec le temps audio de ce message :
    cela sert d'horloge virtuelle pour mesurer les délais sans temps de calcul.
    """
    asr = [e for e in events if e.get("dir") == "out" and e.get("type") in ("asr", "lang")]
    if not any("words" in e for e in asr if e["type"] == "asr"):
        raise SystemExit("Ce test a été enregistré avant l'ajout des horodatages : "
                         "impossible de le rejouer. Refais un test avec cette version.")
    lang = None
    for e in asr:
        if e["type"] == "lang":
            if e.get("retract_from") is not None:
                seg.handle(("retract", e["retract_from"]))
            if e.get("previous"):
                seg.handle(("boundary", "lang"))
            lang = e.get("lang") or lang
            continue
        lang = e.get("lang") or lang
        words = _words(e.get("words", []), lang)
        pending = _words(e.get("pending_words", []), lang)
        end = float(e.get("audio_end", e.get("received_s", 0.0)))
        if on_time:
            on_time(end)
        if e.get("final"):
            if words:
                seg.handle(("words", words, end, []))
            seg.handle(("flush",))
            return
        seg.handle(("words", words, end, pending) if words else ("tick", end, pending))
    seg.handle(("flush",))


def replay(events: list[dict], sat, params: PolicyParams) -> list[dict]:
    """Rejoue les messages ASR enregistrés et renvoie les phrases coupées."""
    segments: list[dict] = []
    emit = lambda m: segments.append(m) if m.get("type") == "segment" else None
    seg = SegmenterWorker(sat, BoundaryPolicy(params), None, _CollectTranslator(), emit)
    feed(events, seg)
    return segments


def _load_run(run: Path) -> tuple[list[dict], dict, list[dict]]:
    events = [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if l]
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    rows = []
    if (run / "segments.csv").exists():
        with open(run / "segments.csv", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
    return events, meta, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="dossier du test (runs/...)")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                    help="modifier un réglage de coupe (ex. th_high=0.7, sat_input=nopunct)")
    args = ap.parse_args()

    events, meta, rows = _load_run(args.run)
    params = PolicyParams(**meta["politique_de_coupe"])
    overrides = dict(kv.split("=", 1) for kv in args.set)
    BoundaryPolicy(params).update(overrides)

    from app.config import SegCfg
    from app.segmentation.sat_segmenter import SatSegmenter
    seg_cfg = meta.get("config", {}).get("segmentation", {})
    sat = SatSegmenter(SegCfg(model=seg_cfg.get("model", "sat-3l-sm"),
                              backend=seg_cfg.get("backend", "onnx")))

    new = [s["text"] for s in replay(events, sat, params)]
    old = [r["texte"] for r in rows]
    print(f"Rejeu de {args.run.name} avec {vars(params)}\n")
    for i in range(max(len(new), len(old))):
        a = old[i] if i < len(old) else "—"
        b = new[i] if i < len(new) else "—"
        mark = "  " if a == b else "≠ "
        print(f"{mark}{i + 1:>2}. {b}")
        if a != b:
            print(f"      enregistré : {a}")
    same = new == old
    print("\nRésultat :", "identique à l'enregistrement ✔" if same
          else ("différent (attendu si tu as modifié des réglages)" if overrides else "DIFFÉRENT ✘"))
    sys.exit(0 if same or overrides else 1)


if __name__ == "__main__":
    main()
