"""Évalue les coupes de phrases d'un ou plusieurs tests enregistrés.

Référence : un fichier texte avec les VRAIES phrases, au choix :
  - une phrase par ligne ;
  - ou un texte suivi, où chaque phrase finit par . ! ? (ex. transcription corrigée).

Utilisation :
    python -m scripts.evaluate --ref reference.txt runs/20260923-181500_A runs/20260923-182000_B
    python -m scripts.evaluate --ref reference.txt runs/*            (tous les tests)
    python -m scripts.evaluate --ref reference.txt runs/... --details  (liste des erreurs)

Méthode :
  1. Les mots du test et ceux de la référence sont normalisés (minuscules, sans
     ponctuation) puis alignés, pour tolérer les erreurs de Whisper.
  2. Chaque coupe du test est ramenée à une position dans la référence.
  3. Une coupe est juste si une vraie fin de phrase se trouve à cette position
     (à `--tolerance` mots près). Chaque vraie fin ne compte qu'une fois.
  4. La dernière fin (fin du flux) est ignorée : elle est toujours trouvée.
  précision = coupes justes / coupes faites ; rappel = coupes justes / vraies fins
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SENT = re.compile(r"(?<=[.!?…])\s+")


def words_of(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def sentences_to_words(sentences: list[str]) -> tuple[list[str], list[int]]:
    """Mots + position (index du dernier mot) de chaque fin de phrase, sauf la dernière."""
    words, ends = [], []
    for s in sentences:
        ws = words_of(s)
        if not ws:
            continue
        words.extend(ws)
        ends.append(len(words) - 1)
    return words, ends[:-1]


def read_reference(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if len(lines) > 1:
        return lines
    return [s for s in _SENT.split(lines[0]) if s.strip()] if lines else []


def read_run(run: Path) -> tuple[list[dict], dict]:
    rows = []
    with open(run / "segments.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    rows.sort(key=lambda r: int(r["id"]))
    summary = {}
    if (run / "summary.json").exists():
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return rows, summary


def align(hyp: list[str], ref: list[str]) -> list[int]:
    """Pour chaque mot du test, la position correspondante dans la référence."""
    mapping = [0] * len(hyp)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, hyp, ref, autojunk=False).get_opcodes():
        for k in range(i1, i2):
            if op == "equal":
                mapping[k] = j1 + (k - i1)
            elif op == "replace":
                span = max(i2 - i1, 1)
                mapping[k] = j1 + min(int((k - i1 + 1) * (j2 - j1) / span) - 1, j2 - j1 - 1)
            else:  # mots en trop dans le test : rattachés au mot de référence précédent
                mapping[k] = max(j1 - 1, 0)
    return mapping


def context(words: list[str], pos: int, n: int = 4) -> str:
    left = " ".join(words[max(pos - n + 1, 0) : pos + 1])
    right = " ".join(words[pos + 1 : pos + 1 + n])
    return f"…{left} ✂ {right}…"


def evaluate(rows: list[dict], ref_words: list[str], ref_ends: list[int], tol: int):
    hyp_words, hyp_ends = sentences_to_words([r["texte"] for r in rows])
    reasons = [r["raison"] for r in rows if words_of(r["texte"])][:-1]
    mapping = align(hyp_words, ref_words)
    remaining = set(ref_ends)
    tp, false_cuts = 0, []
    for hyp_pos, reason in zip(hyp_ends, reasons):
        r = mapping[hyp_pos] if hyp_pos < len(mapping) else len(ref_words) - 1
        match = min((e for e in remaining if abs(e - r) <= tol), key=lambda e: abs(e - r), default=None)
        if match is None:
            false_cuts.append((reason, context(ref_words, r)))
        else:
            remaining.discard(match)
            tp += 1
    missed = [context(ref_words, e) for e in sorted(remaining)]
    n_cuts, n_true = len(hyp_ends), len(ref_ends)
    precision = tp / n_cuts if n_cuts else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"coupes": n_cuts, "vraies_fins": n_true, "justes": tp, "precision": precision,
            "rappel": recall, "f1": f1, "coupes_en_trop": false_cuts, "fins_manquees": missed}


def median_latency(summary: dict) -> str:
    lat = summary.get("latence_traduction_definitive_s") or {}
    values = [v["mediane"] for v in lat.values() if v]
    return f"{min(values):.1f}" if values else "–"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path, help="fichier de référence (phrases justes)")
    ap.add_argument("runs", nargs="+", help="dossiers de tests (runs/...)")
    ap.add_argument("--tolerance", type=int, default=1, help="écart toléré en mots (défaut 1)")
    ap.add_argument("--details", action="store_true", help="afficher chaque erreur")
    ap.add_argument("--csv", type=Path, help="enregistrer le tableau dans ce fichier CSV")
    args = ap.parse_args()

    ref_words, ref_ends = sentences_to_words(read_reference(args.ref))
    print(f"Référence : {len(ref_ends) + 1} phrases, {len(ref_words)} mots "
          f"({len(ref_ends)} fins évaluées, la dernière est ignorée)\n")

    paths = [Path(p) for pattern in args.runs for p in sorted(glob.glob(pattern))]
    results = []
    for run in paths:
        if not (run / "segments.csv").exists():
            print(f"(ignoré : pas de segments.csv dans {run})")
            continue
        rows, summary = read_run(run)
        res = evaluate(rows, ref_words, ref_ends, args.tolerance)
        res["test"] = run.name
        res["reglage_sat"] = (summary.get("label") or "")
        res["latence"] = median_latency(summary)
        res["raisons"] = summary.get("raisons_de_coupe", {})
        results.append(res)

    header = f"{'Test':<34} {'Coupes':>6} {'Justes':>6} {'Précision':>9} {'Rappel':>7} {'F1':>6} {'Latence méd. (s)':>17}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['test'][:34]:<34} {r['coupes']:>6} {r['justes']:>6} {r['precision']:>9.1%} "
              f"{r['rappel']:>7.1%} {r['f1']:>6.2f} {r['latence']:>17}")

    if args.details:
        for r in results:
            print(f"\n=== {r['test']} ===  raisons de coupe : {r['raisons']}")
            for reason, ctx in r["coupes_en_trop"]:
                print(f"  coupe en trop [{reason}] {ctx}")
            for ctx in r["fins_manquees"]:
                print(f"  fin manquée          {ctx}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["test", "coupes", "justes", "precision", "rappel", "f1", "latence_mediane_s"])
            for r in results:
                w.writerow([r["test"], r["coupes"], r["justes"], f"{r['precision']:.3f}",
                            f"{r['rappel']:.3f}", f"{r['f1']:.3f}", r["latence"]])
        print(f"\nTableau enregistré dans {args.csv}")


if __name__ == "__main__":
    main()
