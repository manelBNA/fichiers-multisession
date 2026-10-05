"""Mesure la fidélité des traductions de plusieurs tests par rapport à une traduction
de référence (humaine), sur le texte complet.

    pip install sacrebleu
    python -m scripts.evaluate_translation --ref reference_en.txt --lang en "runs/*"

Scores (texte entier comparé au texte entier) :
  BLEU  0-100, plus haut = plus proche (mots et groupes de mots en commun)
  chrF  0-100, plus haut = plus proche (caractères ; plus tolérant aux variantes de mots)
  TER   0-100+, plus BAS = mieux (proportion de modifications nécessaires)

Attention : si la référence est une traduction libre et soignée, tous les systèmes
automatiques ont des scores modestes ; ce qui compte est la COMPARAISON entre tests.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("Installe d'abord sacrebleu :  pip install sacrebleu")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path, help="traduction de référence (texte complet)")
    ap.add_argument("--lang", default="en", help="langue traduite à évaluer (fichier traduction-<lang>.txt)")
    ap.add_argument("runs", nargs="+", help="dossiers de tests (runs/...)")
    args = ap.parse_args()

    ref = " ".join(args.ref.read_text(encoding="utf-8-sig").split())
    print(f"Référence : {len(ref.split())} mots\n")
    header = f"{'Test':<36} {'BLEU':>6} {'chrF':>6} {'TER':>6} {'Mots':>6}  Mode SaT"
    print(header)
    print("-" * len(header))
    for pattern in args.runs:
        for run in sorted(Path(p) for p in glob.glob(pattern)):
            hyp_file = run / f"traduction-{args.lang}.txt"
            if not hyp_file.exists():
                continue
            hyp = " ".join(hyp_file.read_text(encoding="utf-8-sig").split())
            mode = ""
            if (run / "meta.json").exists():
                meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
                mode = meta.get("politique_de_coupe", {}).get("sat_input", "")
            bleu = sacrebleu.corpus_bleu([hyp], [[ref]]).score
            chrf = sacrebleu.corpus_chrf([hyp], [[ref]]).score
            ter = sacrebleu.corpus_ter([hyp], [[ref]]).score
            print(f"{run.name[:36]:<36} {bleu:>6.1f} {chrf:>6.1f} {ter:>6.1f} {len(hyp.split()):>6}  {mode}")


if __name__ == "__main__":
    main()
