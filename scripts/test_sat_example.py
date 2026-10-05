"""Test 1 : SaT seul sur CPU, sur un exemple de chercheur.

Affiche la probabilité de frontière de chaque mot, la segmentation obtenue,
et le temps de calcul CPU. Compare texte brut (ponctué) et texte normalisé
(minuscules, sans ponctuation, comme un ASR brut).

    python -m scripts.test_sat_example
    python -m scripts.test_sat_example --lang de
    python -m scripts.test_sat_example --lang fr --threshold 0.4 --backend torch
"""
from __future__ import annotations

import argparse
import statistics

from app.config import SegCfg
from app.segmentation.sat_segmenter import SatSegmenter

EXAMPLES = {
    "en": ("Hello everyone. Today I will explain how attention works in transformers. "
           "Attention is a mechanism that lets each word look at the other words in the "
           "sentence, that is, it decides which words are important. So first, let's define "
           "the query, the key and the value. The results show a clear improvement of three points."),
    "de": ("Hallo zusammen. Heute erkläre ich, wie Attention in Transformern funktioniert. "
           "Attention ist ein Mechanismus, bei dem jedes Wort die anderen Wörter im Satz "
           "betrachtet, das heißt, es entscheidet, welche Wörter wichtig sind. Zuerst definieren "
           "wir also die Query, den Key und den Value. Die Ergebnisse zeigen eine klare Verbesserung."),
    "fr": ("Bonjour à tous. Aujourd'hui je vais expliquer comment fonctionne l'attention dans "
           "les transformers. L'attention est un mécanisme qui permet à chaque mot de regarder "
           "les autres mots de la phrase, c'est-à-dire qu'il décide quels mots sont importants. "
           "Donc d'abord, définissons la requête, la clé et la valeur. Les résultats montrent "
           "une nette amélioration de trois points."),
}


def bar(p: float, width: int = 30) -> str:
    return "█" * round(p * width) + "·" * (width - round(p * width))


def show(sat: SatSegmenter, words: list[str], mode: str, threshold: float) -> None:
    probs, ms, text = sat.word_probs(words, mode)
    print(f"\n=== Entrée SaT ({mode}) ===\n{text}\n")
    print(f"{'mot':<14} {'proba':>6}  ")
    for w, p in zip(words, probs):
        mark = "  ✂ coupe" if p >= threshold else ""
        print(f"{w:<14} {p:6.3f}  {bar(p)}{mark}")

    print(f"\n--- Segments (seuil {threshold}) ---")
    cur: list[str] = []
    for w, p in zip(words, probs):
        cur.append(w)
        if p >= threshold:
            print("•", " ".join(cur))
            cur = []
    if cur:
        print("•", " ".join(cur), "  (ouvert)")

    runs = [sat.word_probs(words, mode)[1] for _ in range(10)]
    print(f"\nTemps CPU : médiane {statistics.median(runs):.1f} ms "
          f"pour {len(words)} mots (premier appel {ms:.1f} ms)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sat-3l-sm")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--lang", default="en", choices=sorted(EXAMPLES))
    ap.add_argument("--text", default=None, help="texte personnalisé (remplace l'exemple)")
    args = ap.parse_args()

    sat = SatSegmenter(SegCfg(model=args.model, backend=args.backend))
    print(f"Modèle {args.model} sur CPU ({sat.backend})")
    words = (args.text or EXAMPLES[args.lang]).split()
    show(sat, words, "raw", args.threshold)
    show(sat, words, "normalized", args.threshold)


if __name__ == "__main__":
    main()
