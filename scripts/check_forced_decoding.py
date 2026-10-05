"""Vérifie sur ta machine que Marian sait continuer une traduction imposée.

Les unités de sens reposent sur le « décodage forcé » : Marian doit accepter un
début de traduction imposé et générer seulement la suite. Ce script le vérifie
avec le vrai modèle, en quelques secondes.

    python -m scripts.check_forced_decoding
    python -m scripts.check_forced_decoding --pair de-fr
"""
from __future__ import annotations

import argparse

from app.config import load_config
from app.meaning_units import is_stable
from app.translation.marian_translator import MarianPool

SENTENCE = ("Während in der Vergangenheit die Maximierung mechanischer Kennwerte im "
            "Vordergrund stand, rücken heute der gesamte Lebenszyklus eines Produktes in "
            "den Fokus der Forschung.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="de-en")
    args = ap.parse_args()
    src, tgt = args.pair.split("-")
    pool = MarianPool(load_config().translation)

    words = SENTENCE.split()
    full, _ = pool.translate(src, tgt, SENTENCE)
    print("1. Phrase entière :\n  ", full)

    a = pool.translate_forced(src, tgt, " ".join(words[:11]), "")
    b = pool.translate_forced(src, tgt, " ".join(words[:14]), "")
    print("\n2. Morceau (11 mots) :\n  ", a)
    print("   Morceau + 3 mots :\n  ", b)
    print("   Unité de sens ?", "oui ✔" if is_stable(a, b) else "non (le sens n'est pas encore stable)")

    rest = pool.translate_forced(src, tgt, SENTENCE, a)
    print("\n3. Suite, avec le morceau imposé au début :\n  ", rest)
    print("   Phrase reconstituée :\n  ", f"{a} {rest}")
    ok = bool(rest) and not rest.lower().startswith(a.lower()[:20])
    print("\nDécodage forcé :", "fonctionne ✔ (seule la suite est générée)" if ok
          else "à vérifier ✘ : la suite répète le début ou est vide")


if __name__ == "__main__":
    main()
