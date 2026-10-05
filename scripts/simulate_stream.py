"""Test 2 : simulation du streaming, sans audio ni GPU.

Les mots arrivent par petits groupes, comme les mots validés par Whisper,
avec des horodatages simulés. On voit à chaque étape les probabilités SaT
et la décision de la politique (sat, sat+pause, pause, max_words, flush).

Le scénario contient volontairement :
  - une fin de phrase SANS pause (SaT doit la trouver seul)
  - une hésitation après « the » (la garde syntaxique doit empêcher la coupe)
  - une définition « that is, ... » (ne doit pas être coupée)

    python -m scripts.simulate_stream
"""
from __future__ import annotations

from app.config import SegCfg
from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.segmentation.sat_segmenter import SatSegmenter
from app.types import Word

# (mot, pause après le mot en secondes)
SCRIPT = [
    ("hello", 0.1), ("everyone", 0.8),
    ("today", 0.1), ("i", 0.1), ("will", 0.1), ("explain", 0.1), ("how", 0.1),
    ("attention", 0.1), ("works", 0.1), ("in", 0.1), ("transformers", 0.15),  # pas de pause
    ("attention", 0.1), ("is", 0.1), ("a", 0.1), ("mechanism", 0.1), ("that", 0.1),
    ("lets", 0.1), ("each", 0.1), ("word", 0.1), ("look", 0.1), ("at", 0.1),
    ("the", 0.9),  # hésitation
    ("other", 0.1), ("words", 0.3), ("that", 0.1), ("is", 0.1), ("it", 0.1),
    ("decides", 0.1), ("which", 0.1), ("words", 0.1), ("are", 0.1), ("important", 0.9),
    ("so", 0.1), ("first", 0.1), ("let's", 0.1), ("define", 0.1), ("the", 0.1),
    ("query", 0.1), ("the", 0.1), ("key", 0.1), ("and", 0.1), ("the", 0.1), ("value", 1.0),
]
WORD_S = 0.3   # durée d'un mot
GROUP = 3      # mots validés par passe Whisper


def build_words() -> list[Word]:
    t, out = 0.0, []
    for text, pause in SCRIPT:
        out.append(Word(t, t + WORD_S, " " + text))
        t += WORD_S + pause
    return out


def main() -> None:
    cfg = SegCfg()
    sat = SatSegmenter(cfg)
    policy = BoundaryPolicy(PolicyParams.from_cfg(cfg))
    words = build_words()
    open_words: list[Word] = []

    for g in range(0, len(words), GROUP):
        batch = words[g : g + GROUP]
        open_words.extend(batch)
        nxt = words[g + GROUP].start if g + GROUP < len(words) else None
        audio_end = batch[-1].end + (0.3 if nxt is None else 0.0)
        force = nxt is None
        print(f"\n[t={batch[-1].end:5.2f}s] + {''.join(w.text for w in batch)}")
        # Les 2 mots suivants jouent le rôle de l'hypothèse non validée de Whisper
        context = [w.text for w in words[g + GROUP : g + GROUP + 2]]
        while open_words:
            probs, ms, _ = sat.word_probs([w.text for w in open_words], context=context)
            view = "  ".join(f"{w.text.strip()}:{p:.2f}" for w, p in zip(open_words, probs))
            print(f"   SaT ({ms:.0f} ms) {view}   [suite vue : {' '.join(c.strip() for c in context)}]")
            d = policy.decide(open_words, probs, audio_end, nxt, force, tail_context=bool(context))
            if d is None:
                print("   … on attend")
                break
            i, reason = d
            seg, open_words = open_words[: i + 1], open_words[i + 1 :]
            print(f"   ✂ [{reason}] « {''.join(w.text for w in seg).strip()} »")


if __name__ == "__main__":
    main()
