"""Unités de sens : traduire une longue phrase par morceaux dont le sens est stable.

Principe (Zhang et al., EMNLP 2020, adapté au direct) : un morceau x_c+1 … x_i est
une unité de sens si sa traduction ne change plus quand k mots de plus arrivent.
Les deux traductions commencent par V, la traduction déjà validée (décodage forcé) :

    A = T(x_s … x_i | V)      B = T(x_s … x_i+k | V)
    unité de sens si A (sans sa ponctuation finale) est le début de B

Ce module contient la logique pure (sans thread ni modèle), testable seule :
  - is_stable       : le test de stabilité
  - find_candidates : les frontières candidates dans la phrase en cours
  - MUTracker       : la progression validée, partagée entre segmentation et traduction
"""
from __future__ import annotations

import re
import threading

from app.types import Word

_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
CLAUSE_PUNCT = (",", ";", ":", "—", "–")


def norm_words(text: str) -> list[str]:
    """Mots en minuscules, sans ponctuation (comparaison assouplie)."""
    return [w.lower() for w in _WORD.findall(text)]


def is_stable(delta_a: str, delta_b: str) -> bool:
    """Vrai si la traduction A (après V) est le début de la traduction B (après V)."""
    a, b = norm_words(delta_a), norm_words(delta_b)
    return bool(a) and b[:len(a)] == a


def find_candidates(words: list[Word], probs: list[float], policy, validated: int,
                    already_sent: set[int], k: int, min_words: int,
                    sat_floor: float) -> list[int]:
    """Indices i (dans la phrase en cours) après lesquels tester une unité de sens.

    Conditions : la garde syntaxique autorise une coupe après le mot i ; au moins
    `min_words` mots depuis la dernière unité validée ; un indice de fin de
    proposition (virgule écrite par Whisper, connecteur qui suit, ou probabilité
    SaT ≥ sat_floor) ; et k mots déjà validés après le mot i.
    """
    n = len(words)
    out = []
    for i in range(n):
        if i in already_sent or i + 1 - validated < min_words or n - 1 - i < k:
            continue
        w = words[i]
        if policy._blocks_cut(w):
            continue
        clause_end = w.text.rstrip().endswith(CLAUSE_PUNCT)
        next_starts = i + 1 < n and policy._starts_clause(words[i + 1])
        if clause_end or next_starts or (i < len(probs) and probs[i] >= sat_floor):
            out.append(i)
    return out


class MUTracker:
    """Nombre de mots source déjà validés pour la phrase `key`, par langue cible.

    La segmentation lit le minimum sur les langues cibles (pour choisir les
    candidats suivants) ; la traduction l'écrit à chaque unité validée.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._done: dict[str, dict[str, int]] = {}

    def set(self, key: str, target: str, count: int) -> None:
        with self._lock:
            self._done.setdefault(key, {})[target] = count

    def validated(self, key: str) -> int:
        with self._lock:
            counts = self._done.get(key)
            return min(counts.values()) if counts else 0

    def drop(self, key: str) -> None:
        with self._lock:
            self._done.pop(key, None)
