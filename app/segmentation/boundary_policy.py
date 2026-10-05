"""Décision de frontière : SaT (texte) + pauses (timestamps) + garde syntaxique.

Ordre des règles pour chaque mot i du buffer ouvert :
  1. garde : jamais de coupe après un mot « qui attend une suite » (the, of, and...)
  2. « sat »       : proba >= th_high ET au moins `lookahead_words` mots déjà vus après
  3. « sat+pause » : proba >= th_low  ET pause >= pause_s
  4. « pause »     : pause >= long_pause_s ET proba >= th_pause_floor
Puis coupe de secours « max_words » si le buffer devient trop long,
et « flush » à la fin du flux.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass

from app.config import SegCfg
from app.types import Word

NO_CUT = {
    "en": {
        "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by", "from",
        "into", "about", "between", "through", "and", "or", "but", "nor", "because",
        "if", "than", "as", "whose", "which", "who", "is", "are", "was", "were", "be",
        "been", "being", "am", "has", "have", "had", "do", "does", "did", "will",
        "would", "can", "could", "should", "must", "may", "might", "shall", "i",
        "we", "you", "they", "he", "she", "my", "our", "your", "their", "his",
        "its", "this", "these", "those", "very", "more", "most", "such", "let's",
    },
    # Allemand : pas d'auxiliaires ni de modaux ici, car le verbe conjugué
    # termine souvent la subordonnée (« ..., weil das Modell trainiert wurde. »).
    # Pas non plus les particules séparables (an, auf, aus, mit, nach, zu), qui
    # terminent souvent la phrase (« Sie passen die Parameter automatisch an. »).
    "de": {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
        "einer", "eines", "und", "oder", "aber", "denn", "dass", "weil", "wenn",
        "als", "ob", "von", "vom", "zum", "zur", "im", "in",
        "am", "bei", "beim", "für", "über", "unter",
        "zwischen", "durch", "gegen", "ohne", "ich", "wir", "du", "ihr", "mein",
        "unser", "unsere", "dein", "sein", "seine", "ihre", "dieser", "diese",
        "dieses", "sehr", "mehr",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "du", "de", "d'", "l'", "au", "aux",
        "à", "en", "dans", "sur", "sous", "pour", "par", "avec", "sans", "entre",
        "et", "ou", "mais", "car", "que", "qui", "dont", "si", "est", "sont", "a",
        "ont", "je", "nous", "vous", "ils", "elles", "il", "elle", "on", "ce",
        "cette", "ces", "mon", "ma", "mes", "notre", "nos", "leur", "leurs", "très",
        "plus",
    },
}
CLAUSE_START = {
    "en": {"so", "then", "now", "first", "second", "next", "finally", "however",
           "and", "but", "because", "which", "where", "when"},
    "de": {"also", "dann", "jetzt", "erstens", "zweitens", "schließlich",
           "aber", "und", "weil", "wobei", "wenn"},
    "fr": {"donc", "ensuite", "alors", "puis", "maintenant", "premièrement",
           "enfin", "cependant", "et", "mais", "car", "où", "quand"},
}
_STRIP = re.compile(r"[^\w'\-]+", re.UNICODE)


def _n(text: str) -> str:
    return _STRIP.sub("", text.lower())


def _ends_sentence(w: Word) -> bool:
    return w.text.rstrip().endswith((".", "!", "?", "…"))


@dataclass
class PolicyParams:
    th_high: float = 0.5
    th_low: float = 0.25
    pause_s: float = 0.5
    long_pause_s: float = 1.2
    th_pause_floor: float = 0.05
    lookahead_words: int = 1
    min_words: int = 3
    max_words: int = 30
    sat_input: str = "raw"

    @classmethod
    def from_cfg(cls, c: SegCfg) -> "PolicyParams":
        return cls(c.th_high, c.th_low, c.pause_s, c.long_pause_s, c.th_pause_floor,
                   c.lookahead_words, c.min_words, c.max_words, c.sat_input)


class BoundaryPolicy:
    """Les listes de mots dépendent de la langue de CHAQUE mot (Word.lang)."""

    def __init__(self, params: PolicyParams):
        self.params = params
        self._lock = threading.Lock()

    @staticmethod
    def _blocks_cut(w: Word) -> bool:
        # Un point, un point d'interrogation ou d'exclamation écrit par Whisper
        # l'emporte sur la liste : le mot termine bien la phrase.
        if _ends_sentence(w):
            return False
        return _n(w.text) in NO_CUT.get(w.lang, NO_CUT["en"])

    @staticmethod
    def _starts_clause(w: Word) -> bool:
        return _n(w.text) in CLAUSE_START.get(w.lang, CLAUSE_START["en"])

    def update(self, values: dict) -> PolicyParams:
        """Réglage en direct depuis l'interface."""
        with self._lock:
            for k, v in values.items():
                if hasattr(self.params, k) and k != "sat_input":
                    cur = getattr(self.params, k)
                    setattr(self.params, k, type(cur)(v))
                elif k == "sat_input" and v in ("normalized", "nopunct", "raw"):
                    self.params.sat_input = v
        return self.params

    def as_dict(self) -> dict:
        return asdict(self.params)

    def decide(self, words: list[Word], probs: list[float], audio_end: float,
               next_start: float | None = None, force: bool = False,
               tail_context: bool = True) -> tuple[int, str] | None:
        """Retourne (index du mot après lequel couper, raison) ou None.

        tail_context=False : SaT n'a rien vu après le dernier mot. Il voit alors la
        fin du texte comme une fin de phrase et donne près de 1 au dernier mot :
        on ne s'y fie pas. Ce mot n'est coupé que s'il porte une ponctuation
        finale écrite par Whisper, ou après une pause longue.
        """
        p = self.params
        n = len(words)
        if n == 0:
            return None

        for i in range(n):
            if i + 1 < p.min_words:
                continue
            if self._blocks_cut(words[i]):
                continue
            if i < n - 1:
                gap = words[i + 1].start - words[i].end
            elif next_start is not None:  # des mots non validés suivent : pas une pause
                gap = next_start - words[i].end
            else:
                gap = audio_end - words[i].end
            ahead = n - 1 - i
            if i == n - 1 and not tail_context and not _ends_sentence(words[i]):
                if gap >= p.long_pause_s:
                    return i, "pause"
                continue
            if probs[i] >= p.th_high and ahead >= p.lookahead_words:
                return i, "sat"
            if probs[i] >= p.th_low and gap >= p.pause_s:
                return i, "sat+pause"
            if gap >= p.long_pause_s and probs[i] >= p.th_pause_floor:
                return i, "pause"

        if n >= p.max_words:
            best, best_score = None, -1.0
            last = n - max(p.lookahead_words, 1)
            for i in range(max(p.min_words - 1, 0), last):
                if self._blocks_cut(words[i]):
                    continue
                score = probs[i]
                if words[i].text.rstrip().endswith((",", ";", ":")):
                    score += 0.3
                if i + 1 < n and self._starts_clause(words[i + 1]):
                    score += 0.2
                if score > best_score:
                    best, best_score = i, score
            return (best if best is not None else last - 1), "max_words"

        if force:
            return n - 1, "flush"
        return None
