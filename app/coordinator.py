"""Coordination entre micros (actif seulement avec plusieurs micros).

Automixer : chaque micro entend aussi, plus faiblement, les voisins (diaphonie).
  Principe des mélangeurs automatiques de conférence : pour chaque bloc de
  100 ms, si un autre micro est nettement plus fort au même moment (écart
  > level_margin_db), ce bloc est remplacé par du silence avant Whisper.
  Deux personnes qui parlent en même temps dans leurs propres micros sont
  chacune la plus forte dans le sien : les deux sont gardées.

DuplicateFilter : filet de sécurité. Si une phrase quasi identique vient d'être
  produite par un autre micro sur la même période, la seconde est écartée.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from difflib import SequenceMatcher

import numpy as np


def level_db(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
    return 20 * math.log10(max(rms, 1e-6))


class Automixer:
    """Décide pour chaque bloc de 100 ms s'il est la voix de ce micro ou de la diaphonie.

    Les micros démarrent ensemble : le temps « audio » d'un bloc (numéro du bloc ×
    durée) sert d'horloge commune. Chaque bloc est décidé avec un bloc de retard,
    quand les blocs simultanés des autres micros sont arrivés (les micros n'envoient
    pas leurs blocs exactement dans le même ordre).
    """

    def __init__(self, level_margin_db: float = 6.0, floor_db: float = -50.0,
                 window_s: float = 0.15, chunk_s: float = 0.1):
        self.margin = level_margin_db
        self.floor = floor_db
        self.window = window_s
        self.chunk_s = chunk_s
        self._lock = threading.Lock()
        self._clock: dict[str, float] = {}                         # temps audio par micro
        self._hist: dict[str, deque] = {}                          # (temps, niveau dB)
        self._pending: dict[str, tuple[np.ndarray, float, float]] = {}
        self.gated_frames: dict[str, int] = {}

    def process(self, source_id: str, chunk: np.ndarray) -> np.ndarray | None:
        """Reçoit un bloc ; renvoie le bloc PRÉCÉDENT de ce micro, décidé (ou None)."""
        lvl = level_db(chunk)
        with self._lock:
            t = self._clock.get(source_id, 0.0)
            self._clock[source_id] = t + len(chunk) / 16000
            self._hist.setdefault(source_id, deque(maxlen=100)).append((t, lvl))
            previous = self._pending.get(source_id)
            self._pending[source_id] = (chunk, lvl, t)
        return None if previous is None else self._decide(source_id, *previous)

    def flush(self, source_id: str) -> np.ndarray | None:
        """Fin du flux : décide le dernier bloc en attente."""
        with self._lock:
            previous = self._pending.pop(source_id, None)
        return None if previous is None else self._decide(source_id, *previous)

    def _decide(self, source_id: str, chunk: np.ndarray, lvl: float, t: float) -> np.ndarray:
        with self._lock:
            others = [l for sid, hist in self._hist.items() if sid != source_id
                      for (tt, l) in hist if abs(tt - t) <= self.window and l > self.floor]
        if others and max(others) - lvl > self.margin:
            with self._lock:
                self.gated_frames[source_id] = self.gated_frames.get(source_id, 0) + 1
            return np.zeros_like(chunk)
        return chunk

    def floor_holder(self) -> str | None:
        """Micro le plus fort sur le dernier bloc reçu (au-dessus du bruit de fond)."""
        with self._lock:
            live = [(hist[-1][1], sid) for sid, hist in self._hist.items()
                    if hist and hist[-1][1] > self.floor]
        return max(live)[1] if live else None


class DuplicateFilter:
    def __init__(self, similarity: float = 0.8, window_s: float = 3.0):
        self.similarity = similarity
        self.window = window_s
        self._lock = threading.Lock()
        self._recent: list[tuple[str, float, float, str]] = []   # micro, début, fin, texte

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(text.lower().split())

    def accept(self, source_id: str, start: float, end: float, text: str) -> bool:
        norm = self._norm(text)
        with self._lock:
            self._recent = [r for r in self._recent if r[2] >= end - 30]
            for sid, s, e, t in self._recent:
                overlaps = s - self.window <= end and start <= e + self.window
                if sid != source_id and overlaps and \
                        SequenceMatcher(None, t, norm).ratio() >= self.similarity:
                    return False
            self._recent.append((source_id, start, end, norm))
        return True
