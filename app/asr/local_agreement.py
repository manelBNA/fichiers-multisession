"""Transcription en streaming avec la politique LocalAgreement-2.

Adaptation simplifiée de whisper_streaming (ÚFAL, Macháček et al. 2023) :
à chaque passe, Whisper retranscrit tout le buffer ; on ne valide (commit)
que le plus long préfixe commun entre deux passes successives.
La comparaison est assouplie (minuscules, sans ponctuation, Levenshtein),
comme dans le système MLLP-VRAIN (IWSLT 2025).
"""
from __future__ import annotations

import re

import numpy as np

from app.config import AsrCfg
from app.types import Word

SR = 16000
_PUNCT = re.compile(r"[^\w'\-]+", re.UNICODE)


def norm(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class HypothesisBuffer:
    """Garde l'hypothèse précédente et calcule le préfixe commun validé."""

    def __init__(self, max_lev: int = 1):
        self.max_lev = max_lev
        self.committed_in_buffer: list[Word] = []
        self.buffer: list[Word] = []  # hypothèse précédente non validée
        self.new: list[Word] = []
        self.last_committed_time = 0.0

    def _same(self, a: str, b: str) -> bool:
        na, nb = norm(a), norm(b)
        if na == nb:
            return True
        return min(len(na), len(nb)) > 3 and levenshtein(na, nb) <= self.max_lev

    def insert(self, words: list[Word], offset: float) -> None:
        words = [Word(w.start + offset, w.end + offset, w.text, w.lang) for w in words]
        self.new = [w for w in words if w.start > self.last_committed_time - 0.1]
        # Retirer un chevauchement avec la fin du texte déjà validé (1 à 5 mots)
        if self.new and abs(self.new[0].start - self.last_committed_time) < 1 and self.committed_in_buffer:
            cn, nn = len(self.committed_in_buffer), len(self.new)
            for i in range(1, min(cn, nn, 5) + 1):
                tail = " ".join(norm(w.text) for w in self.committed_in_buffer[-i:])
                head = " ".join(norm(w.text) for w in self.new[:i])
                if tail == head:
                    del self.new[:i]
                    break

    def flush(self) -> list[Word]:
        commit: list[Word] = []
        while self.new and self.buffer and self._same(self.new[0].text, self.buffer[0].text):
            w = self.new.pop(0)
            self.buffer.pop(0)
            commit.append(w)
            self.last_committed_time = w.end
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, t: float) -> None:
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= t:
            self.committed_in_buffer.pop(0)


class OnlineASR:
    def __init__(self, engine, cfg: AsrCfg):
        self.engine = engine
        self.cfg = cfg
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0.0  # temps absolu de audio[0]
        self.hyp = HypothesisBuffer(cfg.lcp_levenshtein)
        self.committed: list[Word] = []
        self.language: str | None = None  # langue imposée à Whisper

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, chunk])

    def audio_end_time(self) -> float:
        return self.offset + len(self.audio) / SR

    def pending_words(self) -> list[Word]:
        return list(self.hyp.buffer)

    def tail(self, seconds: float) -> np.ndarray:
        return self.audio[-int(seconds * SR):]

    def _prompt(self) -> str:
        """Texte validé sorti du buffer, dans la langue courante : contexte pour Whisper."""
        outside = [w.text for w in self.committed if w.end <= self.offset and w.lang == self.language]
        return "".join(outside)[-200:]

    def process_iter(self) -> list[Word]:
        words = self.engine.transcribe(self.audio, self._prompt(), self.language)
        self.hyp.insert(words, self.offset)
        new = self.hyp.flush()
        self.committed.extend(new)
        # Couper le buffer audio sur la fin du dernier mot validé
        if len(self.audio) / SR > self.cfg.buffer_trim_s and self.committed:
            t = self.committed[-1].end
            if t > self.offset:
                self._chunk_at(t)
        return new

    def _chunk_at(self, t: float) -> None:
        self.hyp.pop_committed(t)
        cut = int((t - self.offset) * SR)
        self.audio = self.audio[cut:]
        self.offset = t

    def switch_language(self, new_lang: str, window_s: float) -> tuple[list[Word], float | None]:
        """Changement de langue (nouveau locuteur ou réglage manuel).

        Pendant les secondes nécessaires pour confirmer la nouvelle langue, Whisper
        a transcrit la nouvelle parole dans l'ANCIENNE langue. On :
          1. retire les mots (même validés) qui commencent dans la fenêtre de détection ;
          2. valide les mots en attente situés avant la fenêtre (ancienne langue) ;
          3. coupe l'audio au début de la fenêtre, pour tout retranscrire dans la
             nouvelle langue.
        Retourne (mots validés en plus, temps à partir duquel retirer les mots, ou None).
        """
        cut_t = max(self.audio_end_time() - window_s, self.offset)
        retract_from = None
        wrong = [w for w in self.committed if w.start >= cut_t]
        if wrong and window_s > 0:
            retract_from = wrong[0].start
            self.committed = [w for w in self.committed if w.start < cut_t]
            self.hyp.committed_in_buffer = [w for w in self.hyp.committed_in_buffer if w.start < cut_t]
            self.hyp.last_committed_time = self.committed[-1].end if self.committed else self.offset

        keep = [w for w in self.hyp.buffer if w.end <= cut_t]
        self.hyp.buffer = []
        for w in keep:
            self.hyp.last_committed_time = w.end
        self.hyp.committed_in_buffer.extend(keep)
        self.committed.extend(keep)

        t = max(cut_t, self.hyp.last_committed_time)
        if t > self.offset:
            self._chunk_at(min(t, self.audio_end_time()))
        self.language = new_lang
        return keep, retract_from

    def finish(self) -> list[Word]:
        """Fin du flux : on valide tout ce qui reste."""
        rest = self.hyp.buffer
        self.hyp.buffer = []
        self.committed.extend(rest)
        return rest
