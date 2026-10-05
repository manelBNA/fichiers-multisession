"""SaT (Segment any Text) sur le CPU.

`SaT.predict_proba(texte)` renvoie une probabilité de frontière par CARACTÈRE.
On la convertit en probabilité par MOT : le maximum sur le dernier caractère
du mot et l'espace qui le suit.
"""
from __future__ import annotations

import logging
import re
import threading
import time

import numpy as np

from app.config import SegCfg

log = logging.getLogger("sat")
_STRIP = re.compile(r"[^\w'\-]+", re.UNICODE)


class SatSegmenter:
    def __init__(self, cfg: SegCfg):
        from wtpsplit import SaT

        self.cfg = cfg
        self.lock = threading.Lock()
        self.backend = cfg.backend
        if cfg.backend == "onnx":
            try:
                self.sat = SaT(cfg.model, ort_providers=["CPUExecutionProvider"])
            except Exception as exc:
                log.warning("SaT ONNX indisponible (%s). Repli sur PyTorch CPU.", exc)
                self.sat = SaT(cfg.model)
                self.backend = "torch"
        else:
            self.sat = SaT(cfg.model)  # reste sur le CPU : pas de .to("cuda")
        log.info("SaT '%s' chargé sur CPU (%s)", cfg.model, self.backend)

    def prepare(self, word: str, mode: str | None = None) -> str:
        """Texte donné à SaT pour ce mot.

        raw        : tel que Whisper l'écrit (ponctuation + majuscules)
        nopunct    : sans ponctuation, majuscules gardées (utile en allemand, où la
                     majuscule d'un article comme « Die » signale un début de phrase)
        normalized : minuscules sans ponctuation (simule un ASR brut)
        """
        mode = mode or self.cfg.sat_input
        word = word.strip()
        if mode == "normalized":
            return _STRIP.sub("", word.lower())
        if mode == "nopunct":
            return _STRIP.sub("", word)
        return word

    def word_probs(self, words: list[str], mode: str | None = None,
                   context: list[str] | None = None) -> tuple[list[float], float, str]:
        """Retourne (probabilité par mot, temps CPU en ms, texte envoyé à SaT).

        `context` : mots qui suivent (hypothèse de Whisper pas encore validée). Ils
        sont ajoutés au texte pour que SaT « voie la suite » du dernier mot, mais
        aucune probabilité n'est renvoyée pour eux. Sans eux, SaT voit la fin du
        texte comme une fin de phrase et donne près de 1 au dernier mot.
        """
        n_words = len(words)
        words = list(words) + list(context or [])
        tokens = [self.prepare(w, mode) for w in words]
        spans: list[tuple[int, int] | None] = []
        parts: list[str] = []
        pos = 0
        for tok in tokens:
            if not tok:
                spans.append(None)
                continue
            if parts:
                pos += 1  # espace
            spans.append((pos, pos + len(tok)))
            parts.append(tok)
            pos += len(tok)
        text = " ".join(parts)
        if not text:
            return [0.0] * n_words, 0.0, ""

        t0 = time.perf_counter()
        with self.lock:
            probs = np.asarray(self.sat.predict_proba(text), dtype=float).reshape(-1)
        ms = (time.perf_counter() - t0) * 1000

        if len(probs) != len(text):  # sécurité : ne devrait pas arriver
            probs = np.resize(probs, len(text))

        out: list[float] = []
        starts = [s[0] for s in spans if s] + [len(text)]
        k = 0
        for span in spans:
            if span is None:
                out.append(0.0)
                continue
            _, end = span
            nxt = starts[k + 1]
            window = probs[end - 1 : max(end, nxt)]
            out.append(float(window.max()) if window.size else 0.0)
            k += 1
        return out[:n_words], ms, text

    def warmup(self) -> None:
        self.word_probs(["hello", "world", "this", "is", "a", "test"])
