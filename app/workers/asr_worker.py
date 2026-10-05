"""Thread ASR : consomme l'audio, lance Whisper (GPU), envoie les mots validés.

Langue parlée :
- fixée (en, de, fr) : Whisper transcrit toujours dans cette langue ;
- auto : à chaque passe, la langue des dernières secondes est détectée parmi
  en/de/fr. On ne change de langue qu'après 2 détections concordantes (ou une
  seule très sûre), pour éviter les bascules sur un mot isolé.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Callable

import numpy as np

from app.asr.local_agreement import SR, OnlineASR
from app.config import SUPPORTED_LANGS, AsrCfg
from app.types import SessionSettings

STOP = object()
# Passes Whisper ratées de suite avant de déclarer la session en erreur
MAX_CONSECUTIVE_FAILURES = 5
log = logging.getLogger("asr")


def word_times(words) -> list[list]:
    """[[début, fin, texte], ...] : enregistré dans les journaux pour rejouer la segmentation."""
    return [[round(w.start, 3), round(w.end, 3), w.text] for w in words]


class LangTracker:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.candidate: str | None = None
        self.count = 0

    def update(self, lang: str | None, prob: float) -> str | None:
        """Retourne la langue confirmée, ou None si rien n'est encore sûr."""
        if lang is None:
            return None
        if prob >= 0.9:
            self.candidate, self.count = lang, 0
            return lang
        if prob >= self.threshold:
            self.count = self.count + 1 if lang == self.candidate else 1
            self.candidate = lang
            return lang if self.count >= 2 else None
        self.candidate, self.count = None, 0
        return None


class ASRWorker(threading.Thread):
    def __init__(self, engine, cfg: AsrCfg, settings: SessionSettings, audio_q: queue.Queue,
                 seg_q: queue.Queue, emit: Callable[[dict], None],
                 source_id: str = "main", speaker_id: str = "S1",
                 multi_source: bool = False, thread_tag: str = ""):
        super().__init__(daemon=True, name=f"asr-worker-{thread_tag}{source_id}")
        self.source_id = source_id
        self.speaker_id = speaker_id
        # Plusieurs micros : un seul Whisper est partagé. On saute les passes quand le
        # micro est silencieux, et on espace les passes si le GPU prend du retard.
        # Avec un seul micro, rien ne change (intervalle fixe, aucune passe sautée).
        self.multi_source = multi_source
        self.interval = cfg.min_chunk_s
        self._since_voice = 0.0
        self.engine = engine
        self.cfg = cfg
        self.settings = settings
        self.audio_q = audio_q
        self.seg_q = seg_q
        self.emit = emit
        self.online = OnlineASR(engine, cfg)
        self.tracker = LangTracker(cfg.lang_switch_prob)
        self.received_s = 0.0
        self.alive = True
        # Erreur fatale : la session est prévenue (salle en erreur, place libérée)
        self.on_fatal: Callable[[str, BaseException], None] | None = None

    def _take(self, item) -> bool:
        if item is STOP:
            return True
        self.online.insert_audio(item)
        self.received_s += len(item) / SR
        return False

    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:   # le thread meurt : la session doit le savoir
            log.exception("ASR arrêté par une erreur (%s)", self.name)
            if self.on_fatal is not None:
                self.on_fatal("asr", exc)

    def _loop(self) -> None:
        since_last = 0.0
        stopping = False
        failures = 0
        while self.alive and not stopping:
            try:
                item = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            before = self.received_s
            stopping = self._take(item)
            while not stopping:  # vider ce qui est déjà arrivé
                try:
                    stopping = self._take(self.audio_q.get_nowait())
                except queue.Empty:
                    break
            since_last += self.received_s - before
            if since_last >= self.interval or (stopping and since_last > 0):
                new_audio_s, since_last = since_last, 0.0
                if self.multi_source and not stopping and self._is_silent(new_audio_s):
                    self._tick_silence()
                    continue
                try:
                    self._process()
                    failures = 0
                except Exception as exc:  # une passe ratée ne doit pas arrêter le flux
                    failures += 1
                    log.warning("Passe ASR ratée (%s, %d de suite) : %s", self.name, failures, exc)
                    self.emit({"type": "error", "where": "asr",
                               "message": f"Erreur de transcription : {exc}"})
                    if failures >= MAX_CONSECUTIVE_FAILURES:   # Whisper ne répond plus
                        raise RuntimeError(f"{failures} passes Whisper ratées de suite : "
                                           f"{exc}") from exc

        if not self.alive:
            return
        rest = self.online.finish()
        if rest:
            self.seg_q.put(("words", self._stamp(rest), self.online.audio_end_time(), []))
        self.emit({"type": "asr", "committed": [w.text for w in rest], "pending": "", "ms": 0,
                   "received_s": self.received_s, "committed_until": self._until(),
                   "lang": self.online.language, "words": word_times(rest),
                   "pending_words": [], "final": True})
        self.seg_q.put(("flush",))

    def _is_silent(self, seconds: float) -> bool:
        """Micro silencieux depuis la dernière passe, et plus rien à valider ?"""
        chunk = self.online.tail(max(seconds, 0.1))
        silent = len(chunk) == 0 or float(np.sqrt(np.mean(chunk ** 2))) < 0.004
        return silent and not self.online.pending_words()

    def _tick_silence(self) -> None:
        """Passe sautée : la segmentation apprend quand même que le silence continue."""
        self.seg_q.put(("tick", self.online.audio_end_time(), []))

    def _adapt_interval(self, pass_s: float) -> None:
        if not self.multi_source:
            return
        base = self.cfg.min_chunk_s
        if pass_s > 0.9 * self.interval:          # le GPU ne suit plus : on espace
            self.interval = min(3.0, pass_s * 1.2)
        else:                                     # retour progressif à l'intervalle normal
            self.interval = max(base, self.interval * 0.9)

    def _stamp(self, words):
        """Attribue le micro et l'orateur aux mots (un seul orateur par micro à l'étape 1)."""
        return [replace(w, source=self.source_id, speaker=self.speaker_id) for w in words]

    def _until(self) -> float:
        return self.online.committed[-1].end if self.online.committed else 0.0

    # ---------- langue ----------
    def _update_language(self) -> None:
        source = self.settings.source
        current = self.online.language
        if source != "auto":
            if current is None:
                self.online.language = source
                self.emit({"type": "lang", "lang": source, "prob": 1.0, "mode": "fixed"})
            elif current != source:  # changement manuel dans l'interface
                self._switch(source, 1.0, window_s=0.0, mode="fixed")
            return

        tail = self.online.tail(self.cfg.lang_detect_window_s)
        if len(tail) < SR or float(np.sqrt(np.mean(tail ** 2))) < 0.004:
            return  # trop court ou silence : on garde la langue actuelle
        lang, prob = self.engine.detect(tail, SUPPORTED_LANGS)
        confirmed = self.tracker.update(lang, prob)
        if current is None:
            if lang and prob >= 0.5:
                self.online.language = lang
                self.emit({"type": "lang", "lang": lang, "prob": round(prob, 2), "mode": "auto"})
        elif confirmed and confirmed != current:
            self._switch(confirmed, prob, window_s=self.cfg.lang_detect_window_s, mode="auto")

    def _switch(self, new: str, prob: float, window_s: float, mode: str) -> None:
        previous = self.online.language
        kept, retract_from = self.online.switch_language(new, window_s)
        if retract_from is not None:  # mots transcrits dans la mauvaise langue
            self.seg_q.put(("retract", retract_from))
        if kept:
            self.seg_q.put(("words", self._stamp(kept), self.online.audio_end_time(), []))
            self.emit({"type": "asr", "committed": [w.text for w in kept], "pending": "",
                       "ms": 0, "received_s": round(self.received_s, 2),
                       "committed_until": round(self._until(), 2), "lang": previous,
                       "words": word_times(kept), "pending_words": []})
        self.seg_q.put(("boundary", "lang"))  # nouvelle langue = nouvelle phrase
        self.emit({"type": "lang", "lang": new, "previous": previous,
                   "prob": round(prob, 2), "mode": mode, "retract_from": retract_from})

    # ---------- transcription ----------
    def _process(self) -> None:
        t0 = time.perf_counter()
        self._update_language()
        if self.online.language is None:
            self.emit({"type": "asr", "committed": [], "pending": "", "ms": 0,
                       "received_s": round(self.received_s, 2), "committed_until": 0,
                       "lang": None})
            return
        new = self.online.process_iter()
        ms = (time.perf_counter() - t0) * 1000
        self._adapt_interval(ms / 1000)
        pending = self.online.pending_words()
        self.emit({
            "type": "asr",
            "committed": [w.text for w in new],
            "pending": "".join(w.text for w in pending),
            "ms": round(ms),
            "received_s": round(self.received_s, 2),
            "committed_until": round(self._until(), 2),
            "lang": self.online.language,
            "words": word_times(new),               # pour rejouer la segmentation
            "pending_words": word_times(pending),
            "audio_end": round(self.online.audio_end_time(), 3),
        })
        end = self.online.audio_end_time()
        # Les mots en attente servent de contexte à droite pour SaT
        if new:
            self.seg_q.put(("words", self._stamp(new), end, self._stamp(pending)))
        else:
            self.seg_q.put(("tick", end, self._stamp(pending)))


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
