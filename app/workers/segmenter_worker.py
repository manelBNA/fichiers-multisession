"""Thread de segmentation : SaT (CPU) + BoundaryPolicy.

Reçoit les mots validés de l'ASR, décide des frontières, envoie :
- les phrases fermées au traducteur (traduction définitive + voix)
- le buffer ouvert au traducteur (traduction provisoire)
- les probabilités par mot à l'interface (ruban SaT)
Un changement de langue ferme la phrase en cours (raison « lang »).
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
from collections import Counter
from typing import Callable

from app.meaning_units import find_candidates
from app.segmentation.boundary_policy import BoundaryPolicy
from app.segmentation.sat_segmenter import SatSegmenter
from app.types import Word

log = logging.getLogger("seg")


def majority_lang(words: list[Word]) -> str:
    return Counter(w.lang for w in words).most_common(1)[0][0] if words else "en"


def majority_speaker(words: list[Word]) -> str:
    return Counter(w.speaker for w in words).most_common(1)[0][0] if words else "S1"


class SegmenterWorker(threading.Thread):
    def __init__(self, sat: SatSegmenter, policy: BoundaryPolicy, seg_q: queue.Queue,
                 translator, emit: Callable[[dict], None],
                 next_id: Callable[[], int] | None = None, source_id: str = "main",
                 accept: Callable[[str, float, float, str], bool] | None = None,
                 settings=None, mu_cfg: dict | None = None, tracker=None,
                 thread_tag: str = ""):
        super().__init__(daemon=True, name=f"segmenter-worker-{thread_tag}{source_id}")
        self.source_id = source_id
        # Erreur fatale : la session est prévenue (salle en erreur, place libérée)
        self.on_fatal: Callable[[str, BaseException], None] | None = None
        # Numérotation des phrases : commune à tous les micros d'une session.
        self._next_id = next_id or itertools.count(1).__next__
        # Filtre anti-doublon entre micros (None avec un seul micro)
        self._accept = accept
        # Unités de sens : identifiant de la phrase en cours, candidats déjà envoyés
        self.settings = settings
        self.mu_cfg = mu_cfg
        self.tracker = tracker
        self._sentence_no = 1
        self.open_key = f"{source_id}:1"
        self._mu_sent: set[int] = set()
        self.sat = sat
        self.policy = policy
        self.seg_q = seg_q
        self.translator = translator
        self.emit = emit
        self.open_words: list[Word] = []
        self.audio_end = 0.0
        self.pending: list[Word] = []  # hypothèse non validée de Whisper (contexte à droite)
        self.seg_id = 0
        self._cache: tuple[str, str, list[float], float, str] | None = None
        self.alive = True

    def run(self) -> None:
        try:
            while self.alive:
                try:
                    msg = self.seg_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.handle(msg):
                    return
        except Exception as exc:   # le thread meurt : la session doit le savoir
            log.exception("Segmentation arrêtée par une erreur (%s)", self.name)
            if self.on_fatal is not None:
                self.on_fatal("segmentation", exc)

    def handle(self, msg: tuple) -> bool:
        """Traite un message de l'ASR. Retourne True quand le flux est terminé.

        Séparé de run() pour pouvoir rejouer un enregistrement sans thread
        (tests de non-régression, scripts/replay_run.py).
        """
        kind = msg[0]
        if kind == "words":
            _, words, self.audio_end, self.pending = msg
            self.open_words.extend(words)
        elif kind == "tick":
            _, self.audio_end, self.pending = msg
        elif kind == "refresh":  # réglages modifiés : recalculer
            self._cache = None
        elif kind == "retract":  # mots retranscrits dans la nouvelle langue
            self.open_words = [w for w in self.open_words if w.start < msg[1] - 0.01]
            self._cache = None
            return False
        if kind == "boundary":
            self._segment(force=True, force_reason=msg[1])
            return False
        force = kind == "flush"
        self._segment(force)
        if force:
            self.translator.finish(self.source_id)
            return True
        return False

    def _probs(self) -> tuple[list[float], float, str]:
        ctx = [w.text for w in self.pending]
        key = "".join(w.text for w in self.open_words) + "|" + "".join(ctx)
        mode = self.policy.params.sat_input
        if self._cache and self._cache[0] == key and self._cache[1] == mode:
            return self._cache[2], 0.0, self._cache[4]
        probs, ms, text = self.sat.word_probs([w.text for w in self.open_words], mode, ctx)
        self._cache = (key, mode, probs, ms, text)
        return probs, ms, text

    def _segment(self, force: bool, force_reason: str | None = None) -> None:
        while self.open_words:
            probs, ms, sat_text = self._probs()
            self.emit({
                "type": "sat",
                "words": [w.text.strip() for w in self.open_words],
                "probs": [round(p, 4) for p in probs],
                "ms": round(ms, 1),
                "sat_text": sat_text,
                "context": [w.text.strip() for w in self.pending],
                "policy": self.policy.as_dict(),
            })
            next_start = self.pending[0].start if self.pending else None
            decision = self.policy.decide(self.open_words, probs, self.audio_end,
                                          next_start, force, tail_context=bool(self.pending))
            if decision is None:
                break
            i, reason = decision
            if reason == "flush" and force_reason:
                reason = force_reason
            seg, self.open_words = self.open_words[: i + 1], self.open_words[i + 1 :]
            self._cache = None
            text = "".join(w.text for w in seg).strip()
            if self._accept and not self._accept(self.source_id, seg[0].start, seg[-1].end, text):
                self.emit({"type": "segment_dropped", "text": text, "reason": "duplicate",
                           "start": round(seg[0].start, 2), "end": round(seg[-1].end, 2)})
                self._next_sentence()
                continue
            self.seg_id = self._next_id()
            lang = majority_lang(seg)
            speaker = majority_speaker(seg)
            self.emit({
                "type": "segment", "id": self.seg_id, "text": text, "lang": lang,
                "speaker": speaker, "source": self.source_id, "open_key": self.open_key,
                "word_ends": [round(w.end, 2) for w in seg],
                "reason": reason, "p": round(probs[i], 3),
                "start": round(seg[0].start, 2), "end": round(seg[-1].end, 2),
            })
            self.translator.submit_final(self.seg_id, text, lang, speaker, self.source_id,
                                         self.open_key)
            self._next_sentence()

        if not self.open_words:
            self.emit({"type": "sat", "words": [], "probs": [], "ms": 0, "sat_text": "",
                       "context": [w.text.strip() for w in self.pending],
                       "policy": self.policy.as_dict()})
        elif not force:
            open_text = "".join(w.text for w in self.open_words).strip()
            self.translator.submit_provisional(self.seg_id + 1, open_text,
                                               majority_lang(self.open_words),
                                               majority_speaker(self.open_words),
                                               self.source_id, self.open_key)
            self._check_units(probs)

    def _next_sentence(self) -> None:
        """Une phrase vient d'être coupée : la suivante a un nouvel identifiant."""
        self._sentence_no += 1
        self.open_key = f"{self.source_id}:{self._sentence_no}"
        self._mu_sent = set()

    def _check_units(self, probs: list[float]) -> None:
        """Envoie au traducteur les frontières candidates d'unités de sens (si activées)."""
        s, cfg = self.settings, self.mu_cfg
        if cfg is None or s is None or not getattr(s, "mu_enabled", False) or not self.open_words:
            return
        k = int(getattr(s, "mu_k", cfg.get("k", 3)))
        validated = self.tracker.validated(self.open_key) if self.tracker else 0
        candidates = find_candidates(self.open_words, probs, self.policy, validated,
                                     self._mu_sent, k, int(cfg.get("min_words", 5)),
                                     float(cfg.get("sat_floor", 0.05)))
        texts = [w.text.strip() for w in self.open_words]
        for i in candidates:
            self._mu_sent.add(i)
            self.translator.submit_unit_check(
                self.open_key, self.source_id, majority_speaker(self.open_words),
                majority_lang(self.open_words), texts[: i + 1], texts[: i + 1 + k],
                self.open_words[i].end)
