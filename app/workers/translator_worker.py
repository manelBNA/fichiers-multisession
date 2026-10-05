"""Thread de traduction (Marian, CPU), vers une ou plusieurs langues.

- Chaque phrase est traduite dans toutes les langues cochées sauf la sienne.
- Les traductions DÉFINITIVES passent en priorité, dans l'ordre.
- Pour le PROVISOIRE, seule la demande la plus récente est gardée.
- Le provisoire masque les `mask_k` derniers mots (les plus instables).
- Le contexte (phrase précédente) est gardé séparément pour chaque paire.
- La traduction définitive dans la « langue de la voix » est envoyée à la voix.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from app.config import MtCfg
from app.meaning_units import MUTracker, is_stable, norm_words
from app.translation.marian_translator import MarianPool
from app.types import SessionSettings

log = logging.getLogger("mt")


class TranslatorWorker(threading.Thread):
    def __init__(self, pool: MarianPool, cfg: MtCfg, settings: SessionSettings,
                 emit: Callable[[dict], None], speak: Callable[[int, str, str], None],
                 thread_tag: str = ""):
        super().__init__(daemon=True, name=f"translator-worker{'-' + thread_tag if thread_tag else ''}")
        # Erreur fatale : la session est prévenue (salle en erreur, place libérée)
        self.on_fatal: Callable[[str, BaseException], None] | None = None
        self.pool = pool
        self.cfg = cfg
        self.settings = settings
        self.emit = emit
        self.speak = speak
        self.cond = threading.Condition()
        self.finals: deque[tuple[int, str, str, str]] = deque()
        # Traduction provisoire : seule la demande la plus récente de CHAQUE micro compte
        self.provisional: dict[str, tuple[int, str, str, str, str]] = {}
        self.last_prov: dict[str, tuple[str, tuple]] = {}
        # Contexte (phrase précédente) par ORATEUR et par paire de langues :
        # la réponse d'un orateur n'est jamais traduite avec la phrase d'un autre.
        self.context: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._waiting_sources: set[str] = set()
        # Unités de sens (désactivées par défaut) : tests en attente, et pour chaque
        # (phrase, langue cible) la traduction validée V et le nombre de mots couverts.
        self.unit_checks: deque[tuple] = deque()
        self.mu_state: dict[tuple[str, str], dict] = {}
        self.mu_tracker = MUTracker()
        self.finishing = False
        self.alive = True

    def expect_sources(self, source_ids) -> None:
        """Micros dont on attend la fin avant de terminer (un seul à l'étape 1)."""
        with self.cond:
            self._waiting_sources = set(source_ids)

    def submit_final(self, seg_id: int, text: str, lang: str, speaker: str = "S1",
                     source: str | None = None, key: str | None = None) -> None:
        with self.cond:
            self.finals.append((seg_id, text, lang, speaker, key))
            if key is not None:   # la phrase est finie : ses tests d'unités deviennent inutiles
                self.unit_checks = deque(j for j in self.unit_checks if j[0] != key)
            if source is None:
                self.provisional.clear()
            else:
                self.provisional.pop(source, None)
            self.cond.notify()

    def submit_provisional(self, seg_id: int, text: str, lang: str, speaker: str = "S1",
                           source: str = "main", key: str | None = None) -> None:
        if not self.cfg.provisional:
            return
        with self.cond:
            self.provisional[source] = (seg_id, text, lang, speaker, source, key)
            self.cond.notify()

    def submit_unit_check(self, key: str, source: str, speaker: str, lang: str,
                          words_i: list[str], words_ik: list[str], end_s: float) -> None:
        """Tester si les mots words_i (depuis le début de la phrase) forment une unité de sens."""
        with self.cond:
            self.unit_checks.append((key, source, speaker, lang, words_i, words_ik, end_s))
            self.cond.notify()

    def finish(self, source_id: str | None = None) -> None:
        """Un micro a terminé ; on s'arrête quand tous les micros attendus ont terminé."""
        with self.cond:
            if source_id is not None:
                self._waiting_sources.discard(source_id)
            if source_id is None or not self._waiting_sources:
                self.finishing = True
            self.cond.notify()

    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:   # le thread meurt : la session doit le savoir
            log.exception("Traduction arrêtée par une erreur (%s)", self.name)
            if self.on_fatal is not None:
                self.on_fatal("translation", exc)

    def _loop(self) -> None:
        while self.alive:
            with self.cond:
                while (self.alive and not self.finals and not self.provisional
                       and not self.unit_checks and not self.finishing):
                    self.cond.wait(timeout=0.5)
                if not self.alive:
                    return
                if self.finals:
                    job, final = self.finals.popleft(), True
                elif self.unit_checks:
                    job, final = self.unit_checks.popleft(), "unit"
                elif self.provisional:
                    source = next(iter(self.provisional))       # le plus ancien en attente
                    job, final = self.provisional.pop(source), False
                else:
                    self.emit({"type": "done"})
                    return
            try:
                if final == "unit":
                    self._unit(*job)
                elif final:
                    self._final(*job)
                else:
                    self._provisional(*job)
            except Exception as exc:  # paire indisponible, etc. : on le signale
                self.emit({"type": "error", "where": "translation", "message": str(exc)})

    def _announce_loading(self, lang: str, targets: list[str]) -> None:
        """Prévient l'interface si une paire doit d'abord être téléchargée."""
        waiting = [t for t in targets if not self.pool.is_ready(lang, t)]
        if waiting:
            pairs = ", ".join(f"{lang} → {t}" for t in waiting)
            self.emit({"type": "status", "state": "loading",
                       "text": f"Chargement de la traduction {pairs} (premier usage : "
                               "téléchargement d'environ 300 Mo par paire)…"})

    def _final(self, seg_id: int, text: str, lang: str, speaker: str = "S1",
               key: str | None = None) -> None:
        results: dict[str, str] = {}
        spoken_part: dict[str, str] = {}
        self._announce_loading(lang, self.settings.targets_for(lang))
        for tgt in self.settings.targets_for(lang):
            t0 = time.perf_counter()
            prev = self.context.get((speaker, lang, tgt))
            mu = self.mu_state.pop((key, tgt), None) if key else None
            units, rest = 0, None
            if mu and mu["count"] and self._starts_with_units(text, mu):
                # Des unités de sens ont déjà été affichées : on les garde telles quelles
                # (décodage forcé) et on ne traduit que la suite.
                rest = self._forced(lang, tgt, prev, text, mu["V"])
            if rest is not None:
                out, used_ctx, pivot, units = f"{mu['V']} {rest}".strip(), bool(prev), False, mu["units"]
                spoken_part[tgt] = rest
            else:
                out, used_ctx, pivot = self.pool.translate_with_context(
                    lang, tgt, prev[0] if prev else None, prev[1] if prev else None, text)
            self.context[(speaker, lang, tgt)] = (text, out)
            results[tgt] = out
            msg = {"type": "translation", "id": seg_id, "src": lang, "lang": tgt,
                   "text": out, "final": True, "context": used_ctx, "pivot": pivot,
                   "hidden": 0, "ms": round((time.perf_counter() - t0) * 1000),
                   "speaker": speaker}
            if key is not None:
                msg["key"] = key
            if units:
                msg["units"] = units
            self.emit(msg)
        if key:
            self.mu_tracker.drop(key)
        voice = self.settings.voice_for(lang)
        if voice and voice in results:
            speak_units = self.cfg.meaning_units.get("speak_units", False)
            # Unités déjà prononcées : on ne prononce que la fin de la phrase
            said = spoken_part.get(voice) if speak_units else None
            self.speak(seg_id, voice, said if said is not None else results[voice])

    # ---------- unités de sens ----------
    def _mu_enabled(self) -> bool:
        return bool(getattr(self.settings, "mu_enabled", False))

    def _forced(self, lang: str, tgt: str, prev, source_text: str, validated: str) -> str | None:
        """Traduction de `source_text` imposée à commencer par la traduction validée.

        La phrase précédente (source et traduction) est ajoutée devant, comme contexte.
        """
        src = f"{prev[0]} {source_text}".strip() if prev else source_text
        prefix = f"{prev[1]} {validated}".strip() if prev else validated
        return self.pool.translate_forced(lang, tgt, src, prefix)

    @staticmethod
    def _starts_with_units(text: str, mu: dict) -> bool:
        """La phrase finale commence-t-elle bien par les mots déjà validés ?"""
        return norm_words(text)[:len(mu["src_words"])] == mu["src_words"]

    def _unit(self, key: str, source: str, speaker: str, lang: str, words_i: list[str],
              words_ik: list[str], end_s: float) -> None:
        if not self._mu_enabled():
            return
        text_i, text_ik = " ".join(words_i), " ".join(words_ik)
        for tgt in self.settings.targets_for(lang):
            state = self.mu_state.setdefault((key, tgt), {"V": "", "count": 0, "units": 0,
                                                          "src_words": []})
            min_words = int(self.cfg.meaning_units.get("min_words", 5))
            if len(words_i) - state["count"] < min_words:
                continue
            t0 = time.perf_counter()
            prev = self.context.get((speaker, lang, tgt))
            delta_a = self._forced(lang, tgt, prev, text_i, state["V"])
            delta_b = self._forced(lang, tgt, prev, text_ik, state["V"]) if delta_a else None
            ms = round((time.perf_counter() - t0) * 1000)
            base = {"key": key, "source": source, "speaker": speaker, "src": lang, "lang": tgt,
                    "count": len(words_i), "end": round(end_s, 3), "ms": ms}
            if delta_a is None or delta_b is None or not is_stable(delta_a, delta_b):
                self.emit({"type": "unit_rejected", **base})
                continue
            unit_source = " ".join(words_i[state["count"]:])   # les mots de CETTE unité
            state["V"] = f"{state['V']} {delta_a}".strip()
            state["count"] = len(words_i)
            state["src_words"] = norm_words(text_i)
            state["units"] += 1
            self.mu_tracker.set(key, tgt, len(words_i))
            self.emit({"type": "unit", **base, "text": delta_a, "index": state["units"],
                       "src_text": unit_source})
            voice = self.settings.voice_for(lang)
            if voice == tgt and self.cfg.meaning_units.get("speak_units", False):
                self.speak(None, tgt, delta_a)


    def _provisional(self, seg_id: int, text: str, lang: str, speaker: str = "S1",
                     source: str = "main", key: str | None = None) -> None:
        targets = self.settings.targets_for(lang)
        key = (text, tuple(targets))
        if key == self.last_prov.get(source):
            return
        self.last_prov[source] = key
        self._announce_loading(lang, targets)
        for tgt in targets:
            t0 = time.perf_counter()
            mu = self.mu_state.get((key, tgt)) if key else None
            if mu and mu["count"]:
                # Unités déjà affichées : le provisoire ne montre que la suite
                prev = self.context.get((speaker, lang, tgt))
                rest = self._forced(lang, tgt, prev, text, mu["V"])
                if rest is not None:
                    words = rest.split()
                    k = min(self.cfg.mask_k, max(len(words) - 1, 0))
                    shown = " ".join(words[: len(words) - k]) if k else rest
                    self.emit({"type": "translation", "id": seg_id, "src": lang, "lang": tgt,
                               "text": shown, "final": False, "context": False, "pivot": False,
                               "hidden": k, "ms": round((time.perf_counter() - t0) * 1000),
                               "speaker": speaker, "source": source, "key": key,
                               "after_units": True})
                    continue
            full, pivot = self.pool.translate(lang, tgt, text)
            words = full.split()
            k = min(self.cfg.mask_k, max(len(words) - 1, 0))
            shown = " ".join(words[: len(words) - k]) if k else full
            self.emit({"type": "translation", "id": seg_id, "src": lang, "lang": tgt,
                       "text": shown, "final": False, "context": False, "pivot": pivot,
                       "hidden": k, "ms": round((time.perf_counter() - t0) * 1000),
                       "speaker": speaker, "source": source})
