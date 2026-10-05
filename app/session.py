"""Une session = une connexion WebSocket = des threads en parallèle.

  micro ─► [Pipeline du micro : ASR (GPU) + segmentation SaT (CPU)] ─┐
                                                                     ├─► [Traduction, CPU] ─► [Voix]
  (étape 2 : un pipeline par micro) ─────────────────────────────────┘
  Tous les threads envoient leurs messages JSON au navigateur par emit().
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import threading

from app.config import SUPPORTED_LANGS, Config
from app.registry import ModelRegistry
from app.run_logger import RunLogger
from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.audio_framing import parse_frame
from app.coordinator import Automixer, DuplicateFilter
from app.sources import SpeakerRegistry, resolve_sources
from app.types import SessionSettings
from app.workers.source_pipeline import SourcePipeline
from app.workers.translator_worker import TranslatorWorker

log = logging.getLogger("session")


class Session:
    def __init__(self, registry: ModelRegistry, cfg: Config,
                 loop: asyncio.AbstractEventLoop, out_q: asyncio.Queue, analysis=None,
                 room: str = ""):
        self.registry = registry
        self.room = room                  # salle (plusieurs sessions en parallèle)
        self.pool = None                  # traducteurs propres à la session (sinon ceux du registre)
        self.on_finished = None           # appelé quand la session s'arrête (libère sa place)
        self.analysis = analysis          # analyse automatique après le test (campagne)
        self.analyze = False
        self.cfg = cfg
        self.loop = loop
        self.out_q = out_q
        self.policy = BoundaryPolicy(PolicyParams.from_cfg(cfg.segmentation))
        mu = cfg.translation.meaning_units or {}
        self.settings = SessionSettings(cfg.session.source, list(cfg.session.targets),
                                        cfg.session.voice, bool(mu.get("enabled", False)),
                                        int(mu.get("k", 3)))
        self.profile, self.sources, self.source_warnings = resolve_sources(cfg.sources)
        self.speakers = SpeakerRegistry()
        self.pipelines: dict[str, SourcePipeline] = {}
        self.by_index: dict[int, SourcePipeline] = {}
        self.tagged_audio = False
        self.automixer: Automixer | None = None
        self.segments_info: dict[int, tuple[float, float, str]] = {}   # id -> début, fin, orateur
        self._spoken: list[tuple[float, float, str]] = []
        self.translator: TranslatorWorker | None = None
        self.tts_worker = None
        self.logger: RunLogger | None = None
        self.failed: str | None = None    # erreur fatale d'un thread (session en erreur)
        self._fatal_lock = threading.Lock()

    # Appelé depuis les threads : on repasse dans la boucle asyncio
    def emit(self, msg: dict) -> None:
        if msg.get("type") == "segment":
            self.segments_info[msg["id"]] = (msg.get("start", 0.0), msg.get("end", 0.0),
                                             msg.get("speaker", "S1"))
        logger = self.logger
        if logger is not None:
            logger.log_out(msg)
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, msg)
        if msg.get("type") == "done":  # tout est traduit : on écrit le bilan du test
            self._finish_run(complete=True)

    @property
    def translation_pool(self):
        return self.pool if self.pool is not None else self.registry.pool

    def _release(self) -> None:
        """La session ne travaille plus : sa place peut servir à une autre salle."""
        callback, self.on_finished = self.on_finished, None
        if callback is not None:
            try:
                callback(self)
            except Exception:
                pass

    def _finish_run(self, complete: bool) -> None:
        logger, self.logger = self.logger, None
        if complete:
            self._release()
        if logger is None:
            return
        path = logger.finalize(complete)
        will_analyze = bool(self.analyze and complete and self.analysis is not None)
        msg = {"type": "run_saved", "path": str(path), "complete": complete,
               "analysis": will_analyze}
        if will_analyze:
            self._send(msg)
            self.analysis.submit(str(path), self._send)   # le journal sera écrit après l'analyse
        else:
            from app.test_journal import append
            try:
                msg["journal"] = str(append(path))
            except Exception as exc:  # le journal ne doit jamais empêcher la fin du test
                msg["journal_error"] = str(exc)
            self._send(msg)

    def _send(self, msg: dict) -> None:
        """Envoi vers le navigateur, même depuis un autre thread ; ignoré si la page est fermée."""
        try:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, msg)
        except RuntimeError:
            pass

    @property
    def running(self) -> bool:
        return any(p.alive for p in self.pipelines.values())

    @property
    def workers(self) -> list:
        """Tous les threads de la session (compatibilité avec l'ancien code)."""
        threads = [t for p in self.pipelines.values() for t in p.threads]
        return threads + [t for t in (self.translator, self.tts_worker) if t is not None]

    def _speak(self, seg_id: int, lang: str, text: str) -> None:
        start, end, speaker = self.segments_info.get(seg_id, (0.0, 0.0, "S1"))
        if len(self.pipelines) > 1 and self.cfg.tts.simultaneous_policy == "floor":
            # Paroles simultanées : on ne lit pas une phrase qui chevauche une phrase
            # d'un autre orateur déjà lue (une seule voix à la fois reste compréhensible).
            overlap = any(o_spk != speaker and start < o_end and o_start < end
                          for o_start, o_end, o_spk in self._spoken)
            if overlap:
                self.emit({"type": "speak_skipped", "id": seg_id, "lang": lang,
                           "speaker": speaker, "reason": "simultaneous"})
                return
        self._spoken = [x for x in self._spoken if x[1] > start - 30] + [(start, end, speaker)]
        if self.cfg.tts.engine == "piper" and self.tts_worker is not None:
            self.tts_worker.submit(seg_id, lang, text)
        else:  # le navigateur prononce le texte lui-même
            self.emit({"type": "speak", "id": seg_id, "lang": lang, "text": text,
                       "engine": "browser", "speaker": speaker})

    def _preload(self) -> None:
        """Charge en arrière-plan les paires de traduction nécessaires."""
        s = self.settings
        sources = SUPPORTED_LANGS if s.source == "auto" else (s.source,)
        pairs = [(a, b) for a in sources for b in s.targets_for(a)]
        status = lambda text: self.emit({"type": "status", "state": "loading", "text": text})
        pool = self.translation_pool
        pool.preload(pairs, on_status=status)
        self.emit({"type": "status", "state": "models", "loaded": pool.loaded()})

    def start(self, values: dict | None = None, label: str = "",
              sources: dict | None = None, analyze: bool = False) -> None:
        """`sources` : configuration des micros envoyée par l'interface
        ({"profile": "panel", "sources": [...], "framing": "tagged"}) ; sinon config.yaml."""
        self.close()
        self.failed = None                # un redémarrage repart d'un état sain
        self.analyze = analyze
        if values:
            self.settings.update(values)
        self.profile, self.sources, self.source_warnings = resolve_sources(self.cfg.sources,
                                                                           sources)
        multi = len(self.sources) > 1
        self.tagged_audio = bool(sources and sources.get("framing") == "tagged")
        if self.cfg.logs.enabled:
            self.logger = RunLogger(self.cfg.logs.runs_dir, label, {
                "reglages_langues": self.settings.as_dict(),
                "politique_de_coupe": self.policy.as_dict(),
                "modeles": self.registry.info(),
                "config": {"asr": vars(self.cfg.asr), "segmentation": vars(self.cfg.segmentation),
                           "translation": vars(self.cfg.translation)},
                "profil": self.profile,
                "sources": [vars(src) for src in self.sources],
                "salle": self.room,
            }, room=self.room)
            self.emit({"type": "run_started", "path": str(self.logger.dir)})
        threading.Thread(target=self._preload, daemon=True).start()
        for warning in self.source_warnings:
            self.emit({"type": "status", "state": "warning", "text": warning})

        pool, c = self.translation_pool, self.cfg
        self.translator = TranslatorWorker(pool, c.translation, self.settings, self.emit,
                                           self._speak, thread_tag=self.room)
        self.translator.on_fatal = self._fatal
        id_lock, counter = threading.Lock(), itertools.count(1)

        def next_id() -> int:  # numérotation des phrases commune à tous les micros
            with id_lock:
                return next(counter)

        self.speakers = SpeakerRegistry()
        self.pipelines, self.segments_info, self._spoken = {}, {}, []
        crosstalk = c.sources.crosstalk or {}
        use_mixer = multi and crosstalk.get("enabled", True)
        self.automixer = Automixer(float(crosstalk.get("level_margin_db", 6))) if use_mixer else None
        duplicates = (DuplicateFilter(float(crosstalk.get("text_similarity", 0.8)))
                      if use_mixer else None)
        for src in self.sources:
            # Un orateur par micro (étape 2) : le micro donne l'orateur
            name = src.name if multi else None
            speaker = self.speakers.add(src.id, name=name, lang=src.language)
            self.pipelines[src.id] = SourcePipeline(src, speaker, self.registry, c, self.settings,
                                                    self.policy, self.translator, self.emit,
                                                    next_id, multi_source=multi,
                                                    automixer=self.automixer,
                                                    duplicate_filter=duplicates,
                                                    thread_tag=f"{self.room}:" if self.room else "",
                                                    on_fatal=self._fatal)
        self.by_index = {p.source.index: p for p in self.pipelines.values()}
        self.translator.expect_sources(self.pipelines)
        if c.tts.engine == "piper" and self.registry.tts:
            from app.tts.piper_tts import TTSWorker
            self.tts_worker = TTSWorker(self.registry.tts, self.emit)

        for p in self.pipelines.values():
            p.start()
        self.translator.start()
        if self.tts_worker is not None:
            self.tts_worker.start()
        self.emit({"type": "status", "state": "running", "policy": self.policy.as_dict(),
                   "settings": self.settings.as_dict(), "profile": self.profile,
                   "sources": [vars(src) for src in self.sources],
                   "speakers": self.speakers.as_list()})

    def push_audio(self, data: bytes, source: str | None = None) -> None:
        """Bloc audio reçu du navigateur.

        Format historique : PCM brut, pour le premier micro.
        Format étiqueté (annoncé au démarrage) : en-tête de 2 octets avec le numéro du micro.
        """
        # Copie locale : un thread peut fermer la session (erreur fatale) pendant ce temps
        pipelines, by_index = self.pipelines, self.by_index
        if not pipelines:
            return
        if source is not None:
            pipe = pipelines.get(source)
        elif self.tagged_audio:
            index, data = parse_frame(data, tagged=True)
            pipe = by_index.get(index)
        else:
            pipe = next(iter(pipelines.values()), None)
        if pipe is not None and pipe.alive:
            pipe.push(data)

    def stop(self) -> None:
        """Arrêt propre : chaque ASR valide sa fin, puis segmentation et traduction se vident."""
        for p in self.pipelines.values():
            p.stop()

    def update_policy(self, values: dict) -> None:
        if self.logger:
            self.logger.log_in("policy", values)
        self.policy.update(values)
        for p in self.pipelines.values():
            p.refresh()
        self.emit({"type": "policy", "policy": self.policy.as_dict()})

    def update_settings(self, values: dict) -> None:
        if self.logger:
            self.logger.log_in("settings", values)
        self.settings.update(values)
        if self.running:
            threading.Thread(target=self._preload, daemon=True).start()
        self.emit({"type": "settings", "settings": self.settings.as_dict()})

    def rename_speaker(self, speaker_id: str, name: str) -> None:
        spk = self.speakers.rename(speaker_id, name)
        if spk is not None:
            self.emit({"type": "speakers", "speakers": self.speakers.as_list()})

    def _fatal(self, where: str, exc: BaseException) -> None:
        """Un thread de CETTE session est mort : la session passe en erreur.

        Appelé depuis le thread fautif. Les autres threads de la session sont arrêtés,
        le test est enregistré comme interrompu et la place est libérée (la salle passe
        en « error », voir RoomManager). Les autres salles ne sont pas touchées.
        La régie peut redémarrer : start() reconstruit tous les threads.
        """
        with self._fatal_lock:
            if self.failed is not None or not self.pipelines:
                return                    # déjà en erreur, ou session déjà fermée
            self.failed = f"{where} : {type(exc).__name__}: {exc}"
        log.error("Salle « %s » en erreur (%s) : %s", self.room or "-", where, exc)
        self.emit({"type": "error", "where": where, "fatal": True,
                   "message": f"Erreur grave ({where}) : {exc}. La session est arrêtée ; "
                              "les autres salles continuent. Clique « Démarrer » pour relancer."})
        self.emit({"type": "status", "state": "error", "where": where, "text": self.failed})
        try:
            self.close()
        except Exception:
            log.exception("Fermeture de la session en erreur")
            self._release()

    def close(self) -> None:
        for p in self.pipelines.values():
            p.close()
        for t in (self.translator, self.tts_worker):
            if t is not None:
                t.alive = False
        self.pipelines, self.translator, self.tts_worker = {}, None, None
        self._finish_run(complete=False)  # test interrompu : on garde ce qui a été fait
        self._release()
