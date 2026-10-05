"""Charge les modèles une seule fois, partagés entre les sessions.

Plusieurs sessions (salles) en même temps :
  - Whisper : un seul modèle en mémoire GPU, plusieurs appels en parallèle ;
  - SaT : un seul modèle ONNX, appelé en parallèle ;
  - Marian : un jeu de traducteurs par « emplacement » de session (per_session),
    pour que les salles traduisent sans s'attendre ; ou un seul jeu partagé (shared).
"""
from __future__ import annotations

import logging
import threading
import time

from app.config import Config

log = logging.getLogger("registry")


class ModelRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sat = None
        self.pool = None
        self.whisper = None
        self.tts = None
        self.load_times: dict[str, float] = {}
        self._slot_pools: dict[int | str, object] = {}   # place -> traducteurs (+ « analysis »)
        self._slot_lock = threading.Lock()
        self.status = "Démarrage…"
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return all((self.sat, self.pool, self.whisper))

    def load_safely(self) -> None:
        """Version pour le thread d'arrière-plan : une erreur est affichée, pas avalée."""
        try:
            self.load()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "Échec du chargement des modèles"
            log.exception("Échec du chargement des modèles")

    def load(self) -> None:
        import torch

        # Le CPU est partagé entre SaT, Marian et Piper : on limite les threads PyTorch
        torch.set_num_threads(self.cfg.cpu.torch_threads)

        from app.asr.whisper_engine import WhisperEngine
        from app.segmentation.sat_segmenter import SatSegmenter
        from app.translation.marian_translator import MarianPool

        t0 = time.perf_counter()
        self.status = f"Chargement de SaT {self.cfg.segmentation.model}…"
        self.sat = SatSegmenter(self.cfg.segmentation)
        if self.cfg.event.sat_parallel and self.cfg.event.max_sessions > 1:
            self.sat.lock = threading.BoundedSemaphore(self.cfg.event.max_sessions)
        self.sat.warmup()
        self.load_times["sat"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        self.pool = MarianPool(self.cfg.translation)
        pairs = [tuple(p.split("-")) for p in self.cfg.translation.preload]

        def status(text: str) -> None:
            self.status = text
            log.info(text)

        self.pool.preload(pairs, on_status=status)
        self.load_times["marian"] = round(time.perf_counter() - t0, 1)

        t0 = time.perf_counter()
        self.status = (f"Chargement de Whisper {self.cfg.asr.model} "
                       "(premier lancement : téléchargement d'environ 1,6 Go)…")
        log.info(self.status)
        self.whisper = WhisperEngine(self.cfg.asr, workers=self.cfg.event.whisper_workers)
        self.whisper.warmup()
        self.load_times["whisper"] = round(time.perf_counter() - t0, 1)

        if self.cfg.tts.engine == "piper":
            from app.tts.piper_tts import PiperTTS
            self.tts = PiperTTS(self.cfg.tts)  # voix chargées à la première utilisation
        self.status = "Prêt"
        log.info("Modèles prêts : %s. Ouvre http://localhost:%d", self.load_times,
                 self.cfg.server.port)

    def pool_for(self, slot: int):
        """Traducteurs de l'emplacement de session `slot` (0 = ceux chargés au démarrage)."""
        if slot == 0 or self.cfg.event.marian_instances != "per_session" or self.pool is None:
            return self.pool
        with self._slot_lock:
            pool = self._slot_pools.get(slot)
            if pool is None:
                from app.translation.marian_translator import MarianPool
                pool = MarianPool(self.cfg.translation)   # modèles chargés à la demande
                self._slot_pools[slot] = pool
            return pool

    def analysis_pool(self):
        """Traducteurs de l'analyse automatique (campagne).

        En mode per_session, l'analyse a son propre jeu (chargé à la demande) : ses
        rejeux, longs, ne prennent jamais les verrous des traducteurs d'une salle en
        direct (la place 0 utilise le jeu chargé au démarrage). En mode shared, tout le
        monde partage le même jeu, comme avant.
        """
        if self.cfg.event.marian_instances != "per_session" or self.pool is None:
            return self.pool
        with self._slot_lock:
            pool = self._slot_pools.get("analysis")
            if pool is None:
                from app.translation.marian_translator import MarianPool
                pool = MarianPool(self.cfg.translation)
                self._slot_pools["analysis"] = pool
            return pool

    def info(self) -> dict:
        return {
            "whisper": {"model": self.cfg.asr.model,
                        "device": getattr(self.whisper, "device", None),
                        "gpu_error": getattr(self.whisper, "gpu_error", None)},
            "sat": {"model": self.cfg.segmentation.model, "device": "cpu",
                    "backend": getattr(self.sat, "backend", None)},
            "marian": {"loaded": self.pool.loaded() if self.pool else [], "device": "cpu"},
            "tts": {"engine": self.cfg.tts.engine},
            "event": {"max_sessions": self.cfg.event.max_sessions,
                      "whisper_workers": getattr(self.whisper, "workers", None),
                      "marian_instances": self.cfg.event.marian_instances},
            "load_s": self.load_times,
            "ready": self.ready,
            "status": self.status,
            "error": self.error,
        }
