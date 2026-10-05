"""Whisper (faster-whisper / CTranslate2) sur le GPU, multilingue.

Le GPU est réservé à Whisper : SaT, Marian et la voix tournent sur le CPU.

Robustesse CUDA :
- Sous Windows, les DLL CUDA installées par pip (nvidia-cublas-cu12,
  nvidia-cudnn-cu12) sont ajoutées automatiquement au chemin de recherche.
- Au démarrage, un vrai calcul est lancé sur le GPU : si cuBLAS/cuDNN manquent,
  l'erreur apparaît tout de suite et on se replie sur le CPU.
- Si l'erreur survient quand même plus tard, bascule sur CPU à chaud, sans
  arrêter le flux.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from app.config import AsrCfg
from app.types import Word

log = logging.getLogger("asr")

CUDA_FIX = (
    "Installe les bibliothèques CUDA 12 : "
    "pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\" "
    "puis relance le serveur (voir README, section Windows)."
)


def _register_nvidia_dlls() -> list[str]:
    """Windows : rend visibles les DLL de site-packages/nvidia/*/bin."""
    if os.name != "nt":
        return []
    import site
    import sysconfig

    roots = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"], sys.prefix}
    try:
        roots.update(site.getsitepackages())
        roots.add(site.getusersitepackages())
    except AttributeError:  # certains environnements virtuels
        pass
    added = []
    for root in roots:
        base = Path(root) / "nvidia"
        if not base.is_dir():
            base = Path(root) / "Lib" / "site-packages" / "nvidia"
        if not base.is_dir():
            continue
        for bin_dir in base.glob("*/bin"):
            path = str(bin_dir)
            if path in added:
                continue
            try:
                os.add_dll_directory(path)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            added.append(path)
    return added


_DLL_DIRS = _register_nvidia_dlls()


def _is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "cudart", ".dll", "libcu"))


class WhisperEngine:
    def __init__(self, cfg: AsrCfg, workers: int = 1):
        if cfg.model.endswith(".en"):
            log.warning("Le modèle %s ne connaît que l'anglais : prends un modèle multilingue "
                        "(small, medium, large-v3-turbo) pour l'allemand et le français.", cfg.model)
        if _DLL_DIRS:
            log.info("DLL CUDA ajoutées : %s", ", ".join(_DLL_DIRS))
        self.cfg = cfg
        # Plusieurs sessions : un seul modèle en mémoire, `workers` appels en parallèle
        # (num_workers de faster-whisper). Avec workers = 1 : un appel à la fois, comme avant.
        self.workers = max(1, int(workers))
        self.lock = threading.BoundedSemaphore(self.workers)
        self._fallback_lock = threading.Lock()
        self.gpu_error: str | None = None
        self.device = "cpu"
        self.model = None

        if cfg.device != "cpu":
            try:
                self._load(cfg.device, cfg.compute_type)
                self._probe()  # vrai calcul : révèle tout de suite une DLL manquante
            except Exception as exc:
                self._fallback(exc)
        else:
            self._load("cpu", "int8")
        log.info("Whisper '%s' prêt sur %s", cfg.model, self.device)

    # ---------- chargement ----------
    def _load(self, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(self.cfg.model, device=device, compute_type=compute_type,
                                  num_workers=self.workers)
        self.device = device

    def _probe(self) -> None:
        """Encode 1 s de bruit sans VAD, pour forcer un passage dans cuBLAS/cuDNN."""
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        segments, _ = self.model.transcribe(noise, language="en", beam_size=1, vad_filter=False)
        for _ in segments:
            pass

    def _fallback(self, exc: Exception) -> None:
        with self._fallback_lock:          # plusieurs sessions peuvent échouer en même temps
            if self.device == "cpu" and self.model is not None:
                return                     # une autre session a déjà basculé sur le CPU
            self._fallback_unlocked(exc)

    def _fallback_unlocked(self, exc: Exception) -> None:
        self.gpu_error = str(exc)
        hint = CUDA_FIX if _is_cuda_error(exc) else ""
        log.error("Whisper ne peut pas utiliser le GPU (%s). Repli sur CPU int8, "
                  "beaucoup plus lent. %s", exc, hint)
        self._load("cpu", "int8")

    def _run(self, fn):
        """Exécute un appel Whisper ; bascule sur CPU si le GPU échoue en cours de route."""
        with self.lock:
            try:
                return fn()
            except RuntimeError as exc:
                if self.device == "cpu" or not _is_cuda_error(exc):
                    raise
                self._fallback(exc)
                return fn()

    # ---------- API ----------
    def transcribe(self, audio: np.ndarray, prompt: str, language: str) -> list[Word]:
        """Transcrit le buffer (float32, 16 kHz) dans la langue imposée. Temps relatifs."""
        def call() -> list[Word]:
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                task="transcribe",
                initial_prompt=prompt or None,
                beam_size=self.cfg.beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,
                vad_filter=self.cfg.vad_filter,
            )
            words: list[Word] = []
            for seg in segments:  # le générateur doit être consommé ici (erreurs GPU comprises)
                if seg.no_speech_prob > 0.9:
                    continue
                for w in seg.words or []:
                    words.append(Word(w.start, w.end, w.word, language))
            return words

        return self._run(call)

    def detect(self, audio: np.ndarray, allowed: tuple[str, ...]) -> tuple[str | None, float]:
        """Langue la plus probable parmi `allowed`, probabilité renormalisée entre elles."""
        _, _, all_probs = self._run(lambda: self.model.detect_language(audio=audio))
        cand = [(lang, p) for lang, p in all_probs if lang in allowed]
        total = sum(p for _, p in cand)
        if not cand or total <= 0:
            return None, 0.0
        lang, p = max(cand, key=lambda x: x[1])
        return lang, p / total

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32), "", "en")
