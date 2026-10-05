"""Voix traduite côté serveur avec Piper (ONNX, CPU).

Optionnel : par défaut, la voix est produite par le navigateur (tts.engine: browser).
Piper est sous licence GPL-3.0. Les voix se téléchargent avec :
    python -m piper.download_voices en_US-lessac-medium de_DE-thorsten-medium \
        fr_FR-siwis-medium --download-dir voices
"""
from __future__ import annotations

import base64
import io
import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable

from app.config import ROOT, TtsCfg

log = logging.getLogger("tts")


class PiperTTS:
    def __init__(self, cfg: TtsCfg):
        self.cfg = cfg
        self.dir = Path(cfg.piper_voices_dir)
        if not self.dir.is_absolute():
            self.dir = ROOT / self.dir
        self.voices: dict = {}
        self.lock = threading.Lock()

    def _voice(self, lang: str):
        if lang not in self.voices:
            from piper import PiperVoice

            name = self.cfg.piper_voices.get(lang)
            path = self.dir / f"{name}.onnx"
            if not name or not path.exists():
                raise FileNotFoundError(
                    f"Voix Piper introuvable pour « {lang} » ({path}). Télécharge-la avec : "
                    f"python -m piper.download_voices {name} --download-dir {self.cfg.piper_voices_dir}")
            self.voices[lang] = PiperVoice.load(str(path))
            log.info("Voix Piper '%s' chargée", name)
        return self.voices[lang]

    def synth_wav(self, lang: str, text: str) -> bytes:
        from piper import SynthesisConfig

        with self.lock:
            voice = self._voice(lang)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(
                    length_scale=self.cfg.length_scale))
        return buf.getvalue()


class TTSWorker(threading.Thread):
    """Synthétise les traductions définitives à la suite, sans bloquer la traduction."""

    def __init__(self, tts: PiperTTS, emit: Callable[[dict], None]):
        super().__init__(daemon=True, name="tts-worker")
        self.tts = tts
        self.emit = emit
        self.q: queue.Queue = queue.Queue()
        self.alive = True

    def submit(self, seg_id: int, lang: str, text: str) -> None:
        self.q.put((seg_id, lang, text))

    def run(self) -> None:
        while self.alive:
            try:
                seg_id, lang, text = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                wav = self.tts.synth_wav(lang, text)
            except Exception as exc:
                self.emit({"type": "error", "where": "tts", "message": str(exc)})
                continue
            self.emit({"type": "speak", "id": seg_id, "lang": lang, "text": text,
                       "engine": "piper", "ms": round((time.perf_counter() - t0) * 1000),
                       "audio": base64.b64encode(wav).decode("ascii")})
