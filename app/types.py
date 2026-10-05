"""Types partagés entre les modules."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.config import SUPPORTED_LANGS


@dataclass
class Word:
    start: float      # secondes, temps absolu depuis le début du flux
    end: float
    text: str         # texte brut de Whisper (casse + ponctuation)
    lang: str = "en"  # langue dans laquelle Whisper a transcrit ce mot
    source: str = "main"   # micro d'origine (étape 1 : toujours « main »)
    speaker: str = "S1"    # orateur (étape 1 : toujours « S1 »)


@dataclass
class SessionSettings:
    """Réglages d'une session, partagés par les threads et modifiables en direct."""
    source: str = "auto"                              # auto | en | de | fr
    targets: list = field(default_factory=lambda: ["fr"])
    voice: str = "auto"                               # none | auto | en | de | fr
    mu_enabled: bool = False                          # unités de sens (traduction anticipée)
    mu_k: int = 3                                     # mots d'avance pour le test de stabilité
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, values: dict) -> None:
        with self._lock:
            src = values.get("source", self.source)
            if src == "auto" or src in SUPPORTED_LANGS:
                self.source = src
            if "targets" in values:
                self.targets = [t for t in SUPPORTED_LANGS if t in values["targets"]]
            voice = values.get("voice", self.voice)
            if voice in ("none", "auto") or voice in SUPPORTED_LANGS:
                self.voice = voice
            if "mu_enabled" in values:
                self.mu_enabled = bool(values["mu_enabled"])
            if "mu_k" in values:
                try:
                    self.mu_k = min(4, max(2, int(values["mu_k"])))
                except (TypeError, ValueError):
                    pass

    def targets_for(self, lang: str) -> list[str]:
        """Une phrase est traduite dans chaque langue cochée, sauf la sienne."""
        with self._lock:
            return [t for t in self.targets if t != lang]

    def voice_for(self, lang: str) -> str | None:
        targets = self.targets_for(lang)
        with self._lock:
            voice = self.voice
        if voice == "none" or not targets:
            return None
        if voice == "auto":
            return targets[0]
        return voice if voice in targets else None

    def as_dict(self) -> dict:
        with self._lock:
            return {"source": self.source, "targets": list(self.targets), "voice": self.voice,
                    "mu_enabled": self.mu_enabled, "mu_k": self.mu_k}
