"""Micros (sources) et orateurs.

Profils actifs :
  single : un micro, un orateur (comportement historique, par défaut)
  panel  : plusieurs micros, un orateur par micro (étape 2)
  custom / mixed : plusieurs micros ; les micros en mode « multi » ou « auto »
           (plusieurs orateurs sur un même micro) arrivent à l'étape 3 et sont
           traités en attendant comme « un orateur ».
  interview : un micro, plusieurs orateurs (étape 3) -> « single » en attendant.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass

from app.config import SUPPORTED_LANGS, SourcesCfg

log = logging.getLogger("sources")

PROFILES = ("single", "interview", "panel", "mixed", "custom")
IMPLEMENTED_PROFILES = ("single", "panel", "mixed", "custom")
SOURCE_MODES = ("single", "multi", "auto")
MAX_SOURCES = 6
COLORS = ("#2A4FE0", "#B87C0C", "#1F8A5B", "#B03A6F", "#6B4FC8", "#2B8FA8")
_ID = re.compile(r"[^a-z0-9_-]+")


@dataclass
class SourceConfig:
    id: str = "main"
    name: str = "Micro principal"
    mode: str = "single"            # single | multi | auto
    language: str | None = None     # None = réglage de la session ; « auto » ; en | de | fr
    expected_speakers: int = 1
    index: int = 0                  # numéro dans l'en-tête des blocs audio


@dataclass
class Speaker:
    id: str
    name: str
    source: str
    color: str
    lang: str | None = None
    voice: str | None = None


class SpeakerRegistry:
    """Registre commun à tous les micros (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._speakers: dict[str, Speaker] = {}

    def add(self, source: str, name: str | None = None, lang: str | None = None) -> Speaker:
        with self._lock:
            n = len(self._speakers) + 1
            spk = Speaker(id=f"S{n}", name=name or f"Intervenant {n}", source=source,
                          color=COLORS[(n - 1) % len(COLORS)], lang=lang)
            self._speakers[spk.id] = spk
            return spk

    def get(self, speaker_id: str) -> Speaker | None:
        with self._lock:
            return self._speakers.get(speaker_id)

    def rename(self, speaker_id: str, name: str) -> Speaker | None:
        with self._lock:
            spk = self._speakers.get(speaker_id)
            if spk:
                spk.name = name.strip()[:40] or spk.name
            return spk

    def as_list(self) -> list[dict]:
        with self._lock:
            return [asdict(s) for s in self._speakers.values()]


def _clean_source(raw: dict, index: int, used: set[str], warnings: list[str]) -> SourceConfig:
    sid = _ID.sub("", str(raw.get("id") or f"m{index + 1}").lower())[:16] or f"m{index + 1}"
    while sid in used:
        sid = f"{sid[:13]}_{index + 1}"
    used.add(sid)
    mode = raw.get("mode", "single")
    if mode not in SOURCE_MODES:
        warnings.append(f"Mode inconnu « {mode} » pour {sid} : « single » utilisé.")
        mode = "single"
    if mode != "single":
        warnings.append(f"Plusieurs orateurs sur « {raw.get('name') or sid} » : disponible à "
                        "l'étape 3 ; un seul orateur est utilisé pour ce micro en attendant.")
        mode = "single"
    lang = raw.get("language")
    if lang not in (None, "", "auto", *SUPPORTED_LANGS):
        warnings.append(f"Langue inconnue « {lang} » pour {sid} : réglage de la session utilisé.")
        lang = None
    return SourceConfig(id=sid, name=str(raw.get("name") or f"Micro {index + 1}")[:40],
                        mode=mode, language=lang or None,
                        expected_speakers=max(1, int(raw.get("expected_speakers", 1) or 1)),
                        index=index)


def resolve_sources(cfg: SourcesCfg, override: dict | None = None
                    ) -> tuple[str, list[SourceConfig], list[str]]:
    """Profil effectif, liste des micros, avertissements à afficher.

    `override` : configuration envoyée par l'interface au démarrage
    ({"profile": ..., "sources": [...]}) ; sinon config.yaml.
    """
    warnings: list[str] = []
    requested = (override or {}).get("profile") or cfg.profile
    raw_list = (override or {}).get("sources")
    if raw_list is None:
        raw_list = cfg.list or []

    if requested not in PROFILES:
        warnings.append(f"Profil inconnu « {requested} » : profil « single » utilisé.")
        requested = "single"
    if requested not in IMPLEMENTED_PROFILES:
        warnings.append(f"Le profil « {requested} » arrive à l'étape 3 : "
                        "profil « single » utilisé pour l'instant.")
        requested = "single"
    if requested == "single":
        return "single", [SourceConfig()], warnings

    if len(raw_list) > MAX_SOURCES:
        warnings.append(f"Au plus {MAX_SOURCES} micros : les suivants sont ignorés.")
        raw_list = raw_list[:MAX_SOURCES]
    used: set[str] = set()
    sources = [_clean_source(dict(r), i, used, warnings) for i, r in enumerate(raw_list)]
    if not sources:
        warnings.append("Aucun micro décrit : profil « single » utilisé.")
        return "single", [SourceConfig()], warnings
    return requested, sources, warnings


class SourceSettings:
    """Vue des réglages de session pour un micro : sa langue peut être imposée."""

    def __init__(self, settings, source: SourceConfig):
        self._settings = settings
        self._source = source

    @property
    def source(self) -> str:
        return self._source.language or self._settings.source

    def __getattr__(self, name):
        return getattr(self._settings, name)
