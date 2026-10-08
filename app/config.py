"""Chargement de config.yaml dans des dataclasses typées."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_LANGS = ("en", "de", "fr")
LANG_NAMES = {"en": "anglais", "de": "allemand", "fr": "français"}


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8000
    # HTTPS : obligatoire pour utiliser le micro depuis un AUTRE ordinateur que le serveur
    # (les navigateurs n'autorisent le micro qu'en HTTPS ou sur localhost)
    ssl_certfile: str = ""
    ssl_keyfile: str = ""


@dataclass
class SessionCfg:
    source: str = "auto"
    targets: list = field(default_factory=lambda: ["fr"])
    voice: str = "auto"


@dataclass
class AsrCfg:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    min_chunk_s: float = 1.0
    buffer_trim_s: float = 15.0
    vad_filter: bool = True
    lcp_levenshtein: int = 1
    lang_detect_window_s: float = 3.0
    lang_switch_prob: float = 0.7


@dataclass
class SegCfg:
    model: str = "sat-3l-sm"
    backend: str = "onnx"
    sat_input: str = "raw"
    th_high: float = 0.5
    th_low: float = 0.25
    pause_s: float = 0.5
    long_pause_s: float = 1.2
    th_pause_floor: float = 0.05
    lookahead_words: int = 1
    min_words: int = 3
    max_words: int = 30


@dataclass
class MtCfg:
    model_pattern: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    preload: list = field(default_factory=lambda: ["en-fr", "fr-en"])
    pivot: str = "en"
    num_beams: int = 2
    max_new_tokens: int = 200
    provisional: bool = True
    mask_k: int = 3
    use_context: bool = True
    # Unités de sens : traduire les longues phrases par morceaux stables (désactivé par défaut)
    meaning_units: dict = field(default_factory=lambda: {
        "enabled": False, "k": 3, "min_words": 5, "sat_floor": 0.05, "speak_units": False})


@dataclass
class TtsCfg:
    engine: str = "browser"
    piper_voices_dir: str = "voices"
    piper_voices: dict = field(default_factory=lambda: {
        "en": "en_US-lessac-medium", "de": "de_DE-thorsten-medium", "fr": "fr_FR-siwis-medium"})
    length_scale: float = 0.95
    simultaneous_policy: str = "floor"   # floor : une seule voix à la fois | queue : tout lire


@dataclass
class CpuCfg:
    torch_threads: int = 4


@dataclass
class SourcesCfg:
    """Micros et orateurs. Sans cette section : un micro, un orateur (comportement historique)."""
    profile: str = "single"          # single | interview | panel | mixed | custom
    list: list = field(default_factory=list)   # [{id, name, mode, language, expected_speakers}]
    max_active_asr: int = 2
    crosstalk: dict = field(default_factory=lambda: {
        "enabled": True, "level_margin_db": 6, "text_similarity": 0.8})


@dataclass
class DiarizationCfg:
    engine: str = "none"             # none | embedding | diart | sortformer
    similarity_threshold: float = 0.65
    min_turn_s: float = 1.0
    enrollment: bool = False


@dataclass
class EventCfg:
    """Événement à plusieurs salles : jusqu'à max_sessions sessions en même temps."""
    max_sessions: int = 3              # sessions (salles) actives en même temps
    whisper_workers: int = 3           # appels Whisper en parallèle (un seul modèle en mémoire GPU)
    marian_instances: str = "per_session"   # per_session : traducteurs propres à chaque session | shared
    sat_parallel: bool = True          # SaT (ONNX) appelé en parallèle par les sessions
    default_room: str = "salle-1"      # salle utilisée quand l'interface n'en précise pas
    viewer_history: int = 200          # messages gardés pour un spectateur qui arrive en cours
    max_speakers: int = 2              # speakers (chacun sur son appareil) par salle


@dataclass
class LogCfg:
    enabled: bool = True
    runs_dir: str = "runs"


CONFIG_VERSION = 3  # 3 : section « event » (plusieurs sessions en parallèle, optionnelle)


@dataclass
class Config:
    config_version: int = CONFIG_VERSION
    server: ServerCfg = field(default_factory=ServerCfg)
    session: SessionCfg = field(default_factory=SessionCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    segmentation: SegCfg = field(default_factory=SegCfg)
    translation: MtCfg = field(default_factory=MtCfg)
    tts: TtsCfg = field(default_factory=TtsCfg)
    cpu: CpuCfg = field(default_factory=CpuCfg)
    logs: LogCfg = field(default_factory=LogCfg)
    sources: SourcesCfg = field(default_factory=SourcesCfg)
    diarization: DiarizationCfg = field(default_factory=DiarizationCfg)
    event: EventCfg = field(default_factory=EventCfg)


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path = ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # Un fichier sans config_version est un fichier de la version 1 : même format.
    return Config(
        config_version=int(raw.get("config_version", 1)),
        server=_build(ServerCfg, raw.get("server")),
        session=_build(SessionCfg, raw.get("session")),
        asr=_build(AsrCfg, raw.get("asr")),
        segmentation=_build(SegCfg, raw.get("segmentation")),
        translation=_build(MtCfg, raw.get("translation")),
        tts=_build(TtsCfg, raw.get("tts")),
        cpu=_build(CpuCfg, raw.get("cpu")),
        logs=_build(LogCfg, raw.get("logs")),
        sources=_build(SourcesCfg, raw.get("sources")),
        diarization=_build(DiarizationCfg, raw.get("diarization")),
        event=_build(EventCfg, raw.get("event")),
    )
