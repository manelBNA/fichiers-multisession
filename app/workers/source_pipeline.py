"""Pipeline d'un micro : ASR (Whisper) + segmentation (SaT), avec leurs files.

Une session crée un pipeline par micro ; le traducteur et la voix sont communs.
Les messages envoyés à l'interface reçoivent les champs « source » et
« speaker » (ajoutés seulement s'ils n'y sont pas déjà).
Avec plusieurs micros, l'audio passe d'abord par l'automixer (anti-diaphonie).
"""
from __future__ import annotations

import queue
from typing import Callable

from app.sources import SourceConfig, SourceSettings, Speaker
from app.workers.asr_worker import STOP, ASRWorker, pcm16_to_float
from app.workers.segmenter_worker import SegmenterWorker


class SourcePipeline:
    def __init__(self, source: SourceConfig, speaker: Speaker, registry, cfg, settings,
                 policy, translator, emit: Callable[[dict], None], next_id: Callable[[], int],
                 multi_source: bool = False, automixer=None, duplicate_filter=None,
                 thread_tag: str = "", on_fatal=None):
        self.source = source
        self.speaker = speaker
        self.automixer = automixer
        self.audio_q: queue.Queue = queue.Queue()
        self.seg_q: queue.Queue = queue.Queue()

        def tagged(msg: dict) -> None:
            msg.setdefault("source", source.id)
            msg.setdefault("speaker", speaker.id)
            emit(msg)

        view = SourceSettings(settings, source)   # langue imposée par micro, sinon session
        self.asr = ASRWorker(registry.whisper, cfg.asr, view, self.audio_q, self.seg_q,
                             tagged, source_id=source.id, speaker_id=speaker.id,
                             multi_source=multi_source, thread_tag=thread_tag)
        accept = duplicate_filter.accept if duplicate_filter is not None else None
        self.seg = SegmenterWorker(registry.sat, policy, self.seg_q, translator, tagged,
                                   next_id=next_id, source_id=source.id, accept=accept,
                                   settings=view, mu_cfg=cfg.translation.meaning_units,
                                   tracker=getattr(translator, "mu_tracker", None),
                                   thread_tag=thread_tag)
        self.asr.on_fatal = self.seg.on_fatal = on_fatal

    @property
    def threads(self) -> list:
        return [self.asr, self.seg]

    @property
    def alive(self) -> bool:
        return self.asr.is_alive()

    def start(self) -> None:
        self.asr.start()
        self.seg.start()

    def push(self, data: bytes) -> None:
        chunk = pcm16_to_float(data)
        if self.automixer is not None:
            chunk = self.automixer.process(self.source.id, chunk)   # un bloc de retard
            if chunk is None:
                return
        self.audio_q.put(chunk)

    def stop(self) -> None:
        if self.automixer is not None:
            last = self.automixer.flush(self.source.id)
            if last is not None:
                self.audio_q.put(last)
        self.audio_q.put(STOP)

    def refresh(self) -> None:
        self.seg_q.put(("refresh",))

    def close(self) -> None:
        self.asr.alive = False
        self.seg.alive = False
