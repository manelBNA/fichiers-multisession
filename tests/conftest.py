"""Outils communs aux tests : faux modèles, déterministes et instantanés.

Aucun vrai modèle n'est chargé : les tests tournent en quelques secondes, sans GPU
ni téléchargement, et vérifient la LOGIQUE du projet (validation des mots,
segmentation, traduction, session), pas la qualité des modèles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.types import Word  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"


def make_timeline(parts: list[tuple[str, str]], word_s: float = 0.3, gap_s: float = 0.1,
                  pause_between_parts: float = 1.0):
    """[(texte, langue)] -> [(début, fin, mot, langue)] avec des horodatages réguliers."""
    tl, t = [], 0.0
    for k, (text, lang) in enumerate(parts):
        if k:
            t += pause_between_parts
        for w in text.split():
            tl.append((t, t + word_s, w, lang))
            t += word_s + gap_s
    return tl


class FakeEngine:
    """Faux Whisper : « entend » les mots de la chronologie présents dans le buffer.

    Un mot n'est rendu que s'il se termine au moins 0,2 s avant la fin du buffer,
    et si l'audio est audible à cet endroit (un bloc remis à zéro par l'automixer
    n'est donc pas entendu). Dans une autre langue que la sienne, il est rendu en
    MAJUSCULES (mauvaise langue). Avec plusieurs micros, chaque thread ASR
    (« asr-worker-<micro> ») a sa propre chronologie et sa propre position.
    """
    device = "fake"
    AUDIBLE_RMS = 0.002

    def __init__(self, timeline):
        self.timelines = timeline if isinstance(timeline, dict) else {"main": timeline}
        self.offsets: dict[str, float] = {}
        self.audio_lens: dict[str, float] = {}

    @staticmethod
    def _who() -> str:
        import threading
        name = threading.current_thread().name
        return name.split("asr-worker-", 1)[1] if "asr-worker-" in name else "main"

    def transcribe(self, audio, prompt, language):
        who = self._who()
        offset = self.offsets.get(who, 0.0)
        dur = len(audio) / 16000
        out = []
        for s, e, w, lang in self.timelines.get(who, []):
            if s >= offset - 0.05 and e <= offset + dur - 0.2:
                a, b = int((s - offset) * 16000), int((e - offset) * 16000)
                seg = audio[max(a, 0):max(b, 1)]
                if len(seg) and float(np.sqrt(np.mean(seg ** 2))) < self.AUDIBLE_RMS:
                    continue
                text = w if lang == language else w.upper()
                out.append(Word(s - offset, e - offset, " " + text, language))
        return out

    def detect(self, audio, allowed):
        who = self._who()
        end = self.offsets.get(who, 0.0) + self.audio_lens.get(who, 0.0)
        start = end - len(audio) / 16000
        langs = [l for s, e, w, l in self.timelines.get(who, []) if s < end and e > start]
        if not langs:
            return None, 0.0
        best = max(set(langs), key=langs.count)
        return best, (0.95 if langs.count(best) == len(langs) else 0.75)


class FakeSat:
    """Faux SaT : 0,9 après un mot qui finit par . ? !, sinon 0,02."""
    backend = "fake"

    def word_probs(self, words, mode=None, context=None):
        probs = [0.9 if w.strip().endswith((".", "?", "!")) else 0.02 for w in words]
        return probs, 1.0, " ".join(w.strip() for w in words)


class FakePool:
    """Fausse traduction : « [src>tgt] texte », contexte utilisé dès la 2e phrase d'une paire."""

    def translate(self, src, tgt, text):
        return f"[{src}>{tgt}] {text}", False

    def translate_with_context(self, src, tgt, prev_src, prev_tgt, cur):
        return f"[{src}>{tgt}] {cur}", bool(prev_src), False

    def is_ready(self, src, tgt):
        return True

    def preload(self, pairs, on_status=None):
        pass

    def loaded(self):
        return []


class FakeMUPool(FakePool):
    """Traduction mot à mot (stable) ; un mot de REORDER inverse tout l'ordre (verbe final)."""
    REORDER = {"reduzieren"}

    def mapping(self, text: str) -> list[str]:
        words = [w.lower() for w in text.split()]
        if any(w.strip(".,;:") in self.REORDER for w in words):
            words = list(reversed(words))
        return words

    def translate(self, src, tgt, text):
        return " ".join(self.mapping(text)), False

    def translate_with_context(self, src, tgt, prev_src, prev_tgt, cur):
        return " ".join(self.mapping(cur)), bool(prev_src), False

    def translate_forced(self, src, tgt, text, prefix):
        full = self.mapping(text)
        return " ".join(full[len(prefix.split()) if prefix.strip() else 0:])


class FakeRegistry:
    def __init__(self, timeline):
        self.whisper = FakeEngine(timeline)
        self.sat = FakeSat()
        self.pool = FakePool()
        self.tts = None

    @property
    def ready(self):
        return True

    def info(self):
        return {"fake": True, "ready": True, "whisper": {"model": "fake", "device": "fake",
                                                         "gpu_error": None}}


@pytest.fixture
def patch_online_asr(monkeypatch):
    """Donne au faux Whisper la position absolue du buffer (décalage, durée)."""
    import app.asr.local_agreement as la

    orig_process, orig_tail = la.OnlineASR.process_iter, la.OnlineASR.tail

    def process_iter(self):
        who = FakeEngine._who()
        self.engine.offsets[who] = self.offset
        return orig_process(self)

    def tail(self, seconds):
        who = FakeEngine._who()
        self.engine.offsets[who] = self.offset
        self.engine.audio_lens[who] = len(self.audio) / 16000
        return orig_tail(self, seconds)

    monkeypatch.setattr(la.OnlineASR, "process_iter", process_iter)
    monkeypatch.setattr(la.OnlineASR, "tail", tail)


def run_fake_session(timeline, settings: dict, tmp_path, label: str = "test",
                     extra_s: float = 1.5, timeout_s: float = 20.0, pool=None,
                     analyze: bool = False) -> list[dict]:
    """Joue une chronologie dans une vraie Session (threads compris) et renvoie les messages."""
    import asyncio
    import time

    from app.config import load_config
    from app.session import Session

    cfg = load_config()
    cfg.asr.min_chunk_s = 0.5
    cfg.logs.runs_dir = str(tmp_path / "runs")
    registry = FakeRegistry(timeline)
    if pool is not None:
        registry.pool = pool
    total = (timeline[-1][1] if timeline else 0) + extra_s

    async def main():
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        analysis = None
        if analyze:
            from app.analysis import AnalysisWorker
            analysis = AnalysisWorker(registry, cfg)
            analysis.start()
        session = Session(registry, cfg, loop, out_q, analysis=analysis)
        session.start(settings, label, None, analyze)
        chunk = (np.ones(1600) * 3000).astype(np.int16).tobytes()  # 100 ms de « parole »
        t = 0.0
        while t < total:
            session.push_audio(chunk)
            t += 0.1
            await asyncio.sleep(0.004)
        session.stop()
        msgs, end = [], time.time() + timeout_s
        while time.time() < end:
            try:
                m = await asyncio.wait_for(out_q.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            msgs.append(m)
            if m.get("type") == "run_saved" and not analyze:
                break
            if m.get("type") == "analysis_done":
                break
        session.close()
        return msgs

    return asyncio.run(main())


def stable_view(msgs: list[dict]) -> dict:
    """Partie déterministe des messages : ce qui ne doit JAMAIS changer sans raison."""
    return {
        "segments": [[m["text"], m["lang"]] for m in msgs if m["type"] == "segment"],
        "translations": sorted([m["id"], m["lang"], m["text"]] for m in msgs
                               if m["type"] == "translation" and m.get("final")),
        "speak": sorted([m["id"], m["lang"]] for m in msgs if m["type"] == "speak"),
        "langs": [m["lang"] for m in msgs if m["type"] == "lang"],
        "done": any(m["type"] == "done" for m in msgs),
    }


def run_fake_multi_session(sources: list[dict], timelines: dict, amplitude, settings: dict,
                           tmp_path, total_s: float, label: str = "multi",
                           crosstalk: bool = True, timeout_s: float = 25.0) -> list[dict]:
    """Plusieurs micros : `amplitude(micro, t)` donne le niveau (0 à 1) de chaque micro à
    l'instant t. Les blocs audio sont envoyés au format étiqueté, comme le navigateur."""
    import asyncio
    import time

    from app.audio_framing import make_frame
    from app.config import load_config
    from app.session import Session

    cfg = load_config()
    cfg.asr.min_chunk_s = 0.5
    cfg.logs.runs_dir = str(tmp_path / "runs")
    cfg.sources.crosstalk = {"enabled": crosstalk, "level_margin_db": 6, "text_similarity": 0.8}
    registry = FakeRegistry(timelines)

    async def main():
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        session = Session(registry, cfg, loop, out_q)
        session.start(settings, label, {"profile": "panel", "sources": sources,
                                        "framing": "tagged"})
        t = 0.0
        while t < total_s:
            for i, src in enumerate(sources):
                level = amplitude(src["id"], t)
                pcm = (np.ones(1600) * 30000 * level).astype(np.int16).tobytes()
                session.push_audio(make_frame(i, pcm))
            t += 0.1
            await asyncio.sleep(0.004)
        session.stop()
        msgs, end = [], time.time() + timeout_s
        while time.time() < end:
            try:
                m = await asyncio.wait_for(out_q.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            msgs.append(m)
            if m.get("type") == "run_saved":
                break
        session.close()
        return msgs

    return asyncio.run(main())
