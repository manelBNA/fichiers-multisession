"""Plusieurs sessions (salles) en même temps : isolation, capacité, spectateurs,
cycle de vie (arrêt, redémarrage, erreur d'une salle sans effet sur les autres)."""
import asyncio
import json
import threading
import time
from collections import defaultdict

import numpy as np
import pytest

from app.config import load_config
from app.rooms import Fanout, RoomManager, clean_room_name
from app.session import Session
from tests.conftest import FakeEngine, FakeRegistry, FakeSat, make_timeline

TEXTS = {
    "salle-a": ("Die Entwicklung wird durch Nachhaltigkeit geprägt. Digitale Zwillinge begleiten "
                "dabei sämtliche Prozessschritte.", "de"),
    "salle-b": ("Hello everyone. Today I will explain how attention works in transformers.", "en"),
    "salle-c": ("Bonjour à tous. Je vais maintenant répondre à votre question sur les données.", "fr"),
}


def cfg_for(tmp_path, max_sessions=3):
    cfg = load_config()
    cfg.asr.min_chunk_s = 0.5
    cfg.logs.runs_dir = str(tmp_path / "runs")
    cfg.event.max_sessions = max_sessions
    return cfg


def test_room_names_are_cleaned():
    assert clean_room_name("  Salle A (Amphi 2) ", "salle-1") == "salle-a-amphi-2"
    assert clean_room_name("", "salle-1") == "salle-1"


def test_three_rooms_in_parallel_are_isolated(patch_online_asr, tmp_path):
    timelines = {f"{name}:main": make_timeline([parts]) for name, parts in TEXTS.items()}
    registry = FakeRegistry(timelines)
    cfg = cfg_for(tmp_path)
    total = max(tl[-1][1] for tl in timelines.values()) + 1.5

    async def main():
        loop = asyncio.get_running_loop()
        manager = RoomManager(cfg, registry)
        sessions, queues = {}, {}
        for name, (_, lang) in TEXTS.items():
            room = manager.get(name)
            q: asyncio.Queue = asyncio.Queue()
            manager.subscribe(room, q, "producer")
            s = Session(registry, cfg, loop, Fanout(room), room=name)
            err = manager.start_session(room, s, {"settings": {"source": lang, "targets": ["en", "fr"],
                                                              "voice": "none"}, "label": name})
            assert err is None
            sessions[name], queues[name] = s, q
        assert manager.active == 3
        chunk = (np.ones(1600) * 3000).astype(np.int16).tobytes()
        t = 0.0
        while t < total:                       # les trois salles reçoivent l'audio en même temps
            for s in sessions.values():
                s.push_audio(chunk)
            t += 0.1
            await asyncio.sleep(0.004)
        for s in sessions.values():
            s.stop()
        out = {name: [] for name in sessions}
        done, end = set(), time.time() + 30
        while len(done) < 3 and time.time() < end:
            for name, q in queues.items():
                while not q.empty():
                    m = q.get_nowait()
                    out[name].append(m)
                    if m["type"] == "run_saved":
                        done.add(name)
            await asyncio.sleep(0.05)
        for s in sessions.values():
            s.close()
        return out, manager.active

    out, active_after = asyncio.run(main())
    for name, (text, lang) in TEXTS.items():
        segs = [m["text"] for m in out[name] if m["type"] == "segment"]
        assert " ".join(segs) == text, name                    # chaque salle a son texte
        others = [t for n, (t, _) in TEXTS.items() if n != name]
        assert not any(o.split()[0] in " ".join(segs) for o in others)   # rien des autres salles
        assert any(m["type"] == "run_saved" and m["complete"] for m in out[name])
    assert active_after == 0                                   # toutes les places libérées
    # chaque test est enregistré avec sa salle
    metas = [json.load(open(p, encoding="utf-8")) for p in (tmp_path / "runs").glob("*/meta.json")]
    assert sorted(m["salle"] for m in metas) == sorted(TEXTS)


def test_fourth_session_is_refused_until_a_place_is_free(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.logs.enabled = False
    registry = FakeRegistry([])

    async def main():
        loop = asyncio.get_running_loop()
        manager = RoomManager(cfg, registry)
        sessions = []
        for i in range(3):
            room = manager.get(f"salle-{i + 1}")
            s = Session(registry, cfg, loop, Fanout(room), room=room.name)
            assert manager.start_session(room, s, {"settings": {"source": "en", "targets": ["fr"]}}) is None
            sessions.append(s)
        room4 = manager.get("salle-4")
        s4 = Session(registry, cfg, loop, Fanout(room4), room=room4.name)
        refused = manager.start_session(room4, s4, {})
        assert refused and "Capacité atteinte" in refused and "salle-1" in refused
        sessions[0].close()                                    # une salle s'arrête
        accepted = manager.start_session(room4, s4, {"settings": {"source": "en", "targets": ["fr"]}})
        restarted = manager.start_session(manager.get("salle-2"), sessions[1],
                                          {"settings": {"source": "en", "targets": ["fr"]}})
        state = manager.overview()
        for s in sessions[1:] + [s4]:
            s.close()
        return accepted, restarted, state, manager.active

    accepted, restarted, state, active_after = asyncio.run(main())
    assert accepted is None and restarted is None              # un redémarrage garde sa place
    assert state["active"] == 3 and {r["name"] for r in state["rooms"] if r["running"]} == \
        {"salle-2", "salle-3", "salle-4"}
    assert active_after == 0


def test_viewer_receives_history_then_live_messages():
    from app.rooms import Room
    room = Room("salle-a", history=50)

    class Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, m):
            self.items.append(m)

    producer, viewer = Q(), Q()
    room.subscribers[producer] = "producer"
    room.broadcast({"type": "status", "state": "running", "settings": {"targets": ["en"]}})
    room.broadcast({"type": "segment", "id": 1, "text": "Hallo."})
    room.broadcast({"type": "translation", "id": 1, "lang": "en", "text": "Hello.", "final": True})
    room.broadcast({"type": "translation", "id": 2, "lang": "en", "text": "Wor", "final": False})
    assert [m["type"] for m in room.history] == ["segment", "translation"]   # pas le provisoire
    room.subscribers[viewer] = "viewer"
    room.broadcast({"type": "segment", "id": 2, "text": "Welt."})
    assert viewer.items == [{"type": "segment", "id": 2, "text": "Welt."}] and len(producer.items) == 5
    assert room.viewers == 1 and room.settings == {"targets": ["en"]}
    room.broadcast({"type": "status", "state": "running", "settings": {"targets": ["fr"]}})
    assert list(room.history) == []                            # nouvelle session : historique vidé


def test_translators_per_session_or_shared():
    from app.registry import ModelRegistry
    cfg = load_config()
    reg = ModelRegistry(cfg)
    reg.pool = object()
    a, b = reg.pool_for(1), reg.pool_for(2)
    assert reg.pool_for(0) is reg.pool and a is not reg.pool and a is not b and reg.pool_for(1) is a
    cfg.event.marian_instances = "shared"
    assert reg.pool_for(2) is reg.pool


def test_config_defaults_for_events():
    cfg = load_config()
    assert cfg.event.max_sessions == 3 and cfg.event.whisper_workers == 3
    assert cfg.event.marian_instances == "per_session" and cfg.event.default_room == "salle-1"


# ---------- outils : un événement avec de vraies sessions (threads) et de faux modèles ----------

CHUNK = (np.ones(1600) * 3000).astype(np.int16).tobytes()      # 100 ms de « parole »
TIMELINES = {f"{name}:main": make_timeline([parts]) for name, parts in TEXTS.items()}
TOTAL_S = max(tl[-1][1] for tl in TIMELINES.values()) + 1.5


def start_data(name: str) -> dict:
    _, lang = TEXTS[name]
    return {"settings": {"source": lang, "targets": ["en", "fr"], "voice": "none"}, "label": name}


class Event:
    """Un RoomManager, des salles avec leur régie et leurs spectateurs, de vraies sessions."""

    def __init__(self, cfg, registry, loop):
        self.cfg, self.registry, self.loop = cfg, registry, loop
        self.manager = RoomManager(cfg, registry)
        self.sessions: dict[str, Session] = {}
        self.queues: dict[tuple[str, str], asyncio.Queue] = {}
        self.out: dict[tuple[str, str], list] = defaultdict(list)

    def open(self, name: str, viewers=()) -> None:
        room = self.manager.get(name)
        self.queues[(name, "producer")] = q = asyncio.Queue()
        self.manager.subscribe(room, q, "producer")
        for who in viewers:
            self.queues[(name, who)] = q = asyncio.Queue()
            self.manager.subscribe(room, q, "viewer", who)
        self.sessions[name] = Session(self.registry, self.cfg, self.loop, Fanout(room), room=name)

    def room(self, name: str):
        return self.manager.get(name)

    def state(self, name: str) -> str:
        return self.room(name).state

    def start(self, name: str) -> None:
        assert self.manager.start_session(self.room(name), self.sessions[name], start_data(name)) is None

    def stop(self, name: str) -> None:
        self.manager.stop_session(self.room(name))

    def drain(self) -> None:
        for key, q in self.queues.items():
            while not q.empty():
                self.out[key].append(q.get_nowait())

    def msgs(self, name: str, who: str = "producer", kind: str | None = None) -> list:
        self.drain()
        return [m for m in self.out[(name, who)] if kind is None or m["type"] == kind]

    def text(self, name: str, who: str = "producer", after: int = 0) -> str:
        """Phrases reçues (à partir du `after`-ième message)."""
        return " ".join(m["text"] for m in self.msgs(name, who)[after:] if m["type"] == "segment")

    def saved(self, name: str) -> int:
        return len(self.msgs(name, kind="run_saved"))

    async def feed(self, names, seconds: float) -> None:
        """Audio en « temps accéléré » : 100 ms d'audio toutes les 4 ms, pour `names` seulement."""
        t = 0.0
        while t < seconds:
            for n in names:
                self.sessions[n].push_audio(CHUNK)
            t += 0.1
            await asyncio.sleep(0.004)
        self.drain()

    async def wait(self, cond, timeout: float = 30) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.drain()
            if cond():
                return True
            await asyncio.sleep(0.05)
        self.drain()
        return cond()

    def close(self) -> None:
        for s in self.sessions.values():
            s.close()


def run_event(tmp_path, scenario, registry=None, max_sessions=3):
    cfg = cfg_for(tmp_path, max_sessions)
    registry = registry or FakeRegistry(TIMELINES)

    async def main():
        ev = Event(cfg, registry, asyncio.get_running_loop())
        try:
            return await scenario(ev)
        finally:
            ev.close()

    return asyncio.run(main())


def words_of(name: str) -> set:
    return set(TEXTS[name][0].split())


def states_after(room, state: str) -> list:
    """Suite des états de la salle depuis la dernière fois où elle était dans `state`."""
    seq = [s for _, s in room.transitions]
    i = len(seq) - 1 - seq[::-1].index(state)
    return seq[i:]


# ---------- TESTS 1, 2 et 3 : une, deux, trois salles ----------

@pytest.mark.parametrize("names", [["salle-a"], ["salle-a", "salle-b"],
                                   ["salle-a", "salle-b", "salle-c"]])
def test_one_two_three_sessions_run_and_stop(patch_online_asr, tmp_path, names):
    async def scenario(ev):
        for n in names:
            ev.open(n)
            assert ev.state(n) == "created"
            ev.start(n)
        running = {n: ev.state(n) for n in names}
        active = ev.manager.active
        await ev.feed(names, TOTAL_S)
        for n in names:
            ev.stop(n)
        assert await ev.wait(lambda: all(ev.saved(n) for n in names))
        return (running, active, {n: ev.text(n) for n in names},
                {n: [s for _, s in ev.room(n).transitions] for n in names}, ev.manager.active)

    running, active, texts, transitions, active_after = run_event(tmp_path, scenario)
    assert running == {n: "running" for n in names} and active == len(names)
    for n in names:
        assert texts[n] == TEXTS[n][0], n                       # chaque salle a son texte, entier
        assert transitions[n] == ["starting", "running", "stopping", "stopped"], n
    assert active_after == 0
    # un dossier de test par salle, jamais partagé, avec le nom de la salle
    dirs = sorted(p.parent.name for p in (tmp_path / "runs").glob("*/meta.json"))
    assert len(dirs) == len(names) and all(any(n in d for d in dirs) for n in names)


# ---------- TEST 4 : arrêt d'une salle, les autres continuent ----------

def test_stopping_one_session_does_not_affect_the_others(patch_online_asr, tmp_path):
    names = ["salle-a", "salle-b", "salle-c"]

    async def scenario(ev):
        for n in names:
            ev.open(n)
            ev.start(n)
        await ev.feed(names, 1.5)
        ev.stop("salle-b")
        assert ev.state("salle-b") == "stopping"
        await ev.feed(["salle-a", "salle-c"], 0.3)
        assert await ev.wait(lambda: ev.state("salle-b") == "stopped")
        snapshot = ({n: ev.state(n) for n in names}, ev.manager.active, ev.room("salle-b").slot,
                    {n: len(ev.msgs(n, kind="segment")) for n in names})
        b_asr_then = len(ev.msgs("salle-b", kind="asr"))
        await ev.feed(["salle-a", "salle-c"], TOTAL_S - 1.8)    # A et C continuent seules
        ev.stop("salle-a"), ev.stop("salle-c")
        assert await ev.wait(lambda: ev.saved("salle-a") and ev.saved("salle-c"))
        return (snapshot, {n: ev.text(n) for n in names},
                {n: len(ev.msgs(n, kind="segment")) for n in names}, b_asr_then,
                len(ev.msgs("salle-b", kind="asr")))

    (states, active, b_slot, seg_at_stop), texts, seg_end, b_asr_then, b_asr_end = \
        run_event(tmp_path, scenario)
    assert states == {"salle-a": "running", "salle-b": "stopped", "salle-c": "running"}
    assert active == 2 and b_slot is None                      # la place de B est libérée
    assert texts["salle-a"] == TEXTS["salle-a"][0] and texts["salle-c"] == TEXTS["salle-c"][0]
    assert seg_end["salle-a"] > seg_at_stop["salle-a"] and seg_end["salle-c"] > seg_at_stop["salle-c"]
    assert set(texts["salle-b"].split()) <= words_of("salle-b")  # B arrêtée en cours de texte
    assert b_asr_end == b_asr_then                             # B ne travaille plus


# ---------- TEST 5 : redémarrage d'une salle ----------

def test_restarting_a_stopped_session(patch_online_asr, tmp_path):
    async def scenario(ev):
        for n in ("salle-a", "salle-b"):
            ev.open(n)
            ev.start(n)
        await ev.feed(["salle-a", "salle-b"], 1.0)
        ev.stop("salle-b")
        assert await ev.wait(lambda: ev.state("salle-b") == "stopped")
        restart_at = len(ev.msgs("salle-b"))
        ev.start("salle-b")                                    # même régie, même salle
        seq = states_after(ev.room("salle-b"), "stopped")
        await ev.feed(["salle-a", "salle-b"], TOTAL_S)
        ev.stop("salle-a"), ev.stop("salle-b")
        assert await ev.wait(lambda: ev.saved("salle-a") and ev.saved("salle-b") >= 2)
        return seq, ev.text("salle-b", after=restart_at), ev.text("salle-a"), ev.manager.active

    seq, b_text, a_text, active = run_event(tmp_path, scenario)
    assert seq == ["stopped", "starting", "running"]
    assert b_text == TEXTS["salle-b"][0]                       # le second passage est complet
    assert a_text == TEXTS["salle-a"][0] and active == 0


# ---------- TEST 6 : erreur d'une salle, les autres continuent ----------

class FailingSat(FakeSat):
    """SaT qui plante, mais seulement dans le thread de segmentation d'UNE salle."""

    def __init__(self, room: str):
        self.room, self.broken = room, True

    def word_probs(self, words, mode=None, context=None):
        if self.broken and f"-{self.room}:" in threading.current_thread().name:
            raise RuntimeError("SaT en panne")
        return super().word_probs(words, mode, context)


class FailingEngine(FakeEngine):
    """Whisper qui échoue à chaque passe, mais seulement pour UNE salle."""

    def __init__(self, timeline, room: str):
        super().__init__(timeline)
        self.room = room

    def transcribe(self, audio, prompt, language):
        if f"asr-worker-{self.room}:" in threading.current_thread().name:
            raise RuntimeError("CUDA error: device lost")
        return super().transcribe(audio, prompt, language)


async def error_scenario(ev, names=("salle-a", "salle-b", "salle-c")):
    """La salle A tombe en panne pendant que B et C reçoivent tout leur audio."""
    for n in names:
        ev.open(n)
        ev.start(n)

    def snapshot():
        return ({n: ev.state(n) for n in names}, ev.manager.active,
                ev.room("salle-a").slot, ev.room("salle-a").error)

    t, at_error = 0.0, None
    while t < TOTAL_S:
        await ev.feed(names, 0.5)
        t += 0.5
        if at_error is None and ev.state("salle-a") == "error":
            at_error = snapshot()
    assert await ev.wait(lambda: ev.state("salle-a") == "error")
    at_error = at_error or snapshot()
    ev.stop("salle-b"), ev.stop("salle-c")
    assert await ev.wait(lambda: ev.saved("salle-b") and ev.saved("salle-c"))
    fatal = [m for m in ev.msgs("salle-a", kind="error") if m.get("fatal")]
    return (*at_error, fatal, {n: ev.text(n) for n in names}, {n: ev.state(n) for n in names})


def check_error_isolation(result, where: str):
    at_error, active, a_slot, a_error, fatal, texts, final_states = result
    assert at_error == {"salle-a": "error", "salle-b": "running", "salle-c": "running"}
    assert active == 2 and a_slot is None                      # la place de A est libérée
    assert where in a_error
    assert fatal and fatal[0]["where"] == where                # la régie de A est prévenue
    assert texts["salle-b"] == TEXTS["salle-b"][0] and texts["salle-c"] == TEXTS["salle-c"][0]
    assert final_states == {"salle-a": "error", "salle-b": "stopped", "salle-c": "stopped"}


def test_segmentation_error_in_one_session_spares_the_others(patch_online_asr, tmp_path):
    registry = FakeRegistry(TIMELINES)
    registry.sat = sat = FailingSat("salle-a")
    restarted = {}

    async def scenario(ev):
        result = await error_scenario(ev)
        sat.broken = False                                     # la panne est réparée : on relance A
        before = len(ev.msgs("salle-a"))
        ev.start("salle-a")
        restarted["state"] = ev.state("salle-a")
        await ev.feed(["salle-a"], TOTAL_S)
        ev.stop("salle-a")
        assert await ev.wait(lambda: any(m["type"] == "run_saved"
                                         for m in ev.msgs("salle-a")[before:]))
        restarted["text"] = ev.text("salle-a", after=before)
        restarted["final"] = ev.state("salle-a")
        return result

    check_error_isolation(run_event(tmp_path, scenario, registry), "segmentation")
    assert restarted == {"state": "running", "text": TEXTS["salle-a"][0], "final": "stopped"}


def test_whisper_failures_in_one_session_spare_the_others(patch_online_asr, tmp_path):
    registry = FakeRegistry(TIMELINES)
    registry.whisper = FailingEngine(TIMELINES, "salle-a")
    check_error_isolation(run_event(tmp_path, error_scenario, registry), "asr")


# ---------- TEST 7 : chaque participant ne reçoit que sa salle ----------

def test_participants_only_receive_their_own_session(patch_online_asr, tmp_path):
    rooms = {"salle-a": ("A", "B"), "salle-b": ("C", "D")}

    async def scenario(ev):
        for name, viewers in rooms.items():
            ev.open(name, viewers)
            ev.start(name)
        participants = {r["name"]: r["participants"] for r in ev.manager.list_sessions()["rooms"]}
        await ev.feed(list(rooms), TOTAL_S)
        for name in rooms:
            ev.stop(name)
        assert await ev.wait(lambda: all(ev.saved(n) for n in rooms))
        return participants, {(n, w): ev.msgs(n, w) for n, vs in rooms.items() for w in vs}

    participants, received = run_event(tmp_path, scenario)
    assert participants == {"salle-a": ["A", "B"], "salle-b": ["C", "D"]}
    ids = {key: {id(m) for m in msgs} for key, msgs in received.items()}
    assert ids[("salle-a", "A")] == ids[("salle-a", "B")]          # même salle : mêmes messages
    assert ids[("salle-b", "C")] == ids[("salle-b", "D")]
    assert not ids[("salle-a", "A")] & ids[("salle-b", "C")]       # aucun message en commun
    for (name, who), msgs in received.items():
        own = words_of(name)
        segs = [m["text"] for m in msgs if m["type"] == "segment"]
        assert " ".join(segs) == TEXTS[name][0], (name, who)
        finals = [m["text"] for m in msgs if m["type"] == "translation" and m.get("final")]
        assert finals and all(set(t.split()[1:]) <= own for t in finals), (name, who)
        sat_words = {w for m in msgs if m["type"] == "sat" for w in m["words"]}
        assert sat_words <= own, (name, who)                    # résultats SaT de sa salle seulement


# ---------- TEST 8 : l'audio d'une salle n'est traité que par sa salle ----------

def test_audio_of_one_session_never_reaches_another(patch_online_asr, tmp_path):
    names = ["salle-a", "salle-b", "salle-c"]

    async def scenario(ev):
        for n in names:
            ev.open(n)
            ev.start(n)
        await ev.feed(["salle-a"], TOTAL_S)                    # audio pour A seulement
        received = {n: sum(p.asr.received_s for p in ev.sessions[n].pipelines.values())
                    for n in names}
        queued = {n: sum(p.audio_q.qsize() for p in ev.sessions[n].pipelines.values())
                  for n in names}
        for n in names:
            ev.stop(n)
        assert await ev.wait(lambda: all(ev.saved(n) for n in names))
        heard = {n: [w for m in ev.msgs(n, kind="asr") for w in m.get("committed", [])]
                 for n in names}
        return received, queued, heard, {n: ev.text(n) for n in names}

    received, queued, heard, texts = run_event(tmp_path, scenario)
    assert received["salle-a"] > 0
    assert received["salle-b"] == received["salle-c"] == 0
    assert queued["salle-b"] == queued["salle-c"] == 0
    assert heard["salle-b"] == heard["salle-c"] == []
    assert texts["salle-a"] == TEXTS["salle-a"][0]
    assert texts["salle-b"] == texts["salle-c"] == ""


# ---------- journaux, suppression, filet de sécurité ----------

def test_rooms_started_in_the_same_second_never_share_a_log_folder(tmp_path):
    from app.run_logger import RunLogger
    loggers = [RunLogger(str(tmp_path), "meme-nom", {}, room=r)
               for r in ("salle-a", "salle-b", "salle-a", "salle-a")]
    dirs = [lg.dir for lg in loggers]
    for lg in loggers:
        lg.finalize(False)
    assert len(set(dirs)) == 4
    assert all("_salle-" in d.name and "meme-nom" in d.name for d in dirs)
    assert all((d / "events.jsonl").exists() and (d / "meta.json").exists() for d in dirs)


def test_room_is_deleted_only_when_stopped_and_empty(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.logs.enabled = False
    registry = FakeRegistry([])

    async def main():
        manager = RoomManager(cfg, registry)
        room = manager.get("salle-x")
        q = asyncio.Queue()
        manager.subscribe(room, q, "viewer", "public-1")
        with_viewer = manager.delete_room("salle-x")
        manager.unsubscribe(room, q)
        token = object()
        manager.claim_producer(room, token)
        session = Session(registry, cfg, asyncio.get_running_loop(), Fanout(room), room="salle-x")
        manager.start_session(room, session, {"settings": {"source": "en", "targets": ["fr"]}})
        while_running = manager.delete_room("salle-x")
        session.close()
        with_producer = manager.delete_room("salle-x")
        manager.release_producer(room, token)
        accepted = manager.delete_session("salle-x")
        left = [r["name"] for r in manager.list_sessions()["rooms"]]
        return with_viewer, while_running, with_producer, accepted, left, manager.delete_room("x")

    with_viewer, while_running, with_producer, accepted, left, unknown = asyncio.run(main())
    assert not with_viewer[0] and not while_running[0] and not with_producer[0] and not unknown[0]
    assert "en cours" in while_running[1] and "connexions" in with_producer[1]
    assert accepted == (True, "") and left == []


def test_dead_threads_never_keep_a_place(tmp_path):
    """Filet de sécurité : threads tous morts sans prévenir → salle en erreur, place libérée."""
    cfg = cfg_for(tmp_path)
    cfg.logs.enabled = False

    async def main():
        registry = FakeRegistry([])
        manager = RoomManager(cfg, registry)
        room = manager.get("salle-z")
        session = Session(registry, cfg, asyncio.get_running_loop(), Fanout(room), room="salle-z")
        manager.start_session(room, session, {"settings": {"source": "en", "targets": ["fr"]}})
        for p in session.pipelines.values():                   # mort silencieuse (simulée)
            p.close()
        session.translator.alive = False
        for t in session.workers:
            t.join(2)
        overview = manager.overview()
        session.close()
        return overview

    overview = asyncio.run(main())
    room = next(r for r in overview["rooms"] if r["name"] == "salle-z")
    assert room["state"] == "error" and room["slot"] is None and overview["active"] == 0
