"""Plusieurs speakers dans une salle, chacun sur son appareil (comme Chamk).

- au plus event.max_speakers (2) par salle ; le suivant est refusé avec un message clair ;
- le 1er speaker démarre la session (une place) ; le 2e ajoute son micro en cours de session ;
- chaque phrase porte son orateur (S1, S2) et son texte n'est jamais mélangé avec l'autre ;
- quand un speaker s'arrête, l'autre continue ; la session finit quand le dernier s'arrête ;
- une salle pilotée par une régie refuse les speakers, et inversement.
"""
import asyncio
import json
import time
from collections import defaultdict

import numpy as np
import pytest

from app.config import load_config
from app.rooms import Fanout, RoomManager
from app.session import Session
from tests.conftest import FakeRegistry, make_timeline

CHUNK = (np.ones(1600) * 3000).astype(np.int16).tobytes()      # 100 ms de « parole »
ALICE = "Guten Morgen alle zusammen. Die Konferenz beginnt jetzt."
BOB = "Hallo ich bin Bob. Ich spreche als zweiter Redner."


def cfg_for(tmp_path, max_sessions=3, max_speakers=2):
    cfg = load_config()
    cfg.asr.min_chunk_s = 0.5
    cfg.logs.runs_dir = str(tmp_path / "runs")
    cfg.event.max_sessions = max_sessions
    cfg.event.max_speakers = max_speakers
    return cfg


def remote_start(name: str, lang: str = "de") -> dict:
    return {"settings": {"source": lang, "targets": ["en"], "voice": "none"}, "label": "duo",
            "audio_sources": {"profile": "panel", "framing": "remote",
                              "sources": [{"id": "p1", "name": name, "language": lang}]}}


async def feed(session, source, seconds):
    t = 0.0
    while t < seconds:
        session.push_audio(CHUNK, source=source)
        t += 0.1
        await asyncio.sleep(0.004)


async def collect(q, out, until, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        while not q.empty():
            out.append(q.get_nowait())
        if until():
            return True
        await asyncio.sleep(0.05)
    return until()


def test_two_remote_speakers_share_one_session(patch_online_asr, tmp_path):
    """A démarre, B rejoint en cours ; chaque phrase est attribuée au bon orateur."""
    cfg = cfg_for(tmp_path)
    tl_a = make_timeline([(ALICE, "de")])
    tl_b = make_timeline([(BOB, "de")])
    registry = FakeRegistry({"duo:p1": tl_a, "duo:p2": tl_b})

    async def main():
        loop = asyncio.get_running_loop()
        manager = RoomManager(cfg, registry)
        room = manager.get("duo")
        q: asyncio.Queue = asyncio.Queue()
        manager.subscribe(room, q, "viewer", "public")
        session = Session(registry, cfg, loop, Fanout(room), room="duo")
        assert manager.start_session(room, session, remote_start("Alice")) is None
        src_a, spk_a = session.first_source()
        await feed(session, src_a, 2.0)                       # Alice parle seule un moment
        src_b, spk_b = session.add_source("Bob", "de")        # Bob rejoint en cours de session
        state_with_two = room.state
        await asyncio.gather(feed(session, src_a, tl_a[-1][1] - 0.5),
                             feed(session, src_b, tl_b[-1][1] + 1.5))
        out: list = []
        session.remove_source(src_a)                          # Alice s'arrête…
        await collect(q, out, lambda: False, timeout=1.5)
        done_after_alice = any(m["type"] == "done" for m in out)
        running_after_alice = room.running
        session.remove_source(src_b)                          # …puis Bob
        ok = await collect(q, out, lambda: any(m["type"] == "run_saved" for m in out))
        session.close()
        return (ok, spk_a, spk_b, src_a, src_b, state_with_two, done_after_alice,
                running_after_alice, out, room.state, room.running)

    (ok, spk_a, spk_b, src_a, src_b, state_two, done_early, running_after_alice,
     out, final_state, final_running) = asyncio.run(main())
    assert ok
    assert (spk_a["id"], spk_a["name"], src_a) == ("S1", "Alice", "p1")
    assert (spk_b["id"], spk_b["name"], src_b) == ("S2", "Bob", "p2")
    assert state_two == "running"
    # Alice partie, Bob parle encore : la session ne se termine pas
    assert not done_early and running_after_alice
    segs = [m for m in out if m["type"] == "segment"]
    by_speaker = defaultdict(list)
    for s in segs:
        by_speaker[s["speaker"]].append(s["text"])
    assert " ".join(by_speaker["S1"]) == ALICE                # aucun mot de Bob chez Alice
    assert " ".join(by_speaker["S2"]) == BOB                  # et inversement
    assert {s["source"] for s in segs if s["speaker"] == "S2"} == {"p2"}
    finals = [m for m in out if m["type"] == "translation" and m.get("final")]
    assert {m["id"] for m in finals} == {s["id"] for s in segs}   # chaque phrase traduite
    speakers_msgs = [m for m in out if m["type"] == "speakers"]
    assert [s["name"] for s in speakers_msgs[-1]["speakers"]] == ["Alice", "Bob"]
    assert sum(m["type"] == "done" for m in out) == 1
    assert final_state == "stopped" and not final_running    # place libérée


def test_add_source_is_refused_without_a_remote_session(tmp_path):
    cfg = cfg_for(tmp_path)
    registry = FakeRegistry(make_timeline([(ALICE, "de")]))

    async def main():
        session = Session(registry, cfg, asyncio.get_running_loop(), asyncio.Queue(), room="x")
        with pytest.raises(RuntimeError):
            session.add_source("Bob", "de")                   # aucune session démarrée

    asyncio.run(main())


def test_speaker_limit_and_regie_exclusion(tmp_path):
    cfg = cfg_for(tmp_path, max_speakers=2)
    manager = RoomManager(cfg, FakeRegistry([]))
    room = manager.get("salle-a")
    a, b, c = object(), object(), object()
    assert manager.claim_speaker(room, a, "Alice") is None
    assert manager.claim_speaker(room, b, "Bob") is None
    refused = manager.claim_speaker(room, c, "Carol")
    assert refused and "Limite atteinte : 2 speakers" in refused and "Alice, Bob" in refused
    assert not manager.claim_producer(room, object())         # pas de régie dans une salle de speakers
    room_ov = next(r for r in manager.overview()["rooms"] if r["name"] == "salle-a")
    assert room_ov["speakers"] == ["Alice", "Bob"]
    manager.release_speaker(room, b)
    assert manager.claim_speaker(room, c, "Carol") is None    # une place s'est libérée
    other = manager.get("salle-b")
    assert manager.claim_producer(other, object())
    assert "pilotée par une régie" in manager.claim_speaker(other, object(), "Dan")


# ---------- protocole WebSocket ----------

try:
    from fastapi.testclient import TestClient
except Exception:   # dépend de httpx
    TestClient = None


@pytest.fixture
def speaker_client(monkeypatch, tmp_path):
    if TestClient is None:
        pytest.skip("installer httpx pour ce test")
    import app.main as main
    fake = FakeRegistry({"duo:p1": make_timeline([(ALICE, "de")]),
                         "duo:p2": make_timeline([(BOB, "de")])})
    fake.load_safely = lambda: None
    monkeypatch.setattr(main, "registry", fake)
    main.cfg.logs.runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(main.cfg.event, "max_sessions", 3)
    monkeypatch.setattr(main.cfg.event, "max_speakers", 2)
    monkeypatch.setattr(main, "rooms", RoomManager(main.cfg, fake))
    with TestClient(main.app) as c:
        yield c, main


def receive_until(ws, kind, limit=60):
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if msg["type"] == kind:
            return msg
    raise AssertionError(f"pas de message « {kind} »")


def test_speakers_join_get_the_floor_and_third_is_refused(speaker_client):
    client, _ = speaker_client
    join = json.dumps({"type": "join", "settings": {"source": "de", "targets": ["en"], "voice": "none"}})
    with client.websocket_connect("/ws?room=Duo&role=speaker&name=Alice&lang=de") as a:
        hello = json.loads(a.receive_text())
        assert hello["role"] == "speaker" and hello["room"] == "duo" and hello["max_speakers"] == 2
        a.send_text(join)
        floor_a = receive_until(a, "floor")
        assert floor_a["granted"] and floor_a["speaker"] == "S1" and floor_a["name"] == "Alice"
        with client.websocket_connect("/ws?room=duo&role=speaker&name=Bob&lang=de") as b:
            json.loads(b.receive_text())
            b.send_text(join)
            floor_b = receive_until(b, "floor")
            assert floor_b["granted"] and floor_b["speaker"] == "S2" and floor_b["source"] == "p2"
            with client.websocket_connect("/ws?room=duo&role=speaker&name=Carol") as c:
                err = json.loads(c.receive_text())
                assert err["type"] == "error" and err["where"] == "speaker_limit"
                assert "Limite atteinte" in err["message"]
            with client.websocket_connect("/ws?room=duo&role=producer") as regie:
                err = json.loads(regie.receive_text())
                assert err["type"] == "error" and err["where"] == "room"
            room = next(r for r in client.get("/api/rooms").json()["rooms"] if r["name"] == "duo")
            assert room["speakers"] == ["Alice", "Bob"] and room["running"] and room["state"] == "running"
            b.send_text(json.dumps({"type": "stop"}))
            assert receive_until(b, "floor")["granted"] is False
        a.send_text(json.dumps({"type": "stop"}))
        receive_until(a, "floor")
    for _ in range(100):                                       # la place finit par être libérée
        room = next(r for r in client.get("/api/rooms").json()["rooms"] if r["name"] == "duo")
        if not room["running"]:
            break
        time.sleep(0.05)
    assert not room["running"] and room["speakers"] == []


def test_speaker_cannot_join_a_room_with_a_regie(speaker_client):
    client, _ = speaker_client
    with client.websocket_connect("/ws?room=amphi&role=producer") as regie:
        json.loads(regie.receive_text())
        with client.websocket_connect("/ws?room=amphi&role=speaker&name=Alice") as s:
            err = json.loads(s.receive_text())
            assert err["type"] == "error" and err["where"] == "room"


def test_speaker_page_exists(speaker_client):
    client, _ = speaker_client
    assert client.get("/speaker").status_code == 200
