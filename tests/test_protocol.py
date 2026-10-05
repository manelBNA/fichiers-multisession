"""Protocole WebSocket : l'interface existante doit continuer de fonctionner."""
import json

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # dépend de httpx, absent de certaines installations
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None, reason="installer httpx pour ce test")


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.main as main
    from tests.conftest import FakeRegistry, make_timeline

    fake = FakeRegistry(make_timeline([("Hello there my friends.", "en")]))
    monkeypatch.setattr(main, "registry", fake)
    fake.load_safely = lambda: None
    main.cfg.logs.runs_dir = str(tmp_path / "runs")
    with TestClient(main.app) as c:
        yield c


def test_page_and_info(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/info").json()["ready"] is True


def test_hello_then_commands(client):
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and "policy" in hello and "settings" in hello
        ws.send_text(json.dumps({"type": "policy", "values": {"th_high": 0.6}}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "policy" and msg["policy"]["th_high"] == 0.6
        ws.send_text(json.dumps({"type": "commande_inconnue"}))
        ws.send_text(json.dumps({"type": "settings", "values": {"targets": ["fr"]}}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "settings" and msg["settings"]["targets"] == ["fr"]


# ---------- plusieurs salles ----------

@pytest.fixture
def event_client(monkeypatch, tmp_path):
    import app.main as main
    from app.rooms import RoomManager
    from tests.conftest import FakeRegistry, make_timeline

    fake = FakeRegistry(make_timeline([("Hello there my friends.", "en")]))
    fake.load_safely = lambda: None
    monkeypatch.setattr(main, "registry", fake)
    main.cfg.logs.runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(main.cfg.event, "max_sessions", 1)
    monkeypatch.setattr(main, "rooms", RoomManager(main.cfg, fake))
    with TestClient(main.app) as c:
        yield c, main


def test_producer_and_viewer_share_a_room(event_client):
    client, main = event_client
    with client.websocket_connect("/ws?room=Amphi%20A&role=producer") as prod:
        hello = json.loads(prod.receive_text())
        assert hello["room"] == "amphi-a" and hello["role"] == "producer"
        with client.websocket_connect("/ws?room=amphi-a&role=viewer") as viewer:
            vh = json.loads(viewer.receive_text())
            assert vh["role"] == "viewer" and vh["room"] == "amphi-a" and vh["history"] == []
            prod.send_text(json.dumps({"type": "start", "settings": {"source": "en", "targets": ["fr"]}}))
            seen = [json.loads(viewer.receive_text())["type"] for _ in range(3)]
            assert "status" in seen                      # le spectateur suit la session
            overview = client.get("/api/rooms").json()
            room = next(r for r in overview["rooms"] if r["name"] == "amphi-a")
            assert room["running"] and room["producer"] and room["viewers"] == 1
            prod.send_text(json.dumps({"type": "stop"}))


def test_second_producer_is_refused(event_client):
    client, _ = event_client
    with client.websocket_connect("/ws?room=salle-b") as first:
        json.loads(first.receive_text())
        with client.websocket_connect("/ws?room=salle-b") as second:
            msg = json.loads(second.receive_text())
            assert msg["type"] == "error" and msg["where"] == "room"


def test_capacity_error_when_all_places_are_taken(event_client):
    client, _ = event_client                               # max_sessions = 1 dans ce test
    with client.websocket_connect("/ws?room=salle-1") as a, \
         client.websocket_connect("/ws?room=salle-2") as b:
        json.loads(a.receive_text()); json.loads(b.receive_text())
        a.send_text(json.dumps({"type": "start", "settings": {"source": "en", "targets": ["fr"]}}))
        b.send_text(json.dumps({"type": "start", "settings": {"source": "en", "targets": ["fr"]}}))
        for _ in range(20):
            msg = json.loads(b.receive_text())
            if msg["type"] == "error":
                break
        assert msg["where"] == "capacity" and "Capacité atteinte" in msg["message"]


def test_session_alias_participant_and_state(event_client):
    """/ws/session/<salle> = /ws?room=<salle> ; participant_id et état dans l'aperçu."""
    client, _ = event_client
    with client.websocket_connect("/ws/session/Salle%20B?role=producer&participant_id=regie-b") as prod:
        hello = json.loads(prod.receive_text())
        assert hello["room"] == hello["session_id"] == "salle-b"
        assert hello["participant_id"] == "regie-b" and hello["state"] == "created"
        with client.websocket_connect("/ws?room=salle-b&role=viewer&participant_id=tel 42") as viewer:
            vh = json.loads(viewer.receive_text())
            assert vh["session_id"] == "salle-b" and vh["participant_id"] == "tel-42"
            prod.send_text(json.dumps({"type": "start", "settings": {"source": "en", "targets": ["fr"]}}))
            while json.loads(viewer.receive_text()).get("state") != "running":
                pass
            room = next(r for r in client.get("/api/rooms").json()["rooms"] if r["name"] == "salle-b")
            assert room["state"] == "running" and room["participants"] == ["regie-b", "tel-42"]
            assert client.delete("/api/rooms/salle-b").status_code == 409   # en cours : refusé
            prod.send_text(json.dumps({"type": "stop"}))
            for _ in range(50):
                room = next(r for r in client.get("/api/rooms").json()["rooms"] if r["name"] == "salle-b")
                if room["state"] == "stopped":
                    break
                viewer.receive_text()
            assert room["state"] == "stopped" and not room["running"]
    names = [r["name"] for r in client.get("/api/rooms").json()["rooms"]]
    assert "salle-b" in names
    res = client.delete("/api/rooms/salle-b")                 # arrêtée et vide : supprimée
    assert res.status_code == 200 and "salle-b" not in [r["name"] for r in res.json()["rooms"]]


def test_pages_exist(event_client):
    client, _ = event_client
    assert client.get("/viewer").status_code == 200
    assert client.get("/event").status_code == 200
