"""Serveur FastAPI.

  GET  /          page de test (static/index.html)
  GET  /api/info  modèles chargés et appareils utilisés
  WS   /ws        audio PCM16 16 kHz (binaire) + commandes JSON (texte)
  WS   /ws/session/{salle}   même chose (alias), la salle dans le chemin

Lancer :  python -m app.main
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT, load_config
from app.registry import ModelRegistry
from app.session import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-9s %(message)s")
# Les requêtes HTTP de Hugging Face noient le journal : on ne garde que les avertissements
for noisy in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("server")

cfg = load_config()
registry = ModelRegistry(cfg)
analysis_worker = None   # créé au démarrage : analyse automatique des tests (campagne)
from app.rooms import Fanout, RoomManager  # noqa: E402
rooms = RoomManager(cfg, registry)          # salles de l'événement (plusieurs sessions)


def check_project_files() -> None:
    """Détecte un projet où anciens et nouveaux fichiers sont mélangés."""
    import inspect

    from app.segmentation.boundary_policy import BoundaryPolicy
    from app.workers.asr_worker import ASRWorker
    from app.workers.translator_worker import TranslatorWorker

    checks = {
        "app/workers/translator_worker.py": "settings" in inspect.signature(TranslatorWorker).parameters,
        "app/workers/asr_worker.py": "settings" in inspect.signature(ASRWorker).parameters,
        "app/segmentation/boundary_policy.py": "language" not in inspect.signature(BoundaryPolicy).parameters,
        "app/run_logger.py": (ROOT / "app" / "run_logger.py").exists(),
        "app/sources.py": (ROOT / "app" / "sources.py").exists(),
        "app/workers/source_pipeline.py": (ROOT / "app" / "workers" / "source_pipeline.py").exists(),
        "app/coordinator.py (étape 2)": (ROOT / "app" / "coordinator.py").exists(),
        "app/meaning_units.py (unités de sens)": (ROOT / "app" / "meaning_units.py").exists(),
        "app/analysis.py (campagne)": (ROOT / "app" / "analysis.py").exists(),
        "app/test_journal.py (journal unique)": (ROOT / "app" / "test_journal.py").exists(),
        "app/rooms.py (plusieurs salles)": (ROOT / "app" / "rooms.py").exists(),
        "app/audio_framing.py (étape 2)": (ROOT / "app" / "audio_framing.py").exists(),
        "app/workers/translator_worker.py (orateur)":
            "speaker" in inspect.signature(TranslatorWorker.submit_final).parameters,
    }
    old = [path for path, ok in checks.items() if not ok]
    if old:
        log.error("ANCIENNE VERSION détectée dans : %s. Remplace tout le projet par la "
                  "dernière archive complète (streaming-caption.zip).", ", ".join(old))


check_project_files()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Chargement en arrière-plan : l'interface s'ouvre tout de suite et affiche la progression
    log.info("Interface disponible sur http://localhost:%d", cfg.server.port)
    log.info("Chargement des modèles en arrière-plan (premier lancement : téléchargement, "
             "plusieurs minutes)...")
    task = asyncio.create_task(asyncio.to_thread(registry.load_safely))
    global analysis_worker
    from app.analysis import AnalysisWorker
    analysis_worker = AnalysisWorker(registry, cfg)
    analysis_worker.start()
    yield
    task.cancel()


app = FastAPI(title="Sous-titrage et traduction en direct EN/DE/FR", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/viewer")
async def viewer_page():
    """Écran public d'une salle : sous-titres seulement (téléphones, vidéoprojecteur)."""
    return FileResponse(ROOT / "static" / "viewer.html")


@app.get("/event")
async def event_page():
    """Tableau de bord de l'événement : toutes les salles et leur état."""
    return FileResponse(ROOT / "static" / "event.html")


@app.get("/api/info")
async def info():
    return JSONResponse(registry.info())


@app.get("/api/rooms")
async def rooms_overview():
    return JSONResponse(rooms.overview())


@app.get("/api/journal")
async def journal():
    """Le journal unique de tous les tests (une ligne JSON complète par test)."""
    from app.test_journal import JOURNAL_NAME
    path = (ROOT / cfg.logs.runs_dir) / JOURNAL_NAME
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return JSONResponse({"path": str(path), "tests": len([l for l in text.splitlines() if l]),
                         "content": text})


@app.get("/api/campaign")
async def campaign():
    """Le fichier de campagne (une ligne par test analysé), à transmettre pour le rapport."""
    path = (ROOT / cfg.logs.runs_dir) / "campagne.jsonl"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return JSONResponse({"path": str(path), "tests": len([l for l in text.splitlines() if l]),
                         "content": text})


async def handle_command(ws: WebSocket, session: Session, data: dict, room=None) -> None:
    kind = data.get("type")
    if kind == "start":
        if not registry.ready:
            await ws.send_text(json.dumps({"type": "error", "where": "server",
                "message": "Les modèles sont encore en cours de chargement."}, ensure_ascii=False))
            return
        if room is None:
            room = rooms.get(None)
        # Hors de la boucle asyncio : le démarrage écrit sur le disque (dossier du test,
        # bilan du test précédent) et ne doit pas figer les autres salles.
        error = await asyncio.to_thread(rooms.start_session, room, session, data)
        if error:
            await ws.send_text(json.dumps({"type": "error", "where": "capacity", "message": error,
                                           "rooms": rooms.overview()}, ensure_ascii=False))
    elif kind == "stop":
        if room is not None:
            rooms.stop_session(room)
        else:
            session.stop()
    elif kind == "policy":
        session.update_policy(data.get("values", {}))
    elif kind == "settings":
        session.update_settings(data.get("values", {}))
    elif kind == "rename_speaker":
        session.rename_speaker(str(data.get("speaker", "")), str(data.get("name", "")))


def clean_participant(value: str | None) -> str | None:
    cleaned = re.sub(r"[^\w\-.@]+", "-", (value or "").strip())[:64].strip("-")
    return cleaned or None


@app.delete("/api/rooms/{name}")
async def delete_room(name: str):
    """Supprime une salle arrêtée, sans régie ni spectateur (sinon : 409)."""
    ok, error = rooms.delete_room(name)
    return JSONResponse({"ok": ok, "error": error or None, **rooms.overview()},
                        status_code=200 if ok else 409)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """/ws?room=<salle>&role=producer|viewer[&participant_id=<id>]

    producer (par défaut) : la régie d'une salle ; elle envoie l'audio et pilote la session.
    viewer : un écran public ; il reçoit seulement les messages de sa salle.
    Sans paramètre : salle par défaut, régie (comportement historique).
    """
    await serve_connection(ws, ws.query_params.get("room"))


@app.websocket("/ws/session/{session_id}")
async def ws_session_endpoint(ws: WebSocket, session_id: str):
    """Alias : /ws/session/<salle>?role=producer|viewer[&participant_id=<id>]."""
    await serve_connection(ws, session_id)


async def serve_connection(ws: WebSocket, room_name: str | None) -> None:
    """Une connexion = une seule salle, choisie à l'ouverture."""
    await ws.accept()
    room = rooms.get(room_name)
    role = "viewer" if ws.query_params.get("role") == "viewer" else "producer"
    participant = clean_participant(ws.query_params.get("participant_id"))
    out_q: asyncio.Queue = asyncio.Queue()

    async def sender():
        while True:
            msg = await out_q.get()
            await ws.send_text(json.dumps(msg, ensure_ascii=False))

    if role == "viewer":
        await serve_viewer(ws, room, out_q, sender, participant)
        return

    token = object()
    if not rooms.claim_producer(room, token):
        await ws.send_text(json.dumps({"type": "error", "where": "room", "message":
            f"Une régie est déjà connectée à la salle « {room.name} ». Choisis une autre salle, "
            "ou ferme l'autre régie.", "room": room.name}, ensure_ascii=False))
        await ws.close()
        return
    loop = asyncio.get_running_loop()
    session = Session(registry, cfg, loop, Fanout(room), analysis=analysis_worker, room=room.name)
    room.session = session
    rooms.subscribe(room, out_q, "producer", participant)
    send_task = asyncio.create_task(sender())
    await ws.send_text(json.dumps({"type": "hello", "info": registry.info(),
                                   "policy": session.policy.as_dict(),
                                   "settings": session.settings.as_dict(),
                                   "room": room.name, "session_id": room.name,
                                   "participant_id": participant, "role": "producer",
                                   "state": room.state,
                                   "event": rooms.overview()}, ensure_ascii=False))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                session.push_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    await handle_command(ws, session, json.loads(msg["text"]), room)
                except Exception as exc:  # une commande ratée ne doit pas couper la connexion
                    log.exception("Commande refusée")
                    await ws.send_text(json.dumps({"type": "error", "where": "server",
                        "message": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    except Exception as exc:  # déconnexion brutale, etc.
        log.info("WebSocket fermé : %s", exc)
    finally:
        rooms.unsubscribe(room, out_q)
        send_task.cancel()
        try:
            # Libère aussi la place de la salle. Le bilan du test est écrit sur le disque :
            # hors de la boucle asyncio, pour ne pas figer les autres salles.
            await asyncio.to_thread(session.close)
        finally:
            rooms.release_producer(room, token)


async def serve_viewer(ws: WebSocket, room, out_q: asyncio.Queue, sender,
                       participant: str | None = None) -> None:
    """Écran public : l'historique récent, puis les messages de la salle en direct."""
    await ws.send_text(json.dumps({"type": "hello", "role": "viewer", "room": room.name,
                                   "session_id": room.name, "participant_id": participant,
                                   "running": room.running, "state": room.state,
                                   "settings": room.settings,
                                   "history": list(room.history),
                                   "event": rooms.overview()}, ensure_ascii=False))
    rooms.subscribe(room, out_q, "viewer", participant)
    send_task = asyncio.create_task(sender())
    try:
        while True:                     # un spectateur n'envoie rien d'utile : on attend la fin
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
    except Exception as exc:
        log.info("Écran public fermé : %s", exc)
    finally:
        rooms.unsubscribe(room, out_q)
        send_task.cancel()


def port_is_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    import sys

    import uvicorn

    if not port_is_free(cfg.server.host, cfg.server.port):
        log.error(
            "Le port %d est déjà utilisé : un autre serveur tourne sans doute déjà "
            "(autre fenêtre de terminal ou processus resté actif).\n"
            "  - Ferme l'autre terminal, ou arrête le processus :\n"
            "      Get-NetTCPConnection -LocalPort %d -State Listen | "
            "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n"
            "  - ou change server.port dans config.yaml (ex. 8001).",
            cfg.server.port, cfg.server.port)
        sys.exit(1)
    ssl = {}
    if cfg.server.ssl_certfile and cfg.server.ssl_keyfile:
        ssl = {"ssl_certfile": str(ROOT / cfg.server.ssl_certfile),
               "ssl_keyfile": str(ROOT / cfg.server.ssl_keyfile)}
        log.info("HTTPS activé : https://<adresse-du-serveur>:%d", cfg.server.port)
    log.info("Régie : /?room=<salle> · Écran public : /viewer?room=<salle> · Salles : /event")
    uvicorn.run("app.main:app", host=cfg.server.host, port=cfg.server.port, log_level="info", **ssl)
