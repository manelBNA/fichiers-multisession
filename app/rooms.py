"""Salles d'un événement : plusieurs conférences traduites en même temps.

- Une **salle** = une conférence. Elle a au plus **une régie** (la page qui capte
  l'audio et pilote la session) et autant de **spectateurs** que l'on veut (pages
  qui affichent seulement les sous-titres, par exemple sur les téléphones du public).
- Chaque salle qui travaille occupe une **place** ; il y a `event.max_sessions`
  places (3 par défaut). Une 4e salle qui démarre est refusée avec un message clair.
- Chaque place a ses propres traducteurs Marian (si `marian_instances: per_session`) ;
  Whisper et SaT sont partagés mais appelés en parallèle.
- Tous les messages d'une session sont copiés à la régie et aux spectateurs de sa
  salle, et seulement à eux. Les derniers messages utiles sont gardés pour qu'un
  spectateur qui arrive en cours voie les phrases déjà traduites.
- Chaque salle a un état : created → starting → running → stopping → stopped,
  ou error si un thread de sa session meurt (sa place est alors libérée et la
  régie peut redémarrer). L'état d'une salle ne dépend jamais des autres.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque

HISTORY_TYPES = {"segment", "unit", "speakers", "lang"}
_NAME = re.compile(r"[^a-z0-9_-]+")
log = logging.getLogger("rooms")

# États d'une salle
CREATED, STARTING, RUNNING, STOPPING, STOPPED, ERROR = (
    "created", "starting", "running", "stopping", "stopped", "error")
IDLE_STATES = (CREATED, STOPPED, ERROR)


def clean_room_name(name: str | None, default: str) -> str:
    cleaned = _NAME.sub("-", (name or "").strip().lower()).strip("-")[:30]
    return cleaned or default


class Fanout:
    """Remplace la file d'une session : chaque message part vers toute la salle."""

    def __init__(self, room: "Room"):
        self.room = room

    def put_nowait(self, msg: dict) -> None:
        self.room.broadcast(msg)


class Room:
    def __init__(self, name: str, history: int):
        self.name = name
        self.subscribers: dict = {}            # file asyncio -> rôle (« producer » | « viewer »)
        self.participants: dict = {}           # file asyncio -> identifiant de participant
        self.history: deque = deque(maxlen=history)
        self.state = CREATED
        self.error: str | None = None
        self.transitions: deque = deque(maxlen=50)   # (heure, état) : suivi et tests
        self.producer_token: object | None = None
        self.speakers: dict = {}               # jeton de connexion -> {"name": …} (speakers distants)
        self.session = None
        self.slot: int | None = None
        self.started_at: float | None = None
        self.settings: dict | None = None
        self.segments = 0
        self.last_activity: float | None = None

    # Appelé dans la boucle asyncio (via call_soon_threadsafe de la session)
    def broadcast(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "status" and msg.get("state") == "running":
            self.history.clear()              # nouvelle session : on repart de zéro
            self.settings = msg.get("settings")
            self.segments = 0
        if kind == "segment":
            self.segments += 1
            self.last_activity = time.time()
        if kind in HISTORY_TYPES or (kind == "translation" and msg.get("final")):
            self.history.append(msg)
        for q in list(self.subscribers):
            q.put_nowait(msg)

    def set_state(self, state: str, error: str | None = None) -> None:
        """Appelé sous le verrou du RoomManager."""
        if state != self.state:
            log.info("Salle « %s » : %s → %s%s", self.name, self.state, state,
                     f" ({error})" if error else "")
        self.state = state
        self.error = error if state == ERROR else None
        self.transitions.append((time.time(), state))

    @property
    def viewers(self) -> int:
        return sum(1 for role in self.subscribers.values() if role == "viewer")

    @property
    def running(self) -> bool:
        return self.slot is not None


class RoomManager:
    def __init__(self, cfg, registry):
        self.cfg = cfg
        self.registry = registry
        self.rooms: dict[str, Room] = {}
        self._lock = threading.Lock()
        self._free_slots = list(range(max(1, cfg.event.max_sessions)))

    # ---------- salles ----------
    def get(self, name: str | None) -> Room:
        name = clean_room_name(name, self.cfg.event.default_room)
        with self._lock:
            room = self.rooms.get(name)
            if room is None:
                room = self.rooms[name] = Room(name, self.cfg.event.viewer_history)
            return room

    def subscribe(self, room: Room, queue, role: str, participant: str | None = None) -> None:
        """Une connexion n'est abonnée qu'à UNE salle (une file par connexion)."""
        with self._lock:
            room.subscribers[queue] = role
            if participant:
                room.participants[queue] = participant

    def unsubscribe(self, room: Room, queue) -> None:
        with self._lock:
            room.subscribers.pop(queue, None)
            room.participants.pop(queue, None)

    def delete_room(self, name: str) -> tuple[bool, str]:
        """Supprime une salle seulement si elle est arrêtée, sans régie et sans spectateur."""
        name = clean_room_name(name, "")
        with self._lock:
            room = self.rooms.get(name)
            if room is None:
                return False, f"Salle « {name} » inconnue."
            if room.slot is not None or room.state not in IDLE_STATES:
                return False, f"La salle « {name} » est en cours ({room.state}) : arrête-la d'abord."
            if room.producer_token is not None or room.speakers or room.subscribers:
                return False, f"La salle « {name} » a encore des connexions (régie, speakers ou écrans)."
            del self.rooms[name]
        log.info("Salle « %s » supprimée", name)
        return True, ""

    delete_session = delete_room           # vocabulaire « session »

    def claim_producer(self, room: Room, token: object) -> bool:
        """Une seule régie par salle (sinon deux sources audio se mélangeraient)."""
        with self._lock:
            if room.producer_token is not None or room.speakers:
                return False
            room.producer_token = token
            return True

    def claim_speaker(self, room: Room, token: object, name: str) -> str | None:
        """Place de speaker (chacun sur son appareil). Renvoie un message d'erreur si refusé."""
        limit = max(1, self.cfg.event.max_speakers)
        with self._lock:
            if room.producer_token is not None:
                return (f"La salle « {room.name} » est pilotée par une régie : "
                        "les speakers ne peuvent pas s'y connecter.")
            if len(room.speakers) >= limit:
                names = ", ".join(s["name"] for s in room.speakers.values())
                return (f"Limite atteinte : {limit} speakers déjà connectés dans la salle "
                        f"« {room.name} » ({names}). Rejoins-la comme participant (/viewer).")
            room.speakers[token] = {"name": name}
            return None

    def release_speaker(self, room: Room, token: object) -> None:
        with self._lock:
            room.speakers.pop(token, None)

    def release_producer(self, room: Room, token: object) -> None:
        with self._lock:
            if room.producer_token is token:
                room.producer_token = None
                room.session = None

    # ---------- places de session ----------
    @property
    def active(self) -> int:
        return self.cfg.event.max_sessions - len(self._free_slots)

    def _acquire_slot(self, room: Room) -> int | None:
        with self._lock:
            if room.slot is not None:
                return room.slot
            if not self._free_slots:
                return None
            room.slot = self._free_slots.pop(0)
            room.started_at = time.time()
            return room.slot

    def _release_slot_unlocked(self, room: Room) -> None:
        if room.slot is not None:
            self._free_slots.append(room.slot)
            self._free_slots.sort()
            room.slot = None
            room.started_at = None

    def release_slot(self, room: Room) -> None:
        with self._lock:
            self._release_slot_unlocked(room)
            if room.state not in IDLE_STATES:
                room.set_state(STOPPED)

    def _session_finished(self, room: Room, session) -> None:
        """La session ne travaille plus (fin normale, régie fermée ou erreur fatale)."""
        with self._lock:
            self._release_slot_unlocked(room)
            if session.failed:
                room.set_state(ERROR, session.failed)
            elif room.state != ERROR:
                room.set_state(STOPPED)

    def start_session(self, room: Room, session, data: dict) -> str | None:
        """Démarre la session de la salle ; renvoie un message d'erreur si c'est impossible.

        Peut tourner hors de la boucle asyncio (asyncio.to_thread) : session.start()
        écrit sur le disque (dossier du test).
        """
        session.on_finished = None              # un redémarrage garde la place de la salle
        slot = self._acquire_slot(room)
        if slot is None:
            with self._lock:
                busy = ", ".join(r.name for r in self.rooms.values() if r.running)
            return (f"Capacité atteinte : {self.cfg.event.max_sessions} sessions déjà en cours "
                    f"({busy}). Arrête une salle ou attends qu'elle se termine.")
        with self._lock:
            room.set_state(STARTING)
        pool_for = getattr(self.registry, "pool_for", None)
        session.pool = pool_for(slot) if pool_for else None
        if hasattr(session.pool, "clear_failures"):
            session.pool.clear_failures()       # relancer la salle = réessayer les paires en échec
        room.session = session
        try:
            session.start(data.get("settings"), str(data.get("label", ""))[:60],
                          data.get("audio_sources"), bool(data.get("analyze", False)))
        except Exception as exc:
            log.exception("Démarrage impossible dans la salle « %s »", room.name)
            with self._lock:
                self._release_slot_unlocked(room)
                room.set_state(ERROR, f"{type(exc).__name__}: {exc}")
            raise
        session.on_finished = lambda s, r=room: self._session_finished(r, s)
        with self._lock:
            if room.state == STARTING:          # sinon : erreur fatale dès le démarrage
                room.set_state(RUNNING)
        return None

    def stop_session(self, room: Room) -> None:
        """Arrêt propre : la fin de l'audio est transcrite et traduite, puis la place est
        libérée (état « stopping » jusque-là). Les autres salles ne sont pas touchées."""
        session = room.session
        with self._lock:
            if room.state in (STARTING, RUNNING):
                room.set_state(STOPPING)
        if session is not None:
            session.stop()

    def reap(self) -> None:
        """Filet de sécurité : une salle dont tous les threads sont morts sans avoir
        prévenu ne doit pas garder sa place indéfiniment."""
        with self._lock:
            rooms = [r for r in self.rooms.values()
                     if r.slot is not None and r.state in (RUNNING, STOPPING)]
        for room in rooms:
            session = room.session
            workers = list(getattr(session, "workers", []) or []) if session else []
            started = [t for t in workers if getattr(t, "ident", None) is not None]
            if session is None or (started and len(started) == len(workers)
                                   and not any(t.is_alive() for t in started)):
                log.error("Salle « %s » : plus aucun thread actif, place libérée", room.name)
                if session is not None and not session.failed:
                    session.failed = "threads arrêtés sans fin de session"
                with self._lock:
                    self._release_slot_unlocked(room)
                    room.set_state(ERROR, "threads arrêtés sans fin de session")

    # ---------- tableau de bord ----------
    @staticmethod
    def _translation_errors(room: Room) -> list[str]:
        """Paires Marian en échec de chargement dans le jeu de traducteurs de la salle."""
        pool = getattr(room.session, "pool", None) if room.running else None
        failures = getattr(pool, "failure_messages", None)
        return failures() if failures else []

    def overview(self) -> dict:
        self.reap()
        with self._lock:
            rooms = list(self.rooms.values())
            participants = {r.name: sorted(set(r.participants.values())) for r in rooms}
            speakers = {r.name: [s["name"] for s in r.speakers.values()] for r in rooms}
        now = time.time()
        return {
            "max_sessions": self.cfg.event.max_sessions,
            "max_speakers": self.cfg.event.max_speakers,
            "active": self.active,
            "rooms": [{
                "name": r.name,
                "session_id": r.name,
                "state": r.state,
                "error": r.error,
                "running": r.running,
                "slot": r.slot,
                "producer": r.producer_token is not None,
                "speakers": speakers.get(r.name, []),
                "viewers": r.viewers,
                "participants": participants.get(r.name, []),
                "settings": r.settings,
                "segments": r.segments,
                "running_s": round(now - r.started_at) if r.started_at else None,
                "idle_s": round(now - r.last_activity) if r.last_activity else None,
                "translation_errors": self._translation_errors(r),
            } for r in sorted(rooms, key=lambda r: r.name)],
        }

    list_rooms = list_sessions = overview  # vocabulaire « session »
