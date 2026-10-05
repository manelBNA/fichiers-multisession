"""Enregistrement automatique de chaque test.

Chaque clic sur « Démarrer l'écoute » crée un dossier :

  runs/20260923-181500_B_th05/
    meta.json         réglages, politique de coupe, modèles, heure de début
    events.jsonl      TOUT ce qui s'est passé, une ligne JSON par événement,
                      avec le temps écoulé depuis le début (t, en secondes)
    segments.csv      une ligne par phrase : raison de la coupe, probabilité,
                      horodatages, traductions, latences (ouvrable dans Excel)
    original.txt      texte original complet
    traduction-xx.txt traduction complète par langue
    summary.json      résumé chiffré : nombre de phrases, raisons de coupe,
                      temps de calcul, latences (moyenne, médiane, 90e centile)

Latences (en secondes, estimées) :
  latence_coupe_s              fin du dernier mot prononcé -> coupe de la phrase
  latence_traduction_s_<lang>  fin du dernier mot prononcé -> traduction définitive
Le temps « audio » est converti en temps réel avec le décalage le plus petit
observé entre l'heure d'un message ASR et la quantité d'audio reçue (le plus
proche du vrai début de l'audio : chaque passe Whisper ajoute son temps de calcul).
Fiable quand l'audio arrive en temps réel (micro, fichier lu dans le navigateur).
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.config import LANG_NAMES, ROOT


def _stats(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"n": len(values), "moyenne": round(statistics.mean(values), 3),
            "mediane": round(statistics.median(values), 3), "p90": round(p90, 3),
            "max": round(ordered[-1], 3)}


class RunLogger:
    def __init__(self, runs_dir: str, label: str, meta: dict, room: str = ""):
        root = Path(runs_dir)
        if not root.is_absolute():
            root = ROOT / root
        root.mkdir(parents=True, exist_ok=True)
        self.dir = self._new_dir(root, label, room)

        self.label = label
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.closed = False
        self.events = open(self.dir / "events.jsonl", "w", encoding="utf-8")

        self.meta = {"label": label, "debut": datetime.now().isoformat(timespec="seconds"), **meta}
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

        self.offsets: list[float] = []     # temps réel - temps audio, un par message ASR
        self.commit_lags: list[float] = [] # audio reçu - fin du dernier mot validé
        self.segments: dict[int, dict] = {}
        self.translations: dict[int, dict[str, dict]] = {}
        self.ms = {"asr": [], "sat": [], "mt_final": [], "mt_provisoire": []}
        self.langs_seen: list[str] = []
        self.policy_changes: list[dict] = []
        self.settings_changes: list[dict] = []
        self.errors: list[dict] = []
        self.lang_events: list[dict] = []
        self.units: list[dict] = []   # unités de sens validées (avec leur heure)
        self.units_rejected = 0
        self.dropped = 0          # phrases écartées (doublons entre micros)
        self.voice_skipped = 0    # phrases non lues (paroles simultanées)

    @staticmethod
    def _new_dir(root: Path, label: str, room: str) -> Path:
        """Dossier propre à ce test : « date_salle_nom ».

        Plusieurs salles peuvent démarrer dans la même seconde avec le même nom de
        test : le dossier est créé de façon atomique (exist_ok=False) et reçoit un
        suffixe -2, -3… s'il existe déjà. Deux tests n'écrivent jamais au même endroit.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [stamp] + [re.sub(r"[^\w\-]+", "_", p or "").strip("_")[:40] for p in (room, label)]
        base = "_".join(p for p in parts if p)
        for n in range(1, 1000):
            path = root / (base if n == 1 else f"{base}-{n}")
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        raise RuntimeError(f"Impossible de créer un dossier de test unique dans {root}")

    def _t(self) -> float:
        return round(time.time() - self.t0, 3)

    def _write(self, record: dict) -> None:
        self.events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.events.flush()

    # ---------- entrées : ce que l'utilisateur a changé pendant le test ----------
    def log_in(self, kind: str, values: dict) -> None:
        with self.lock:
            if self.closed:
                return
            t = self._t()
            self._write({"t": t, "dir": "in", "type": kind, "values": values})
            target = self.policy_changes if kind == "policy" else self.settings_changes
            target.append({"t": t, "values": values})

    # ---------- sorties : tout ce que le serveur a produit ----------
    def log_out(self, msg: dict) -> None:
        with self.lock:
            if self.closed:
                return
            t = self._t()
            record = {k: v for k, v in msg.items() if k != "audio"}  # pas l'audio Piper
            self._write({"t": t, "dir": "out", **record})
            kind = msg.get("type")

            if kind == "asr":
                if msg.get("ms"):
                    self.ms["asr"].append(msg["ms"])
                if msg.get("received_s"):
                    self.offsets.append(t - msg["received_s"])
                    if msg.get("committed_until"):
                        self.commit_lags.append(msg["received_s"] - msg["committed_until"])
            elif kind == "sat" and msg.get("ms"):
                self.ms["sat"].append(msg["ms"])
            elif kind == "segment":
                seg = dict(msg)
                seg["t"] = t
                self.segments[msg["id"]] = seg
            elif kind == "translation":
                if msg.get("final"):
                    self.ms["mt_final"].append(msg.get("ms"))
                    self.translations.setdefault(msg["id"], {})[msg["lang"]] = {**msg, "t": t}
                    if msg["lang"] not in self.langs_seen:
                        self.langs_seen.append(msg["lang"])
                else:
                    self.ms["mt_provisoire"].append(msg.get("ms"))
            elif kind == "lang":
                self.lang_events.append({"t": t, **record})
            elif kind == "error":
                self.errors.append({"t": t, **record})
            elif kind == "unit":
                self.units.append({**record, "t": t})
            elif kind == "unit_rejected":
                self.units_rejected += 1
            elif kind == "segment_dropped":
                self.dropped += 1
            elif kind == "speak_skipped":
                self.voice_skipped += 1

    # ---------- fin du test : fichiers de synthèse ----------
    def finalize(self, complete: bool = True) -> Path:
        with self.lock:
            if self.closed:
                return self.dir
            self.closed = True
            self.events.close()
            duration = self._t()
        offset = min(self.offsets) if self.offsets else None
        for seg in self.segments.values():
            seg["seg_latency_s"] = (round(seg["t"] - (seg["end"] + offset), 3)
                                    if offset is not None and seg.get("end") is not None else None)

        rows, final_latencies = [], {lang: [] for lang in self.langs_seen}
        for sid in sorted(self.segments):
            seg = self.segments[sid]
            row = {"id": sid, "langue": seg.get("lang"), "raison": seg.get("reason"),
                   "proba": seg.get("p"), "debut_s": seg.get("start"), "fin_s": seg.get("end"),
                   "mots": len(seg.get("text", "").split()),
                   "latence_coupe_s": seg.get("seg_latency_s"), "texte": seg.get("text", "")}
            for lang in self.langs_seen:
                tr = self.translations.get(sid, {}).get(lang)
                latency = None
                if tr and offset is not None and seg.get("end") is not None:
                    latency = round(tr["t"] - (seg["end"] + offset), 3)
                    final_latencies[lang].append(latency)
                row[f"traduction_{lang}"] = tr["text"] if tr else ""
                row[f"contexte_{lang}"] = tr.get("context") if tr else ""
                row[f"pivot_{lang}"] = tr.get("pivot") if tr else ""
                row[f"calcul_ms_{lang}"] = tr.get("ms") if tr else ""
                row[f"latence_traduction_s_{lang}"] = latency
            # Colonnes ajoutées à la fin : les scripts existants lisent par nom de colonne
            row["source"] = seg.get("source", "main")
            row["orateur"] = seg.get("speaker", "S1")
            rows.append(row)

        # Délai d'affichage par mot : pour chaque mot prononcé, temps jusqu'à l'affichage
        # de sa traduction définitive (unité de sens si elle le couvre, sinon phrase entière).
        word_delays: dict[str, list[float]] = {lang: [] for lang in self.langs_seen}
        units_by = {}
        for u in self.units:
            units_by.setdefault((u.get("key"), u.get("lang")), []).append(u)
        for row in rows:
            seg = self.segments[row["id"]]
            ends = seg.get("word_ends") or []
            for lang in self.langs_seen:
                row[f"delai_moyen_mot_s_{lang}"] = None       # même colonnes pour toutes les lignes
                tr = self.translations.get(row["id"], {}).get(lang)
                if not tr or offset is None or not ends:
                    continue
                units = sorted(units_by.get((seg.get("open_key"), lang), []),
                               key=lambda u: u["count"])
                delays = []
                for j, end in enumerate(ends):
                    shown = next((u["t"] for u in units if u["count"] > j), tr["t"])
                    delays.append(round(shown - (end + offset), 3))
                word_delays[lang].extend(delays)
                row[f"delai_moyen_mot_s_{lang}"] = round(sum(delays) / len(delays), 3)
            row["unites_de_sens"] = len(units_by.get((seg.get("open_key"),
                                                      self.langs_seen[0] if self.langs_seen else None), []))

        if rows:  # utf-8-sig et « ; » : s'ouvre directement dans Excel en français
            fieldnames: list[str] = []
            for row in rows:                                  # union des colonnes, dans l'ordre
                fieldnames += [k for k in row if k not in fieldnames]
            with open(self.dir / "segments.csv", "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", restval="")
                writer.writeheader()
                writer.writerows(rows)

        ordered = [self.segments[s] for s in sorted(self.segments)]
        (self.dir / "original.txt").write_text(" ".join(s["text"] for s in ordered),
                                               encoding="utf-8")
        for lang in self.langs_seen:
            parts = [s["text"] if s.get("lang") == lang
                     else self.translations.get(s["id"], {}).get(lang, {}).get("text", "")
                     for s in ordered]
            (self.dir / f"traduction-{lang}.txt").write_text(
                " ".join(p for p in parts if p), encoding="utf-8")

        summary = {
            "label": self.label,
            "termine_normalement": complete,
            "duree_s": duration,
            "phrases": len(rows),
            "mots_par_phrase": _stats([r["mots"] for r in rows]),
            "raisons_de_coupe": dict(Counter(r["raison"] for r in rows)),
            "langues_parlees": dict(Counter(r["langue"] for r in rows)),
            "phrases_par_orateur": dict(Counter(r["orateur"] for r in rows)),
            "phrases_par_source": dict(Counter(r["source"] for r in rows)),
            "phrases_ecartees_doublons": self.dropped,
            "delai_affichage_par_mot_s": {k: _stats(v) for k, v in word_delays.items()},
            "unites_de_sens": {"validees": len(self.units), "rejetees": self.units_rejected,
                               "latence_s": _stats([u["t"] - (u["end"] + offset)
                                                    for u in self.units
                                                    if offset is not None and "end" in u])},
            "voix_non_lues_paroles_simultanees": self.voice_skipped,
            "langues_traduites": {lang: LANG_NAMES.get(lang, lang) for lang in self.langs_seen},
            "temps_calcul_ms": {k: _stats(v) for k, v in self.ms.items()},
            "latence_coupe_s": _stats([r["latence_coupe_s"] for r in rows]),
            "retard_mots_valides_s": _stats(self.commit_lags),
            "latence_traduction_definitive_s": {k: _stats(v) for k, v in final_latencies.items()},
            "changements_de_langue": len([e for e in self.lang_events if e.get("previous")]),
            "changements_de_reglages_pendant_le_test": len(self.policy_changes) + len(self.settings_changes),
            "erreurs": self.errors,
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        return self.dir
