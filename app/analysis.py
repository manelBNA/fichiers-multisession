"""Analyse automatique d'un test enregistré (campagne de tests).

Après chaque test, on rejoue HORS LIGNE le flux de mots enregistré (mêmes mots,
mêmes horodatages) avec plusieurs politiques, vers chaque langue cible :

  B0     pauses seules (SaT désactivé)            -> référence basse
  P0     SaT, phrases entières (réglages du test)  -> système actuel
  MU_k2  P0 + unités de sens, k = 2
  MU_k3  P0 + unités de sens, k = 3
  MU_k4  P0 + unités de sens, k = 4

Mesures (délais en temps AUDIO : dus à l'algorithme, sans le temps de calcul) :
  - délai d'affichage par mot (moyen, médian, 90e centile) et du 1er mot des phrases,
    au total et par longueur de phrase ;
  - unités de sens validées / testées, part des phrases découpées ;
  - écart de traduction avec P0 (chrF entre les deux sorties : 100 = identique) ;
  - si des références existent dans references/ : chrF et BLEU par rapport à la
    traduction de référence, et F1 des fins de phrases par rapport au texte source.

Résultats : analyse.json et analyse.md dans le dossier du test, plus une ligne dans
runs/campagne.jsonl (le fichier unique à transmettre pour le rapport).
"""
from __future__ import annotations

import copy
import json
import logging
import queue
import statistics
import threading
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from app.config import ROOT, SUPPORTED_LANGS, Config
from app.segmentation.boundary_policy import PolicyParams

log = logging.getLogger("analysis")

POLICIES = [("B0", "Pauses seules"), ("P0", "SaT, phrases entières"),
            ("MU_k2", "Unités de sens, k=2"), ("MU_k3", "Unités de sens, k=3"),
            ("MU_k4", "Unités de sens, k=4")]
BUCKETS = [("courtes (<15 mots)", 0, 15), ("moyennes (15-25)", 15, 26), ("longues (>25)", 26, 10_000)]


def _r(v, n=2):
    return round(v, n) if isinstance(v, (int, float)) else v


def _chrf(hyp: str, ref: str) -> float:
    try:
        import sacrebleu
        return sacrebleu.corpus_chrf([hyp], [[ref]]).score
    except ImportError:                      # approximation si sacrebleu est absent
        return 100 * SequenceMatcher(None, hyp, ref).ratio()


def _bleu(hyp: str, ref: str) -> float | None:
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu([hyp], [[ref]]).score
    except ImportError:
        return None


def _find_reference(label: str, lang: str) -> str | None:
    """Référence facultative dans references/ :
    <nom>.<langue>.txt pour une traduction (ex. mon_expose.en.txt),
    <nom>.src.txt pour le texte source (une phrase par ligne ou texte ponctué).
    <nom> = nom du test, ou nom du fichier audio pour une campagne (camp_01_mon_expose -> mon_expose).
    """
    import re
    if not label:
        return None
    for name in (label, re.sub(r"^camp_\d+_", "", label)):
        path = ROOT / "references" / f"{name}.{lang}.txt"
        if path.exists():
            return " ".join(path.read_text(encoding="utf-8-sig").split())
    return None


def _policy_params(name: str, base: PolicyParams) -> PolicyParams:
    p = copy.deepcopy(base)
    if name == "B0":   # SaT n'intervient plus : seules les pauses et la coupe de secours
        p.th_high, p.th_low, p.th_pause_floor = 1.01, 1.01, 0.0
    return p


def analyze_run(run_dir: Path, sat, pool, cfg: Config, progress=None) -> dict:
    """Analyse complète d'un test ; écrit analyse.json, analyse.md et la ligne de campagne."""
    from scripts.replay_mu import simulate
    from scripts.replay_run import _load_run

    t_start = time.perf_counter()
    events, meta, rows = _load_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
        if (run_dir / "summary.json").exists() else {}
    base = PolicyParams(**meta["politique_de_coupe"])
    label = meta.get("label", "")
    source = meta.get("reglages_langues", {}).get("source", "auto")
    if source not in SUPPORTED_LANGS:
        spoken = summary.get("langues_parlees") or {}
        source = max(spoken, key=spoken.get) if spoken else \
            next((e["lang"] for e in events if e.get("type") == "lang"), "en")
    targets = [t for t in SUPPORTED_LANGS if t != source]

    words = sum(len(r.get("texte", "").split()) for r in rows)
    duration = summary.get("duree_s") or 0
    result = {
        "test": label or run_dir.name, "dossier": run_dir.name,
        "date": meta.get("debut"), "langue_source": source, "cibles": {},
        "duree_s": _r(duration, 1), "mots": words,
        "debit_mots_par_s": _r(words / duration, 2) if duration else None,
        "reglages_du_test": {"politique_de_coupe": meta.get("politique_de_coupe"),
                             "modeles": meta.get("modeles", {}).get("whisper")},
        "direct": {k: summary.get(k) for k in ("phrases", "raisons_de_coupe", "temps_calcul_ms",
                                               "latence_traduction_definitive_s",
                                               "retard_mots_valides_s")},
    }

    src_ref = _find_reference(label, "src")
    total = len(targets) * len(POLICIES)
    done = 0
    for tgt in targets:
        per_policy: dict[str, dict] = {}
        ref = _find_reference(label, tgt)
        for name, _ in POLICIES:
            mu = name.startswith("MU")
            k = int(name[-1]) if mu else 3
            r = simulate(events, sat, pool, _policy_params(name, base), source, tgt,
                         cfg.translation, mu, k)
            sentences = r["par_phrase"]
            by_len = {}
            for bname, lo, hi in BUCKETS:
                sel = [s for s in sentences if lo <= s["mots"] < hi and s["delai_moyen"] is not None]
                by_len[bname] = {"phrases": len(sel), "delai_moyen": _r(statistics.mean(
                    [s["delai_moyen"] for s in sel])) if sel else None}
            entry = {
                "phrases": r["phrases"],
                "mots_par_phrase": _r(statistics.mean([s["mots"] for s in sentences]), 1)
                if sentences else None,
                "delai_mot_moyen": _r(r["delai_mot_moyen"]),
                "delai_mot_median": _r(r["delai_mot_median"]),
                "delai_mot_p90": _r(r["delai_mot_p90"]),
                "delai_premier_mot": _r(r["delai_premier_mot"]),
                "par_longueur": by_len,
                "texte": r["texte"],
            }
            if mu:
                entry.update({"unites_validees": r["unites_validees"],
                              "unites_testees": r["unites_testees"],
                              "phrases_decoupees_pct": _r(100 * sum(1 for s in sentences if s["unites"])
                                                          / len(sentences), 1) if sentences else 0})
            if ref:
                entry["chrf_ref"] = _r(_chrf(r["texte"], ref), 1)
                entry["bleu_ref"] = _r(_bleu(r["texte"], ref), 1)
            if src_ref and name in ("B0", "P0"):
                from scripts.evaluate import evaluate, read_reference, sentences_to_words
                ref_words, ref_ends = sentences_to_words(read_reference_text(src_ref))
                ev = evaluate([{"texte": t, "raison": ""} for t in r["segments"]],
                              ref_words, ref_ends, 1)
                entry["segmentation_f1"] = _r(ev["f1"], 3)
            per_policy[name] = entry
            done += 1
            if progress:
                progress(done, total, f"{source}→{tgt} · {name}")

        p0 = per_policy["P0"]
        for name, e in per_policy.items():
            if name != "P0" and p0["texte"] and e["texte"]:
                e["ecart_traduction_vs_P0_chrf"] = _r(_chrf(e["texte"], p0["texte"]), 1)
            if name.startswith("MU") and p0["delai_mot_moyen"]:
                e["gain_delai_pct"] = _r(100 * (p0["delai_mot_moyen"] - e["delai_mot_moyen"])
                                         / p0["delai_mot_moyen"], 1)
        best = max((n for n in per_policy if n.startswith("MU")),
                   key=lambda n: per_policy[n].get("gain_delai_pct") or -1e9)
        verdict = _verdict(per_policy[best])
        for e in per_policy.values():
            e.pop("texte", None)
        result["cibles"][tgt] = {"politiques": per_policy, "meilleure_mu": best,
                                 "verdict_auto": verdict, "reference_utilisee": bool(ref)}

    result["duree_analyse_s"] = _r(time.perf_counter() - t_start, 1)
    (run_dir / "analyse.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (run_dir / "analyse.md").write_text(_markdown(result), encoding="utf-8")
    campaign = run_dir.parent / "campagne.jsonl"
    with open(campaign, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def read_reference_text(text: str) -> list[str]:
    import re
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


def _verdict(e: dict) -> str:
    """Critères fixés à l'avance (plan de test) : gain ≥ 15 % et traduction peu modifiée."""
    gain = e.get("gain_delai_pct") or 0
    ecart = e.get("ecart_traduction_vs_P0_chrf")
    changed = ecart is not None and ecart < 90
    if gain >= 15 and not changed:
        return "gain net"
    if gain >= 15:
        return "gain net, traduction modifiée : vérifier la fidélité"
    if gain >= 5:
        return "gain faible"
    return "pas de gain"


def _markdown(r: dict) -> str:
    lines = [f"# Analyse automatique : {r['test']}", "",
             f"- Langue source : {r['langue_source']} · durée {r['duree_s']} s · "
             f"{r['mots']} mots · débit {r['debit_mots_par_s']} mots/s",
             "- Délais en temps audio (sans le temps de calcul).", ""]
    for tgt, c in r["cibles"].items():
        lines += [f"## {r['langue_source']} → {tgt}", "",
                  "| Politique | Phrases | Délai/mot moyen | médian | p90 | 1er mot | Unités | Gain | Écart vs P0 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, label in POLICIES:
            e = c["politiques"][name]
            units = f"{e['unites_validees']}/{e['unites_testees']}" if "unites_validees" in e else "–"
            lines.append(f"| {label} | {e['phrases']} | {e['delai_mot_moyen']} | "
                         f"{e['delai_mot_median']} | {e['delai_mot_p90']} | {e['delai_premier_mot']} | "
                         f"{units} | {e.get('gain_delai_pct', '–')} % | "
                         f"{e.get('ecart_traduction_vs_P0_chrf', '–')} |")
        lines += ["", f"Meilleure configuration : **{c['meilleure_mu']}** · verdict automatique : "
                      f"**{c['verdict_auto']}**", ""]
    return "\n".join(lines)


class AnalysisWorker(threading.Thread):
    """Une seule analyse à la fois, en arrière-plan, après chaque test enregistré."""

    def __init__(self, registry, cfg: Config):
        super().__init__(daemon=True, name="analysis-worker")
        self.registry, self.cfg = registry, cfg
        self.jobs: queue.Queue = queue.Queue()

    def submit(self, run_dir: str, emit) -> None:
        self.jobs.put((Path(run_dir), emit))

    def run(self) -> None:
        while True:
            run_dir, emit = self.jobs.get()
            def progress(done, total, what):
                emit({"type": "analysis_progress", "path": str(run_dir), "done": done,
                      "total": total, "what": what})
            from app.test_journal import append as journal_append
            try:
                emit({"type": "analysis_started", "path": str(run_dir)})
                # Traducteurs propres à l'analyse (per_session) : jamais ceux d'une salle en direct
                pool_for = getattr(self.registry, "analysis_pool", None)
                pool = pool_for() if pool_for else self.registry.pool
                result = analyze_run(run_dir, self.registry.sat, pool, self.cfg, progress)
                journal = journal_append(run_dir, result)
                emit({"type": "analysis_done", "path": str(run_dir),
                      "test": result["test"], "langue_source": result["langue_source"],
                      "resume": {t: {"meilleure_mu": c["meilleure_mu"],
                                     "gain_delai_pct": c["politiques"][c["meilleure_mu"]].get("gain_delai_pct"),
                                     "verdict": c["verdict_auto"]}
                                 for t, c in result["cibles"].items()},
                      "duree_analyse_s": result["duree_analyse_s"], "journal": str(journal)})
            except Exception as exc:   # une analyse ratée ne doit pas bloquer la campagne
                log.exception("Analyse impossible")
                try:
                    journal_append(run_dir, {"erreur": str(exc)})
                except Exception:
                    pass
                emit({"type": "analysis_done", "path": str(run_dir), "error": str(exc)})
