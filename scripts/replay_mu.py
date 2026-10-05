"""Mesure hors ligne de l'apport des unités de sens, sur un test enregistré.

Rejoue le flux de mots d'un test (avec ses horodatages) avec le vrai SaT et le vrai
Marian, sans et avec unités de sens (plusieurs valeurs de k), et compare :
  - le délai d'affichage par mot : temps audio entre la fin d'un mot prononcé et
    l'affichage de sa traduction définitive. Horloge VIRTUELLE (temps audio) :
    c'est le délai dû à l'algorithme seul, sans le temps de calcul ;
  - le nombre d'unités testées et validées ;
  - la fidélité (chrF, BLEU) si une traduction de référence est donnée.

    python -m scripts.replay_mu runs/20260924-101500_de --lang en --k 2 3 4 --ref reference_en.txt

Seuls les tests enregistrés avec les horodatages (étape 0) peuvent être rejoués.
"""
from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path

from app.config import MtCfg
from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.types import SessionSettings
from app.workers.segmenter_worker import SegmenterWorker
from app.workers.translator_worker import TranslatorWorker
from scripts.replay_run import _load_run, feed


class _SyncTranslator:
    """Exécute immédiatement, dans le même thread, les demandes de la segmentation."""

    def __init__(self, tr: TranslatorWorker):
        self.tr = tr
        self.mu_tracker = tr.mu_tracker

    def submit_final(self, seg_id, text, lang, speaker="S1", source=None, key=None):
        self.tr._final(seg_id, text, lang, speaker, key)

    def submit_provisional(self, *a, **k):
        pass   # le provisoire ne change pas le délai de la traduction définitive

    def submit_unit_check(self, *args):
        self.tr._unit(*args)

    def finish(self, *a, **k):
        pass


def simulate(events: list[dict], sat, pool, params: PolicyParams, source_lang: str,
             target: str, mt_cfg: MtCfg, mu_enabled: bool, k: int) -> dict:
    clock = {"t": 0.0}
    log: list[tuple[float, dict]] = []
    emit = lambda m: log.append((clock["t"], m))
    settings = SessionSettings(source_lang, [target], "none", mu_enabled, k)
    tr = TranslatorWorker(pool, mt_cfg, settings, emit, lambda *a: None)
    seg = SegmenterWorker(sat, BoundaryPolicy(copy.deepcopy(params)), None, _SyncTranslator(tr),
                          emit, settings=settings, mu_cfg=mt_cfg.meaning_units,
                          tracker=tr.mu_tracker)
    feed(events, seg, on_time=lambda t: clock.__setitem__("t", t))

    segments = [m for _, m in log if m["type"] == "segment"]
    finals = {m["id"]: (t, m) for t, m in log if m["type"] == "translation" and m.get("final")}
    units = [(t, m) for t, m in log if m["type"] == "unit"]
    delays, first_word, per_sentence = [], [], []
    for s in segments:
        if s["id"] not in finals:
            continue
        t_final = finals[s["id"]][0]
        mine = sorted((u for u in units if u[1]["key"] == s.get("open_key")),
                      key=lambda u: u[1]["count"])
        sent_delays = []
        for j, end in enumerate(s.get("word_ends", [])):
            shown = next((t for t, u in mine if u["count"] > j), t_final)
            d = max(0.0, shown - end)
            delays.append(d)
            sent_delays.append(d)
            if j == 0:
                first_word.append(d)
        per_sentence.append({"mots": len(s.get("word_ends", [])), "unites": len(mine),
                             "delai_moyen": statistics.mean(sent_delays) if sent_delays else None,
                             "raison": s.get("reason")})
    text = " ".join(finals[i][1]["text"] for i in sorted(finals))
    return {
        "phrases": len(segments),
        "unites_validees": len(units),
        "unites_testees": len(units) + sum(1 for _, m in log if m["type"] == "unit_rejected"),
        "delai_mot_median": statistics.median(delays) if delays else None,
        "delai_mot_moyen": statistics.mean(delays) if delays else None,
        "delai_mot_p90": sorted(delays)[int(0.9 * (len(delays) - 1))] if delays else None,
        "delai_premier_mot": statistics.mean(first_word) if first_word else None,
        "texte": text,
        "par_phrase": per_sentence,
        "segments": [s["text"] for s in segments],
    }


def _scores(hyp: str, ref: str | None) -> tuple[str, str]:
    if not ref:
        return "–", "–"
    try:
        import sacrebleu
    except ImportError:
        return "sacrebleu ?", "–"
    return (f"{sacrebleu.corpus_chrf([hyp], [[ref]]).score:.1f}",
            f"{sacrebleu.corpus_bleu([hyp], [[ref]]).score:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="dossier du test (runs/...)")
    ap.add_argument("--lang", default="en", help="langue cible à mesurer")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4], help="mots d'avance à tester")
    ap.add_argument("--ref", type=Path, help="traduction de référence (texte complet)")
    ap.add_argument("--save", type=Path, help="enregistrer les traductions obtenues dans ce dossier")
    args = ap.parse_args()

    events, meta, _ = _load_run(args.run)
    params = PolicyParams(**meta["politique_de_coupe"])
    source_lang = meta.get("reglages_langues", {}).get("source", "de")
    if source_lang == "auto":
        source_lang = next((e["lang"] for e in events if e.get("type") == "lang"), "de")

    from app.config import SegCfg, load_config
    from app.segmentation.sat_segmenter import SatSegmenter
    from app.translation.marian_translator import MarianPool
    seg_cfg = meta.get("config", {}).get("segmentation", {})
    sat = SatSegmenter(SegCfg(model=seg_cfg.get("model", "sat-3l-sm"),
                              backend=seg_cfg.get("backend", "onnx")))
    mt_cfg = load_config().translation
    pool = MarianPool(mt_cfg)
    ref = " ".join(args.ref.read_text(encoding="utf-8-sig").split()) if args.ref else None

    print(f"Rejeu de {args.run.name} : {source_lang} → {args.lang}, délais en temps audio "
          "(sans temps de calcul)\n")
    header = (f"{'Politique':<22}{'Unités':>12}{'Délai/mot méd.':>16}{'moyen':>8}{'p90':>8}"
              f"{'1er mot':>9}{'chrF':>7}{'BLEU':>7}")
    print(header)
    print("-" * len(header))
    configs = [("Sans unités de sens", False, 3)] + [(f"Unités de sens, k={k}", True, k) for k in args.k]
    for name, on, k in configs:
        r = simulate(events, sat, pool, params, source_lang, args.lang, mt_cfg, on, k)
        chrf, bleu = _scores(r["texte"], ref)
        units = f"{r['unites_validees']}/{r['unites_testees']}" if on else "–"
        f = lambda v: f"{v:.2f}" if v is not None else "–"
        print(f"{name:<22}{units:>12}{f(r['delai_mot_median']):>16}{f(r['delai_mot_moyen']):>8}"
              f"{f(r['delai_mot_p90']):>8}{f(r['delai_premier_mot']):>9}{chrf:>7}{bleu:>7}")
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
            slug = "sans_unites" if not on else f"unites_k{k}"
            (args.save / f"traduction-{args.lang}-{slug}.txt").write_text(r["texte"], encoding="utf-8")
    print("\nUnités : validées / testées. Délai par mot : secondes entre la fin du mot prononcé "
          "et l'affichage de sa traduction définitive.")


if __name__ == "__main__":
    main()
