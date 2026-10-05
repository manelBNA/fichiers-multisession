"""Unités de sens : test de stabilité, frontières candidates, traduction par morceaux."""
from app.config import MtCfg
from app.meaning_units import MUTracker, find_candidates, is_stable
from app.segmentation.boundary_policy import BoundaryPolicy, PolicyParams
from app.types import SessionSettings, Word
from app.workers.translator_worker import TranslatorWorker
from tests.conftest import FakeMUPool, make_timeline, run_fake_session

SENT = ("Während in der Vergangenheit die Maximierung mechanischer Kennwerte im Vordergrund "
        "stand, rücken heute der gesamte Lebenszyklus eines Produktes in den Fokus der Forschung.")


def words(text, lang="de"):
    return [Word(i * .7, i * .7 + .5, " " + w, lang) for i, w in enumerate(text.split())]


def translator(mu=True, targets=("en",), k=3):
    msgs, spoken = [], []
    s = SessionSettings("de", list(targets), "none", mu, k)
    tr = TranslatorWorker(FakeMUPool(), MtCfg(), s, msgs.append,
                          lambda i, lang, text: spoken.append(text))
    return tr, msgs, spoken


# ---------- logique pure ----------

def test_stability():
    assert is_stable("While in the past the focus,", "while in the past the focus today the")
    assert not is_stable("While the focus", "the focus while")
    assert not is_stable("", "anything")


def test_candidates_need_clause_end_guard_lookahead_and_distance():
    ws = words(SENT)
    policy = BoundaryPolicy(PolicyParams())
    probs = [0.0] * len(ws)
    got = find_candidates(ws, probs, policy, validated=0, already_sent=set(), k=3,
                          min_words=5, sat_floor=0.05)
    assert got == [10]                                         # « stand, » (virgule)
    assert find_candidates(ws[:13], probs, policy, 0, set(), 3, 5, 0.05) == []   # 2 mots après
    assert find_candidates(ws, probs, policy, 8, set(), 3, 5, 0.05) == []        # trop près
    ws2 = words("wir haben heute der, gesamte Zyklus hier und da betrachtet")
    assert 3 not in find_candidates(ws2, [0.0] * 10, policy, 0, set(), 3, 1, 0.05)  # garde « der »


def test_tracker_minimum_over_targets():
    t = MUTracker()
    t.set("main:1", "en", 11)
    t.set("main:1", "fr", 6)
    assert t.validated("main:1") == 6
    t.drop("main:1")
    assert t.validated("main:1") == 0


# ---------- traducteur ----------

def test_unit_validated_then_sentence_completed():
    tr, msgs, _ = translator()
    ws = SENT.split()
    tr._unit("main:1", "main", "S1", "de", ws[:11], ws[:14], 7.5)
    (unit,) = [m for m in msgs if m["type"] == "unit"]
    assert unit["text"] == " ".join(w.lower() for w in ws[:11]) and unit["count"] == 11
    assert unit["src_text"] == " ".join(ws[:11])
    tr._final(1, SENT, "de", "S1", "main:1")
    (final,) = [m for m in msgs if m["type"] == "translation"]
    assert final["text"] == " ".join(FakeMUPool().mapping(SENT))   # V + suite = phrase entière
    assert final["units"] == 1 and final["key"] == "main:1"


def test_unit_rejected_when_verb_final_changes_the_translation():
    tr, msgs, _ = translator()
    ws = "Dadurch lassen sich sowohl Ausschuss als auch Verluste, erheblich reduzieren".split()
    tr._unit("main:1", "main", "S1", "de", ws[:8], ws[:11], 5.0)
    assert [m["type"] for m in msgs] == ["unit_rejected"]
    assert "main:1" not in {k for k, _ in tr.mu_state if tr.mu_state[(k, _)]["count"]}


def test_disabled_does_nothing():
    tr, msgs, _ = translator(mu=False)
    ws = SENT.split()
    tr._unit("main:1", "main", "S1", "de", ws[:11], ws[:14], 7.5)
    assert msgs == []


def test_final_mismatch_falls_back_to_normal_translation():
    tr, msgs, _ = translator()
    ws = SENT.split()
    tr._unit("main:1", "main", "S1", "de", ws[:11], ws[:14], 7.5)
    tr._final(1, "Ganz andere Worte hier jetzt.", "de", "S1", "main:1")
    (final,) = [m for m in msgs if m["type"] == "translation"]
    assert final["text"] == "ganz andere worte hier jetzt." and "units" not in final


# ---------- session complète ----------

def test_session_with_meaning_units(patch_online_asr, tmp_path):
    msgs = run_fake_session(make_timeline([(SENT, "de")], word_s=0.4, gap_s=0.2),
                            {"source": "de", "targets": ["en"], "voice": "none",
                             "mu_enabled": True, "mu_k": 3}, tmp_path, pool=FakeMUPool())
    types = [m["type"] for m in msgs]
    assert "unit" in types and types.index("unit") < types.index("segment")
    (seg,) = [m for m in msgs if m["type"] == "segment"]
    (unit,) = [m for m in msgs if m["type"] == "unit"]
    assert unit["key"] == seg["open_key"] and len(seg["word_ends"]) == len(SENT.split())
    (final,) = [m for m in msgs if m["type"] == "translation" and m["final"]]
    assert final["text"] == " ".join(FakeMUPool().mapping(SENT)) and final["units"] == 1


def test_session_without_meaning_units_sends_no_unit(patch_online_asr, tmp_path):
    msgs = run_fake_session(make_timeline([(SENT, "de")], word_s=0.4, gap_s=0.2),
                            {"source": "de", "targets": ["en"], "voice": "none"},
                            tmp_path, pool=FakeMUPool())
    assert not any(m["type"] in ("unit", "unit_rejected") for m in msgs)


# ---------- rejeu hors ligne ----------

def test_offline_replay_shows_the_latency_gain(patch_online_asr, tmp_path):
    import json

    from app.config import MtCfg
    from app.segmentation.boundary_policy import PolicyParams
    from scripts.replay_mu import simulate
    from tests.conftest import FakeSat

    msgs = run_fake_session(make_timeline([(SENT, "de")], word_s=0.4, gap_s=0.2),
                            {"source": "de", "targets": ["en"], "voice": "none"}, tmp_path,
                            pool=FakeMUPool())
    (run,) = [m["path"] for m in msgs if m["type"] == "run_saved"]
    events = [json.loads(l) for l in open(f"{run}/events.jsonl", encoding="utf-8")]
    params = PolicyParams(**json.load(open(f"{run}/meta.json", encoding="utf-8"))["politique_de_coupe"])
    off = simulate(events, FakeSat(), FakeMUPool(), params, "de", "en", MtCfg(), False, 3)
    on = simulate(events, FakeSat(), FakeMUPool(), params, "de", "en", MtCfg(), True, 3)
    assert off["unites_validees"] == 0 and on["unites_validees"] == 1
    assert on["texte"] == off["texte"]                      # même traduction finale
    assert on["delai_mot_moyen"] < off["delai_mot_moyen"]   # mais affichée plus tôt
    assert on["delai_premier_mot"] < off["delai_premier_mot"] - 2
