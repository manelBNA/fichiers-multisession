"""Marian (Hugging Face) sur le CPU : un modèle par paire de langues.

MarianTranslator : un modèle (ex. opus-mt-fr-de)
  - translate()              : traduction simple (provisoire ou sans contexte)
  - translate_with_context() : traduit « phrase précédente + phrase courante »
    puis retire la partie déjà traduite ; repli sans contexte si l'extraction échoue.

MarianPool : charge les paires à la demande (en-fr, fr-de, de-en...).
  Si une paire directe n'existe pas, passe par une langue pivot (src -> en -> tgt).
"""
from __future__ import annotations

import logging
import re
import threading
import time

from app.config import MtCfg

log = logging.getLogger("mt")
_SENT = re.compile(r"(?<=[.!?…])\s+")

# Un seul chargement Marian à la fois dans TOUT le processus (toutes places, toutes salles,
# analyse comprise) : deux from_pretrained de transformers en parallèle peuvent échouer avec
# « Cannot copy out of meta tensor; no data! », et après cet échec même un chargement seul
# échoue. Le verrou ne couvre QUE la création du modèle : la traduction reste parallèle.
_LOAD_LOCK = threading.Lock()
LOAD_ATTEMPTS = 3              # tentatives par demande de chargement
RETRY_DELAYS_S = (1.0, 3.0)    # pause entre deux tentatives (hors du verrou global)
RETRY_AFTER_S = 30.0           # après 3 échecs : nouvel essai automatique au plus tôt dans 30 s


class MarianLoadError(RuntimeError):
    """Chargement d'une paire en échec (temporaire) : signalé, jamais masqué par le pivot."""


def _model_not_found(exc: BaseException) -> bool:
    """Le modèle n'existe pas sur Hugging Face (paire non publiée) : échec définitif."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in ("RepositoryNotFoundError", "EntryNotFoundError"):
            return True
        if "is not a valid model identifier" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _load_translator(name: str, cfg: "MtCfg") -> "MarianTranslator":
    t0 = time.perf_counter()
    with _LOAD_LOCK:
        waited = time.perf_counter() - t0
        if waited > 0.05:
            log.info("Chargement de %s : %.1f s d'attente du verrou de chargement Marian", name, waited)
        return MarianTranslator(name, cfg)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def _close(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?", "…")) else text + "."


class MarianTranslator:
    def __init__(self, model_name: str, cfg: MtCfg):
        from transformers import MarianMTModel, MarianTokenizer

        self.name = model_name
        self.cfg = cfg
        self.lock = threading.Lock()
        self.tok = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to("cpu").eval()
        log.info("Marian '%s' chargé sur CPU", model_name)

    def translate(self, text: str) -> str:
        import torch

        text = text.strip()
        if not text:
            return ""
        with self.lock, torch.inference_mode():
            batch = self.tok([text], return_tensors="pt", truncation=True)
            out = self.model.generate(**batch, num_beams=self.cfg.num_beams,
                                      max_new_tokens=self.cfg.max_new_tokens)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]

    def translate_with_context(self, prev_src: str | None, prev_tgt: str | None,
                               cur: str) -> tuple[str, bool]:
        if not (self.cfg.use_context and prev_src and prev_tgt):
            return self.translate(cur), False
        joint = self.translate(_close(prev_src) + " " + cur)
        n_prev = len(split_sentences(prev_tgt)) or 1
        sents = split_sentences(joint)
        if len(sents) > n_prev:
            return " ".join(sents[n_prev:]), True
        return self.translate(cur), False

    def translate_forced(self, text: str, prefix: str) -> str:
        """Traduit `text` en imposant au décodeur de commencer par `prefix` (décodage forcé).

        Retourne seulement la SUITE générée après le préfixe. Utilisé par les unités
        de sens : la traduction déjà validée ne peut plus changer.
        """
        import torch

        text = text.strip()
        if not text:
            return ""
        with self.lock, torch.inference_mode():
            batch = self.tok([text], return_tensors="pt", truncation=True)
            start = self.model.config.decoder_start_token_id
            ids = self.tok(text_target=prefix.strip(), add_special_tokens=False).input_ids \
                if prefix.strip() else []
            forced = torch.tensor([[start] + list(ids)], dtype=torch.long)
            out = self.model.generate(**batch, decoder_input_ids=forced,
                                      num_beams=self.cfg.num_beams,
                                      max_new_tokens=self.cfg.max_new_tokens)
        new_ids = out[0][forced.shape[1]:]
        return self.tok.decode(new_ids, skip_special_tokens=True).strip()

    def warmup(self) -> None:
        self.translate("Hello.")


class MarianPool:
    def __init__(self, cfg: MtCfg):
        self.cfg = cfg
        self.models: dict[tuple[str, str], MarianTranslator] = {}
        self.missing: set[tuple[str, str]] = set()     # modèle inexistant : passage par le pivot
        self.failures: dict[tuple[str, str], dict] = {}  # échecs temporaires : nouvel essai prévu
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def is_ready(self, src: str, tgt: str) -> bool:
        return (src, tgt) in self.models

    def _lock_for(self, key: tuple[str, str]) -> threading.Lock:
        # Un verrou PAR paire : un téléchargement lent ou bloqué (ex. de-fr)
        # ne bloque plus le chargement des autres paires (ex. de-en).
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def get(self, src: str, tgt: str) -> MarianTranslator | None:
        """Modèle de la paire ; None si le modèle n'existe pas (pivot).

        Lève MarianLoadError si le chargement échoue LOAD_ATTEMPTS fois de suite : la paire
        n'est pas marquée inexistante, un nouvel essai a lieu après RETRY_AFTER_S.
        """
        key = (src, tgt)
        if key in self.models:
            return self.models[key]
        with self._lock_for(key):
            if key in self.models:
                return self.models[key]
            if key in self.missing:
                return None
            failed = self.failures.get(key)
            if failed and time.monotonic() < failed["retry_at"]:
                wait = failed["retry_at"] - time.monotonic()
                raise MarianLoadError(f"{failed['message']} Nouvel essai automatique dans {wait:.0f} s.")
            name = self.cfg.model_pattern.format(src=src, tgt=tgt)
            log.info("Chargement de %s (premier usage : téléchargement d'environ 300 Mo)…", name)
            last = None
            for attempt in range(1, LOAD_ATTEMPTS + 1):
                try:
                    model = _load_translator(name, self.cfg)
                    model.warmup()
                except Exception as exc:
                    if _model_not_found(exc):
                        log.warning("Paire %s -> %s : le modèle %s n'existe pas (%s)", src, tgt, name, exc)
                        self.missing.add(key)
                        self.failures.pop(key, None)
                        return None
                    last = exc
                    log.error("Chargement de %s : échec %d/%d", name, attempt, LOAD_ATTEMPTS,
                              exc_info=exc)
                    if attempt < LOAD_ATTEMPTS:
                        time.sleep(RETRY_DELAYS_S[min(attempt, len(RETRY_DELAYS_S)) - 1])
                    continue
                if failed or attempt > 1:
                    log.info("Chargement de %s réussi (tentative %d) : paire %s -> %s disponible",
                             name, attempt, src, tgt)
                self.failures.pop(key, None)
                self.models[key] = model
                return model
            message = (f"Traduction {src} → {tgt} indisponible : le chargement de {name} a échoué "
                       f"{LOAD_ATTEMPTS} fois ({type(last).__name__}: {last}).")
            self.failures[key] = {"message": message, "retry_at": time.monotonic() + RETRY_AFTER_S,
                                  "attempts": (failed["attempts"] if failed else 0) + LOAD_ATTEMPTS}
            log.error("%s Nouvel essai dans %.0f s.", message, RETRY_AFTER_S)
            raise MarianLoadError(f"{message} Nouvel essai automatique dans {RETRY_AFTER_S:.0f} s.")

    def failure_messages(self) -> list[str]:
        """Paires en échec de chargement (tableau de bord)."""
        return [f["message"] for f in list(self.failures.values())]

    def clear_failures(self) -> None:
        """Nouvelle session : les paires en échec seront rechargées dès leur premier usage."""
        if self.failures:
            log.info("Nouvel essai demandé pour : %s",
                     ", ".join(f"{s}-{t}" for s, t in list(self.failures)))
        self.failures.clear()

    def route(self, src: str, tgt: str) -> list[MarianTranslator]:
        """Modèle direct si possible, sinon chaîne par la langue pivot (modèle inexistant).

        Un échec de chargement (MarianLoadError) n'est PAS remplacé par le pivot : il remonte.
        """
        direct = self.get(src, tgt)
        if direct:
            return [direct]
        pivot = self.cfg.pivot
        if pivot not in (src, tgt):
            first, second = self.get(src, pivot), self.get(pivot, tgt)
            if first and second:
                return [first, second]
        raise RuntimeError(f"Aucun modèle pour {src} -> {tgt}")

    def translate(self, src: str, tgt: str, text: str) -> tuple[str, bool]:
        chain = self.route(src, tgt)
        for model in chain:
            text = model.translate(text)
        return text, len(chain) > 1

    def translate_with_context(self, src: str, tgt: str, prev_src: str | None,
                               prev_tgt: str | None, cur: str) -> tuple[str, bool, bool]:
        """Retourne (traduction, contexte utilisé, passé par le pivot)."""
        chain = self.route(src, tgt)
        if len(chain) == 1:
            out, used = chain[0].translate_with_context(prev_src, prev_tgt, cur)
            return out, used, False
        text, _ = self.translate(src, tgt, cur)  # pivot : sans contexte
        return text, False, True

    def translate_forced(self, src: str, tgt: str, text: str, prefix: str) -> str | None:
        """Décodage forcé ; None si la paire passe par le pivot (non géré)."""
        model = self.get(src, tgt)
        if model is None:
            return None
        return model.translate_forced(text, prefix)

    def preload(self, pairs: list[tuple[str, str]], on_status=None) -> None:
        for src, tgt in pairs:
            if (src, tgt) in self.models:
                continue
            if on_status:
                on_status(f"Chargement de la traduction {src} → {tgt}…")
            try:
                self.route(src, tgt)
            except RuntimeError as exc:
                if on_status:
                    on_status(str(exc))

    def loaded(self) -> list[str]:
        return sorted(f"{s}-{t}" for s, t in self.models)
