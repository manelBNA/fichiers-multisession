# Multisession : plusieurs conférences traduites en même temps

Ce document explique comment le projet fait tourner **jusqu'à 3 sessions (salles) en
parallèle**, chacune avec sa propre chaîne Whisper → SaT → Marian, sans qu'aucune donnée
ne passe d'une salle à l'autre. Il indique aussi quels fichiers sont responsables de
chaque partie du mécanisme.

Vocabulaire : dans le code, une **session** de conférence s'appelle une **salle**
(`room`). Le nom de la salle sert d'identifiant de session (`session_id`).

---

## 1. Vue d'ensemble

```
                         Serveur FastAPI (un seul processus)
                         ───────────────────────────────────
 Modèles chargés UNE fois, partagés (app/registry.py)
   Whisper GPU ── 3 appels en parallèle (sémaphore + num_workers=3)
   SaT ONNX   ── appelé en parallèle par les salles
   Marian     ── un jeu de traducteurs PAR PLACE (place 0, 1, 2) + un pour l'analyse

 RoomManager (app/rooms.py) : 3 places, l'état de chaque salle
 │
 ├── Salle « amphi-a » [running] ── place 0
 │     Régie (micro) ──audio──► Session ──► threads ASR → SaT → Marian (pool place 0)
 │                                 │
 │                                 └─► Fanout ─► régie + écrans publics d'amphi-a SEULEMENT
 │
 ├── Salle « amphi-b » [running] ── place 1   (même chaîne, ses propres threads et files)
 │
 └── Salle « amphi-c » [stopped] ── aucune place (libérée)
```

Le principe :

- **Les modèles sont partagés.** On ne charge pas 3 copies de Whisper : cela coûterait
  3 × ~2 Go de VRAM, sans aller plus vite sur un seul GPU.
- **Tout le reste appartient à une seule salle** : les threads, les files d'attente,
  l'état, le contexte de traduction, le jeu de traducteurs Marian, les connexions
  WebSocket et le dossier de journal.

---

## 2. Les trois pages (frontend)

| Page | URL | Rôle |
|---|---|---|
| Régie | `/?room=amphi-a` | Capte l'orateur (micro ou fichier) et pilote la salle. **Une seule régie par salle.** |
| Écran public | `/viewer?room=amphi-a` | Affiche les sous-titres. Autant d'écrans que voulu (vidéoprojecteur, téléphones). |
| Tableau de bord | `/event` | Les 3 places, l'état de chaque salle, les liens, la suppression d'une salle arrêtée. |

---

## 3. Le parcours d'une salle, étape par étape

### 3.1 Connexion

Le navigateur ouvre une connexion WebSocket sur l'une de ces adresses :

```
/ws?room=amphi-a&role=producer        (régie)
/ws?room=amphi-a&role=viewer          (écran public)
/ws/session/amphi-a?role=viewer       (même chose, la salle dans le chemin)
   + &participant_id=<identifiant>    (facultatif)
```

`serve_connection()` (dans `app/main.py`) procède ainsi :
1. Il retrouve la salle avec `rooms.get(nom)`, ou la crée. Le nom est nettoyé :
   « Amphi A » devient `amphi-a`.
2. Il crée **une file asyncio propre à cette connexion** et l'abonne à **cette salle
   seulement** (`rooms.subscribe`). Une connexion ne peut donc jamais recevoir les
   messages d'une autre salle.
3. Pour une régie, il réserve la salle (`claim_producer`) : une seconde régie sur la même
   salle est refusée, sinon deux sources audio se mélangeraient. Il crée ensuite un objet
   `Session` relié à la salle par un `Fanout`.

### 3.2 Démarrage : prendre une place

Quand la régie envoie `{"type": "start"}`, le serveur appelle
`rooms.start_session(room, session, data)`. Cet appel s'exécute hors de la boucle asyncio,
car il écrit sur le disque. Il fait les choses suivantes :

1. Il prend une **place libre** (`_acquire_slot`). S'il n'y en a plus, il renvoie le
   message « Capacité atteinte : 3 sessions déjà en cours (…) ». Une 4e salle est donc
   refusée.
2. L'état de la salle passe à `starting`.
3. Il donne à la session **les traducteurs Marian de sa place** (`registry.pool_for(place)`).
4. Il lance `session.start()`, qui crée **tous les threads de la salle** (voir 3.3).
5. L'état de la salle passe à `running`.

### 3.3 Les threads d'une salle

Chaque salle a sa propre chaîne de threads :

```
audio de la régie
   │  session.push_audio()  (dans la boucle asyncio : seulement mettre en file)
   ▼
audio_q ──► [Thread ASR « asr-worker-amphi-a:main »]     Whisper partagé (GPU)
               │ mots validés
               ▼
            seg_q ──► [Thread SaT « segmenter-worker-amphi-a:main »]   SaT partagé (CPU)
                        │ phrases terminées / phrase en cours
                        ▼
                     [Thread Marian « translator-worker-amphi-a »]     pool Marian de la place
                        │ traductions
                        ▼
                     [Thread Voix Piper] (facultatif)
```

- Les files `audio_q` et `seg_q` sont créées **pour chaque salle** (et pour chaque micro
  de la salle) dans `SourcePipeline`. Il n'existe **aucune file audio globale**.
- Le contexte de traduction (phrase précédente) est gardé dans le `TranslatorWorker` de
  la salle. Il ne se mélange donc jamais d'une salle à l'autre.
- Les réglages (langues, voix, politique de coupe) appartiennent à la `Session` de la
  salle. Une salle DE→FR et une salle DE→EN ne s'influencent pas.
- Le nom de la salle apparaît dans le nom des threads, ce qui rend les journaux lisibles.

### 3.4 Diffusion des résultats : seulement vers la salle

Les threads produisent des messages (transcription, résultats SaT, phrases, traductions).
Ils les envoient par `session.emit()`, qui :

1. écrit le message dans le **journal de la salle** (`RunLogger`) ;
2. le repasse dans la boucle asyncio (`call_soon_threadsafe`) vers le **`Fanout` de la
   salle** ;
3. `Room.broadcast()` le copie dans la file de **chaque connexion abonnée à cette salle**,
   et à aucune autre ;
4. garde les phrases et les traductions définitives dans un **historique de la salle**
   (200 messages). Un écran public qui arrive en cours voit ainsi les phrases déjà
   traduites.

### 3.5 Arrêt

La régie envoie `{"type": "stop"}`, et `rooms.stop_session(room)` :

1. passe la salle en `stopping` ;
2. termine la fin de l'audio : le dernier morceau est transcrit, segmenté et traduit ;
3. quand tout est traduit (message `done`), écrit le bilan du test, **libère la place** et
   passe la salle en `stopped`.

Les autres salles ne sont jamais touchées. La place libérée peut aussitôt servir à une
autre salle. Si la régie ferme sa page, la session est aussi fermée et sa place libérée.

### 3.6 Redémarrage

La régie clique de nouveau sur « Démarrer ». `session.start()` ferme proprement
l'ancienne chaîne et en reconstruit une neuve : `stopped` → `starting` → `running`.

---

## 4. États d'une salle

```
 created ──start──► starting ──► running ──stop──► stopping ──(tout traduit)──► stopped
                        │            │                                            │
                        └────────────┴──► error (panne d'un thread de la salle)   │
                                              │                                   │
                                              └────────── start (redémarrage) ◄───┘
```

| État | Signification | Place occupée ? |
|---|---|---|
| `created` | Salle connue, jamais démarrée | non |
| `starting` | Threads en cours de création | oui |
| `running` | La salle traduit | oui |
| `stopping` | Arrêt demandé, fin de la traduction en cours | oui |
| `stopped` | Arrêtée normalement | non |
| `error` | Un thread de la salle est tombé en panne | **non** (libérée) |

On consulte cet état sur `GET /api/rooms` (champs `state`, `error`, `slot`,
`participants`…) et sur le tableau de bord `/event`.

---

## 5. Isolation des erreurs

Chaque thread de salle (ASR, SaT, Marian) est protégé :

| Thread | Erreur passagère | Erreur grave |
|---|---|---|
| ASR (Whisper) | Une passe ratée est signalée, la suivante est tentée | 5 passes ratées de suite, ou erreur hors d'une passe |
| SaT | — | Toute exception |
| Marian | Une phrase ratée (paire indisponible…) est signalée | Exception hors d'une phrase |

En cas d'erreur grave, le thread appelle `session._fatal()`, qui :

1. écrit un **journal détaillé** de l'erreur (avec la trace) ;
2. prévient la régie (`{"type": "error", "fatal": true}`) et les écrans
   (`{"type": "status", "state": "error"}`) ;
3. arrête les autres threads **de cette salle seulement** ;
4. enregistre le test comme interrompu ;
5. **libère la place** et passe la salle en `error`. La régie peut relancer.

**Filet de sécurité** : si tous les threads d'une salle meurent sans prévenir,
`RoomManager.reap()` (appelé à chaque consultation de `/api/rooms`) le détecte, passe la
salle en `error` et libère sa place. Un thread mort ne garde donc jamais une place
silencieusement.

Exemple :

```
Salle 1 → error     (SaT de la salle 1 a planté, place libérée)
Salle 2 → running   (continue normalement)
Salle 3 → running   (continue normalement)
```

---

## 6. Ressources et concurrence

| Ressource | Mécanisme | Fichier |
|---|---|---|
| Whisper (GPU) | Un seul modèle. `num_workers = event.whisper_workers` (3) et un sémaphore à 3 places : 3 salles transcrivent en même temps | `app/asr/whisper_engine.py` |
| SaT (CPU) | Un seul modèle ONNX. Son verrou est remplacé par un sémaphore à `max_sessions` places (`event.sat_parallel`) | `app/registry.py`, `app/segmentation/sat_segmenter.py` |
| Marian (CPU) | `marian_instances: per_session` : un jeu de traducteurs par **place**, chargé à la demande et réutilisé par la salle suivante. `shared` : un seul jeu (moins de mémoire, mais les salles s'attendent) | `app/registry.py` (`pool_for`) |
| Analyse de campagne | Son propre jeu Marian : elle ne bloque jamais les traducteurs d'une salle en direct | `app/registry.py` (`analysis_pool`), `app/analysis.py` |
| Boucle asyncio | Seule la mise en file de l'audio s'y fait. Les calculs se font dans les threads, et les écritures disque du démarrage et de la fermeture passent par `asyncio.to_thread` | `app/main.py` |
| Journaux | Un dossier par test : `runs/<date>_<salle>_<nom>/`, créé de façon atomique (suffixe `-2`, `-3` en cas de collision) | `app/run_logger.py` |

**Mémoire bornée** : au plus 3 jeux Marian pour les places, plus 1 pour l'analyse. Le
nombre de modèles ne grandit pas avec le nombre de salles créées. À l'arrêt d'une salle,
ses threads, ses files et son journal sont fermés ; les modèles restent chargés pour
éviter de les recharger à chaque démarrage.

**Points de vigilance matériels** :
- 3 salles sur un seul GPU : mettre `asr.beam_size: 1`. Avec une petite carte (4 Go),
  préférer `asr.model: small` ou `asr.compute_type: int8_float16`.
- Le processeur est partagé : régler `cpu.torch_threads` à environ « nombre de cœurs / 3 ».
- Un repli de Whisper sur le CPU (erreur CUDA) touche toutes les salles, puisque le modèle
  est partagé.

---

## 7. Fichiers responsables du multisession

### Cœur du mécanisme

| Fichier | Responsabilité |
|---|---|
| `app/rooms.py` | **Gestionnaire des sessions** (`RoomManager`) : salles, places (capacité de 3), états, `start_session`, `stop_session`, `delete_room`, `list_sessions`, `reap`. `Room` : abonnés, participants, historique, `broadcast()` limité à la salle. `Fanout` : relie une `Session` à sa salle. |
| `app/session.py` | **Une session = une salle en cours** : crée et arrête les threads de la salle, reçoit l'audio (`push_audio`), diffuse les messages (`emit`), gère l'erreur fatale (`_fatal`), libère la place (`_release`). |
| `app/main.py` | **Points d'entrée** : WebSocket `/ws` et `/ws/session/{id}` (une connexion = une salle), commandes `start`/`stop`, `GET /api/rooms`, `DELETE /api/rooms/{salle}`, pages `/`, `/viewer`, `/event`. |
| `app/registry.py` | **Modèles partagés** : Whisper, SaT et Marian chargés une fois ; `pool_for(place)` (Marian par place) ; `analysis_pool()`. |

### Chaîne de traitement d'une salle

| Fichier | Responsabilité |
|---|---|
| `app/workers/source_pipeline.py` | Pipeline d'un micro : crée les files `audio_q` et `seg_q` de la salle et ses threads ASR et SaT. |
| `app/workers/asr_worker.py` | Thread ASR de la salle (Whisper), tolérance aux passes ratées, signalement d'erreur grave. |
| `app/workers/segmenter_worker.py` | Thread SaT de la salle (frontières de phrases), signalement d'erreur grave. |
| `app/workers/translator_worker.py` | Thread Marian de la salle : traductions, contexte propre à la salle, signalement d'erreur grave. |
| `app/asr/whisper_engine.py` | Whisper partagé, appels parallèles (sémaphore, `num_workers`). |
| `app/segmentation/sat_segmenter.py` | SaT partagé (verrou ou sémaphore). |
| `app/translation/marian_translator.py` | `MarianPool` : un jeu de traducteurs chargé à la demande (un par place). |

### Journaux, analyse, configuration

| Fichier | Responsabilité |
|---|---|
| `app/run_logger.py` | Dossier de journal unique par test et par salle (`events.jsonl`, `meta.json`, …). |
| `app/analysis.py` | Analyse automatique, avec son propre jeu Marian. |
| `app/config.py` | Section `event` (`EventCfg`) : `max_sessions`, `whisper_workers`, `marian_instances`, `sat_parallel`, `default_room`, `viewer_history`. |
| `config.yaml` | Valeurs utilisées (section `event:`). |

### Interface

| Fichier | Responsabilité |
|---|---|
| `static/index.html` | Régie : choix de la salle, démarrage et arrêt, relance après une erreur. |
| `static/viewer.html` | Écran public : choix de la salle (rejoindre ou quitter), sous-titres de cette salle. |
| `static/event.html` | Tableau de bord : places, état de chaque salle, liens, suppression. |

### Tests

| Fichier | Ce qui est vérifié |
|---|---|
| `tests/test_multisession.py` | 1, 2 et 3 salles en parallèle ; arrêt d'une salle pendant que les autres continuent ; redémarrage ; panne SaT et panne Whisper d'une salle sans effet sur les autres ; participants qui ne reçoivent que leur salle ; audio d'une salle jamais traité par une autre ; dossiers de journal uniques ; suppression ; filet de sécurité ; 4e salle refusée. |
| `tests/test_protocol.py` | WebSocket : régie et écran dans la même salle, seconde régie refusée, capacité, alias `/ws/session/{id}`, `participant_id`, états, `DELETE /api/rooms`. |

```bash
python -m pytest tests/test_multisession.py tests/test_protocol.py
```

---

## 8. Configuration (`config.yaml`)

```yaml
event:
  max_sessions: 3            # salles qui traduisent en même temps (places)
  whisper_workers: 3         # transcriptions Whisper en parallèle (un seul modèle GPU)
  marian_instances: per_session   # per_session | shared
  sat_parallel: true         # SaT appelé en parallèle par les salles
  default_room: salle-1      # salle utilisée sans ?room=
  viewer_history: 200        # messages montrés à un écran public qui arrive en cours
```

---

## 9. Ce qui garantit qu'aucune donnée ne passe d'une salle à l'autre

| Donnée | Garantie |
|---|---|
| Audio | Il arrive par la connexion de la régie, qui n'a qu'**une** `Session`. Il va dans les files `audio_q` de cette session uniquement. |
| Transcription, SaT, traduction | Calculées par les threads de la salle, à partir de ses propres files. Le contexte de traduction est propre à la salle. |
| Diffusion | `Room.broadcast()` n'écrit que dans les files des connexions abonnées à **cette** salle. |
| Participants | Chaque connexion a sa propre file, abonnée à une seule salle, choisie à l'ouverture. |
| Journaux | Un dossier unique par test, avec le nom de la salle. |
| Erreurs | Une panne arrête uniquement les threads de sa salle et ne libère que sa place. |
