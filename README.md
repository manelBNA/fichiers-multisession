# Sous-titrage et traduction en direct : anglais, allemand, français

Pendant que tu parles, tu vois **en même temps** la transcription, les sous-titres
traduits et tu entends **la voix traduite**. Les trois langues fonctionnent dans tous
les sens : anglais, allemand et français, en source comme en cible.

| Étape | Modèle | Matériel | Rôle |
|---|---|---|---|
| 1 | Whisper multilingue (faster-whisper) | **GPU** | Détecte la langue, transcrit, valide les mots stables |
| 2 | SaT `sat-3l-sm` | **CPU** | Probabilité de fin de phrase pour chaque mot (sans code de langue) |
| 3 | Marian `opus-mt-{src}-{tgt}` | **CPU** | Traduction provisoire puis définitive, avec contexte, par paire |
| 4 | Voix du navigateur, ou Piper | **Client** ou **CPU** | Prononce la traduction définitive |

## Les modes

- **Langue parlée fixe** (anglais, allemand ou français) : Whisper transcrit toujours dans cette langue.
- **Détection automatique** : la langue des 3 dernières secondes est détectée parmi les trois.
  Le changement n'est accepté qu'après deux détections concordantes (ou une seule très sûre).
- **Sous-titres en** : coche une ou plusieurs langues. Chaque phrase est traduite dans
  toutes les langues cochées **sauf la sienne**.
- **Voix traduite** : « Vers la langue cible » prononce la traduction dans la première
  langue cochée différente de la langue parlée.

**Conversation bidirectionnelle**, par exemple entre un anglophone et un francophone :
langue parlée « Détection automatique », sous-titres en anglais **et** français, voix
« Vers la langue cible ». Quand l'un parle anglais, on voit et on entend le français ;
quand l'autre répond en français, on voit et on entend l'anglais.

## Architecture

```
 Navigateur                                  Serveur Python (FastAPI)
┌──────────────────────┐  PCM 16 kHz    ┌──────────────────────────────────────────────────┐
│ Micro ou fichier     │ ─────────────► │ WebSocket /ws                                    │
│  ├─ lecture audio    │  (100 ms/bloc) │   ▼ audio_q                                      │
│  └─ AudioWorklet     │                │ [Thread ASR] Whisper GPU : langue + LocalAgreement│
│                      │                │   ▼ seg_q : mots validés (avec leur langue)       │
│ Transcription        │                │ [Thread Segmentation] SaT CPU + politique        │
│ Ruban SaT            │     JSON       │   ▼ phrase finie / phrase en cours               │
│ Phrases + traductions│ ◄───────────── │ [Thread Traduction] Marian CPU, une paire par    │
│ Bande de sous-titres │                │   langue cible, contexte gardé par paire         │
│ Voix traduite ◄──────┼── « speak » ── │   ▼ traduction définitive dans la langue voix    │
│ Réglages en direct   │ ─────────────► │ [Thread Voix] Piper CPU (optionnel)              │
└──────────────────────┘ settings/policy└──────────────────────────────────────────────────┘
```

Aucun thread ne bloque les autres : pendant que Whisper transcrit la seconde suivante sur le
GPU, SaT et Marian travaillent sur le CPU, et la voix prononce la phrase précédente.

### Quand la langue change

Pendant les ~2 secondes nécessaires pour confirmer une nouvelle langue, Whisper a transcrit
la nouvelle parole dans l'ancienne langue. Au changement, le serveur retire ces mots (même
validés), coupe l'audio au début de la fenêtre de détection et retranscrit dans la bonne
langue. La phrase en cours est fermée (raison « Changement de langue »).

### Paires de traduction

Les six paires `en-de`, `de-en`, `en-fr`, `fr-en`, `de-fr`, `fr-de` sont chargées **à la
demande** (environ 300 Mo de RAM chacune), et celles de `translation.preload` dès le
démarrage. Si une paire directe manque, la traduction passe par l'anglais (indiqué
« via l'anglais » dans l'interface).

### Comment une frontière est décidée

Pour chaque mot du buffer ouvert, dans cet ordre (`app/segmentation/boundary_policy.py`) :

1. **Garde syntaxique** : jamais de coupe après un mot qui attend une suite (*the, of, der, und, le, de…*),
   selon la langue de chaque mot. En allemand, les auxiliaires ne bloquent pas la coupe, car
   le verbe conjugué termine souvent la subordonnée.
2. **sat** : probabilité ≥ seuil sûr **et** au moins 1 mot déjà reçu après.
3. **sat+pause** : probabilité ≥ seuil avec pause **et** pause ≥ 0,5 s.
4. **pause** : pause très longue (≥ 1,2 s) avec une probabilité non nulle.
5. **max_words** : coupe de secours si le buffer dépasse 30 mots.
6. **lang** / **flush** : changement de langue ou fin du flux.

### Comment le contexte est préservé

- **Dans la phrase** : traduction provisoire de toute la phrase en cours, derniers mots masqués (`mask_k`).
- **Entre les phrases** : la traduction définitive traduit « phrase précédente + phrase courante »
  puis retire la partie déjà traduite. Le contexte est gardé **séparément pour chaque paire**,
  donc les deux côtés d'une conversation ne se mélangent pas.

## Structure du projet

```
streaming-caption/
├── README.md
├── requirements.txt
├── requirements-dev.txt            # outils de test
├── pytest.ini
├── config.yaml                     # tous les réglages
├── app/
│   ├── main.py                     # FastAPI : page, /api/info, WebSocket /ws
│   ├── config.py                   # lecture de config.yaml
│   ├── registry.py                 # charge les modèles une seule fois
│   ├── run_logger.py               # enregistrement automatique de chaque test
│   ├── sources.py                  # micros, orateurs, profils
│   ├── coordinator.py              # anti-diaphonie et doublons entre micros
│   ├── meaning_units.py            # unités de sens : test de stabilité, candidats
│   ├── analysis.py                 # analyse automatique des tests (campagne)
│   ├── test_journal.py             # journal unique de tous les tests
│   ├── rooms.py                    # salles de l'événement : régie, écrans publics, places
│   ├── audio_framing.py            # format des blocs audio (un ou plusieurs micros)
│   ├── session.py                  # relie les threads pour une connexion
│   ├── types.py                    # Word (avec sa langue), réglages de session
│   ├── asr/
│   │   ├── whisper_engine.py       # Whisper GPU : transcription + détection de langue
│   │   └── local_agreement.py      # mots stables + changement de langue
│   ├── segmentation/
│   │   ├── sat_segmenter.py        # SaT CPU : proba par caractère → par mot
│   │   └── boundary_policy.py      # règles de coupe
│   ├── translation/
│   │   └── marian_translator.py    # Marian CPU : un modèle par paire, pivot anglais
│   ├── tts/
│   │   └── piper_tts.py            # voix côté serveur (optionnel)
│   └── workers/
│       ├── source_pipeline.py      # pipeline d'un micro : ASR + segmentation
│       ├── asr_worker.py           # thread 1 : audio, langue, transcription
│       ├── segmenter_worker.py     # thread 2 : frontières
│       └── translator_worker.py    # thread 3 : traductions + envoi à la voix
├── static/
│   └── index.html                  # interface : langues, sous-titres, voix
└── scripts/
    ├── evaluate.py                 # précision, rappel, F1 des coupes de plusieurs tests
    ├── evaluate_translation.py     # BLEU, chrF, TER des traductions
    ├── replay_run.py               # rejouer la segmentation d'un test enregistré
    ├── replay_mu.py                # mesurer hors ligne l'apport des unités de sens
    ├── check_forced_decoding.py    # vérifier le décodage forcé avec le vrai Marian
    ├── test_sat_example.py         # test 1 : SaT seul (exemples en, de, fr)
    └── simulate_stream.py          # test 2 : streaming simulé, sans audio
```

## Installation

Python 3.10 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

# PyTorch CPU suffit : Whisper utilise CTranslate2 pour le GPU, pas PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU pour Whisper** : faster-whisper a besoin de CUDA 12 et cuDNN 9. Si elles
ne sont pas installées sur le système, tu peux les ajouter par pip :

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

Si CUDA reste introuvable, Whisper se replie automatiquement sur le CPU (plus lent),
et le message de démarrage l'indique.

**Modèle Whisper** : il doit être **multilingue** (`small`, `medium`, `large-v3-turbo`),
pas une version `.en`. `large-v3-turbo` en float16 demande environ 2 Go de VRAM ;
`small` est plus léger mais moins précis en allemand et en français.

**Voix traduite** : par défaut, c'est le navigateur qui parle (rien à installer, Chrome ou
Edge recommandés, qui ont des voix allemandes et françaises de bonne qualité).
Pour une voix identique sur toutes les machines, active Piper (CPU, licence GPL-3.0) :

```bash
pip install piper-tts
python -m piper.download_voices en_US-lessac-medium de_DE-thorsten-medium fr_FR-siwis-medium --download-dir voices
```

puis mets `tts.engine: piper` dans `config.yaml`.

Au premier lancement, les modèles sont téléchargés depuis Hugging Face : environ 2,5 Go
au total (Whisper `large-v3-turbo` environ 1,6 Go, chaque paire Marian environ 300 Mo).
Avec une connexion lente, cela peut prendre 30 minutes ou plus. L'interface s'ouvre tout
de suite et affiche la progression ; le bouton « Démarrer l'écoute » s'active quand tout
est prêt. Ensuite, les modèles restent en cache sur le disque.

Pour accélérer le téléchargement, crée un jeton gratuit sur Hugging Face et définis la
variable `HF_TOKEN` avant de lancer le serveur (Windows : `set HF_TOKEN=hf_...`).
Pour un premier test rapide, `asr.model: small` (environ 480 Mo) suffit.

## Tester, du plus simple au plus complet

### Test 1 : SaT seul, sur l'exemple

```bash
python -m scripts.test_sat_example
python -m scripts.test_sat_example --lang de
python -m scripts.test_sat_example --lang fr --threshold 0.4
python -m scripts.test_sat_example --text "hello today i will explain how attention works so first"
```

Affiche la probabilité de chaque mot, les coupes et le temps CPU,
pour le texte ponctué **et** pour le texte normalisé (comme un ASR brut).

### Test 2 : streaming simulé, sans audio ni GPU

```bash
python -m scripts.simulate_stream
```

Les mots arrivent 3 par 3 avec des pauses simulées. Le scénario contient une fin de
phrase sans pause, une hésitation après « the » et une définition « that is… ».

### Test 3 : l'application complète

```bash
python -m app.main
```

Ouvre `http://localhost:8000` dans Chrome ou Edge, choisis la langue parlée, les langues
de sous-titres et la voix, puis clique « Démarrer l'écoute ».

- **Micro** : parle, par exemple « Hello, today I will explain… ». Les sous-titres
  apparaissent pendant que tu parles, et la voix traduite prononce chaque phrase terminée.
- **Fichier audio** : l'original est joué (volume réglable, 40 % par défaut) et envoyé
  au serveur en temps réel, pendant que la voix traduite est jouée par-dessus.

Les langues peuvent être changées **pendant** l'écoute, sans redémarrer.

**Écho** : avec le micro et des haut-parleurs, la voix traduite peut être captée et
retranscrite. Utilise un casque, ou coche « Couper le micro pendant la voix traduite »
(le micro envoie alors du silence tant que la voix parle).

Le micro fonctionne sur `localhost`. Depuis une autre machine, le navigateur exige HTTPS.

## Enregistrement automatique des tests

Chaque clic sur « Démarrer l'écoute » crée un dossier dans `runs/`, nommé avec la date,
l'heure et le **nom du test** saisi dans l'interface (ex. `runs/20260923-181500_B_seuil05/`).
Tout est écrit automatiquement ; le dossier est complété quand la traduction est finie
(ou tel quel si le test est interrompu).

| Fichier | Contenu |
|---|---|
| `meta.json` | réglages de langues, politique de coupe, modèles utilisés, configuration |
| `events.jsonl` | tout ce qui s'est passé, une ligne par événement, avec le temps `t` en secondes |
| `segments.csv` | une ligne par phrase : raison de la coupe, probabilité, horodatages, traductions, latences (s'ouvre dans Excel) |
| `original.txt`, `traduction-xx.txt` | textes complets |
| `summary.json` | nombre de phrases, raisons de coupe, temps de calcul et latences (moyenne, médiane, 90e centile) |

Les latences sont estimées en comparant l'heure réelle à la quantité d'audio reçue : elles
sont fiables quand l'audio arrive en temps réel (micro ou fichier lu dans le navigateur).
Pour désactiver l'enregistrement : `logs.enabled: false` dans `config.yaml`.

## Évaluer les coupes (précision, rappel, F1)

1. Prépare une **référence** : les vraies phrases de l'extrait, une par ligne, ou un texte
   suivi où chaque phrase finit par un point (par exemple le texte original corrigé à la main).
2. Lance l'évaluation sur un ou plusieurs tests :

```bash
python -m scripts.evaluate --ref reference.txt runs/20260923-181500_A runs/20260923-182000_B
python -m scripts.evaluate --ref reference.txt "runs/*" --details --csv resultats.csv
```

Le script aligne les mots du test sur la référence (il tolère les erreurs de Whisper),
puis affiche pour chaque test le nombre de coupes, les coupes justes, la précision, le
rappel, le F1 et la latence médiane de la traduction définitive. Avec `--details`, il liste
chaque coupe en trop (avec sa raison) et chaque fin de phrase manquée, dans leur contexte.

## Tests de non-régression

Le dossier `tests/` vérifie la logique du projet avec des faux modèles (sans GPU,
sans téléchargement, en quelques secondes) :

```bash
pip install -r requirements-dev.txt
python -m pytest
```

| Fichier | Ce qui est vérifié |
|---|---|
| `test_boundary_policy.py` | Règles de coupe sur les cas réels (hésitations, particules allemandes, dernier mot sans suite…) |
| `test_local_agreement.py` | Validation des mots stables, changement de langue |
| `test_sat_segmenter.py` | Probabilités par mot, mots de contexte, modes d'entrée de SaT |
| `test_translator.py` | Langues cibles, contexte par paire et par orateur, masquage du provisoire |
| `test_session_golden.py` | Session complète comparée à une référence enregistrée (`tests/golden/`) |
| `test_replay.py` | Le rejeu d'un test enregistré redonne les mêmes phrases |
| `test_protocol.py` | Messages WebSocket de l'interface |
| `test_step1_sources.py` | Micros et orateurs, profils, colonnes du journal |
| `test_step2_multi_sources.py` | Plusieurs micros : tours de parole, diaphonie, paroles simultanées, format audio |
| `test_meaning_units.py` | Unités de sens : stabilité, candidats, traduction par morceaux, rejeu |
| `test_campaign.py` | Campagne : analyse automatique après le test, rapport, fichier de campagne |
| `test_journal.py` | Journal unique : chaque test ajouté avec tous ses détails |
| `test_multisession.py` | 1, 2, 3 salles en parallèle, capacité, arrêt et redémarrage d'une salle, panne d'une salle sans effet sur les autres, isolation de l'audio et des participants, dossiers de test, suppression d'une salle |

Après une modification **volontaire** du comportement, on réenregistre la référence :
`UPDATE_GOLDEN=1 python -m pytest tests/test_session_golden.py`.

### Rejouer un vrai test, sans audio

Chaque test enregistre maintenant les mots validés avec leurs horodatages. On peut donc
refaire sa segmentation avec le vrai SaT, pour vérifier qu'une mise à jour ne change rien,
ou pour essayer d'autres réglages sur exactement le même flux :

```bash
python -m scripts.replay_run runs/20260924-101500_mon_test
python -m scripts.replay_run runs/20260924-101500_mon_test --set th_high=0.7 --set sat_input=nopunct
```

## Événement à plusieurs salles (jusqu'à 3 sessions en même temps)

Une **salle** = une conférence. Chaque salle a :
- une **régie** : la page qui capte l'orateur (micro ou fichier) et pilote la session :
  `http://<serveur>:8000/?room=amphi-a` ;
- un **écran public**, à ouvrir sur autant d'appareils que voulu (vidéoprojecteur,
  téléphones du public) : `http://<serveur>:8000/viewer?room=amphi-a`. Le public choisit
  sa langue et la taille du texte ; il voit les phrases déjà traduites en arrivant ;
- le **tableau de bord** de l'événement : `http://<serveur>:8000/event` (places occupées,
  état des salles, liens vers les régies et les écrans publics).

Jusqu'à `event.max_sessions` salles (3 par défaut) traduisent **en même temps**, chacune avec
sa propre chaîne Whisper → SaT → unités de sens → Marian :

| Modèle | Fonctionnement à plusieurs salles |
| --- | --- |
| Whisper (GPU) | un seul modèle en mémoire, `whisper_workers` transcriptions en parallèle |
| SaT (processeur) | un seul modèle, appelé en parallèle par les salles |
| Marian (processeur) | des traducteurs propres à chaque place (`per_session`) : les salles ne s'attendent pas |

Une 4e salle qui démarre est refusée avec un message clair, tant qu'aucune place n'est libre.
Une salle ne peut avoir qu'une régie à la fois. Chaque test enregistré indique sa salle
(`meta.json`, journal des tests) et a son propre dossier, nommé avec la salle
(`runs/<date>_<salle>_<nom du test>/`, suffixe `-2`, `-3`… si deux tests démarrent dans
la même seconde) : deux salles n'écrivent jamais au même endroit.

**État de chaque salle** (tableau de bord, `/api/rooms`) : `created` → `starting` →
`running` → `stopping` (fin de la transcription et de la traduction) → `stopped`.
Si un thread d'une salle tombe en panne (segmentation, traduction, ou 5 passes Whisper
ratées de suite), la salle passe en `error` : son test est enregistré comme interrompu,
**sa place est libérée** et sa régie est prévenue ; elle peut relancer avec « Démarrer ».
Les autres salles continuent sans interruption. Une salle arrêtée, sans régie ni écran
public, peut être supprimée du tableau de bord (`DELETE /api/rooms/<salle>`).

**Connexions** : `/ws?room=<salle>&role=producer|viewer` (inchangé), ou l'équivalent
`/ws/session/<salle>?role=…`. Paramètre facultatif `participant_id=<identifiant>` : il
apparaît dans la liste des participants de la salle (`/api/rooms`). Une connexion ne
reçoit que les messages de sa salle.

L'analyse automatique (campagne) a ses propres traducteurs Marian en mode `per_session` :
elle ne prend jamais ceux d'une salle en direct (elle partage encore le processeur).

```yaml
event:
  max_sessions: 3            # salles qui traduisent en même temps
  whisper_workers: 3         # transcriptions Whisper en parallèle sur le GPU
  marian_instances: per_session   # per_session | shared (moins de mémoire, mais les salles s'attendent)
  sat_parallel: true
  default_room: salle-1
  viewer_history: 200        # messages montrés à un écran public qui arrive en cours
```

### Matériel et réglages pour 3 salles

- **GPU** : 3 salles = jusqu'à 3 transcriptions en même temps. Avec `beam_size: 5`, une
  passe prend environ 0,8 s par seconde d'audio : trois salles dépasseraient la capacité
  du GPU. **Mettre `asr.beam_size: 1`**, et surveiller le temps des passes dans le journal.
- **Processeur** : `cpu.torch_threads` est partagé par les traducteurs des 3 salles ; régler
  environ « nombre de cœurs / 3 » (par exemple 4 sur 12 cœurs).
- **Mémoire** : chaque place charge ses propres modèles Marian (environ 300 Mo par paire de
  langues).

### Régies sur d'autres ordinateurs : HTTPS obligatoire pour le micro

Les navigateurs n'autorisent le micro qu'en **HTTPS** (ou sur l'ordinateur du serveur lui-même).
Pour des régies installées dans les salles, créer un certificat (par exemple avec `mkcert`),
puis :

```yaml
server:
  ssl_certfile: certs/serveur.pem
  ssl_keyfile: certs/serveur-key.pem
```

Les écrans publics n'utilisent pas de micro : ils fonctionnent aussi en HTTP. En mode fichier,
la régie fonctionne sans HTTPS.

## Plusieurs micros

Panneau **Micros** de l'interface, ou section `sources` de `config.yaml`.

| Configuration | Usage |
|---|---|
| **Un micro** (`single`, par défaut) | Comportement historique, inchangé |
| **Plusieurs micros, un orateur chacun** (`panel`) | Table ronde, micros individuels : le micro donne l'orateur |
| Un micro, plusieurs orateurs (`interview`) | Étape 3 (en attendant : un seul orateur) |

Pour chaque micro : le nom de l'orateur, l'entrée (un micro de l'ordinateur, ou un **fichier
audio** pour tester : tous les fichiers démarrent ensemble), et la langue (celle de la session,
détection automatique, ou une langue imposée). Clique sur « Autoriser et lister les micros »
pour voir leurs noms. Les configurations peuvent être **enregistrées comme profils** dans le
navigateur et rechargées en un clic.

Ce que fait le serveur avec plusieurs micros :
- **un pipeline par micro** (Whisper + SaT) : deux personnes qui parlent en même temps dans
  deux micros ne se mélangent jamais ;
- **anti-diaphonie** : un micro nettement plus faible que le plus fort au même instant
  (6 dB par défaut) est mis en silence, car il n'entend que la voix du voisin ; une phrase
  quasi identique arrivée sur un autre micro est écartée ;
- **Whisper partagé** : les passes sont sautées sur un micro silencieux, et espacées si le
  GPU prend du retard ;
- **contexte de traduction et voix propres à chaque orateur**. Paroles simultanées :
  une seule voix à la fois (`tts.simultaneous_policy: floor`) ou tout lire (`queue`) ;
- l'audio est envoyé avec un en-tête de 2 octets (version, numéro du micro), annoncé au
  démarrage : le format d'un seul micro reste inchangé.

Dans l'interface, chaque orateur a sa couleur ; un clic sur son nom permet de le renommer.
La bande de sous-titres affiche une ligne par orateur actif. Dans `segments.csv`, les colonnes
`source` et `orateur` indiquent le micro et l'orateur de chaque phrase ; `summary.json` compte
les phrases par orateur, les doublons écartés et les voix non lues.

**Conseils** : casques, ou haut-parleurs éloignés des micros ; un micro par personne, proche
de sa bouche ; une carte son multicanale fonctionne comme plusieurs micros.

## Unités de sens (traduction anticipée des longues phrases)

Désactivées par défaut. Case « Unités de sens » dans « Réglages de coupe », ou
`translation.meaning_units.enabled: true` dans `config.yaml`.

**Principe** : au lieu d'attendre la fin d'une longue phrase, on traduit tout de suite un
morceau dont le sens est déjà stable. À une frontière candidate (virgule, connecteur,
probabilité SaT non nulle, garde syntaxique respectée, au moins 5 mots), on attend k mots
de plus, puis Marian traduit le morceau (A) et le morceau + k mots (B), en imposant la
traduction déjà validée au début (décodage forcé). Si A est le début de B, le morceau est une
**unité de sens** : sa traduction est affichée en définitif, et la phrase finale la reprend
telle quelle. Sinon (verbe final allemand, réordonnancement), on attend la fin comme avant.

**Mesurer le gain avant de l'utiliser en direct** (sur un test enregistré avec cette version) :

```bash
python -m scripts.check_forced_decoding            # vérifie le décodage forcé avec le vrai Marian
python -m scripts.replay_mu runs/<test> --lang en --k 2 3 4 --ref reference_en.txt
```

`replay_mu` rejoue le même flux de mots sans et avec unités de sens et affiche, pour chaque
réglage : unités validées / testées, **délai d'affichage par mot** (temps audio entre la fin
d'un mot prononcé et l'affichage de sa traduction définitive), délai du premier mot de chaque
phrase, et chrF / BLEU par rapport à la référence.

En direct, `summary.json` donne aussi le délai d'affichage par mot et le nombre d'unités
validées et rejetées ; `segments.csv` a une colonne `unites_de_sens` par phrase.

## Campagne de tests automatique

Pour tester sans effort sur beaucoup d'enregistrements : panneau **Campagne de tests**.

1. Choisis tes fichiers audio (plusieurs à la fois, n'importe quelle langue parmi en, de, fr).
   Laisse « Langue parlée » sur Détection automatique si les langues sont mélangées.
2. Clique sur **Lancer la campagne**. Chaque fichier est joué à son tour (voix désactivée),
   enregistré sous le nom `camp_NN_<nom du fichier>`, puis **analysé automatiquement** :
   le flux de mots est rejoué hors ligne avec 5 politiques (pauses seules, SaT, unités de
   sens k = 2, 3, 4), vers chaque autre langue.
3. À la fin, clique sur **Télécharger le journal des tests** (ou **Copier le journal**) :
   ce fichier unique contient tous les détails de tous les tests, à transmettre pour le rapport.

### Le journal unique : `runs/journal_des_tests.jsonl`

Chaque test (campagne ou test manuel) y ajoute **une ligne complète** : nom, date, réglages
(langues, politique de coupe, modèles, configuration, micros), réglages modifiés pendant le
test, résultats en direct (`summary.json`), **chaque phrase** (raison de coupe, probabilité,
horodatages, traductions, latences, unités de sens, délai par mot) et, en campagne,
l'**analyse automatique** complète. Un test de campagne est écrit une seule fois, quand son
analyse est terminée.

Pour chaque test, le dossier `runs/<test>/` contient aussi `analyse.json` (tous les chiffres)
et `analyse.md` (tableaux lisibles). Mesures : délai d'affichage par mot (moyen, médian,
90e centile, par longueur de phrase), délai du premier mot, unités validées / testées, part
des phrases découpées, écart de traduction avec la politique actuelle, gain de délai et
verdict automatique (critères fixés à l'avance : gain ≥ 15 % avec une traduction quasi
inchangée).

**Références facultatives** (pour mesurer la fidélité réelle) : dépose dans un dossier
`references/` à la racine du projet, pour un fichier `mon_expose.wav` :
`mon_expose.src.txt` (texte source, une phrase par ligne) → F1 de la segmentation ;
`mon_expose.en.txt`, `mon_expose.fr.txt`… (traductions de référence) → chrF et BLEU.

L'analyse d'un test de quelques minutes prend de l'ordre de quelques minutes sur CPU
(10 rejeux avec Marian) ; le fichier suivant attend la fin de l'analyse, pour que les
mesures en direct ne soient pas perturbées.

## Lire l'interface

- **Langue entendue** : la langue détectée et sa confiance ; une étiquette marque chaque
  changement de langue dans la transcription.
- **Transcription** : noir = validé, gris = encore instable.
- **Probabilités SaT** : une barre par mot de la phrase en cours. Trait plein bleu = seuil sûr,
  pointillé ambré = seuil avec pause. Bleu = au-dessus du seuil sûr, ambré = au-dessus du seuil avec pause.
- **Phrases et traductions** : la raison de chaque coupe, la langue, et une ligne par langue de sous-titres.
- **Bande noire** : une ligne par langue de sous-titres, plus l'original. Blanc = définitif,
  gris = provisoire ; « voix » indique la langue en train d'être prononcée.
- **Phrases à prononcer** : si ce nombre grimpe, la voix accélère automatiquement.
- **Réglages de coupe** : modifiables en direct. Essaie de passer le texte envoyé à SaT de
  « ASR brut » à « tel que Whisper l'écrit » pour voir l'effet de la ponctuation.

## Réglages utiles (`config.yaml`)

| Paramètre | Effet |
|---|---|
| `session.*` | langues par défaut au chargement de la page |
| `asr.model` | modèle multilingue : `small` rapide, `large-v3-turbo` plus précis |
| `asr.lang_switch_prob` | plus haut = changements de langue plus sûrs mais plus lents |
| `translation.preload` | paires chargées au démarrage (les autres le sont à la demande) |
| `tts.engine` | `browser` (par défaut) ou `piper` |
| `asr.compute_type` | `int8_float16` pour économiser la VRAM |
| `asr.min_chunk_s` | plus petit = plus réactif, mais plus de charge GPU |
| `segmentation.backend` | `onnx` est plus rapide sur CPU que `torch` |
| `segmentation.th_high` / `th_low` | plus haut = moins de coupes, mais plus sûres |
| `segmentation.lookahead_words` | 0 = plus rapide mais plus d'erreurs en fin de buffer |
| `translation.mask_k` | mots masqués dans le provisoire (moins de clignotement) |
| `cpu.torch_threads` | à ajuster selon le nombre de cœurs (Marian, SaT et Piper se partagent le CPU) |

## Windows : activer le GPU pour Whisper

Whisper passe par CTranslate2, qui a besoin de **CUDA 12** (cuBLAS) et de **cuDNN 9**.
Sans elles, l'erreur est `Library cublas64_12.dll is not found or cannot be loaded`.
Le serveur se replie alors sur le CPU (très lent avec `large-v3-turbo`) et l'interface l'indique.

1. **Vérifie le pilote NVIDIA** : `nvidia-smi` doit afficher « CUDA Version: 12.x » ou plus.
   Sinon, mets à jour le pilote.
2. **Installe les bibliothèques CUDA dans le venv** (le plus simple) :

   ```bash
   .venv\Scripts\activate
   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   ```

   Le serveur ajoute automatiquement leurs dossiers `site-packages\nvidia\*\bin` au chemin
   de recherche des DLL (message « DLL CUDA ajoutées » au démarrage).
3. **Vérifie** :

   ```bash
   python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
   ```

   Le résultat doit être 1 ou plus.
4. **Relance complètement le serveur** (Ctrl+C puis `python -m app.main`).

Autres possibilités si l'étape 2 ne suffit pas : installer le CUDA Toolkit 12 et cuDNN 9
de NVIDIA (en ajoutant leurs dossiers `bin` au PATH), ou copier les DLL cuBLAS et cuDNN
pour CUDA 12 dans un dossier présent dans le PATH.

En attendant, pour tester sur CPU, mets `asr.device: cpu` et `asr.model: small` dans
`config.yaml` : plus léger, il peut suivre le temps réel sur un bon processeur.

## Dépannage

- **« Could not load library cudnn… » ou « cublas64_12.dll is not found »** : voir la section Windows ci-dessus.
- **Le traceback affiche des lignes qui ne correspondent pas au code** : le serveur tournait encore
  avec l'ancienne version. Arrête-le complètement, supprime les dossiers `__pycache__`, relance.
- **Pas de son capté en mode fichier** : utiliser Chrome ou Edge ; certains navigateurs bloquent
  la lecture automatique, clique alors sur lecture dans le lecteur.
- **Retard qui augmente** : Whisper est trop lent pour le GPU (la détection de langue ajoute
  une passe). Prendre un modèle plus petit, augmenter `min_chunk_s`, ou fixer la langue parlée.
- **Erreur au chargement de SaT avec `transformers`** : passer par `wtpsplit` (c'est ce que fait
  le projet) ; si l'erreur persiste, installer une version de `transformers` compatible avec `wtpsplit`.
- **Trop de coupes au milieu des phrases** : monter `th_high`, ou garder `lookahead_words` à 1 ou plus.
- **Pas de voix** : vérifier que le volume de la voix n'est pas à 0 et que le navigateur a des
  voix pour la langue (Chrome et Edge en ont). Avec Piper, vérifier que les voix sont dans `voices/`.
- **La langue bascule à tort** : monter `asr.lang_switch_prob`, ou fixer la langue parlée.
- **Le CPU sature avec deux langues de sous-titres** : chaque phrase est traduite deux fois ;
  mettre `translation.provisional: false` supprime les traductions provisoires.
