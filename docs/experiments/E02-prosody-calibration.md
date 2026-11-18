# E02 — Calibration prosodique à l'inférence (protocole et harnais)

- **Statut** : **BLOQUÉE** (protocole, corpus inédit de 64 phrases, harnais reproductible et smoke test vérifiés ; calibration perceptuelle et recommandations par voix en attente d'un checkpoint entraîné sur `fr4k`).
- **Branche / worktree** : `exp/e02-prosody-protocol` (basée sur `fr` `8a4f42a`)
- **Budget GPU d'entraînement** : `0 h`

> [!IMPORTANT]
> **Pourquoi E02 est déclarée bloquée plutôt que calibrée maintenant :**
> Le seul checkpoint français disponible à ce stade ([inference.pt](../../.local-runs/colab/opt1-1117/out/run_info/inference.pt), pilote `opt1-1117`, `step=1317`, `261` énoncés d'entraînement) présente un **WER de 21,09 % sur `novel`** et **58,17 % sur `train`** (dont **75,00 %** sur `M1`) dans [report.json](../../.local-runs/colab/opt1-1117/out/eval/report.json). Ajuster finement `pitch_range`, `pitch_jitter` ou `voiced_vowels` sur un modèle qui n'a pas encore convergé intelligibilité et alignement serait non représentatif et contraire au critère d'arrêt de E02. Aucune configuration prosodique par voix n'est donc recommandée avant l'entraînement sur `fr4k`.

---

## 1. Baseline observée et critère de déblocage

### 1.1 Artefact français de référence (`opt1-1117`)

| Paramètre / métrique | Valeur observée |
|---|---|
| Checkpoint | `.local-runs/colab/opt1-1117/out/run_info/inference.pt` |
| SHA-256 | `f673dd421478b86e36f60a4965818eba304eb8517fc15de8ba0690d60ca0afa2` |
| État | `step = 1317`, `4 533 067` paramètres d'inférence, `frontend = espeak` (`vocab_size = 54`), voix `F1` / `M1` |
| Évaluation ASR utilisée | `faster-whisper small` (`speed = 1.0`, `pitch_range = 1.0`, `seed = 0`) |

### 1.2 Mesures du rapport `opt1-1117` (`report.json`)

| Jeu (`samples`) | Voix | `generated_wer` | `reference_wer` | `pitch_std_st` (gén. / réf.) | `voiced_fraction` (gén. / réf.) | `duration_ratio` | `pause_ratio` |
|---|---|---:|---:|---:|---:|---:|---:|
| `novel` (8) | `all` | `0.2109` | — | `2.3175` / — | `0.7240` / — | — | — |
| `novel` (4 / 4) | `F1` / `M1` | `0.2135` / `0.2083` | — | `2.1575` / `2.4775` | `0.7708` / `0.6772` | — | — |
| `test` (4) | `all` | `0.1146` | `0.0323` | `2.2325` / `3.0275` | `0.5930` / `0.5678` | `1.0805` | `1.001` |
| `test` (2 / 2) | `F1` / `M1` | `0.0625` / `0.1666` | `0.0312` / `0.0333` | `2.3200` / `2.1450` (`3.105` / `2.950`) | `0.6915` / `0.4945` (`0.693` / `0.4425`) | `1.185` / `0.976` | `1.0655` / `0.9365` |
| `validation` (4) | `all` | `0.3833` | `0.0000` | `2.5200` / `3.3600` | `0.5890` / `0.5917` | `1.1398` | `1.320` |
| `validation` (2 / 2) | `F1` / `M1` | `0.3750` / `0.3917` | `0.0000` / `0.0000` | `2.1250` / `2.9150` (`3.565` / `3.155`) | `0.6340` / `0.5440` (`0.692` / `0.4915`) | `1.133` / `1.1465` | `1.4435` / `1.1965` |
| `train` (4) | `all` | `0.5817` | `0.0192` | `2.1625` / `2.1325` | `0.6065` / `0.6125` | `1.034` | `1.270` |
| `train` (2 / 2) | `F1` / `M1` | `0.4134` / `0.7500` | `0.0384` / `0.0000` | `1.9100` / `2.4150` (`2.205` / `2.060`) | `0.6805` / `0.5325` (`0.646` / `0.579`) | `0.985` / `1.083` | `0.9905` / `1.5495` |

### 1.3 Seuil de déblocage appliqué dans le harnais

Le script [eval_e02_prosody.py](../../scripts/eval_e02_prosody.py) vérifie `--report` (`summary.novel.all.generated_wer <= 0.15` par défaut) et refuse de lancer la campagne de calibration si le modèle ne respecte pas le critère minimal d'intelligibilité (`E02 BLOCKED: checkpoint baseline novel WER=0.2109 > 0.15`), sauf passage explicite de `--allow-unready` pour un contrôle technique du harnais.

---

## 2. Hypothèse falsifiable et mécanismes inspectés

### 2.1 Hypothèse pour la phase de calibration (post-`fr4k`)

> **Hypothèse (E02)** : Sur un checkpoint français convergent (`novel WER <= 0.15`), la régression F0 tend à sous-estimer la dispersion mélodique naturelle (`2.23–2.52` demi-tons contre `3.03–3.36` demi-tons en référence sur les locuteurs français mesurés). Une calibration modérée des contrôles d'inférence existants (`pitch_range`, `pitch_jitter`, `voiced_vowels`), testée **facteur par facteur** avant toute combinaison, permet d'améliorer la naturalité perçue en écoute comparative à l'aveugle sans dégrader le WER/CER français, sans sur-voiser artificiellement ni ajouter d'artefacts de vibration ou de surcoût de latence.

**Critères de réfutation** :
- Dégradation significative du WER/CER par rapport à la baseline `(pitch_range=1.0, pitch_jitter=0.0, voiced_vowels=True)`.
- Apparition d'artefacts (vibrato synthétique avec `pitch_jitter >= 0.1`, cassures de registre ou intonation caricaturale avec `pitch_range >= 1.15`).
- Absence de gain perceptuel à l'écoute en aveugle : dans ce cas, conserver les réglages neutres ou ajuster uniquement par voix selon les retours perceptuels.

### 2.2 Fonctionnement exact des paramètres dans `TinyVoca.synthesize`

L'inspection de [model.py](../../src/tiny_voca/model.py) (lignes `317-379`) confirme l'ordre d'application suivant :

1. **`predict_durations`** (`L317-325`) : chaque phonème dure au moins `1` trame ; les tokens de bord, ponctuations et frontières de mots (`" "`) deviennent silencieux (`0` trame) lorsque `output[..., 1] <= 0`.
2. **`voiced_vowels`** (`L359-368`, défaut `True`) : force `prosody[..., 1] >= 2.0` sur les trames dont le token propriétaire appartient à `vowel_tokens`, **avant** `widen_pitch` et `hard_prosody` (cela modifie donc à la fois le conditionnement de `decode` et la fréquence/voisement utilisés pour la moyenne F0).
3. **`widen_pitch`** (`L326-336`, défaut `1.0`) : calcule la moyenne F0 sur les trames voisées et applique `mean + (log_f0 - mean) * pitch_range`. Le contour élargi passe ensuite par `hard_prosody` (borné à `[-2.0, 1.5]`) vers `acoustic_stack` (`decode`), ainsi que vers l'oscillateur.
4. **`pitch_wobble`** (`L338-347`, `L376-377`, défaut `0.0`) : injecte `semitones * ln(2)/12 * (2 * slow_17 + fast) / sqrt(5)` **uniquement** sur la fréquence d'entrée du vocoder après `decode` (sans impacter `acoustic_stack`).

---

## 3. Corpus inédit et protocole reproductible

### 3.1 Jeu de 64 phrases françaises inédites

Le fichier [french_e02_prosody_sentences.json](../../tests/french_e02_prosody_sentences.json) contient **64 phrases inédites** réparties équitablement en **8 catégories prosodiques** (`8` phrases chacune) :

| Catégorie | Nombre | Rôle prosodique ciblé |
|---|---:|---|
| `assertions_neutres` | 8 | Déclinaison finale et intonation déclarative neutre |
| `questions_totales` | 8 | Montée mélodique finale (`... ?`) |
| `questions_partielles` | 8 | Mot interrogatif initial et profil descendant/mixte |
| `exclamatives` | 8 | Pic expressif et dynamique F0 (`... !`) |
| `enumerations_pauses` | 8 | Ruptures rythmiques aux virgules et pauses internes |
| `subordonnees_longues` | 8 | Frontières de groupes prosodiques majeurs |
| `liaisons_e_caduc` | 8 | Enchaînements vocaliques, schwas et contrastes courts |
| `dialogues_incises` | 8 | Changements de registre et incises ponctuées |

### 3.2 Isolation et couverture phonémique vérifiées

Les vérifications automatisées (`eval_e02_prosody.py --dry-run` et `tests/test_e02.py`) confirment :

- **0 chevauchement textuel** (`overlap_ids = []`) vis-à-vis de **4 272** énoncés exclus (`4 000` phrases distinctes des splits `train`/`validation`/`test` de `data/scripts_fr_4k` + `272` cas de [french_phoneme_benchmark.json](../../tests/french_phoneme_benchmark.json)). *(En worktree git sans le dossier `data/` ignoré par git, l'isolation vérifie automatiquement les 272 cas de référence du dépôt ; dans le dépôt principal complet, elle vérifie les 4 272 phrases.)*
- **0 phonème inconnu** (`unknowns = []`) sous le frontal E01 (`tokens_min = 39`, `tokens_median = 51.5`, `tokens_max = 83`).

### 3.3 Plan orthogonal (une variation à la fois avant combinaison)

Dans `ORTHOGONAL_VARIANTS` ([eval_e02_prosody.py](../../scripts/eval_e02_prosody.py)) :

1. `baseline` : `pitch_range=1.0`, `pitch_jitter=0.0`, `voiced_vowels=True`
2. `range_0.9` : `pitch_range=0.9`, `pitch_jitter=0.0`, `voiced_vowels=True`
3. `range_1.15` : `pitch_range=1.15`, `pitch_jitter=0.0`, `voiced_vowels=True`
4. `range_1.3` : `pitch_range=1.3`, `pitch_jitter=0.0`, `voiced_vowels=True`
5. `jitter_0.1` : `pitch_range=1.0`, `pitch_jitter=0.1`, `voiced_vowels=True`
6. `jitter_0.2` : `pitch_range=1.0`, `pitch_jitter=0.2`, `voiced_vowels=True`
7. `voiced_vowels_false` : `pitch_range=1.0`, `pitch_jitter=0.0`, `voiced_vowels=False`

Les métriques collectées par variante et par voix (`all`, `F1`, `M1`) sont : `all_finite`, `f0_std_st_mean`, `f0_iqr_st_mean`, `voiced_frac_mean`, `vowel_voiced_frac_mean`, `pause_ratio_mean`, `internal_pause_ms_mean`, `phones_per_sec_mean`, `total_audio_seconds` et `median_latency_ms`, accompagnées de fichiers WAV témoins (`qt_01`, `ex_01`, `en_01`, `sl_01`). À débloquer avec une évaluation ASR WER/CER externe et une écoute en aveugle une fois le modèle `fr4k` disponible.

---

## 4. Vérification technique du harnais (smoke test diagnostic uniquement)

Pour vérifier l'absence de bug du harnais sans consommer de GPU (`0 h`), un **smoke test diagnostic** (`--allow-unready --limit 8`) a été exécuté sur les 8 premières phrases (`qt_01` à `qt_08`, catégorie `questions_totales`) avec les deux voix (`16` énoncés, `49,14 s` audio par variante, `23,79 s` `F1` + `25,35 s` `M1`) sur `opt1-1117`, enregistrant `.local-runs/e02_smoke/e02_results.json` et 14 WAVs pour `qt_01`.

> [!WARNING]
> **Ces chiffres ne constituent pas une calibration de E02.** Ils proviennent d'un sous-ensemble restreint (`8` questions totales sans virgule interne) sur un checkpoint sous-entraîné (`opt1-1117`, `step=1317`), uniquement pour valider la mécanique du pipeline.

| Variante (`range / jitter / voiced_vowels`) | `all_finite` | `f0_std_st` (`all` / `F1` / `M1`) | `f0_iqr_st` (`all`) | `voiced_frac` (`all` / `F1` / `M1`) | `vowel_voiced` | `pause_ratio` | `internal_pause_ms` | `phones/s` | Latence méd. (`ms`) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `baseline` (`1.0 / 0.0 / True`) | `true` | `2.163` (`2.155` / `2.171`) | `3.044` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `163.06` |
| `range_0.9` (`0.9 / 0.0 / True`) | `true` | `1.946` (`1.939` / `1.954`) | `2.739` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `129.09` |
| `range_1.15` (`1.15 / 0.0 / True`) | `true` | `2.487` (`2.478` / `2.496`) | `3.500` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `155.68` |
| `range_1.3` (`1.3 / 0.0 / True`) | `true` | `2.811` (`2.801` / `2.822`) | `3.957` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `121.58` |
| `jitter_0.1` (`1.0 / 0.1 / True`) | `true` | `2.177` (`2.171` / `2.183`) | `3.081` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `113.05` |
| `jitter_0.2` (`1.0 / 0.2 / True`) | `true` | `2.196` (`2.192` / `2.200`) | `3.097` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `129.96` |
| `voiced_vowels_false` (`1.0 / 0.0 / False`) | `true` | `2.163` (`2.155` / `2.171`) | `3.044` | `0.6818` (`0.7292` / `0.6344`) | `1.0` | `0.1699` | `0.0` | `13.36` | `123.00` |

Observations mécaniques sur ce smoke test :
- `pitch_range` dilate exactement `f0_std_st_mean` proportionnellement autour de la moyenne (`2.163 × 0.9 = 1.946`, `× 1.15 = 2.487`, `× 1.3 = 2.811`), sans changer les durées ni le taux de voisement.
- `pitch_jitter` augmente légèrement la dispersion F0 (`+0.014` st à `0.1`, `+0.033` st à `0.2`) ; son impact réel (grain vs micro-instabilité) relève uniquement de l'écoute perceptuelle.
- Sur ces 8 énoncés précis (`qt_01..qt_08`), `voiced_vowels=False` donne des métriques identiques à `baseline` car toutes les trames vocaliques étaient déjà prédites voisées (`vowel_voiced_frac_mean = 1.0`), et `internal_pause_ms_mean = 0.0` reflète l'absence de virgule interne dans `qt_01..qt_08` (les pauses de `pause_ratio_mean = 0.1699` proviennent des tokens de bord `<bos>`/`<eos>`).

---

## 5. Livrables et procédure à suivre après l'entraînement `fr4k`

### 5.1 Livrables produits (prêts pour revue)

- [tests/french_e02_prosody_sentences.json](../../tests/french_e02_prosody_sentences.json) — 64 phrases inédites (8 catégories).
- [scripts/eval_e02_prosody.py](../../scripts/eval_e02_prosody.py) — harnais reproductible avec garde de blocage automatique (`--max-novel-wer 0.15`).
- [tests/test_e02.py](../../tests/test_e02.py) — tests unitaires (isolation du corpus, `0` inconnu, validation du `--dry-run` et du blocage `E02 BLOCKED`).
- [docs/experiments/E02-prosody-calibration.md](../../docs/experiments/E02-prosody-calibration.md) — le présent protocole et constat de blocage.

### 5.2 Commandes pour exécuter E02 sur le futur checkpoint `fr4k`

```bash
# 1. Vérifier l'isolation et le blocage automatique si WER > 0.15
PYTHONPATH=src:scripts ./env/bin/python scripts/eval_e02_prosody.py \
  --checkpoint <fr4k_inference.pt> \
  --report <fr4k_eval_report.json> \
  --out-dir .local-runs/e02_fr4k

# 2. Mesurer le WER/CER français sur les WAV générés par variante et par voix (F1 / M1),
#    puis conduire une écoute comparative en aveugle avant d'arrêter une configuration par voix.
```
