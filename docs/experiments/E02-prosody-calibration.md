# E02 — Calibration prosodique à l'inférence (protocole, harnais et validation `fr_ref_20k`)

- **Statut** : **VALIDÉE** sur `fr_ref_20k` (`step = 20 000`, `novel WER = 0.1165 <= 0.15`) — réglages neutres conservés (`pitch_range=1.0`, `pitch_jitter=0.0`, `voiced_vowels=True`) après sweep orthogonal (`896` inférences), évaluation ASR stratifiée (`faster-whisper small`) et écoute comparative ; option `pitch_range=1.15` sur `M1` à recontrôler après le Palier 2 (`35 000` pas).
- **Branche / dépôt** : `fr` (`yoch/tiny-voca`)
- **Budget GPU d'entraînement E02** : `0 h` (calibration exécutée localement sur CPU après entraînement de `fr_ref_20k`)

> [!IMPORTANT]
> **Déblocage et conclusion E02 sur `fr_ref_20k` :**
> Le modèle français de référence Palier 1 ([inference.pt](../../.local-runs/colab/fr_ref_20k/fr_ref_20k_final/inference.pt), `step = 20 000`) atteint un **WER `novel` de `0.1165`** (`<= 0.15`, [report.json](../../.local-runs/colab/fr_ref_20k/fr_ref_20k_final/eval/report.json)), ce qui lève le blocage initial constaté sur le pilote `opt1-1117` (`step = 1317`, `novel WER = 0.2109`). Sur le corpus inédit de 64 phrases (`F1` et `M1`), les variantes de calibration n'introduisent pratiquement aucune différence perceptible à l'écoute utilisateur ; la configuration neutre `(pitch_range=1.0, pitch_jitter=0.0, voiced_vowels=True)` est donc retenue par défaut.

---

## 1. Critère de déblocage et checkpoints français mesurés

### 1.1 Artefact français Palier 1 (`fr_ref_20k`, déblocage E02)

| Paramètre / métrique | Valeur observée |
|---|---|
| Checkpoint | `.local-runs/colab/fr_ref_20k/fr_ref_20k_final/inference.pt` |
| SHA-256 | `e4642456ece7efc6578e04b48b0df09055674f50bcf59ddca5760985398d8301` |
| Empreinte règles phonétiques | `70856b196b07f0a0584e8392c593bb90e669ce883433cb79f35c88d26f` (`frontend_token_mismatches = []`) |
| État | `step = 20 000`, `4 533 067` paramètres d'inférence, voix `F1` / `M1` |
| Rapport `eval/report.json` | `novel WER = 0.1165` (`F1: 0.1206`, `M1: 0.1123`) ; `test WER = 0.0893` (réf. `0.0491` ; `F1: 0.0312/0.0312`, `M1: 0.1473/0.0670`) ; `validation WER = 0.2020` (réf. `0.1990`) ; `train WER = 0.0573` (réf. `0.0677`) |

### 1.2 Historique de blocage initial sur le pilote `opt1-1117` (avant entraînement `fr4k`)

| Paramètre / métrique | Valeur observée |
|---|---|
| Checkpoint | `.local-runs/colab/opt1-1117/out/run_info/inference.pt` |
| SHA-256 | `f673dd421478b86e36f60a4965818eba304eb8517fc15de8ba0690d60ca0afa2` |
| État | `step = 1317`, `4 533 067` paramètres d'inférence, `frontend = espeak` (`vocab_size = 54`), voix `F1` / `M1` |
| Évaluation ASR utilisée | `faster-whisper small` (`speed = 1.0`, `pitch_range = 1.0`, `seed = 0`) |

### 1.3 Mesures historiques du rapport `opt1-1117` (`report.json`)

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

### 1.4 Seuil de déblocage appliqué dans le harnais

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

## 4. Vérification technique historique du harnais (`opt1-1117`, smoke test)

Pour vérifier l'absence de bug du harnais sans consommer de GPU (`0 h`), un **smoke test diagnostic** (`--allow-unready --limit 8`) a été exécuté initialement sur les 8 premières phrases (`qt_01` à `qt_08`, catégorie `questions_totales`) avec les deux voix (`16` énoncés, `49,14 s` audio par variante, `23,79 s` `F1` + `25,35 s` `M1`) sur `opt1-1117`, enregistrant `.local-runs/e02_smoke/e02_results.json` et 14 WAVs pour `qt_01`.

> [!WARNING]
> **Ces chiffres historiques ne constituent pas la calibration de E02.** Ils proviennent d'un sous-ensemble restreint (`8` questions totales sans virgule interne) sur le checkpoint sous-entraîné `opt1-1117` (`step=1317`), uniquement pour valider la mécanique du pipeline avant l'entraînement `fr4k`.

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

## 5. Calibration complète sur `fr_ref_20k` (`step = 20 000`)

Le sweep orthogonal complet a été exécuté localement sur CPU (`0` unité GPU) via `scripts/eval_e02_prosody.py --with-asr` avec l'intégralité des **64 phrases inédites × 2 voix × 7 variantes (`896` inférences, `128` énoncés/variante, `487,57 s` de calcul)** et évaluation ASR **`faster-whisper small`** (`cpu_threads=4`) sur un sous-ensemble stratifié équilibré de **32 phrases par variante et par voix** (`4` phrases × `8` catégories, `194` appels ASR uniques grâce à la déduplication SHA-256 PCM ; artefact local `.local-runs/e02_fr_ref_20k/e02_results.json`, `e02_transcripts.jsonl` et `56` fichiers WAV dans `wavs/`).

### 5.1 Mesures prosodiques et intelligibilité ASR (`faster-whisper small`)

Métriques `baseline` complémentaires (`all` / `F1` / `M1`) : `f0_iqr_st = 3.376`, `voiced_frac = 0.6517` (`0.6767` / `0.6267`), `vowel_voiced = 1.0`, `pause_ratio = 0.2066` (`0.1889` / `0.2242`), `internal_pause_ms = 261.2`, `phones/s = 13.79` (`14.13` / `13.44`).

| Variante (`range / jitter / voiced_vowels`) | `f0_std_st` (`all` / `F1` / `M1`) | `wer_mean` (`all`) | `corpus_wer` | `cer_mean` (`all`) | `F1` (`wer` / `cer`) | `M1` (`wer` / `cer`) | VS `baseline` (`improved / regressed`) |
|---|---:|---:|---:|---:|---|---|---|
| `baseline` (`1.0 / 0.0 / True`) | `2.561` (`2.439` / `2.682`) | `0.0734` | `0.0699` | `0.0290` (`corpus_cer 0.0271`) | `0.0696` / `0.0259` | `0.0772` / `0.0320` | — |
| `range_0.9` (`0.9 / 0.0 / True`) | `2.305` (`2.195` / `2.414`) | `0.0726` | `0.0699` | `0.0292` | `0.0737` / `0.0272` | `0.0715` / `0.0311` | `1 / 1` |
| `range_1.15` (`1.15 / 0.0 / True`) | `2.945` (`2.805` / `3.085`) | `0.0594` | `0.0591` | `0.0226` (`corpus_cer 0.0229`) | `0.0612` / `0.0206` | `0.0576` / `0.0246` | `3 / 1` |
| `range_1.3` (`1.3 / 0.0 / True`) | `3.329` (`3.171` / `3.487`) | `0.0645` | `0.0645` | `0.0225` | `0.0612` / `0.0166` | `0.0678` / `0.0284` | `3 / 3` |
| `jitter_0.1` (`1.0 / 0.1 / True`) | `2.561` (`2.440` / `2.682`) | `0.0696` | `0.0699` | `0.0267` | `0.0737` / `0.0272` | `0.0654` / `0.0262` | `2 / 2` |
| `jitter_0.2` (`1.0 / 0.2 / True`) | `2.566` (`2.446` / `2.686`) | `0.0695` | `0.0672` | `0.0267` | `0.0675` / `0.0223` | `0.0715` / `0.0311` | `2 / 1` |
| `voiced_vowels_false` (`1.0 / 0.0 / False`) | `2.561` (`2.439` / `2.682`) | `0.0734` | — | `0.0290` | — | — | `0 / 0` (`vowel_voiced = 0.9999`, `0.9999` / `0.9998`) |

### 5.2 Diagnostic par catégorie prosodique (`64 × 2` baseline)

L'inspection détaillée des `64 × 2` synthèses `baseline` montre une structure prosodique cohérente :
- **Pauses aux virgules** : `0` virgule à durée nulle (`comma_ms = 305.8–469.3 ms`), et allongement pré-pausal net en fin d'énoncé (`last_ph = 114.7–145.3 ms` contre `68.5–78.5 ms` en moyenne).
- **Contours F0 terminaux (`f0_end_delta`)** : descente déclarative sur `assertions`, `dialogues`, `enumerations`, `exclamatives`, `liaisons`, `subordonnees` (`-1.55` à `-2.77` st) et montée interrogative marquée sur `questions_totales` (`F1 +4.82 st`, `M1 +1.79 st`) et `questions_partielles` (`F1 +4.90 st`, `M1 +2.49 st`).
- **Silences de bord et écrêtage** : `F1` présente `<bos> = 276–309 ms` (`<eos> = 145–176 ms`, `1` pic `> 0.99`), tandis que `M1` présente un silence initial plus long (`<bos> = 552–624 ms`, `<eos> = 39–55 ms`, `9` pics `> 0.99`).

### 5.3 Écoute utilisateur et recommandation par voix

- **`voiced_vowels`** : avec `99,99 %` des trames vocaliques déjà voisées (`0` écart ASR `improved/regressed`, `cache = 194` vs `192` signatures PCM uniques), `voiced_vowels=False` est pratiquement indiscernable ; conserver **`True`**.
- **`pitch_jitter`** : `0.1` et `0.2` n'apportent aucun gain audible ni amélioration ASR significative ; conserver **`0.0`**.
- **`pitch_range`** : `range_1.15` rapproche `M1` de sa dispersion naturelle (`3.085` st vs `2.88` st en référence) et réduit le WER stratifié (`0.0734 -> 0.0594`, CER `0.0290 -> 0.0226`), mais `range_1.0` maintient `F1` (`2.439` st) dans sa plage naturelle (`2.28–2.43` st). Lors de l'écoute comparative, l'utilisateur ne discerne pratiquement aucune différence entre les variantes.

**Décision conservatrice E02 (`fr_ref_20k`)** : conserver les paramètres neutres par défaut pour `F1` et `M1` (`pitch_range=1.0`, `pitch_jitter=0.0`, `voiced_vowels=True`), avec la piste `pitch_range=1.15` sur `M1` notée pour recontrôle après le Palier 2 (`35 000` pas).

---

## 6. Livrables et reproductibilité

### 6.1 Livrables du dépôt (`yoch/tiny-voca`, branche `fr`)

- [tests/french_e02_prosody_sentences.json](../../tests/french_e02_prosody_sentences.json) — 64 phrases inédites (8 catégories).
- [scripts/eval_e02_prosody.py](../../scripts/eval_e02_prosody.py) — harnais reproductible avec garde de blocage (`--max-novel-wer 0.15`), intégration `--with-asr` (`faster-whisper` + `jiwer`), déduplication PCM SHA-256 et échantillonnage stratifié `--asr-per-category`.
- [tests/test_e02.py](../../tests/test_e02.py) — tests unitaires (isolation du corpus, `0` inconnu, `--dry-run`, blocage `E02 BLOCKED`, `_stratified_ids` et propagation des options CLI `--with-asr`).
- [docs/experiments/E02-prosody-calibration.md](../../docs/experiments/E02-prosody-calibration.md) — protocole et validation complète sur `fr_ref_20k`.

### 6.2 Commande de reproduction sur un checkpoint français

```bash
PYTHONPATH=src:scripts ./env/bin/python scripts/eval_e02_prosody.py \
  --checkpoint .local-runs/colab/fr_ref_20k/fr_ref_20k_final/inference.pt \
  --report .local-runs/colab/fr_ref_20k/fr_ref_20k_final/eval/report.json \
  --with-asr --asr-model small --asr-per-category 4 \
  --out-dir .local-runs/e02_fr_ref_20k
```
