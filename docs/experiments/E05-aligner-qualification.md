# E05 — Qualification de l'aligneur français et ré-annotation ciblée `fr4k` (`ANNOTATION_VERSION = 4`)

- **Statut** : **VALIDÉ**
- **Branche** : `fr`
- **Coût GPU utilisé** : **`0,06` unité Colab** (`voice-tts-e05-1118`, `--gpu T4` standard à `1,15 u/h`, sans `--high-mem`, arrêtée immédiatement après récupération ; **`0 h` d'entraînement GPU**). `1 269` tenseurs `.pt` (`378` validation + `375` test + `516` train modifiés) extraits en local CPU (`0` unité Colab) sans recalculer les `6 232` enregistrements `train` inchangés.

---

## 1. Hypothèse falsifiable et vérifications de traçabilité (Priorité 1)

> **Hypothèse (E05)** :
> 1. Séparer la compatibilité vocabulaire (`frontend_fingerprint()` = `31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6`) du hachage effectif des règles phonétiques (`frontend_rules_fingerprint()`, `FRONTEND_RULES_VERSION = 2`, `ANNOTATION_VERSION = 4`) préserve le chargement en inférence des checkpoints antérieurs (`opt1-1117`) tout en bloquant explicitement la réutilisation silencieuse des anciennes annotations `fr4k` dans `AnnotatedDataset` et `train.py --resume`.
> 2. Les frontières CTC françaises actuelles (`AlignmentSettings(onset_offset_seconds=-0.019)`) souffrent sur `M1` d'un biais apparent de `pyin` (non de l'aligneur CTC) ; une sonde spectrale physique symétrique `LF/HF` sur `24 kHz` doit le trancher sans ambiguïté.
> 3. Ré-aligner les **`577` énoncés modifiés par E01** (`516` train, `29` validation, `32` test) réduit nettement la fraction de phonèmes faibles (`weak_phone_fraction`) tout en conservant `>= 99 %` de parité exacte sur les énoncés inchangés.

**Résultat Traçabilité (Priorité 1)** :
- Vérifié dans [text.py](../../src/tiny_voca/text.py), [config.py](../../src/tiny_voca/config.py), [annotate.py](../../src/tiny_voca/annotate.py), [data.py](../../src/tiny_voca/data.py), [train.py](../../src/tiny_voca/train.py), [export.py](../../src/tiny_voca/export.py) et [export_checkpoint.py](../../scripts/export_checkpoint.py) :
  - `opt1-1117` (`inference.pt`) se charge sans erreur avec `check_frontend=True`.
  - L'ancien manifeste `.local-runs/colab/fr4k/annotations/annotations/train.jsonl` est refusé explicitement (`ValueError: annotations were produced with outdated phonetic rules or annotation version; re-run annotation`).
  - `train.py --resume` rejette tout snapshot à règles périmées, tandis que `--init-from ... --new-data` reste autorisé pour un éventuel fine-tuning.

---

## 2. Qualification physique de l'aligneur sur `validation` (`373` acceptés) et `test` (`372` acceptés)

Plutôt que de se fier au seul détecteur de voisement `librosa.pyin` (`calibration.json` avait suggéré `-0,0102 s` à cause d'un décalage apparent de `+13,6 ms` sur `M1`), le script [eval_e05_aligner.py](../../scripts/eval_e05_aligner.py) mesure trois marqueurs acoustiques symétriques directement sur la forme d'onde `24 kHz` (pas de fenêtre asynchrone) :

1. **Transitions `Fricative {f,s,ʃ} ↔ Voyelle`** : passage à `0 dB` du rapport logarithmique symétrique `LF (80–700 Hz) / HF (>= 3,2 kHz)` (fenêtre Hann `256` échantillons = `10,7 ms`, pas `1 ms`).
2. **Attaques `Occlusive {p,t,k} → Voyelle` (`S>V`)** : bascule `LF (80–600 Hz) / HF (>= 2,5 kHz)` (fenêtre `128` échantillons = `5,33 ms`, pas `1 ms`).
3. **Bords de silence (`BOS` / `EOS`)** : franchissement du seuil RMS adaptatif (`win=256`, pas `64` = `2,67 ms`).

### 2.1 Comparaison `spectral_lf_hf` vs `pyin` et `silence_rms` (`validation` / `test`, `stage = "ctc"`, `-0,019 s`)

*(Convention de signe : `+` signifie que la transition physique survient **après** la frontière alignée.)*

| Sonde & Transition | Voix | `validation` (`n`) Méd / Moy (`ms`) | `val` `<= 10,7 ms` / `<= 21,3 ms` | `test` (`n`) Méd / Moy (`ms`) | `test` `<= 10,7 ms` / `<= 21,3 ms` |
|---|---|---:|---:|---:|---:|
| **`S>V` (`LF/HF`, burst)** | **`F1`** | `352` : **`-0,67`** / `+0,51` *(pré-E01 `+0,45`)* | `62,5 %` / **`87,2 %`** | `342` : **`-1,17`** / `-0,53` | `62,3 %` / **`88,3 %`** |
| **`S>V` (`LF/HF`, burst)** | **`M1`** | `383` : **`-0,67`** / **`-0,22`** | `61,4 %` / **`85,1 %`** | `383` : **`-0,67`** / **`+0,33`** | `63,7 %` / **`85,9 %`** |
| **`F>V` (`LF/HF`)** | `F1` | `304` : `+5,17` / `+5,26` | `68,8 %` / **`93,4 %`** | `273` : `+6,00` / `+6,57` | `59,0 %` / **`89,4 %`** |
| **`F>V` (`LF/HF`)** | `M1` | `295` : `+7,67` / `+8,02` | `57,3 %` / **`88,8 %`** | `274` : `+8,67` / `+8,59` | `55,5 %` / **`86,9 %`** |
| **`V>F` (`LF/HF`)** | `F1` | `151` : `-11,67` / `-12,07` | `45,7 %` / **`81,5 %`** | `143` : `-7,67` / `-9,26` | `49,7 %` / **`88,8 %`** |
| **`V>F` (`LF/HF`)** | `M1` | `150` : `-6,50` / `-8,49` | `56,7 %` / **`83,3 %`** | `143` : `-5,33` / `-5,57` | `65,0 %` / **`93,0 %`** |
| `F>V` (`pyin` seul) | `F1` / `M1` | `303`: `0,0` (`-0,35`) / `241`: **`+10,67` (`+13,63`)** | `90,1 %` / `61,8 %` | `273`: `0,0` (`+0,90`) / `229`: **`+10,67` (`+14,53`)** | `87,6 %` / `58,5 %` |
| `silence_rms` (`bos_end`) | `all` | `373` : **`0,00`** / `-9,74` | **`95,7 %`** / `95,7 %` | `372` : **`0,00`** / `-14,06` | **`95,4 %`** / `95,7 %` |
| `silence_rms` (`eos_start`) | `all` | `347` : `-8,00` / `-1,89` | `77,5 %` / **`98,0 %`** | `348` : `-8,00` / `-5,12` | `77,0 %` / **`98,9 %`** |

**Conclusion scientifique de qualification :**
- Sur `S>V`, **`F1` et `M1` sont centrées à `-0,67 ms` médian (`<= 0,06` trame !) et `~0,0 ms` de biais moyen sur les deux splits** (`85,1 %`–`88,3 %` dans `<= 2` trames), avec **`0,0 ms` d'écart entre voix féminine et masculine**.
- Sur `M1` `F>V` / `V>F`, le point médian symétrique spectral se situe à `(8,02 - 8,49) / 2 = -0,24 ms` (`validation`), alors que `pyin` accuse `+13,6 ms` à `+14,5 ms` de retard et rate `18 %` (`54/295` val, `45/274` test) des transitions `M1` (ainsi que la majorité des rejets `vowels_not_voiced` `M1`). **C'est bien `pyin` qui retarde l'amorce sur voix grave `M1`, pas `wav2vec2`**.
- Décision : **conserver strictement `AlignmentSettings(onset_offset_seconds=-0.019)`** (décaler à `-0,0102 s` dégraderait `S>V` et `BOS` de `+9 ms`) et conserver `stage = "ctc"` (`refine` reste non requis ; sur CPU local `FrameClassifier` coûterait `~220 min` pour `6000` steps sans gain démontré sur les sondes).

---

## 3. Ré-annotation ciblée (`577` énoncés modifiés) et parité contrôlée

### 3.1 Parité des énoncés inchangés (`T4` vs manifeste `fr4k` initial)

- Sur l'ensemble des **`349` énoncés inchangés de `validation`** ré-alignés sur `T4` avec `facebook/wav2vec2-lv-60-espeak-cv-ft` (`rev ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`) :
  - **`347 / 349` (`99,43 %`) ont des durées trame-à-trame strictement identiques**.
  - Les `2` seuls écarts (`471f133ac7904fd1_F1` et `9a986f29d71a313d_F1`) valent `L1 = 4` trames sur une seule bascule de pas CTC (`20 ms`) en fp16.
- Sur le jeu de contrôle curé de **`32` énoncés** (`16` témoins inchangés + `16` paires modifiées E01) : **`16 / 16` (`100 %`) de parité exacte** sur les témoins inchangés ; sur les `16` énoncés modifiés, **`weak_phone_fraction` chute de `0,0596 → 0,0274` (`-54,0 %`)** et **`token_probability_mean` passe de `0,8400 → 0,8667`**.

### 3.2 Gain sur l'ensemble des `577` énoncés modifiés par E01 (`516` train + `29` val + `32` test)

Typologie des `577` phrases concernées (`7,69 %` des `7 501` énoncés) : `498` liaisons *et*, `134` géminées `-t_t`/`-z_z`, `108` élisions/mots `(en)`, `37` nombres/unités/abréviations, `2` finales `-ouent`.

| Jeu (`n = 577` modifiés) | `token_probability_mean` (moy / min) | `weak_phone_fraction_mean` (moy / max) | Rejetés (`gate`) |
|---|---:|---:|---:|
| **Avant E01 (`de41e47`)** | `0,8316` / `0,6403` | `0,0603` / `0,2500` | `1` |
| **Après E01/E05 (`fr` / `annotations_e01`)** | **`0,8522`** / **`0,6904`** | **`0,0369` (`-38,8 %`)** / **`0,2000`** | `2` *(`c85237893742fbb1_M1` passe de `weak 0,10→0,048` et `prob 0,741→0,785`, mais sa borne recalée classe `vowel_voiced=0,4444 < 0,50`)* |

Exemples individuels vérifiés avant / après :
- `cf09b0efae837be9_F1` (*"...son enthousiasme."*) : `weak` **`0,250 → 0,000`**, `prob` **`0,640 → 0,885`**
- `cf09b0efae837be9_M1` : `weak` **`0,214 → 0,000`**, `prob` **`0,651 → 0,859`**
- `519d5a34a7e97e7e_F1` (*"...l'enthousiasme..."*) : `weak` **`0,204 → 0,040`**, `prob` **`0,701 → 0,849`**
- `38500ec0421966a8_M1` (*"...l'euthanasie..."*) : `weak` **`0,174 → 0,046`**, `prob` **`0,700 → 0,823`**
- `1b5e1aadd743ca94_M1` (*"...dans le dos."*) : `weak` **`0,176 → 0,062`**, `prob` **`0,696 → 0,775`**

---

## 4. État final du jeu qualifié `annotations_e01`

Répertoire local : `.local-runs/colab/fr4k/annotations_e01/` (`fingerprint = 4f66e209c581ce9f82c8c14d8b15e35763459295ee85545b2776ceb01eef3f2c`, `ANNOTATION_VERSION = 4`, `frontend_rules = 70856b196b07f0a0584e8392c593bb90e69f36e669ce883433cb79f35c88d26f`) :

| Split | Acceptés (`F1` / `M1`) | Rejetés | Durée acceptée | Remarque acoustique |
|---|---:|---:|---:|---|
| `train` | **`6 672`** (`3 350` / `3 322`) | `76` | `5,755 h` | `.pt` des `516` modifiés extraits localement ; les `6 232` restants seront extraits avant l'entraînement sans toucher à `annotations_e01` |
| `validation` | **`373`** (`187` / `186`) | `5` | `0,314 h` | `378/378` `.pt` vérifiés et chargés par `AnnotatedDataset` |
| `test` | **`372`** (`187` / `185`) | `3` | `0,314 h` | `375/375` `.pt` vérifiés et chargés par `AnnotatedDataset` |
| **Total** | **`7 417`** | `84` | **`6,383 h`** | Prêt pour la phase d'entraînement budgétée ([E06-compact-architecture.md](E06-compact-architecture.md)) |
