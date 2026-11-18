# E06 & Feuille de route entraînement — Architecture compacte (`2,66 M`) et budget de référence français

- **Statut** : **PRÉPARÉE (`0` entraînement GPU lancé à ce stade)** — Configurations, tests unitaires, benchmarks CPU (`1T` / `4T` avec `omit_mask=True`) et plan budgété établis ; en attente de la finalisation E05 puis d'autorisation GPU dédiée.
- **Branche** : `fr`
- **Budget GPU consommé pour cette préparation** : `0 h` (`0` unité Colab)

---

## 1. Comparaison architecturale locale (`CPU`, `omit_mask=True`, `0` entraînement GPU)

### 1.1 Paramètres et taille ONNX FP32

| Configuration | Fichier config | `channels` | `heads` (`dim/tête`) | `feedforward` | Params inférence | Params `mel_head` | `.onnx` FP32 (`omit_mask=True`) | Nœuds ONNX |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Référence FR (`192/640`)** | [fr_baseline_192_640.json](../../configs/fr_baseline_192_640.json) | 192 | 4 (`48`) | 640 | `4 533 067` | `15 440` | `21 307 929 o` (`21,31 MB`) | 2 217 |
| **E06 compacte (`160/384`)** | [e06_160_384.json](../../configs/e06_160_384.json) | 160 | 4 (`40`) | 384 | **`2 657 899`** (`-41,37 %`) | `12 880` | **`13 813 353 o` (`13,81 MB`, `-35,17 %`)** | 2 217 |

*(La diminution sur `.onnx` est de `-35,17 %` et non `-41,37 %` car les noyaux fixes STFT/cepstre ~`2,87 MB` sont identiques entre les deux architectures. L'empreinte acoustique `Config.acoustic_fingerprint` reste strictement identique : les tenseurs `.pt` qualifiés E05 sont réutilisables à 100 %.)*

### 1.2 Latences CPU ONNX Runtime (`CPUExecutionProvider`, `ORT_ENABLE_ALL`)

Mesures sur **Intel Core i7-3770 @ 3.40 GHz** (Ivy Bridge sans AVX2/VNNI), script [benchmark_e06_arch.py](../../scripts/benchmark_e06_arch.py), `warmup = 5`, `runs = 30`, passes séquentielles isolées (aucune autre tâche active en parallèle ; biais durée non entraîné fixé à `[1.6, 1.0]` pour produire des longueurs réalistes : `short = 0,811 s`, `medium = 2,560 s`, `long = 5,632 s`) :

- **1 thread** (`loadavg` : `4,71 → 4,57`) :

| Phrase (`tokens`, `durée`) | Réf. `192/640` Méd / Min / p95 (`ms`) | Réf. `RTFx` | E06 `160/384` Méd / Min / p95 (`ms`) | E06 `RTFx` | Gain médian |
|---|---:|---:|---:|---:|---:|
| `short` (`19`, `0,811 s`) | `32,86` / `30,77` / `38,33` | `24,7x` | **`28,04`** / `24,11` / `40,28` | **`28,9x`** | **`-14,7 %`** |
| `medium` (`60`, `2,560 s`) | `101,68` / `97,76` / `119,37` | `25,2x` | **`77,33`** / `75,00` / `81,00` | **`33,1x`** | **`-23,9 %`** |
| `long` (`132`, `5,632 s`) | `286,15` / `261,09` / `323,51` | `19,7x` | **`215,34`** / `200,84` / `229,35` | **`26,2x`** | **`-24,7 %`** |

- **4 threads** (`loadavg` : `4,75 → 4,51`) :

| Phrase (`tokens`, `durée`) | Réf. `192/640` Méd / Min / p95 (`ms`) | Réf. `RTFx` | E06 `160/384` Méd / Min / p95 (`ms`) | E06 `RTFx` | Gain médian |
|---|---:|---:|---:|---:|---:|
| `short` (`19`, `0,811 s`) | `18,75` / `13,68` / `28,65` | `43,2x` | **`14,69`** / `11,77` / `16,42` | **`55,2x`** | **`-21,7 %`** |
| `medium` (`60`, `2,560 s`) | `45,86` / `41,35` / `58,23` | `55,8x` | **`41,22`** / `34,16` / `57,64` | **`62,1x`** | **`-10,1 %`** |
| `long` (`132`, `5,632 s`) | `118,85` / `104,92` / `146,65` | `47,4x` | **`98,26`** / `87,78` / `114,76` | **`57,3x`** | **`-17,3 %`** |

*(Constat : même avec une baisse de `-41,4 %` de paramètres, l'accélération réelle FP32 sur CPU se situe entre `-10 %` et `-25 %` car l'attention $O(T^2)$ et les convolutions STFT pèsent d'autant plus une fois les projections linéaires allégées. L'option INT8 `acoustic_stack` et le DSP C restent expérimentaux hors défaut `.onnx`, sans nouvelle campagne d'optimisation à ce stade.)*

---

## 2. Priorité 3 — Budget mesuré et critères d'arrêt pour la référence TTS française (`fr_baseline_192_640`)

### 2.1 Base empirique mesurée sur Colab `L4` (`opt1-1117/out/run_info/metrics.jsonl`)

Sur la VM Colab **`L4`** à **`1,94` unités/h** (`batch_frames = 9000`, `max_batch = 32`, `crop_samples = 8192`, `crops = 2`, données `fr` format `4`) :
- **Phase pré-GAN (`step <= gan_start`)** : médiane mesurée **`9,1 steps/s`** (`~1 000` steps en `110 s`).
- **Phase GAN (`step > gan_start`)** : médiane mesurée **`1,28 steps/s`** (`~4 608 steps/h`).

> [!IMPORTANT]
> **Règle de coût GPU sur `features/`** : Ne jamais exécuter `tiny_voca.prepare` (`librosa.pyin` CPU séquentiel) sur une VM GPU `L4` (ce qui avait coûté `~98 min ≈ 3,17` unités sur le run précédent). Préparer les `.pt` manquants de `train` (`~2,8 Go` au complet) en local CPU (`0` unité) ou en archive avant d'ouvrir la session GPU d'entraînement.

### 2.2 Découpage par paliers avec critères d'arrêt stricts (`seed = 20261113`, données qualifiées E05)

Recette standard alignée sur `train.py` (`--gan-start 8000 --teacher-hold 3000 --teacher-ramp 7000 --teacher-final 0 --gan-ramp 3000 --gan-ratio 0.7 --keep-every 5000`, `train` = `fr4k` qualifié E05, entraînement avec masquage de padding complet — jamais `omit_mask=True` en batch paddé) :

| Palier | Plage de steps | Durée GPU `L4` estimée | Coût `L4` estimé (`1,94/h`) | Critère d'évaluation & arrêt |
|---|---|---:|---:|---|
| **Palier 1 — Référence initiale (arrêt obligatoire à 20k)** | `0 → 20 000` (`8k` pré-GAN + `12k` GAN) | `~15 min + 2h36 = ~2h51` *(+ transfert/éval `~20 min`, plafond `3h30`)* | **`~5,5` unités** *(max `~6,8` u.)* | Exécuter `tiny_voca.evaluate` sur `test` + `novel`. **Arrêt définitif si `validation_predicted_mel` stagne/diverge ou si effondrement des durées/GAN**. Débloque **E02** si `novel WER <= 0,15`. |
| **Palier 2 — Extension qualité conditionnelle (`--resume`)** | `20 000 → 35 000` (`+15k` GAN) | `~3h15` *(plafond `3h45`)* | **`~6,3` unités** *(max `~7,3` u.)* | Uniquement si le Palier 1 montre une amélioration nette mais perfectible (`0,08 < novel WER <= 0,20`, pertes `mel/stft` encore décroissantes). |
| **Palier 3 — Maturation finale optionnelle (`--resume`)** | `35 000 → 50 000` (`+15k` GAN) | `~3h15` *(plafond `3h45`)* | **`~6,3` unités** *(max `~7,3` u.)* | Uniquement si Palier 2 justifie la convergence finale (nota : les `104 000` steps du README anglais coûteraient `~21 h ≈ 41` unités sur `L4` sans paliers). |

---

## 3. Priorité 4 — Condition de reprise d'E02

Conformément à [E02-prosody-calibration.md](E02-prosody-calibration.md), **E02** ne reprend qu'après validation du Palier 1 (ou 2) de la référence française (`novel WER <= 0,15` dans `report.json`), via :

```bash
PYTHONPATH=src:scripts ./env/bin/python scripts/eval_e02_prosody.py \
  --checkpoint <fr4k_ref_inference.pt> \
  --report <fr4k_ref_report.json> \
  --out-dir .local-runs/e02_fr4k_ref
```

---

## 4. Priorité 5 — Protocole comparatif E06 (`160/384`) vs Référence (`192/640`)

Pour éviter tout biais, **E06 (`2,66 M`)** sera entraînée **strictement à données et protocole identiques** après validation du Palier 1 de la référence :
- **Données** : exactement les mêmes manifestes `annotations_e01/{train,validation,test}.jsonl` et le même `seed = 20261113`.
- **Hyperparamètres** : `--config configs/e06_160_384.json`, même `--batch-frames 9000 --max-batch 32 --gan-start 8000 --teacher-hold 3000 --teacher-ramp 7000 --gan-ramp 3000`, comparée au même palier `20 000` steps (coût `L4` attendu `<= ~5,5` unités, potentiellement légèrement inférieur à `5,5` u. grâce au coût Transformer réduit).
- **Critères d'évaluation E06 vs Référence (`192/640`)** :
  1. `validation_predicted_mel` et `validation_predicted_stft` au même nombre de steps (`20k`).
  2. Intelligibilité `tiny_voca.evaluate` : `novel WER` et `test WER` (`faster-whisper`).
  3. Qualité acoustique (`mel_dtw`, `pitch_std`, `duration_ratio`, écoute comparative en aveugle).
  4. Gain d'efficience : débit `steps/s` / coût GPU réel + latence ONNX CPU (`1T` / `4T`, FP32 `13,81 MB`, et variante `int8_acoustic` en annexe expérimentale).
  5. **Décision** : adopter `160/384` seulement si la baisse de paramètres (`-41,4 %`) et de latence (`-10 %` à `-25 %`) ne dégrade pas l'intelligibilité ni la naturalité perceptuelle.
