# E03 — Profilage ONNX Runtime et quantification dynamique INT8

## 1. Contexte, périmètre et baseline exacte

Cette expérience caractérise les performances CPU natives (latence, facteur temps réel $\text{RTF}_x$, mémoire RSS pic, taille binaire et fidélité numérique/acoustique) de **tiny-voca** en français, de PyTorch FP32 jusqu'aux graphes ONNX optimisés et quantifiés dynamiquement en INT8.

> [!IMPORTANT]
> **Périmètre matériel et absence de mesures navigateur/WASM** : Toutes les mesures de ce rapport ont été effectuées en natif Linux x86\_64 sur un processeur **Intel Core i7-3770 @ 3.40 GHz** (Ivy Bridge, 4 cœurs physiques / 8 threads logiques, extensions `sse4_1`, `sse4_2`, `avx`, `f16c` — **sans AVX2, FMA ni AVX-512 VNNI**). **Aucune mesure WASM/navigateur n'est incluse** ; il ne faut pas extrapoler linéairement ces temps natifs vers `onnxruntime-web` (WASM mono-thread sans accélération `MatMulInteger` native VNNI/AVX2).

### Environnement logiciel et checkpoint de référence

- **Branche / worktree** : `exp/e03-onnx-perf` (basée sur `dcc4ab2`), sans modification du cœur modèle ni de l'export.
- **Environnement Python** : `/home/yoch/voice-tts/env/bin/python` (`Python 3.12.3`, `torch==2.15.0+cpu`, `onnx==1.23.2`, `onnxruntime==1.32.0`, `CPUExecutionProvider`).
- **Checkpoint français** : `/home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt`
  - **SHA-256** : `f673dd421478b86e36f60a4965818eba304eb8517fc15de8ba0690d60ca0afa2`
  - **Step** : `1317` (format `4`, frontend `espeak` / `libespeak-ng`, voix `F1` et `M1`, `vocab_size=54`, `sample_rate=24000`, `hop=256`).
  - **Paramètres d'inférence** : `4,533,067` paramètres (`18,132,268` octets = `18.132 MB` en FP32), dont :
    - `4,353,088` poids (`96.03%`) dans les 50 couches `nn.Linear` (`MatMul` avec initialiseur),
    - `4,325,376` poids (`95.42%`) dans les 44 couches `nn.Linear` des 3 piles Transformer (`encoder`, `prosody_stack`, `acoustic_stack`),
    - `1,966,080` poids (`43.37%`) dans les 20 couches `nn.Linear` de `acoustic_stack` seul,
    - `27,712` poids (`0.61%`) dans les 6 petites têtes linéaires (`duration.out`, `frame_position`, `prosody_head`, `prosody_projection`, `vocoder.harmonic`, `vocoder.noise`).
  - **Buffers précalculés ajoutés par `Exportable`** ([export.py](../../src/tiny_voca/export.py)) : `2,868,224` octets (`2.868 MB`), dont `2,636,800` octets (`2.637 MB`) pour les noyaux d'analyse/synthèse STFT (`synth_*`, `noise_*`), `230,400` octets (`0.230 MB`) pour les matrices cepstrales (`harmonic_*`, `noise_*`) et `1,024` octets pour `ramp`.

### Protocole de benchmark reproductible

Script reproductible : [benchmark_onnx.py](../../scripts/benchmark_onnx.py) (résultats complets enregistrés dans `/home/yoch/voice-tts/.local-runs/e03/benchmark_results.json`).

- **Corpus de test français (3 longueurs, 0 phonème inconnu)** :
  - `short` (**19 tokens**, `163` trames / `1.739 s` sur `F1`, `212` trames / `2.261 s` sur `M1`) : *"Bonjour, comment allez-vous ?"*
  - `medium` (**60 tokens**, `377` trames / `4.021 s` sur `F1`, `406` trames / `4.331 s` sur `M1`) : *"As-tu vraiment payé quarante-deux euros et cinquante centimes pour ce beau livre ?"*
  - `long` (**132 tokens**, `855` trames / `9.120 s` sur `F1`, `829` trames / `8.843 s` sur `M1`) : *"Après un long voyage à travers les montagnes enneigées, les voyageurs fatigués atteignirent enfin le petit village où ils furent chaleureusement accueillis avec un repas chaud."*
- **Échauffement et répétitions** : `6` runs d'échauffement (`>= 5`) suivis de `35` exécutions chronométrées (`>= 30`) par configuration `(variante, threads, phrase)`, mesurées en **1 thread** et **4 threads** (`OMP/MKL/OPENBLAS_NUM_THREADS` + `intra_op_num_threads` fixés dans chaque sous-processus, `inter_op_num_threads=1`).
- **Isolation RSS** : chaque couple `(variante, threads, phrase)` s'exécute dans un sous-processus Python isolé (`--worker-json`) afin que `VmHWM` (`/proc/self/status`) et `ru_maxrss` (`resource.getrusage`) ne soient contaminés par aucune autre variante.
- **Note sur la charge hôte** : machine de travail partagée (`loadavg_start = 2.48`, `loadavg_end = 6.08` sur la session complète ; `e2e_loadavg = 4.01 -> 4.99` sur la passe E2E dédiée). Des réveils ponctuels de processus tiers expliquent certains pics de `p95` / `iqr` sur `4 threads` ; les valeurs `min` et le profilage noyau ORT sont rapportés en complément pour distinguer le coût intrinsèque de l'ordonnancement hôte.

---

## 2. Hypothèses falsifiables

1. **H1 (Répartition du coût par bloc)** : Les piles Transformer trame-par-trame (`predict_prosody` + `decode`) représentent > 60 % du temps PyTorch FP32, mais la réécriture ONNX (`Exportable`) augmente sensiblement la part du vocoder en remplaçant `torch.stft`/`torch.istft` (FFT) par des convolutions denses (`Conv1d`/`ConvTranspose1d`).
2. **H2 (Quantification INT8 sélective et fidélité)** : En ciblant uniquement les `MatMul` des projections linéaires neuronales (en excluant les 4 `MatMul` cepstraux `/MatMul`..`/MatMul_3`, `Sin`/`Cos`/`Exp`, et `Conv`/`ConvTranspose`), `quantize_dynamic` divise par ~2.5 la taille du `.onnx` (~21.4 MB → ~8.5 MB). Toutefois, quantifier `encoder` et `prosody_stack` perturbera les durées discrètes et la phase cumulée du F0, tandis que quantifier uniquement `acoustic_stack` préservera exactement les durées et maintiendra l'erreur RMS de forme d'onde sous 1 %.
3. **H3 (Accélération sur CPU Ivy Bridge sans AVX2/VNNI)** : En l'absence d'instructions VNNI/AVX2, la quantification dynamique INT8 (`DynamicQuantizeLinear` + `MatMulInteger`) apportera un gain de vitesse modéré (plafonné autour de 1.1×–1.4×) en raison du coût de quantification dynamique des activations et de la part croissante des opérateurs restés en FP32 (attention $O(T^2)$ et convolutions STFT).

---

## 3. Audit du graphe ONNX et ciblage de la quantification

L'inspection du graphe exporté `tiny-voca-fp32-extnoise.onnx` (`2,338` nœuds au total) recense exactement :

| Opérateur | Total | Sous-ensemble et rôle | Ciblé INT8 ? |
|---|---:|---|---|
| `MatMul` | 76 | **50** projections `nn.Linear` avec initialiseur (`encoder`: 16, `duration.out`: 1, `frame_position`: 1, `prosody_stack`: 8, `prosody_head`: 1, `prosody_projection`: 1, `acoustic_stack`: 20, `harmonic`: 1, `noise`: 1) | **Oui** (selon variante) |
| `MatMul` | — | **22** produits dynamiques d'attention (`Q @ K^T` et `Attn @ V`, sans poids constant) | **Non** (exclu) |
| `MatMul` | — | **4** projections cepstre → spectre DFT (`/MatMul`, `/MatMul_1`, `/MatMul_2`, `/MatMul_3` dans `_response`, alimentées par `Constant`) | **Non** (exclu) |
| `Gemm` | 0 | Absent du graphe TorchScript exporté | — |
| `Conv` | 17 | **13** convolutions neurales (`depthwise` × 11, `duration` × 2) + **4** analyses STFT (`/Conv`..`/Conv_3`) | **Non** (exclu) |
| `ConvTranspose` | 4 | **4** synthèses iSTFT (`/ConvTranspose`..`/ConvTranspose_3`) | **Non** (exclu) |
| `Sin` / `Cos` / `Exp` | 7 / 6 / 5 | Rotary embeddings (`encoder`/`prosody`/`acoustic`), phase `_pulses` et réponse cepstrale `_response` | **Non** (exclu) |

> [!NOTE]
> `onnxruntime.quantization.quant_pre_process` échoue sur ce graphe avec `Incomplete symbolic shape inference` (à cause de l'expansion dynamique `total = ends[0, -1]; torch.arange(total)`). Le ciblage explicite via `nodes_to_quantize` après audit du graphe garantit qu'aucun opérateur DSP (`/MatMul`..`/MatMul_3`, `Conv`, `ConvTranspose`, `Sin`, `Cos`, `Exp`) n'est modifié.

Quatre variantes INT8 (`QuantType.QInt8`, `reduce_range=True` pour éviter la saturation `vpmaddubsw` sur x86 sans VNNI) ont été générées et vérifiées :
- `tiny-voca-int8-linear.onnx` : les **50** projections `nn.Linear` (`per_channel=False`).
- `tiny-voca-int8-linear-pc.onnx` : les **50** projections `nn.Linear` (`per_channel=True`).
- `tiny-voca-int8-transformer.onnx` : les **44** projections `nn.Linear` des 3 piles Transformer (`95.42%` des poids ; les 6 têtes scalaires/cepstrales restent en FP32).
- `tiny-voca-int8-acoustic.onnx` : les **20** projections `nn.Linear` de `acoustic_stack` seul (`43.37%` des poids ; `encoder`, `prosody_stack`, têtes et vocoder restent en FP32).

---

## 4. Mesure séparée du front-end (`phonemize`)

Le front-end français (`tiny_voca.text.phonemize` via `libespeak-ng`) est exécuté hors graphe ONNX, une fois par phrase :

| Phrase | Tokens | Inconnus | Médiane (ms) | p95 (ms) | Min (ms) | `VmHWM` isolé (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `short` | 19 | 0 | 0.367 | 0.415 | 0.344 | 300.52 |
| `medium` | 60 | 0 | 0.434 | 0.549 | 0.409 | 300.38 |
| `long` | 132 | 0 | 1.292 | 2.063 | 0.907 | 300.49 |

Le front-end natif (`libespeak-ng`) coûte **0.37 à 1.29 ms** (< 0.6 % de la latence du modèle acoustique). (Note : dans la démo web, c'est Pyodide + dizione en WASM qui assure cette étape, non mesuré ici.)

---

## 5. Profilage interne PyTorch FP32 par bloc et profilage ORT

### 5.1 Décomposition PyTorch FP32 : `TinyVoca` natif vs `Exportable` (`external_noise=True`)

Temps médians (ms) et part relative (%) sur `35` exécutions (`warmup=6`, voix `F1`) :

#### 1 thread (`intra=1`)

| Bloc | `short` Natif | `short` Exportable | `medium` Natif | `medium` Exportable | `long` Natif | `long` Exportable |
|---|---:|---:|---:|---:|---:|---:|
| `encode` (Text Encoder) | 8.46 (8.2%) | 8.52 (7.8%) | 15.97 (6.2%) | 15.80 (6.1%) | 31.48 (4.1%) | 31.10 (4.1%) |
| `predict_durations` | 0.99 (1.0%) | 0.90 (0.8%) | 1.45 (0.6%) | 1.40 (0.5%) | 2.33 (0.3%) | 2.35 (0.3%) |
| `expand` | 0.77 (0.8%) | 0.58 (0.5%) | 1.12 (0.4%) | 0.96 (0.4%) | 2.01 (0.3%) | 2.00 (0.3%) |
| `predict_prosody` (Stack A) | 20.69 (20.0%) | 19.52 (17.9%) | 50.97 (19.9%) | 51.12 (19.6%) | 166.40 (21.9%) | 162.53 (21.3%) |
| `decode` (Stack B) | 48.14 (46.5%) | 46.80 (43.0%) | 125.94 (49.2%) | 125.66 (48.2%) | 416.48 (54.7%) | 403.53 (52.8%) |
| `SourceFilter`: `cepstrum_response` / `_response` | 4.56 (4.4%) | 4.88 (4.5%) | 10.19 (4.0%) | 11.27 (4.3%) | 24.76 (3.2%) | 29.09 (3.8%) |
| `SourceFilter`: `pulse_train` / `_pulses` | 6.28 (6.1%) | 2.30 (2.1%) | 17.98 (7.0%) | 4.83 (1.9%) | 44.88 (5.9%) | 12.49 (1.6%) |
| `SourceFilter`: `_filter` harmonique | 5.08 (4.9%) | 16.23 (14.9%) | 12.69 (5.0%) | 28.22 (10.8%) | 27.91 (3.7%) | 66.65 (8.7%) |
| `SourceFilter`: `_filter` bruit | 8.45 (8.2%) | 9.09 (8.3%) | 19.53 (7.6%) | 21.31 (8.2%) | 44.53 (5.8%) | 53.95 (7.1%) |
| **Total `_filter` (H + B)** | **13.53 (13.1%)** | **25.31 (23.3%)** | **32.22 (12.6%)** | **49.52 (19.0%)** | **72.44 (9.5%)** | **120.60 (15.8%)** |
| **Somme des blocs (ms)** | **103.44** | **108.81** | **255.82** | **260.57** | **760.79** | **763.68** |

#### 4 threads (`intra=4`)

| Bloc | `short` Natif | `short` Exportable | `medium` Natif | `medium` Exportable | `long` Natif | `long` Exportable |
|---|---:|---:|---:|---:|---:|---:|
| `encode` (Text Encoder) | 6.24 (11.2%) | 5.84 (10.7%) | 9.90 (8.8%) | 9.28 (8.4%) | 22.28 (6.3%) | 15.21 (5.4%) |
| `predict_durations` | 0.78 (1.4%) | 0.71 (1.3%) | 1.10 (1.0%) | 1.04 (0.9%) | 1.81 (0.5%) | 1.42 (0.5%) |
| `expand` | 0.74 (1.3%) | 0.59 (1.1%) | 0.95 (0.8%) | 0.84 (0.8%) | 2.09 (0.6%) | 1.38 (0.5%) |
| `predict_prosody` (Stack A) | 9.92 (17.8%) | 9.92 (18.2%) | 20.81 (18.4%) | 19.33 (17.6%) | 72.01 (20.4%) | 53.69 (19.1%) |
| `decode` (Stack B) | 23.34 (41.8%) | 22.31 (41.0%) | 48.94 (43.4%) | 47.87 (43.6%) | 178.66 (50.6%) | 136.72 (48.6%) |
| `SourceFilter`: `cepstrum_response` / `_response` | 2.73 (4.9%) | 2.37 (4.4%) | 5.15 (4.6%) | 5.02 (4.6%) | 12.35 (3.5%) | 10.89 (3.9%) |
| `SourceFilter`: `pulse_train` / `_pulses` | 3.57 (6.4%) | 1.59 (2.9%) | 7.84 (6.9%) | 2.54 (2.3%) | 23.40 (6.6%) | 4.60 (1.6%) |
| `SourceFilter`: `_filter` harmonique | 2.72 (4.9%) | 6.54 (12.0%) | 5.24 (4.6%) | 13.22 (12.0%) | 10.80 (3.1%) | 30.93 (11.0%) |
| `SourceFilter`: `_filter` bruit | 5.83 (10.4%) | 4.50 (8.3%) | 12.97 (11.5%) | 10.70 (9.7%) | 29.85 (8.4%) | 26.73 (9.5%) |
| **Total `_filter` (H + B)** | **8.55 (15.3%)** | **11.04 (20.3%)** | **18.21 (16.1%)** | **23.92 (21.8%)** | **40.64 (11.5%)** | **57.66 (20.5%)** |
| **Somme des blocs (ms)** | **55.87** | **54.36** | **112.90** | **109.84** | **353.24** | **281.57** |

**Enseignements clés PyTorch :**
- `predict_prosody` + `decode` comptent ensemble pour **59.2%–76.6%** du temps total selon la longueur (`short` → `long`, croissance quadratique de la self-attention sur $T$ trames). `predict_durations` + `expand` pèsent **< 2.7%** (1.0%–1.8% en 1T).
- L'oscillateur vectorisé en float32 `Exportable._pulses` est **3.6×–5.1× plus rapide** (`12.49 ms` à 1T, `4.60 ms` à 4T sur `long`) que `pulse_train` en float64 (`44.88 ms` / `23.40 ms`).
- À l'inverse, remplacer `torch.stft`/`torch.istft` par des `Conv1d`/`ConvTranspose1d` denses dans `Exportable._filter` multiplie par **1.54×–1.87×** (1T) et **1.29×–1.42×** (4T) le coût du filtrage, portant la part de `_filter` à **15.8%–23.3%** en 1 thread et **20.3%–21.8%** en 4 threads.

### 5.2 Profilage par groupe de nœuds dans ONNX Runtime (`ORT_ENABLE_ALL`, `tiny-voca-fp32.onnx`)

Moyenne sur 15 exécutions profilées après 5 runs d'échauffement filtrés via `model_run` (note : l'instrumentation `enable_profiling` ajoute un léger surcoût par nœud) :

| Groupe de nœuds ORT | 1T `short` | 1T `medium` | 1T `long` | 4T `short` | 4T `medium` | 4T `long` |
|---|---:|---:|---:|---:|---:|---:|
| `neural_linear_matmul` (`qkv/out/expand/proj/têtes`) | 24.26 ms (33.1%) | 52.89 ms (27.7%) | 121.91 ms (19.1%) | 10.38 ms (22.1%) | 18.25 ms (18.0%) | 42.40 ms (11.9%) |
| `attention_qk_softmax` (`Q@K^T`, `Softmax`, `Attn@V`) | 13.01 ms (17.8%) | 44.50 ms (23.3%) | 214.24 ms (33.6%) | 13.46 ms (28.6%) | 26.66 ms (26.4%) | 111.33 ms (31.3%) |
| **`attention_isnan_where`** (`IsNaN`+`Where` du masque SDPA) | **1.56 ms (2.1%)** | **17.71 ms (9.3%)** | **116.26 ms (18.2%)** | **1.99 ms (4.2%)** | **17.66 ms (17.5%)** | **110.33 ms (31.1%)** |
| **`vocoder_stft_conv`** (8 `Conv`/`ConvTranspose` STFT/iSTFT) | **14.96 ms (20.4%)** | **32.84 ms (17.2%)** | **80.66 ms (12.7%)** | **6.69 ms (14.2%)** | **13.67 ms (13.5%)** | **33.73 ms (9.5%)** |
| `neural_norm_gelu_rotary_misc` | 10.08 ms (13.8%) | 23.59 ms (12.4%) | 55.25 ms (8.7%) | 5.61 ms (11.9%) | 10.47 ms (10.4%) | 21.32 ms (6.0%) |
| `vocoder_pulses_and_top_level_ops` | 5.55 ms (7.6%) | 11.56 ms (6.1%) | 32.10 ms (5.0%) | 6.33 ms (13.5%) | 9.49 ms (9.4%) | 24.88 ms (7.0%) |
| `vocoder_response_cepstrum` (`/MatMul`..`_3`, `Exp`, `Sin`, `Cos`) | 2.44 ms (3.3%) | 5.28 ms (2.8%) | 12.58 ms (2.0%) | 1.78 ms (3.8%) | 3.71 ms (3.7%) | 8.91 ms (2.5%) |
| `neural_depthwise_predictor_conv` (13 convs neurales) | 1.29 ms (1.8%) | 2.29 ms (1.2%) | 4.39 ms (0.7%) | 0.75 ms (1.6%) | 1.04 ms (1.0%) | 2.15 ms (0.6%) |
| `vocoder_stft_pad` (`/Pad`, `/Pad_1`) | 0.09 ms (0.1%) | 0.14 ms (0.1%) | 0.34 ms (0.1%) | 0.08 ms (0.2%) | 0.15 ms (0.1%) | 0.34 ms (0.1%) |
| **Total noyaux ORT (`kern_sum`)** | **73.25 ms** | **190.80 ms** | **637.73 ms** | **47.07 ms** | **101.12 ms** | **355.38 ms** |

> [!WARNING]
> **Goulot majeur inattendu découvert par le profilage ORT (`attention_isnan_where`)** : Dans [model.py](../../src/tiny_voca/model.py#L61), `F.scaled_dot_product_attention(..., attn_mask=mask[:, None, None, :])` est exporté en ONNX par TorchScript avec une garde `IsNaN` + `Where` sur l'intégralité des matrices d'attention `[1, 4, T, T]`. Or, dans `Exportable` (`B=1`, sans padding), `token_mask` et `frame_mask` valent partout `True` ! Ces nœuds redondants consomment **18.2 %** (`116.26 ms` en 1T) à **31.1 %** (`110.33 ms` en 4T) du temps noyau ORT sur phrase longue.

Détail des 8 convolutions STFT/iSTFT dans ORT (`medium` = 377 trames, `long` = 855 trames) :
- `/ConvTranspose` (iSTFT harmonique `514→1, k=512, s=128`) est le nœud STFT le plus coûteux : `9.59 ms` (1T) / `3.90 ms` (4T) sur `medium`, et `25.34 ms` (1T) / `9.04 ms` (4T) sur `long`.
- `/ConvTranspose_2` (iSTFT bruit `258→1, k=256, s=64`) : `5.66 ms` (1T) / `2.57 ms` (4T) sur `medium` ; `13.34 ms` (1T) / `6.44 ms` (4T) sur `long`.
- `/Conv` + `/Conv_1` (analyse STFT harmonique réel+imag) : `9.94 ms` (1T) / `3.18 ms` (4T) sur `medium` ; `23.39 ms` (1T) / `7.93 ms` (4T) sur `long`.
- `/Conv_2` + `/Conv_3` (analyse STFT bruit réel+imag) : `4.94 ms` (1T) / `1.73 ms` (4T) sur `medium` ; `11.67 ms` (1T) / `4.24 ms` (4T) sur `long`.

---

## 6. Benchmarks end-to-end (hors front-end) : 1 thread et 4 threads

### 6.1 Résultats à 1 thread (`intra=1`, 35 runs après 6 warmup, sous-processus isolés)

| Variante | `short` Méd / p95 (ms) | `short` RTFx | `short` RSS (MiB) | `medium` Méd / p95 (ms) | `medium` RTFx | `medium` RSS (MiB) | `long` Méd / p95 (ms) | `long` RTFx | `long` RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_fp32_synthesize` | 125.25 / 195.43 | 13.9x | 362.97 | 338.23 / 496.43 | 11.9x | 371.88 | 1063.03 / 1393.22 | 8.6x | 404.37 |
| `torch_fp32_exportable` | 111.71 / 134.23 | 15.6x | 367.95 | 269.00 / 294.70 | 14.9x | 379.46 | 797.08 / 925.96 | 11.4x | 407.39 |
| `ort_fp32_rel_opt_all` (`ext=False`) | **61.91** / 93.49 | **28.1x** | 356.86 | **187.05** / 257.01 | **21.5x** | 374.46 | 859.57 / 1189.29 | 10.6x | 449.55 |
| `ort_fp32_ext_opt_off` (`DISABLE_ALL`) | 78.09 / 178.81 | 22.3x | 362.95 | 249.44 / 463.29 | 16.1x | 380.16 | 785.50 / 1036.92 | 11.6x | 443.41 |
| `ort_fp32_ext_opt_basic` (`ENABLE_BASIC`) | 77.70 / 110.50 | 22.4x | 362.11 | 226.59 / 324.63 | 17.8x | 378.96 | 772.71 / 1176.09 | 11.8x | 441.98 |
| `ort_fp32_ext_opt_all` (`ENABLE_ALL`) | 75.97 / 98.63 | 22.9x | 361.07 | 257.28 / 498.99 | 15.6x | 378.83 | 753.02 / 907.55 | 12.1x | 438.53 |
| `ort_int8_linear` (50 `Linear`) | 73.40 / 92.79 | 23.7x | 341.30 | **199.30** / 221.19 | **20.1x** | **352.02** | 669.72 / 786.33 | 13.5x | **425.25** |
| `ort_int8_linear_pc` (50 `Linear` per-ch) | 74.49 / 87.28 | 23.3x | 341.34 | 205.76 / 270.04 | 19.5x | 352.55 | 673.68 / 772.86 | 13.4x | 425.15 |
| `ort_int8_transformer` (44 Xfmr) | **68.95** / 72.33 | **25.2x** | **341.25** | 203.08 / 243.82 | 19.9x | 352.64 | 692.51 / 910.43 | 13.1x | 426.08 |
| `ort_int8_acoustic` (20 `acoustic_stack`) | 74.77 / 83.47 | 23.2x | 352.29 | 207.42 / 268.42 | 19.4x | 370.00 | **667.27** / 830.52 | **13.7x** | 436.25 |

*(Valeurs `min` en 1T pour le contexte de variance hôte : sur `medium`, `ort_fp32_ext_opt_all` atteint `202.40 ms` vs `187.70 ms` pour `ort_int8_linear` ; sur `long`, `674.14 ms` vs `628.28 ms` pour `ort_int8_acoustic`.)*

### 6.2 Résultats à 4 threads (`intra=4`, 35 runs après 6 warmup, sous-processus isolés)

| Variante | `short` Méd / p95 (ms) | `short` RTFx | `short` RSS (MiB) | `medium` Méd / p95 (ms) | `medium` RTFx | `medium` RSS (MiB) | `long` Méd / p95 (ms) | `long` RTFx | `long` RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_fp32_synthesize` | 83.90 / 150.26 | 20.7x | 366.16 | 156.94 / 295.44 | 25.6x | 374.64 | 456.98 / 884.90 | 20.0x | 406.87 |
| `torch_fp32_exportable` | 61.74 / 87.66 | 28.2x | 373.97 | **102.58** / 183.87 | **39.2x** | 378.95 | **280.42** / 374.84 | **32.5x** | 414.77 |
| `ort_fp32_rel_opt_all` (`ext=False`) | 48.31 / 68.86 | 36.0x | 357.55 | 144.12 / 226.87 | 27.9x | 374.20 | 468.73 / 650.89 | 19.5x | 449.57 |
| `ort_fp32_ext_opt_off` (`DISABLE_ALL`) | 43.44 / 89.78 | 40.0x | 362.96 | 128.01 / 212.67 | 31.4x | 378.45 | 488.79 / 909.85 | 18.7x | 443.30 |
| `ort_fp32_ext_opt_basic` (`ENABLE_BASIC`) | 39.10 / 60.71 | 44.5x | 361.23 | 136.12 / 183.79 | 29.5x | 376.50 | 471.46 / 705.43 | 19.3x | 441.86 |
| `ort_fp32_ext_opt_all` (`ENABLE_ALL`) | 35.18 / 46.08 | 49.4x | 360.12 | 182.65 / 266.88 | 22.0x | 377.93 | 533.16 / 838.68 | 17.1x | 440.46 |
| `ort_int8_linear` (50 `Linear`) | 48.55 / 88.00 | 35.8x | 341.98 | **103.03** / 139.66 | **38.9x** | **351.32** | 386.51 / 489.36 | 23.5x | **425.93** |
| `ort_int8_linear_pc` (50 `Linear` per-ch) | 40.10 / 58.98 | 43.4x | 342.29 | 127.89 / 183.93 | 31.4x | 351.54 | **354.41** / 469.75 | **25.6x** | 426.10 |
| `ort_int8_transformer` (44 Xfmr) | **32.61** / 44.78 | **53.3x** | **341.94** | 113.44 / 145.28 | 35.6x | 351.48 | 443.53 / 701.17 | 20.5x | 427.21 |
| `ort_int8_acoustic` (20 `acoustic_stack`) | 38.30 / 48.70 | 45.4x | 353.33 | 112.10 / 142.10 | 35.9x | 368.38 | 380.00 / 588.01 | 24.0x | 437.31 |

**Analyse des performances et de la mémoire :**
1. **Niveaux d'optimisation ORT (`DISABLE_ALL` vs `ENABLE_BASIC` vs `ENABLE_ALL`)** : En 1 thread, `ORT_ENABLE_ALL` améliore la latence sur `short`/`long` (`78.09 → 75.97 ms` et `785.50 → 753.02 ms`). En 4 threads sur cette passe, `ort_fp32_ext_opt_all` a subi une contention ponctuelle sur `medium`/`long` (`min` = `144.85 / 345.52 ms`, `iqr` = `47.61 / 137.34 ms`, alors que le profilage `ort_prof` mesurait `109.22 / 365.26 ms` malgré l'instrumentation). Dans l'ensemble, les fusions ORT (`BiasGelu`, `FusedMatMul`) n'apportent qu'un gain modeste (~5–15 %) car elles n'éliminent ni les `Conv`/`ConvTranspose` STFT ni le motif `IsNaN`+`Where` issu de `scaled_dot_product_attention`.
2. **Impact de la quantification INT8 sur la vitesse et le RSS** :
   - Sur `1 thread`, les variantes INT8 offrent un speedup régulier de **1.09×–1.29×** sur `medium` et `long` par rapport à `ort_fp32_ext_opt_all` (`199–207 ms` vs `257 ms` sur `medium` ; `667–693 ms` vs `753 ms` sur `long`, soit **1.54×–1.59×** plus rapide que `torch_fp32_synthesize`).
   - Sur `4 threads`, l'INT8 surpasse nettement les runs ORT FP32 sur `medium`/`long` (`354–387 ms` vs `469–533 ms`), atteignant jusqu'à **53.3×** temps réel sur `short` (`ort_int8_transformer`) et **45.4× / 35.9× / 24.0×** avec `ort_int8_acoustic` (`min` = `30.44 / 94.13 / 313.57 ms`).
   - Le gain RSS pic (`VmHWM`) est systématique et parfaitement reproductible : **−19.8 MiB** sur `short`, **−26.8 MiB** sur `medium` et **−13.3 MiB** sur `long` pour `ort_int8_linear` / `ort_int8_transformer` par rapport à `ort_fp32_ext_opt_all` (`425.25 MiB` vs `438.53 MiB` sur `long`). On note aussi qu'ORT FP32 consomme ~34 MiB de RSS pic en plus que PyTorch FP32 sur `long` (`438.5–449.6 MiB` vs `404.4–406.9 MiB`) du fait de l'arène mémoire des tenseurs intermédiaires et du bruit `[1, 768000]`.

---

## 7. Inventaire des modèles candidats (tailles et checksums SHA-256)

Tous les modèles candidats sont stockés dans `/home/yoch/voice-tts/.local-runs/e03/` (non suivis par git) :

| Fichier `.onnx` | Taille (octets) | Taille (MB / MiB) | SHA-256 | Nœuds quantifiés (`MatMulInteger` / `DQL`) | Nœuds protégés intacts (`Conv`/`ConvT`/`Sin`/`Cos`/`Exp`/`MatMul` FP32) |
|---|---:|---:|---|---|---|
| `tiny-voca-fp32.onnx` | 21 369 430 | 21.369 / 20.380 | `512ad32fa1d6bd72f7ed06f34e0e50e002fec3641492a8a8b9afa765e8b46881` | 0 / 0 | 17 / 4 / 7 / 6 / 5 / 76 |
| `tiny-voca-fp32-extnoise.onnx` | 21 369 917 | 21.370 / 20.380 | `1b3500d638be2e2384fe80375c429bc182e33de87deeca8515e8b0e6ae295b00` | 0 / 0 | 17 / 4 / 7 / 6 / 5 / 76 |
| `tiny-voca-int8-linear.onnx` | 8 535 143 | 8.535 / 8.140 | `8df6fc48a954c09bdf7caee34ff7e6b05fb774b6992df6fdb638f88cae83605f` | 50 / 49 | 17 / 4 / 7 / 6 / 5 / 26 |
| `tiny-voca-int8-linear-pc.onnx` | 8 625 989 | 8.626 / 8.226 | `d7c29a89b4f63545104b3b57819ddff1e9717c13d5c4024222875fa69746bce2` | 50 / 49 | 17 / 4 / 7 / 6 / 5 / 26 |
| `tiny-voca-int8-transformer.onnx` | 8 612 940 | 8.613 / 8.214 | `260c85c137def19c2e7af867a72ed1bf7362f114e577cacb4324a62b7b3a91d3` | 44 / 44 | 17 / 4 / 7 / 6 / 5 / 32 |
| `tiny-voca-int8-acoustic.onnx` | 15 662 148 | 15.662 / 14.937 | `fe3e2c12aa5a0f29cd29f9fe008cda4ac6b5b94d9d78e5e1a1bdcba3d6f5209b` | 20 / 20 | 17 / 4 / 7 / 6 / 5 / 56 |

*(Rappel : `tiny-voca-int8-linear.onnx` réduit la taille de **60.1%** par rapport au FP32, mais les `2.637 MB` de noyaux de convolution STFT/iSTFT non quantifiables représentent alors **30.9%** du fichier total.)*

---

## 8. Analyse fidélité numérique et acoustique vs PyTorch FP32 (`external_noise=True`)

Comparaison bit-à-bit / échantillon-par-échantillon contre `TinyVoca.synthesize` FP32 avec le **même flux de bruit blanc** (`seed=0`) sur les 6 énoncés (`short`, `medium`, `long` × `F1`, `M1`). Tous les candidats produisent **100 % d'échantillons finis** (`NaN=0`, `Inf=0`).

| Candidat | Phrase (Voix) | Durées identiques (`L1` trames) | Δ Longueur (trames) | Ratio RMS | Err. RMS rel. | Err. max / crête réf. | `mel_dtw` (L1 log-mel) |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_fp32_exportable` | `short` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 1.0000 / 1.0000 | 0.00051 / 0.00006 | 0.00261 / 0.00020 | 0.0004 / 0.0000 |
| `torch_fp32_exportable` | `medium` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 1.0000 / 1.0000 | 0.00014 / 0.00008 | 0.00127 / 0.00025 | 0.0002 / 0.0001 |
| `torch_fp32_exportable` | `long` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 1.0000 / 1.0000 | 0.00015 / 0.00032 | 0.00096 / 0.00371 | 0.0002 / 0.0001 |
| `ort_fp32_ext_opt_off` | `short` / `medium` / `long` | Oui (0) sur 6/6 | 0 | 1.0000 | 0.00019–0.00088 | 0.00077–0.00871 | 0.0002–0.0008 |
| `ort_fp32_ext_opt_basic` | `short` / `medium` / `long` | Oui (0) sur 6/6 | 0 | 1.0000 | 0.00019–0.00088 | 0.00077–0.00871 | 0.0002–0.0008 |
| `ort_fp32_ext_opt_all` | `short` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 1.0000 / 1.0000 | 0.00237 / 0.00018 | 0.01237 / 0.00060 | 0.0013 / 0.0002 |
| `ort_fp32_ext_opt_all` | `medium` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 0.9999 / 1.0000 | 0.00302 / 0.00036 | 0.04091 / 0.00176 | 0.0009 / 0.0003 |
| `ort_fp32_ext_opt_all` | `long` (F1 / M1) | Oui (0) / Oui (0) | 0 / 0 | 1.0000 / 1.0002 | 0.00031 / 0.00415 | 0.00144 / 0.04584 | 0.0003 / 0.0005 |
| `ort_int8_linear` | `short` (F1 / M1) | Oui (0) / Non (1) | 0 / +1 | 1.0080 / 1.0089 | 1.45281 / N/A | 1.84379 / N/A | 0.1557 / 0.1970 |
| `ort_int8_linear` | `medium` (F1 / M1) | Non (1) / Non (5) | −1 / −3 | 1.0073 / 1.0065 | N/A / N/A | N/A / N/A | 0.2100 / 0.2315 |
| `ort_int8_linear` | `long` (F1 / M1) | Non (7) / Non (8) | −5 / −8 | 1.0106 / 1.0096 | N/A / N/A | N/A / N/A | 0.2595 / 0.2497 |
| `ort_int8_linear_pc` | `short` (F1 / M1) | Oui (0) / Non (2) | 0 / +2 | 0.9974 / 0.9929 | 1.47298 / N/A | 1.27514 / N/A | 0.1442 / 0.2074 |
| `ort_int8_linear_pc` | `medium` (F1 / M1) | Non (1) / Non (1) | −1 / −1 | 0.9974 / 0.9973 | N/A / N/A | N/A / N/A | 0.2150 / 0.2108 |
| `ort_int8_linear_pc` | `long` (F1 / M1) | Non (8) / Non (10) | −6 / −8 | 0.9996 / 0.9976 | N/A / N/A | N/A / N/A | 0.2592 / 0.2470 |
| `ort_int8_transformer` | `short` (F1 / M1) | Oui (0) / Non (1) | 0 / +1 | 1.0007 / 0.9999 | 1.53392 / N/A | 1.84606 / N/A | 0.1347 / 0.1751 |
| `ort_int8_transformer` | `medium` (F1 / M1) | Non (2) / Non (3) | +2 / −3 | 0.9995 / 1.0033 | N/A / N/A | N/A / N/A | 0.2090 / 0.2202 |
| `ort_int8_transformer` | `long` (F1 / M1) | Non (2) / Non (4) | −2 / −4 | 1.0006 / 1.0034 | N/A / N/A | N/A / N/A | 0.2144 / 0.2265 |
| **`ort_int8_acoustic`** | `short` (F1 / M1) | **Oui (0) / Oui (0)** | **0 / 0** | 1.0014 / 1.0004 | **0.00594 / 0.00676** | **0.01323 / 0.01077** | **0.0069 / 0.0055** |
| **`ort_int8_acoustic`** | `medium` (F1 / M1) | **Oui (0) / Oui (0)** | **0 / 0** | 1.0007 / 0.9997 | **0.00622 / 0.00621** | **0.04225 / 0.00861** | **0.0062 / 0.0060** |
| **`ort_int8_acoustic`** | `long` (F1 / M1) | **Oui (0) / Oui (0)** | **0 / 0** | 1.0004 / 1.0015 | **0.00545 / 0.00751** | **0.00759 / 0.04222** | **0.0058 / 0.0062** |

*(À titre d'ordre de grandeur, `ort_fp32_rel_opt_all` avec bruit aléatoire indépendant présente `mel_dtw = 0.2229–0.2552` contre la référence `seed=0` uniquement à cause du tirage de bruit différent ; ses durées sont identiques à 6/6.)*

**Verdict fidélité vs vitesse :**
- **H2 confirmée de façon frappante** : Quantifier `encoder` (`ort_int8_linear`, `ort_int8_linear_pc`, `ort_int8_transformer`) déplace le seuil d'arrondi de `duration` de 1 à 10 trames (`L1`) sur 5 énoncés sur 6, rendant la comparaison échantillon-par-échantillon inapplicable (`N/A`), et quantifier `prosody_stack` décale légèrement le contour de `log_f0`, ce qui provoque une dérive de phase cumulée dans `_pulses` (`err_rms_rel ≈ 1.45–1.53` sur `short F1`, soit une décorrélation de phase de forme d'onde, et `mel_dtw = 0.135–0.260`).
- **Point optimal identifié (`tiny-voca-int8-acoustic.onnx`)** : En ne quantifiant que les 20 projections de `acoustic_stack` (43.37 % des poids), l'encodeur de durées, le prédicteur de prosodie (F0/voisement) et tout le vocoder restent en FP32. On obtient **100 % d'identité des durées (6/6)**, une erreur RMS relative de **0.55 %–0.75 %** (`max/peak <= 4.2%`) et un `mel_dtw` de **0.0055–0.0069**, indiscernables du FP32, tout en réduisant le fichier de **26.7 %** (`15.66 MB`) et en obtenant `13.7x–45.4x` temps réel.
- **Note `per_channel` vs `per_tensor`** : `ort_int8_linear_pc` corrige le léger biais d'énergie RMS (`0.993–1.000` vs `1.007–1.011` en `per_tensor`), mais ne supprime pas la dérive discrète de durée ni le coût sur les petites projections.

---

## 9. Recommandations chiffrées pour E04 (convolutions STFT et optimisations d'export)

Les profils PyTorch (`Exportable` vs `TinyVoca`) et ORT (`ort_node_profile`) fournissent des cibles quantitatives précises pour **E04** :

1. **Cible n°1 — Remplacer les 8 convolutions denses STFT/iSTFT (`_filter`) par une FFT/iFFT radix-2** :
   - **Coût temps actuel** : En PyTorch `Exportable`, `_filter` (harmonique + bruit) consomme **25.31–120.60 ms (15.8 %–23.3 %)** à 1 thread et **11.04–57.66 ms (20.3 %–21.8 %)** à 4 threads, soit **1.3×–1.9× plus lent** que `torch.stft`/`torch.istft` natif (`13.53–72.44 ms` 1T / `8.55–40.64 ms` 4T). Dans ONNX Runtime (`ORT_ENABLE_ALL`), ces 8 nœuds `Conv`/`ConvTranspose` représentent **14.96–80.66 ms (12.7 %–20.4 %)** en 1 thread et **6.69–33.73 ms (9.5 %–14.2 %)** en 4 threads (dont `/ConvTranspose` harmonique seul pèse `25.34 ms` en 1T / `9.04 ms` en 4T sur `long`).
   - **Coût stockage actuel** : Les noyaux STFT/iSTFT (`synth_analysis_*`, `synth_synthesis`, `noise_analysis_*`, `noise_synthesis`) occupent **2,636,800 octets (`2.637 MB`)** dans le graphe ONNX, soit **12.3 %** du modèle FP32 (`21.37 MB`) et **30.9 %** du modèle quantifié `tiny-voca-int8-linear.onnx` (`8.54 MB`).
   - **Gain attendu en E04** : Passer d'une DFT matricielle $O(N^2)$ (`Conv1d` 512→257 et 256→129) à une FFT radix-2 $O(N \log N)$ (soit ~10× moins d'opérations arithmétiques pour $N=512$, sans matrices de poids STFT) doit réduire l'empreinte binaire de **−2.64 MB** (ramenant le modèle FP32 vers `~18.7 MB` et l'INT8 complet vers `~5.9 MB`) et diviser par ~2 à ~3 le temps de filtrage vocoder (gain direct estimé de **−7 % à −13 %** de latence totale FP32 en natif, et proportionnellement plus élevé après levée du verrou SDPA ci-dessous).

2. **Cible n°2 (gain rapide complémentaire) — Supprimer le masque `attn_mask` redondant en inférence mono-énoncé (`B=1`)** :
   - `F.scaled_dot_product_attention` avec un masque booléen génère `IsNaN` + `Where` sur `[1, 4, T, T]`, coûtant **17.5 % (`medium` 4T) à 31.1 % (`long` 4T, `110.33 ms`)** du temps noyau ORT.
   - En supprimant `attn_mask` dans `Exportable` (`B=1`, aucune trame paddée), ce surcoût quadratique tombe à `0 ms`, faisant passer la part relative des convolutions STFT sur phrase longue de `9.5 %` à `~13.8 %` (4T) et `~15.5 %` (1T), ce qui renforce d'autant le retour sur investissement d'E04 sur les phrases longues.

3. **Validation tests unitaires après intégration** :
   - `PYTHONPATH=src:tests /home/yoch/voice-tts/env/bin/python -m unittest test_export test_model -v` → **17 tests exécutés, 17 OK** (`55.512 s`).
