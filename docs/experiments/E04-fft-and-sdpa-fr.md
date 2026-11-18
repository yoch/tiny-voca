# E04 — Remplacement des convolutions STFT par FFT C et suppression du masque SDPA `B=1`

- **Statut** : **MIXTE VALIDÉ PAR VOLET**
  - **Suppression du `attn_mask` redondant en `B=1` (`omit_mask=True` dans `Exportable`)** : **VALIDÉ POUR INTÉGRATION** (gain pur massif, parité bit-à-bit exacte `SNR = ∞ dB`, zéro dépendance externe).
  - **Moteur FFT C empaqueté réel (`scripts/e04/dsp.c` : `compact` / `fused`)** : **VALIDÉ NUMÉRIQUEMENT ET EN 1-THREAD** (`SNR = 123–130 dB`, `-15 %` à `-31 %` sur `dense_unmasked` à 1 thread, `-24 %` à `-43 %` vs baseline `dense_masked`, et `-40.3 %` taille totale avec INT8 sélectif `acoustic_stack`), **conservé expérimental/optionnel** car mono-thread (gain réduit/instable à 4 threads sous charge sur phrases longues) et nécessite un binaire natif/WASM compagnon.

> [!IMPORTANT]
> **Périmètre matériel et navigateur** : Toutes les mesures ci-dessous ont été exécutées sur **Intel Core i7-3770 @ 3.40 GHz** (SSE4.1, SSE4.2, AVX, F16C — **sans AVX2 ni VNNI**, comblant la lacune explicitement relevée dans la PR #1 `codex/e04-fft-vocoder` d'`yoch/tiny-voca`). **Aucune mesure WASM/navigateur locale** n'a été conduite sur cet hôte (`clang` cible `wasm32` présent, mais `wasm-ld`/`ziglang` et `onnxruntime-web` absents hors-ligne) ; les données EPYC + Chromium WASM antérieures proviennent de `origin/codex/e04-fft-vocoder` et ne doivent pas être extrapolées sans test dédié.

---

## 1. Investigation initiale et acquis de `yoch/tiny-voca` (`codex/e04-fft-vocoder`)

### 1.1 Décomposition théorique vs mesurée sur le modèle français (`opt1-1117`, `step=1317`)
- Sur le graphe ONNX de référence français (`dense-masked.onnx`, `21 369 917 B`, `2 338` nœuds, SHA-256 `b85edda0b26dcdd1e4e3e27a08b31036f43f4d2de77aaadde1a353c6690b319b`) :
  - `8` opérateurs `Conv`/`ConvTranspose` pour STFT/iSTFT (`2 636 800 B` de poids fixes) : harmonique `N=512, hop=128`, bruit `N=256, hop=64`.
  - `4` `MatMul` + `Exp`/`Sin`/`Cos` cepstraux (`230 400 B` de matrices DFT).
  - **`11` `IsNaN` + `22` `Where` SDPA** générés par `F.scaled_dot_product_attention(..., attn_mask=mask[:, None, None, :])` bien que `token_mask` et `frame_mask` valent partout `True` en `B=1`.

### 1.2 État E04-A issu de `origin/codex/e04-fft-vocoder`
La PR #1 (`codex/e04-fft-vocoder`, commits `60b0a5c`, `ad71242`, `6ddf59e`) sur `yoch/tiny-voca` a déjà établi deux faits importants (vérifiés sur `README.md` et `e04b/REPORT.md`) :
1. **Les opérateurs ONNX standard (`DFT` opsets 17/20 et graphes `radix2` purs) régressent** sur CPU (`29.62 ms` dense → `56.66 ms` DFT / `275.65 ms` radix2 sur filtre double, `78.41 → 98.55 / 155.10 ms` E2E sur Core i7-3770) ainsi que sur EPYC AVX2 et Chromium WASM : cette voie est donc **abandonnée conformément au critère d'arrêt d'E04**.
2. **Un noyau C dédié FFT réel empaqueté (`e04b/dsp.c`)** sans dépendance tierce (licence Apache-2.0) avec séparation `compact` (têtes linéaires + oscillateur `_pulses` en ONNX ; cepstre + FFT/iFFT + overlap-add en C) réduit la latence sur EPYC 1T (`-18 %`) et Chromium WASM (`-25.5–28.1 %` 1T, `-19.3–24.3 %` 4T à 2 s), **sans mesure antérieure sur CPU sans AVX2** ni combinaison avec les découvertes d'E03 (`no_mask` + `int8_acoustic`).

---

## 2. Hypothèses falsifiables (E04 sur modèle français + Core i7-3770)

1. **H1 (`omit_mask=True` dans `Exportable`)** : En passant `mask=None` dans `Attention`, `Block`, `Stack`, `Predictor` lors de l'export `B=1` ([model.py](../../src/tiny_voca/model.py), [export.py](../../src/tiny_voca/export.py)), les `11` `IsNaN` et `22` `Where` SDPA disparaissent (`0` nœud `IsNaN`), la parité acoustique contre `dense_masked` reste bit-à-bit exacte (`SNR > 100 dB`, `mel_dtw = 0`), la latence baisse fortement sur phrases moyennes/longues, et `VmHWM` diminue.
2. **H2 (`dsp.c` sans AVX2)** : Compilé en `-O3 -march=native -ffp-contract=off` (AVX/SSE4.2 sans AVX2), le noyau FFT C (`fused` / `compact`) maintient une parité `SNR > 115 dB` contre `dense_masked` sur toutes longueurs de bord (`hop±1`, `fft±1`) et réduit encore la latence 1-thread par rapport à `dense_unmasked`.
3. **H3 (Synergie E03 + E04 : `compact_int8_acoustic_unmasked`)** : Combiner `omit_mask=True`, la séparation `compact` et la quantification sélective INT8 des 20 `MatMul` d'`acoustic_stack` réduit l'empreinte disque d'environ `40 %` (~`21.4 → ~12.8 MB`) et le RSS pic de `> 50 MiB` sur phrase longue, tout en conservant `100 %` des durées identiques et une erreur RMS relative `< 1 %`.

---

## 3. Matrice orthogonale mesurée (`warmup=5`, `runs=30`, ordres `forward` et `reverse`)

Script reproductible : [benchmark_e04.py](../../scripts/benchmark_e04.py), noyau C : [dsp.c](../../scripts/e04/dsp.c) + [runtime.py](../../scripts/e04/runtime.py), résultats bruts dans `/home/yoch/voice-tts/.local-runs/e04/benchmark_e04.json`.  
Phrases françaises identiques à E03 (`F1`) : `short` (`19` tok, `1.739 s`), `medium` (`60` tok, `4.021 s`), `long` (`132` tok, `9.120 s`).  
*(Note charge hôte : `loadavg_start = 2.63`, `loadavg_end = 6.26` ; chaque configuration `(ordre, threads, phrase, variante)` s'exécute dans un sous-processus Python isolé pour mesurer `VmHWM`.)*

### 3.1 Inventaire des candidats et empreinte binaire

| Variante | Fichier `.onnx` + `.so` | Taille totale (MB) | Nœuds | `Conv` / `ConvT` | `IsNaN` / `Where` | `MatMul` / `MatMulInt` | SHA-256 (préfixe 16 hex) |
|---|---|---:|---:|---:|---:|---:|---|
| `dense_masked` | `21 369 917 B` | **21.370** | 2 338 | 17 / 4 | **11 / 30** | 76 / 0 | `b85edda0b26dcdd1` |
| `dense_unmasked` | `21 343 280 B` | **21.343** | 2 167 | 17 / 4 | **0 / 8** | 76 / 0 | `680e00ccdf5e36aa` |
| `dense_unmasked_int8_acoustic` | `15 621 777 B` | **15.622** | 2 247 | 17 / 4 | 0 / 8 | 56 / 20 | `9e1c849274ce6fe8` |
| `fused_fp32_unmasked` | `18 696 173 + 32 432 B` | **18.729** | 2 056 | **13 / 0** | 0 / 8 | 76 / 0 | `be76ff0a8edcd7fc` |
| `compact_fp32_unmasked` | `18 451 838 + 32 432 B` | **18.484** | **1 891** | **13 / 0** | **0 / 5** | 72 / 0 | `f6aae34f913ed2fd` |
| `compact_int8_acoustic_unmasked` | `12 718 277 + 32 432 B` | **12.751** | 1 971 | **13 / 0** | 0 / 5 | 52 / 20 | `ead07ada0a3d1aaf` |

*(SHA-256 complets dans `benchmark_e04.json` ; `dsp.so` = `32 432 B`, SHA-256 `a63cbde6f0fca09e1ba780271b62d832d2c32febc227cbf10b1dd288c4fc1492`.)*

### 3.2 Profilage noyau ORT (`long`, `132` tok, `ORT_ENABLE_ALL`, moyenne sur `15` runs après `5` warmup)

| Groupe ORT | `dense_masked` 1T | `dense_unmasked` 1T | `compact_fp32_unmasked` 1T | `dense_masked` 4T | `dense_unmasked` 4T | `compact_fp32_unmasked` 4T |
|---|---:|---:|---:|---:|---:|---:|
| `attention_qk_softmax` | 267.04 (34.3%) | 206.26 (39.1%) | 210.19 (50.6%) | 144.41 (31.0%) | 112.18 (44.2%) | 140.50 (57.1%) |
| **`attention_isnan_where`** | **148.72 (19.1%)** | **0.00 (0.0%)** | **0.00 (0.0%)** | **160.71 (34.5%)** | **0.00 (0.0%)** | **0.00 (0.0%)** |
| `neural_linear_matmul` | 144.80 (18.6%) | 126.24 (23.9%) | 131.62 (31.7%) | 52.35 (11.3%) | 46.59 (18.4%) | 62.49 (25.4%) |
| **`vocoder_stft_conv`** | **91.77 (11.8%)** | **84.79 (16.1%)** | **0.00 (0.0%)** | **41.13 (8.8%)** | **40.21 (15.8%)** | **0.00 (0.0%)** |
| `neural_norm_gelu_rotary_misc` | 68.50 (8.8%) | 56.73 (10.8%) | 57.74 (13.9%) | 27.53 (5.9%) | 22.01 (8.7%) | 28.41 (11.6%) |
| `vocoder_pulses_and_top_level_ops` | 37.22 (4.8%) | 34.98 (6.6%) | 9.64 (2.3%) | 24.65 (5.3%) | 21.22 (8.4%) | 9.09 (3.7%) |
| `vocoder_response_cepstrum` | 15.33 (2.0%) | 14.00 (2.7%) | 1.83 (0.4%) | 12.01 (2.6%) | 9.50 (3.7%) | 2.83 (1.2%) |
| `neural_depthwise_predictor_conv` | 5.49 (0.7%) | 4.62 (0.9%) | 4.69 (1.1%) | 2.40 (0.5%) | 2.00 (0.8%) | 2.72 (1.1%) |
| `vocoder_stft_pad` | 0.24 (0.0%) | 0.17 (0.0%) | 0.00 (0.0%) | 0.23 (0.1%) | 0.19 (0.1%) | 0.00 (0.0%) |
| **Total `model_run` ORT** | **790.32 ms** | **534.83 ms** | **421.69 ms** | **474.30 ms** | **261.19 ms** | **254.66 ms** |

**Lecture directe :**
- À 1 thread, `omit_mask=True` fait chuter `model_run` ORT de **`790.32 → 534.83 ms` (`-32.3 %`)**, et à 4 threads de **`474.30 → 261.19 ms` (`-44.9 %`)**.
- Sans `IsNaN`+`Where`, les `8` `Conv`/`ConvTranspose` STFT deviennent `16.07 %` (1T) et `15.84 %` (4T) du temps noyau ORT ; la séparation `compact` ramène ces nœuds STFT à `0 ms` et les ops top-level/cepstrales de `49.0 → 11.5 ms` en 1T (coût C `compact_filter` mesuré à part ci-dessous : `~37–40 ms` en 1T sur `long`).

### 3.3 Latence end-to-end (inférence + DSP + copie audio, `30` runs × `2` ordres)

#### 1 thread (`intra=1`) — Médiane `fwd / rev` (et `min fwd / rev`), `dsp_ms`, `VmHWM`

| Phrase | Variante | Méd. `fwd / rev` (ms) | `min fwd / rev` (ms) | `p95 fwd / rev` (ms) | `dsp` (ms) | RTFx (`fwd / rev`) | `VmHWM` (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `short` | `dense_masked` | 71.45 / 91.57 | 65.17 / 71.12 | 85.17 / 149.87 | 0.00 | 24.3x / 19.0x | 360.3 / 357.9 |
| `short` | `dense_unmasked` | 66.73 / 93.63 | 62.87 / 78.04 | 81.76 / 115.39 | 0.00 | 26.1x / 18.6x | 359.6 / 359.7 |
| `short` | `dense_unmasked_int8_acoustic` | 68.45 / 81.99 | 64.60 / 68.36 | 75.03 / 96.09 | 0.00 | 25.4x / 21.2x | 349.8 / 349.9 |
| `short` | **`fused_fp32_unmasked`** | **56.45 / 61.25** | 51.65 / 53.15 | 73.87 / 75.25 | 4.51 / 4.35 | **30.8x / 28.4x** | 353.2 / 352.7 |
| `short` | `compact_fp32_unmasked` | 56.95 / 71.20 | 49.94 / 56.17 | 64.09 / 93.78 | 6.70 / 7.73 | 30.5x / 24.4x | 351.4 / 351.9 |
| `short` | `compact_int8_acoustic_unmasked` | 57.18 / 82.88 | 54.80 / 54.55 | 63.97 / 136.01 | 6.66 / 8.58 | 30.4x / 21.0x | **343.9 / 343.9** |
| `medium` | `dense_masked` | 196.91 / 265.88 | 186.98 / 205.56 | 256.19 / 344.82 | 0.00 | 20.3x / 15.0x | 377.3 / 374.7 |
| `medium` | `dense_unmasked` | 188.71 / 224.27 | 162.06 / 172.16 | 233.76 / 293.09 | 0.00 | 21.2x / 17.8x | 373.2 / 374.5 |
| `medium` | `dense_unmasked_int8_acoustic` | 216.28 / 184.41 | 174.16 / 168.83 | 283.29 / 245.12 | 0.00 | 18.5x / 21.7x | 358.9 / 358.1 |
| `medium` | **`fused_fp32_unmasked`** | **157.04 / 149.48** | 139.19 / 135.23 | 177.33 / 195.60 | 10.66 / 9.86 | **25.5x / 26.8x** | 363.4 / 363.5 |
| `medium` | **`compact_fp32_unmasked`** | 176.46 / **143.68** | 140.93 / 134.48 | 206.71 / 170.28 | 17.18 / 15.46 | 22.7x / **27.8x** | 363.1 / 360.8 |
| `medium` | `compact_int8_acoustic_unmasked` | 183.12 / 165.23 | 148.15 / 151.70 | 271.86 / 210.75 | 16.96 / 17.29 | 21.8x / 24.2x | **352.7 / 350.9** |
| `long` | `dense_masked` | 848.96 / 865.02 | 689.62 / 723.46 | 1450.08 / 1090.17 | 0.00 | 10.7x / 10.5x | 437.3 / 436.8 |
| `long` | `dense_unmasked` | 883.67 / 645.04 | 563.49 / 551.13 | 1634.13 / 802.80 | 0.00 | 10.3x / 14.1x | 410.9 / 411.3 |
| `long` | `dense_unmasked_int8_acoustic` | 720.82 / 607.32 | 574.57 / 506.47 | 822.14 / 690.73 | 0.00 | 12.7x / 15.0x | 390.7 / 389.7 |
| `long` | `fused_fp32_unmasked` | 613.38 / 499.81 | 496.02 / 421.88 | 945.87 / 665.06 | 27.74 / 24.61 | 14.9x / 18.2x | 414.5 / 414.0 |
| `long` | **`compact_fp32_unmasked`** | **506.53 / 495.64** | 461.06 / 423.72 | 625.36 / 635.26 | 40.07 / 37.02 | **18.0x / 18.4x** | 413.5 / 413.0 |
| `long` | `compact_int8_acoustic_unmasked` | 537.59 / 497.15 | 470.94 / 429.59 | 740.23 / 596.64 | 38.34 / 38.76 | 17.0x / 18.3x | **382.5 / 382.4** |

#### 4 threads (`intra=4`) — Médiane `fwd / rev` (et `min fwd / rev`), `dsp_ms`, `VmHWM`

| Phrase | Variante | Méd. `fwd / rev` (ms) | `min fwd / rev` (ms) | `p95 fwd / rev` (ms) | `dsp` (ms) | RTFx (`fwd / rev`) | `VmHWM` (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `short` | `dense_masked` | 84.22 / 51.45 | 42.94 / 31.24 | 196.35 / 60.41 | 0.00 | 20.6x / 33.8x | 358.8 / 360.9 |
| `short` | `dense_unmasked` | 55.10 / 43.40 | 38.10 / 29.84 | 132.01 / 76.32 | 0.00 | 31.6x / 40.1x | 360.1 / 358.2 |
| `short` | `dense_unmasked_int8_acoustic` | 62.54 / 34.91 | 42.22 / 27.84 | 98.58 / 66.62 | 0.00 | 27.8x / 49.8x | 350.5 / 350.9 |
| `short` | **`fused_fp32_unmasked`** | **42.77 / 33.18** | 32.81 / 26.45 | 60.45 / 49.45 | 5.76 / 4.53 | **40.7x / 52.4x** | 353.8 / 353.7 |
| `short` | `compact_fp32_unmasked` | 54.87 / 35.68 | 40.53 / 25.90 | 81.53 / 46.65 | 11.33 / 7.28 | 31.7x / 48.7x | 352.0 / 352.0 |
| `short` | `compact_int8_acoustic_unmasked` | 44.95 / 47.06 | 34.92 / 26.47 | 69.39 / 69.32 | 8.41 / 9.23 | 38.7x / 36.9x | **344.4 / 344.4** |
| `medium` | `dense_masked` | 165.08 / 161.98 | 134.04 / 111.14 | 237.03 / 320.71 | 0.00 | 24.2x / 24.7x | 375.8 / 375.3 |
| `medium` | `dense_unmasked` | 124.28 / **88.89** | 102.18 / 73.39 | 180.20 / 125.23 | 0.00 | 32.2x / **45.0x** | 371.7 / 373.3 |
| `medium` | `dense_unmasked_int8_acoustic` | 131.37 / 92.75 | 109.59 / 75.18 | 201.23 / 127.02 | 0.00 | 30.4x / 43.1x | 357.3 / 357.7 |
| `medium` | `fused_fp32_unmasked` | 116.53 / **84.23** | 86.65 / 65.10 | 144.60 / 109.20 | 15.59 / 10.76 | 34.3x / **47.5x** | 364.1 / 364.0 |
| `medium` | `compact_fp32_unmasked` | **97.89** / 116.98 | 83.43 / 98.97 | 132.13 / 156.32 | 18.65 / 26.59 | **40.9x** / 34.2x | 361.7 / 362.4 |
| `medium` | `compact_int8_acoustic_unmasked` | 113.73 / 93.44 | 90.89 / 62.93 | 169.78 / 119.68 | 27.46 / 17.61 | 35.2x / 42.8x | **351.9 / 351.9** |
| `long` | `dense_masked` | 677.12 / 535.79 | 564.71 / 468.59 | 946.19 / 717.91 | 0.00 | 13.5x / 17.0x | 441.1 / 440.1 |
| `long` | `dense_unmasked` | 398.83 / 258.51 | 314.75 / 217.73 | 590.35 / 358.38 | 0.00 | 22.9x / 35.3x | 413.2 / 412.2 |
| `long` | `dense_unmasked_int8_acoustic` | 436.52 / 326.64 | 311.72 / 222.57 | 1161.95 / 484.82 | 0.00 | 20.9x / 27.9x | 391.0 / 390.1 |
| `long` | `fused_fp32_unmasked` | 395.51 / **256.40** | 294.91 / **189.98** | 602.51 / 483.24 | 43.55 / 27.03 | 23.1x / **35.6x** | 414.1 / 414.2 |
| `long` | `compact_fp32_unmasked` | 325.65 / 327.77 | 278.95 / 220.08 | 463.10 / 654.42 | 51.80 / 56.73 | 28.0x / 27.8x | 414.0 / 413.9 |
| `long` | `compact_int8_acoustic_unmasked` | **316.25** / 357.49 | 273.39 / 208.74 | 407.03 / 882.25 | 48.85 / 58.22 | **28.8x** / 25.5x | **382.2 / 382.9** |

---

## 4. Fidélité acoustique & numérique sur `6` énoncés français (`F1` + `M1`)

Comparaison contre `dense_masked` ORT FP32 **avec flux de bruit identique (`seed=0`)** (`0` NaN/Inf sur `100 %` des énoncés) :

| Variante | Durées identiques (`dense` & `torch`) | Ratio RMS (`dense`) | Err. RMS rel. (`dense`) | Err. max / crête (`dense`) | SNR (`dense`) | `mel_dtw` (`dense`) |
|---|---|---|---|---|---:|---:|
| `dense_masked` (réf.) | **6/6** (`0` trame) | 1.0000 | 0.000000 | 0.000000 | — | 0.00000 |
| **`dense_unmasked`** | **6/6** (`0` trame) | **1.0000** | **0.000000** | **0.000000** | **Exact (∞ dB)** | **0.00000** |
| `fused_fp32_unmasked` | **6/6** (`0` trame) | **1.0000** | **0.000000** | `0.000000–0.000001` | **127.51–129.55 dB** | **0.00000** |
| `compact_fp32_unmasked` | **6/6** (`0` trame) | **1.0000** | `0.000001` | `0.000001` | **123.19–124.10 dB** | **0.00000** |
| `dense_unmasked_int8_acoustic` | **6/6** (`0` trame) | `1.0000–1.0014` | `0.005435–0.006843` | `0.007741–0.010717` | 43.29–45.30 dB | `0.00507–0.00636` |
| `compact_int8_acoustic_unmasked` | **6/6** (`0` trame) | `1.0000–1.0014` | `0.005435–0.006843` | `0.007741–0.010717` | 43.29–45.30 dB | `0.00507–0.00636` |

Extraits audio comparatifs sauvegardés dans `/home/yoch/voice-tts/.local-runs/e04/audio/` (`medium_F1_*.wav`, `long_M1_*.wav`).

---

## 5. Décision d'architecture et validation

1. **À intégrer dans le tronc français (`fr`)** : **`omit_mask=True`** sur `Exportable` ([model.py](../../src/tiny_voca/model.py), [export.py](../../src/tiny_voca/export.py), option `--keep-mask` conservée pour diagnostic).
   - Élimine `100 %` des nœuds `IsNaN` (`11 → 0`) et des `Where` SDPA, réduit `model_run` ORT de **`-32.3 %` (1T)** et **`-44.9 %` (4T)** sur phrase longue, diminue `VmHWM` de **`-26 MiB`**, **zéro changement d'API**, **zéro dépendance externe**, **parité mathématique exacte (`0.0` erreur RMS)**.
2. **À conserver comme option / pilote expérimental (`scripts/e04/`)** : **`compact` / `fused` + `dsp.c`** (et `compact_int8_acoustic_unmasked`).
   - Sur CPU **mono-thread** (cas critique pour latence Web/embarqué), la FFT C est un **vrai gain reproductible** : `dense_unmasked` → `compact_fp32_unmasked` / `fused_fp32_unmasked` économise **`15 %–31 %`** de temps total (**`-24 %` à `-43 %`** face au baseline `dense_masked`, avec `18.0–30.8x` RTFx sur `long`→`short`).
   - Avec INT8 sélectif sur `acoustic_stack` (`compact_int8_acoustic_unmasked`), la taille totale chute à **`12.751 MB`** (`-40.3 %`) et `VmHWM` à **`382 MiB`** (`-55 MiB` sur `long`) tout en gardant **6/6 durées identiques** et `0.54–0.68 %` d'erreur RMS (`mel_dtw <= 0.0064`).
   - En **4 threads CPU natifs**, la variance hôte observée et le caractère mono-thread du C DSP ne dégagent pas de supériorité nette de `compact` face à `dense_unmasked` (`fused` reste légèrement meilleur ou comparable sur `min`, ex. `189.98` vs `217.73 ms` sur `long`), d'où le maintien du mode fichier unique `.onnx` (`dense_unmasked`) comme défaut natif multi-thread simple.

### Tests exécutés
- `PYTHONPATH=src:tests:scripts /home/yoch/voice-tts/env/bin/python -m unittest discover -s tests -v` → **`Ran 81 tests in 73.748s` — `OK (skipped=2)`** (les 2 tests ignorés sont identiques à la baseline : `test_acoustic_preparation_interrupt_resume_provenance_and_bad_rows` et `test_with_dizione_itself`). Inclut les 3 tests dédiés [test_e04.py](../../tests/test_e04.py), ainsi que `test_export`, `test_model`, `test_frontend`, `test_infer`, `test_alignment`, `test_data` et `test_web`.
