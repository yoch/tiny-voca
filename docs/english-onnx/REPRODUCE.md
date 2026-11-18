# Reproduction CPU et Chromium

Utiliser la branche `codex/english-onnx-standalone` et Python 3.12.
Les exports archivés utilisent PyTorch 2.15.0+cpu, ONNX 1.23.2 et
opset 17. Les mesures locales utilisent ORT 1.32.0, NumPy 2.5.5,
SciPy 1.18.1 et, pour les empreintes vocales, PyTorch et torchaudio
2.8.0+cpu. Dizione est au commit `8cd889e4de7be6bd1dd2e6891344a0729fa14dd4`.
Chromium utilise Playwright 1.65.1, ORT Web 1.20.1 et Pyodide 0.26.4.

## Poids et outils

```bash
set -euo pipefail
export EN_ONNX_WORK=/tmp/tiny-voca-english
export PYTHONPATH=src:scripts/english_onnx
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DIZIONE_DATA="$EN_ONNX_WORK/phonemizer"
mkdir -p "$DIZIONE_DATA" "$EN_ONNX_WORK"/{graphs,asr,speaker,web}
base=https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14
curl -fL "$base/tiny-voca.pt" -o "$EN_ONNX_WORK/tiny-voca.pt"
curl -fL "$base/onnx/model.onnx" -o "$EN_ONNX_WORK/graphs/public.onnx"
for f in lexicon.tsv align.json g2p.json g2p.npz homographs.json; do
  curl -fL "$base/phonemizer/$f" -o "$DIZIONE_DATA/$f"
done
(
  cd "$EN_ONNX_WORK"
  echo 'ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70  tiny-voca.pt' | sha256sum -c -
  echo 'fe6d918791a15b82e37465981d7f27a070e46b46635e5a3482fc8dff87a1ae46  graphs/public.onnx' | sha256sum -c -
)
git clone https://github.com/n4ze3m/dizione.git "$EN_ONNX_WORK/dizione"
git -C "$EN_ONNX_WORK/dizione" checkout 8cd889e4de7be6bd1dd2e6891344a0729fa14dd4
python -m pip install -e "$EN_ONNX_WORK/dizione"
(cd "$EN_ONNX_WORK/dizione/src" && zip -q -r "$EN_ONNX_WORK/dizione.zip" dizione -x '*/__pycache__/*')
export DIZIONE_ZIP="$EN_ONNX_WORK/dizione.zip"
npm --prefix "$EN_ONNX_WORK/web" install onnxruntime-web@1.20.1 playwright@1.65.1 pyodide@0.26.4
export NODE_PATH="$EN_ONNX_WORK/web/node_modules"
"$EN_ONNX_WORK/web/node_modules/.bin/playwright" install chromium
```

Installer les dépendances Python indiquées en tête, le projet, `soundfile`,
`faster-whisper==1.2.1`, `jiwer==4.0.0`, `transformers==4.57.6` et `pytest`
dans l’environnement de mesure. Les actifs Pyodide doivent contenir le
wheel NumPy 1.26.4 référencé par leur lockfile. Le placer dans le dossier
`$NODE_PATH/pyodide` si le paquet NPM le télécharge séparément.

```bash
whisper=https://huggingface.co/Systran/faster-whisper-base.en/resolve/3d3d5dee26484f91867d81cb899cfcf72b96be6c
for f in config.json tokenizer.json vocabulary.txt model.bin; do
  curl -fL "$whisper/$f" -o "$EN_ONNX_WORK/asr/$f"
done
voice=https://huggingface.co/wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/f0c48c298fd835726c27956a5d617bad7115627e
curl -fL "$voice/voxceleb_resnet34_LM.onnx" -o "$EN_ONNX_WORK/speaker/wespeaker.onnx"
```

## Export et criblage

```bash
root="$EN_ONNX_WORK/graphs"
python scripts/english_onnx/build.py --checkpoint "$EN_ONNX_WORK/tiny-voca.pt" --public "$root/public.onnx" \
  --output "$root" base mask packed scan exact matmulola \
  prune624 prune576 prune512 prune384 prune160 weighted512 solo384first solo384last \
  svd144 svd128 svd96 svd128out qkv128 attention80 prosody576 encoder576 combined512 \
  reconstruct576 reconstruct512 reconstruct448 reconstruct384 reconstruct160
python scripts/english_onnx/evaluate.py --public "$root/public.onnx" --output "$root/public-fixed.onnx"
for method in int8 qdq fp16; do
  python scripts/english_onnx/quantize.py --root "$root" exact "$method"
done
python scripts/english_onnx/quantize.py --root "$root" reconstruct512 int8
python scripts/english_onnx/evaluate.py --original "$root/public-fixed.onnx" --output "$EN_ONNX_WORK/screen.json" \
  --cases "$EN_ONNX_WORK/calibration.json" "$root"/{mask,packed,scan,exact,matmulola,prune512,reconstruct512,exact-int8}-fixed.onnx
```

Les graphes nommés `*-fixed.onnx` servent aux comparaisons numériques avec
un bruit commun. Les modèles livrables utilisent les six entrées
d’origine et génèrent leur bruit dans ONNX. Ils renvoient `audio` float32
et `durations` int64. `voice=0` correspond à F2 et `voice=1` à M2.
`pitch_shift` est exprimé en log-Hz, `speed` et `pitch_range` sont des
facteurs, et les durées sont en trames de 256 échantillons à 24 kHz.

## Fidélité, latence et écoute A/B

Adapter `taskset` et `EN_ONNX_CPUS` aux quatre CPU autorisés. Exécuter les
mesures séquentiellement. Les mesures définitives utilisent les trois
cas fixés dans [latency-cases.json](latency-cases.json).

```bash
taskset -c 0-3 python scripts/english_onnx/evaluate.py --original "$root/public-fixed.onnx" --split holdout \
  --asr "$EN_ONNX_WORK/asr" --speaker "$EN_ONNX_WORK/speaker/wespeaker.onnx" --wavs "$EN_ONNX_WORK/wavs" \
  --output "$EN_ONNX_WORK/holdout.json" \
  --independent exact exact-fp16 exact-int8 reconstruct576 reconstruct512 reconstruct512-int8 -- \
  "$root"/{exact,matmulola,exact-fp16,exact-int8,exact-qdq,prune624,reconstruct576,reconstruct512,reconstruct512-int8}-fixed.onnx

variants=(public exact matmulola exact-int8 exact-fp16 reconstruct512 reconstruct512-int8)
fixture=docs/english-onnx/latency-cases.json
bash scripts/english_onnx/bench_cpu.sh "$root" "$fixture" "$EN_ONNX_WORK/cpu" "${variants[@]}"
EN_ONNX_RUNS=16 taskset -c 0-3 node scripts/english_onnx/web.cjs "$root" "$fixture" "$EN_ONNX_WORK/web.json" "${variants[@]}"
EN_ONNX_FIXED=1 taskset -c 0-3 node scripts/english_onnx/web.cjs "$root" "$EN_ONNX_WORK/calibration.json" \
  "$EN_ONNX_WORK/web-parity.json" "${variants[@]/%/-fixed}"
python scripts/english_onnx/summary.py --quality "$EN_ONNX_WORK/holdout.json" \
  --cpu "$EN_ONNX_WORK/cpu" --web "$EN_ONNX_WORK/web.json" --output "$EN_ONNX_WORK/summary.json"

python scripts/english_onnx/listening.py --wavs "$EN_ONNX_WORK/wavs" --output "$EN_ONNX_WORK/AB.zip" \
  --key "$EN_ONNX_WORK/AB-key.json" exact-int8 reconstruct512-int8
python -m pytest -q tests/test_standalone_onnx.py tests/test_export.py tests/test_model.py
```

Pour l’écoute, distribuer seulement `AB.zip`, décompresser puis ouvrir
`index.html`. Enregistrer les réponses avant de divulguer la clé.
