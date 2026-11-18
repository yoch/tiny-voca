# Reproduce the French alignment probe

Use this worktree at the pinned French fork revision plus the E05 scripts.
The run is CPU-only. It does not train a recognizer or a TTS model. It needs
Python 3.12 with the Tiny Voca dependencies, SciPy, ONNX Runtime 1.32.0,
eSpeak NG 1.52.0 and FFmpeg 7.1.5 to match the recorded run.

Rhapsodie is the external timing reference. Its site publishes the audio,
phone/word TextGrids, provenance and license. Use the original WAV files for
timing. The MP3 package is useful for the codec-clock audit only. [Rhapsodie downloads](https://rhapsodie.modyco.fr/info-telechargement/)

```sh
export TINY_VOCA_LANGUAGE=fr TINY_VOCA_FRONTEND=espeak PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /tmp/e05/model /tmp/e05/rhapsodie
base=https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/c69750f5043e5e1f8a71ab95dd3b98338c280c92
curl -fL "$base/onnx/model_int8.onnx" -o /tmp/e05/model/alignment.onnx
curl -fL "$base/vocab.json" -o /tmp/e05/model/vocab.json
echo '74174710e34035bbb7f611601d016c32fc575de7a6f53b1078107dc10a84e7ae  /tmp/e05/model/alignment.onnx' | sha256sum -c -
echo 'd732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0  /tmp/e05/model/vocab.json' | sha256sum -c -
curl -fL https://rhapsodie.modyco.fr/rp/prosodie/finaux/Rhap-proso-all-textgrids.zip -o /tmp/e05/rhapsodie/grids.zip
echo '0cdcbbf3c8cc557d971133812cb62f56e8b6476257280d1ccf957b159279be02  /tmp/e05/rhapsodie/grids.zip' | sha256sum -c -
curl -fL https://rhapsodie.modyco.fr/rp/waves/Rhap-sound-wav.zip -o /tmp/e05/rhapsodie/wavs.zip
unzip -jo /tmp/e05/rhapsodie/wavs.zip Rhap-M0004.wav Rhap-M0008.wav Rhap-M2001.wav Rhap-M0013.wav \
  Rhap-M0010.wav Rhap-M0012.wav Rhap-M0021.wav Rhap-M2005.wav -d /tmp/e05/rhapsodie
python scripts/e05/prepare_rhapsodie.py /tmp/e05/rhapsodie/grids.zip /tmp/e05/rhapsodie /tmp/e05/wav-corpus
taskset -c 0-3 python scripts/e05/alignment_probe.py \
  --corpus /tmp/e05/wav-corpus --model /tmp/e05/model/alignment.onnx \
  --vocab /tmp/e05/model/vocab.json --cache /tmp/e05/posteriors --output /tmp/e05/wav-results.json
python scripts/e05/test_probe.py -q
```

For the audio clock audit, place matching `Rhap-*.mp3` files from the site's
MP3 archive alongside the WAV files and run `python scripts/e05/audio_clock.py
/tmp/e05/rhapsodie`. A positive delay means the decoded MP3 waveform arrives
later than the WAV waveform on a common sample grid.

The WAV archive is about 678 MB. The measured run retrieved only the eight
members via ZIP byte ranges and checked the ZIP CRCs. Each member's SHA-256 is
in `g2p_edits` in the results. Choose four allowed CPU IDs if the machine's
affinity does not include 0–3. Use an empty cache for a fresh latency measure.
The probe caches one CTC posterior matrix per audio and model digest. Retiming
the same clips uses that cache, so it needs no repeated neural inference.
Development and holdout differ by recording ID. The held-out clips cover
different utterances from development, and the script does not use their scores
for selecting the onset parameter or pause policy. The INT8 graph is a pinned
screening proxy for the parent phoneme recognizer used in French training.

For target-voice references, reuse the same probe with a separate corpus
directory. Put the clips there as 24 kHz mono float WAVs, right-pad to a
multiple of 256 samples, and supply a `manifest.json` array. Each row records a
unique `id`, `split` (`dev` or `holdout`), `recording` group, `file`, `text`,
`source_sha256`, and `start` and `end` coordinates in the original recording.
`reference` is the ordered list of `{phone, start, end}` intervals, in seconds
relative to the clip. `oracle` is the human phone sequence with the normal
Tiny Voca boundary and punctuation tokens. Keep ambiguous labels and scorer
disagreements in the annotation review so that the timing subset uses
adjudicated boundaries.
