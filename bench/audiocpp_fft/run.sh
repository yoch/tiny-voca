#!/usr/bin/env bash
# Setup + baseline for the audio.cpp tiny-voca waveform-head benchmark. Run from the repository root.
# Creates bench/audiocpp_fft/work/ (ignored): audio.cpp @ 337af0c, upstream tiny-voca code (main of this repo) with
# its public weights, real dumps of the waveform-head inputs, the reference and optimized harness builds, and
# results/baseline.txt (timings, parity).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
W=$B/work; R=$B/results
mkdir -p "$W" "$R"
cd "$W"
if [ ! -d audio.cpp ]; then
  git clone -q https://github.com/0xShug0/audio.cpp audio.cpp
  git -C audio.cpp checkout -q 337af0c
fi
A=$W/audio.cpp
# Python side: upstream tiny-voca (its "own" front end, dizione) and the public English weights.
if [ ! -d venv ]; then
  python3 -m venv venv
  venv/bin/pip -q install --upgrade pip
  git -C "$B/../.." archive origin/main src pyproject.toml README.md LICENSE | (mkdir -p up && tar -x -C up)
  venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu
  venv/bin/pip -q install numpy soundfile huggingface_hub "./up[own]"
fi
if [ ! -f dumps/cases.txt ]; then
  CK=$(venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('n4ze3m/tiny-voca', 'tiny-voca.pt'))")
  (cd up && TINY_VOCA_FRONTEND=own PYTHONPATH=src ../venv/bin/python "$B/dump.py" "$CK" "$W/dumps" en)
fi
FLAGS="-O3 -march=native -std=c++17 -DNDEBUG -DENGINE_SPEECHFFT_NO_MULTITHREADING"
COMMON="-I$B/stub -I$A/include -I$A/src/framework/audio $B/bench_ac.cpp $A/src/framework/audio/fft.cpp -lpthread"
g++ $FLAGS $A/src/community_models/tiny_voca/source_filter.cpp $COMMON -o bench_ref &
g++ $FLAGS -I$B/opt/include $B/opt/src/source_filter.cpp $COMMON -o bench_opt &
wait
mkdir -p out_ref out_opt
{
  echo "== $(date -u +%FT%TZ)  $(lscpu | grep 'Model name' | sed 's/  */ /g')  nproc $(nproc)  $(g++ --version | head -1)"
  echo "-- reference (audio.cpp 337af0c), FFT single-threaded"; ./bench_ref dumps 15 out_ref
  echo "-- optimized (opt/), FFT single-threaded";             ./bench_opt dumps 15 out_opt
  echo "-- parity optimized vs reference"; venv/bin/python "$B/compare.py" out_ref out_opt
} | tee "$R/baseline.txt"
