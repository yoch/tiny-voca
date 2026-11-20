#!/usr/bin/env bash
# Reproduce the llama.cpp tiny-voca host-synth benchmark on this machine.
# usage: bash bench/llama_cpp_fft/run.sh [OUT_DIR]      (default: bench/llama_cpp_fft/results/<host>-<utc>)
# Needs: git, cmake, a C++17 compiler, curl, python3. Takes ~15-30 min (two llama.cpp builds).
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${1:-$KIT/results/$(hostname | tr -c 'a-zA-Z0-9-\n' '-')-$STAMP}
WORK=${WORK:-/tmp/tv-bench}
JOBS=${JOBS:-$(nproc)}
COMMIT=9d6f04f
mkdir -p "$OUT" "$WORK"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== host"
{ date -u; uname -a; lscpu; nproc; free -h; gcc --version | head -1; cmake --version | head -1; } > "$OUT/host.txt" 2>&1 || true
grep -E 'Model name|^CPU\(s\)|Thread|Core|Flags' "$OUT/host.txt" | cut -c1-200 || true

echo "== llama.cpp $COMMIT"
if [ ! -d "$WORK/llama.cpp" ]; then
  git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
fi
cd "$WORK/llama.cpp"
git fetch -q origin master || true
git checkout -q -f "$COMMIT" 2>/dev/null || { git fetch -q origin "$COMMIT"; git checkout -q -f "$COMMIT"; }
git reset -q --hard && git clean -qfd
git log -1 --format='%H %ad' --date=short | tee "$OUT/llama_commit.txt"

# both builds: -t forwarded to mtmd (so thread counts are controlled) + timing hooks; "fast" adds the synth patch
git apply "$KIT/tts-threads-9d6f04f.patch"
git apply "$KIT/timing-instrumentation.patch"
build() {
  cmake -S . -B "build-$1" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > "$OUT/cmake-$1.log" 2>&1
  cmake --build "build-$1" --target llama-tts -j "$JOBS" > "$OUT/build-$1.log" 2>&1
  tail -1 "$OUT/build-$1.log"
}
echo "== build base"; build base
git apply "$KIT/tiny-voca-rfft-v2.patch"
echo "== build fast"; build fast
git diff --stat | tee "$OUT/applied_patches.txt"

echo "== models"
G=$WORK/gguf; mkdir -p "$G"
for f in tiny-voca-F32.gguf mmproj-tiny-voca-F32.gguf; do
  [ -s "$G/$f" ] || curl -sfL -o "$G/$f" "https://huggingface.co/ggml-org/tiny-voca-GGUF/resolve/main/$f"
done
sha256sum "$G"/*.gguf | tee "$OUT/gguf_sha256.txt"

tts() {  # build threads text voice wavout
  LD_LIBRARY_PATH="$WORK/llama.cpp/build-$1/bin" TV_TIMING=1 "$WORK/llama.cpp/build-$1/bin/llama-tts" \
    -m "$G/tiny-voca-F32.gguf" -mm "$G/mmproj-tiny-voca-F32.gguf" -p "$3" --tts-voice "$4" --seed 7 -t "$2" -o "$5" 2>&1 | grep '^TV_TIMING'
}
declare -A TEXT
TEXT[short]="Thanks a lot!"
TEXT[medium]="Could you send me the report by Friday afternoon, please?"
TEXT[long]="After a long and difficult journey across the mountains, the travellers finally reached the small village, where they were welcomed with warm food, a place to sleep and many questions about the road behind them."

echo "== interleaved timings (median of 10, base and fast alternated)"
CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')
THREADS="1 4"; [ "${CORES:-0}" -gt 4 ] && THREADS="1 4 $CORES"
: > "$OUT/timings.txt"
for t in $THREADS; do
  for k in short medium long; do
    for v in F2 M2; do
      for r in $(seq 10); do
        for b in base fast; do echo "$b $(tts $b $t "${TEXT[$k]}" $v /dev/null)"; done
      done | awk -v k=$k -v v=$v -v t=$t '{split($4,a,"=");split($5,s,"="); n[$1]=n[$1] " " a[2]; y[$1]=y[$1] " " s[2]; fr=$3}
        function med(str,  arr, c) { c = split(str, arr, " "); asort(arr); return arr[int((c + 1) / 2)] }
        END { nb=med(n["base"]); sb=med(y["base"]); nf=med(n["fast"]); sf=med(y["fast"]);
              printf "threads=%s %-6s %s %s | base stacks=%.1f synth=%.1f | fast stacks=%.1f synth=%.1f | synth %+.0f%%, stacks+synth %+.0f%%\n",
                     t, k, v, fr, nb, sb, nf, sf, 100*(sf/sb-1), 100*((nf+sf)/(nb+sb)-1) }' | tee -a "$OUT/timings.txt"
    done
  done
done

echo "== WAV parity base vs fast (8 utterances)"
mkdir -p "$WORK/wav"
i=0
for txt in "Hi." "Thanks a lot!" "${TEXT[medium]}" "${TEXT[long]}"; do
  for v in F2 M2; do
    i=$((i+1))
    for b in base fast; do tts $b 4 "$txt" $v "$WORK/wav/u${i}_$b.wav" > /dev/null; done
  done
done
python3 - "$WORK/wav" <<'EOF' | tee "$OUT/parity.txt"
import math, sys, wave, array
d = sys.argv[1]
for i in range(1, 9):
    def read(p):
        w = wave.open(p); a = array.array("h", w.readframes(w.getnframes())); return a
    a, b = read(f"{d}/u{i}_base.wav"), read(f"{d}/u{i}_fast.wav")
    n = min(len(a), len(b))
    s = sum(x * x for x in a); e = sum((a[j] - b[j]) ** 2 for j in range(n))
    print(f"u{i} same_length={len(a) == len(b)} max_abs_diff_lsb={max(abs(a[j] - b[j]) for j in range(n))} "
          f"snr_db={'inf' if e == 0 else round(10 * math.log10(s / e), 1)}")
EOF

echo "== standalone harness (old synth copy vs variants A/B/C/D, threads T/K)"
cd "$KIT"
g++ -O3 -DNDEBUG -std=c++17 -w -pthread -o "$WORK/bench" bench.cpp
"$WORK/bench" dumps/u*.bin | tee "$OUT/harness.txt"
THREADS=1 "$WORK/bench" dumps/u1.bin dumps/u3.bin dumps/u5.bin dumps/u7.bin | tee "$OUT/harness_threads.txt"
echo "== done: $OUT"
