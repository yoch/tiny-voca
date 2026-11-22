#!/usr/bin/env bash
# interleaved llama-tts timings of several builds (bin dirs with TV_TIMING hooks), then WAV parity vs the first build
# usage: run_tts.sh OUT BINS_ROOT GGUF_DIR REPS build1 build2 ...    (build1 = reference)
set -uo pipefail
OUT=$1; BINS=$2; G=$3; REPS=$4; shift 4; BUILDS=("$@")
mkdir -p "$OUT"
declare -A TEXT
TEXT[short]="Thanks a lot!"
TEXT[medium]="Could you send me the report by Friday afternoon, please?"
TEXT[long]="After a long and difficult journey across the mountains, the travellers finally reached the small village, where they were welcomed with warm food, a place to sleep and many questions about the road behind them."
tts() {  # build threads text voice wavout -> "frames net synth wall_ms"
  local pin=""; [ "$2" = 1 ] && pin="taskset -c 2"
  local t0=$(date +%s%N)
  local line=$(LD_LIBRARY_PATH="$BINS/$1" TV_TIMING=1 $pin "$BINS/$1/llama-tts" -m "$G/tiny-voca-F32.gguf" -mm "$G/mmproj-tiny-voca-F32.gguf" \
      -p "$3" --tts-voice "$4" --seed 7 -t "$2" -o "$5" 2>&1 | grep '^TV_TIMING')
  local t1=$(date +%s%N)
  echo "$line wall_ms=$(( (t1 - t0) / 1000000 ))"
}
RAW=$OUT/tts_raw.txt; : > "$RAW"
nb=${#BUILDS[@]}
for t in 1 4; do
  for k in short medium long; do
    for v in F2 M2; do
      for r in $(seq "$REPS"); do
        for i in $(seq 0 $((nb - 1))); do
          b=${BUILDS[$(( (i + r) % nb ))]}   # rotate the order every repetition
          echo "threads=$t text=$k voice=$v rep=$r build=$b $(tts "$b" "$t" "${TEXT[$k]}" "$v" /dev/null)" >> "$RAW"
        done
      done
      echo "done threads=$t $k $v"
    done
  done
done
echo "== WAV parity"
mkdir -p "$OUT/wav"
i=0
for txt in "Hi." "Thanks a lot!" "${TEXT[medium]}" "${TEXT[long]}"; do
  for v in F2 M2; do
    i=$((i+1))
    for b in "${BUILDS[@]}"; do tts "$b" 4 "$txt" "$v" "$OUT/wav/u${i}_$b.wav" > /dev/null; done
    tts "${BUILDS[1]}" 4 "$txt" "$v" "$OUT/wav/u${i}_${BUILDS[1]}_again.wav" > /dev/null
  done
done
echo TTS_DONE
