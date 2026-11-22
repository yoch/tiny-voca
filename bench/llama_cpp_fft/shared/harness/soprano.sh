# soprano.sh OUT REPS build... : interleaved llama-tts runs of Soprano-1.1-80M (Q8_0), wall time, 4 threads and 1 (pinned),
# then the WAV of each build for parity (first build = reference)
OUT=$1; REPS=$2; shift 2; B=("$@"); G=/tmp/tv/gguf; mkdir -p $OUT/wav; RAW=$OUT/soprano_raw.txt; : > $RAW
TXT="After a long and difficult journey across the mountains, the travellers finally reached the small village, where they were welcomed with warm food, a place to sleep and many questions about the road behind them."
run() { local pin=""; [ "$2" = 1 ] && pin="taskset -c 2"; local t0=$(date +%s%N)
  LD_LIBRARY_PATH=/tmp/tv/bins/$1 $pin /tmp/tv/bins/$1/llama-tts -m $G/Soprano-1.1-80M-Q8_0.gguf -mm $G/mmproj-Soprano-1.1-80M-F16.gguf -p "$TXT" --seed 7 -t $2 -o $3 > $OUT/last_$1.log 2>&1
  echo "wall_ms=$(( ($(date +%s%N) - t0) / 1000000 ))"; }
for t in 4 1; do for r in $(seq $REPS); do for i in $(seq 0 $((${#B[@]} - 1))); do b=${B[$(( (i + r) % ${#B[@]} ))]}
  echo "threads=$t rep=$r build=$b $(run $b $t /dev/null)" >> $RAW; done; done; done
for b in "${B[@]}"; do run $b 4 $OUT/wav/soprano_$b.wav > /dev/null; done
run ${B[0]} 4 $OUT/wav/soprano_${B[0]}_again.wav > /dev/null
echo SOPRANO_DONE
