#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo 'usage: bench_cpu.sh MODELS CASES OUTPUT NAME [...]' >&2
  exit 2
fi
models=$1
cases=$2
output=$3
shift 3
mkdir -p "$output"
variants=("$@")
for order in forward reverse; do
  if [ "$order" = forward ]; then threads=(1 4); else threads=(4 1); fi
  for count in "${threads[@]}"; do
    for ((i=0; i<${#variants[@]}; ++i)); do
      index=$i
      if [ "$order" = reverse ]; then index=$(("${#variants[@]}" - 1 - i)); fi
      variant=${variants[$index]}
      taskset -c "${EN_ONNX_CPUS:-0-3}" "${PYTHON:-python}" scripts/english_onnx/bench.py \
        --model "$models/$variant.onnx" --cases "$cases" --threads "$count" \
        --output "$output/$order-$count-$variant.json"
    done
  done
done
