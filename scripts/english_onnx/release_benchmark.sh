#!/usr/bin/env bash
set -euo pipefail
work=${1:?Provide the release experiment directory}
python_bin=${PYTHON:-python}
cpus=${EN_ONNX_CPUS:-0-3}
mkdir -p "$work/ola-cpu"
# Preselected medium/long cases and one forward/reverse pass per backend.
for order in forward reverse; do
  if [ "$order" = forward ]; then
    names=(exact-int8 matmulola-int8)
    threads=(1 4)
  else
    names=(matmulola-int8 exact-int8)
    threads=(4 1)
  fi
  for count in "${threads[@]}"; do
    for name in "${names[@]}"; do
      taskset -c "$cpus" "$python_bin" scripts/english_onnx/bench.py \
        --model "$work/graphs/$name.onnx" --cases "$work/fixtures/ola.json" \
        --threads "$count" --runs 15 --output "$work/ola-cpu/$order-$count-$name.json"
    done
  done
done
EN_ONNX_FIXED=0 EN_ONNX_RUNS=12 EN_ONNX_THREADS=1,4 taskset -c "$cpus" node scripts/english_onnx/web.cjs \
  "$work/graphs" "$work/fixtures/ola.json" "$work/ola-web.json" exact-int8 matmulola-int8
EN_ONNX_FIXED=1 EN_ONNX_THREADS=1,4 taskset -c "$cpus" node scripts/english_onnx/web.cjs \
  "$work/graphs" "$work/fixtures/ola.json" "$work/ola-web-parity.json" exact-int8-fixed matmulola-int8-fixed
EN_ONNX_FIXED=0 EN_ONNX_RUNS=20 EN_ONNX_THREADS=4 taskset -c "$cpus" node scripts/english_onnx/web.cjs \
  "$work/graphs" "$work/fixtures/short.json" "$work/short-web.json" public exact-int8
