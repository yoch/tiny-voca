#!/usr/bin/env bash
# wrap.sh NAME candidate.cpp > tu.cpp : the host synth section of a tiny-voca.cpp (pulse train .. synth) in namespace NAME
set -e
n=$1; f=$2
a=$(grep -n '^// band-limited pulse train' "$f" | cut -d: -f1)
b=$(grep -n '^// host: unknown-word model' "$f" | cut -d: -f1)
echo '#include "shim.h"'
grep '^#include <' "$f"
echo "namespace $n {"
echo 'struct mtmd_graph_tiny_voca : tiny_voca_consts { static void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio); };'
echo "#line $a \"$(basename $f)\""
sed -n "${a},$((b-2))p" "$f"
echo "}"
echo "extern synth_fn reg_$n; synth_fn reg_$n = $n::mtmd_graph_tiny_voca::synth;"
