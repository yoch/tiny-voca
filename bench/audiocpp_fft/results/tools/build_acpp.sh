#!/usr/bin/env bash
# Builds audio.cpp (CPU only, Release) in work/acpp/build: audiocpp_cli, tiny_voca_unit_test, tiny_voca_synth_probe.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); cd $W/acpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON > $W/acpp_cmake.log 2>&1
taskset -c 0,1,3 cmake --build build -j3 --target audiocpp_cli tiny_voca_unit_test tiny_voca_synth_probe > $W/acpp_build.log 2>&1
echo BUILD_DONE >> $W/acpp_build.log
