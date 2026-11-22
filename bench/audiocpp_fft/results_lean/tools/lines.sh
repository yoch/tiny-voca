#!/usr/bin/env bash
# lines.sh DIR [REFDIR]: changed lines of DIR's source_filter.h/.cpp against the original audio.cpp files
# (diff -u, lines starting with - or +, file headers excluded), split into code and comment-only/blank lines.
W=$(cd "$(dirname "$0")/.." && pwd); R=${2:-$W/abl/ref}; H=include/engine/community_models/tiny_voca/source_filter.h
{ diff -u $R/src/source_filter.cpp $1/src/source_filter.cpp; diff -u $R/$H $1/$H; } | grep '^[-+]' | grep -v '^\(---\|+++\) ' |
awk '{t++; l=substr($0,2); if (l ~ /^[ \t]*(\/\/.*)?$/) c++; else k++} END {printf "%d code + %d comment/blank (= %d)\n", k, c, t}'
