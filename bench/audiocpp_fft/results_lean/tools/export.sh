#!/usr/bin/env bash
# export.sh: copies the candidates work/abl/{LEAN,MID,MIDG} to lean/{lean,mid,fast} and writes the patches against
# audio.cpp 337af0c (a/ = original, b/ = candidate, paths as in audio.cpp).
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); B=$W/..; T=$(mktemp -d)
H=include/engine/community_models/tiny_voca/source_filter.h; S=src/community_models/tiny_voca/source_filter.cpp
for p in "lean LEAN" "mid MID" "fast MIDG"; do
  set -- $p; d=$B/lean/$1; mkdir -p $d $T/$1/a/$(dirname $H) $T/$1/a/$(dirname $S) $T/$1/b/$(dirname $H) $T/$1/b/$(dirname $S)
  cp $W/abl/$2/src/source_filter.cpp $d/source_filter.cpp; cp $W/abl/$2/$H $d/source_filter.h
  cp $W/audio.cpp/$H $T/$1/a/$H; cp $W/audio.cpp/$S $T/$1/a/$S; cp $W/abl/$2/$H $T/$1/b/$H; cp $W/abl/$2/src/source_filter.cpp $T/$1/b/$S
  (cd $T/$1 && diff -ruN a b > $d/$1.patch || true)
  (cd $W/audio.cpp && git apply --check $d/$1.patch && echo "$1: $($W/tools/lines.sh $W/abl/$2), patch applies")
done
rm -rf $T
