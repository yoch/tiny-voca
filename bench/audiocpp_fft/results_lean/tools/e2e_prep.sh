#!/usr/bin/env bash
# e2e_prep.sh: after build_acpp.sh (unchanged sources), copies audiocpp_cli + tiny_voca_unit_test to e2e_bins/ref,
# then for each candidate copies its source_filter.h/.cpp into work/acpp, rebuilds the two targets and copies them to
# e2e_bins/NAME; restores the original files at the end. Runs each unit test.
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); B=$W/..; cd $W/acpp
H=include/engine/community_models/tiny_voca/source_filter.h; S=src/community_models/tiny_voca/source_filter.cpp
find_bin() { find build -type f -name "$1" -perm -u+x | head -1; }
grab() {
  mkdir -p $W/e2e_bins/$1
  cp "$(find_bin audiocpp_cli)" "$(find_bin tiny_voca_unit_test)" $W/e2e_bins/$1/
  echo "$1: $($W/e2e_bins/$1/tiny_voca_unit_test 2>&1 | tail -1)"
}
grab ref
for p in "lean $B/lean/lean" "mid $B/lean/mid" "fin $B/opt"; do
  set -- $p
  if [ -f $2/source_filter.cpp ]; then cp $2/source_filter.cpp $S; cp $2/source_filter.h $H; else cp $2/src/source_filter.cpp $S; cp $2/$H $H; fi
  cmake --build build -j4 --target audiocpp_cli tiny_voca_unit_test > $W/e2e_build_$1.log 2>&1
  grab $1
done
cp $W/audio.cpp/$S $S; cp $W/audio.cpp/$H $H
cmake --build build -j4 --target audiocpp_cli tiny_voca_unit_test > $W/e2e_build_restore.log 2>&1
echo PREP_DONE
