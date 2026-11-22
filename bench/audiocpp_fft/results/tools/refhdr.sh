# makes refhdr/: the reference header with the stub settings (for pair.sh)
W=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $W/refhdr/engine/community_models/tiny_voca
cp $W/audio.cpp/include/engine/community_models/tiny_voca/source_filter.h $W/refhdr/engine/community_models/tiny_voca/
cp $W/../stub/engine/community_models/tiny_voca/assets.h $W/refhdr/engine/community_models/tiny_voca/
