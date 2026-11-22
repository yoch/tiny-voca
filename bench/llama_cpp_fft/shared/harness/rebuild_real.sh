# rebuild_real.sh name audio tv : rebuild llama-tts + test-mtmd-impl in /tmp/tv/llama.cpp with
#   tools/mtmd/mtmd-audio.cpp = shared/src/mtmd-audio.<audio>.cpp and tools/mtmd/models/tiny-voca.cpp = lean/src/<tv>.cpp
# and copy build/bin to /tmp/tv/bins/<name>; the clone is restored afterwards (the timing patch in mtmd.cpp stays)
set -e
B=/home/user/tiny-voca/bench/llama_cpp_fft
cd /tmp/tv/llama.cpp
name=$1; audio=$2; tv=$3
cp $B/shared/src/mtmd-audio.$audio.cpp tools/mtmd/mtmd-audio.cpp
cp $B/lean/src/$tv.cpp tools/mtmd/models/tiny-voca.cpp
cmake --build build --target llama-tts test-mtmd-impl -j 4 > /tmp/tv/blogs/$name.log 2>&1
rm -rf /tmp/tv/bins/$name; cp -r build/bin /tmp/tv/bins/$name
echo "built $name (audio=$audio tv=$tv): $(grep -c 'warning:' /tmp/tv/blogs/$name.log) warnings"
git checkout -q tools/mtmd/mtmd-audio.cpp tools/mtmd/models/tiny-voca.cpp
