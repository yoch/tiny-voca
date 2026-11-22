# rb.sh name... : rebuild libmtmd with src/<name>.cpp as tiny-voca.cpp, copy bin dir to /tmp/tv/bins/<name>
set -e
cd /tmp/tv/llama.cpp
for n in "$@"; do
  cp /home/user/tiny-voca/bench/llama_cpp_fft/lean/src/$n.cpp tools/mtmd/models/tiny-voca.cpp
  cmake --build build --target llama-tts -j 4 > /tmp/tv/blogs/$n.log 2>&1
  rm -rf /tmp/tv/bins/$n; cp -r build/bin /tmp/tv/bins/$n
  echo "built $n: $(grep -c 'warning:' /tmp/tv/blogs/$n.log) warnings"
done
git checkout -q tools/mtmd/models/tiny-voca.cpp
