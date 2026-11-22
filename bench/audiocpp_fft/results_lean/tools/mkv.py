# mkv.py NAME BASE HDRFROM edits... : builds abl/NAME from abl/BASE applying named edits.
import sys, os, shutil
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blocks.py')).read())
fin = open('fin/src/source_filter.cpp').read()
name, base, hdr = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(f'{base}/src/source_filter.cpp').read()
H = 'include/engine/community_models/tiny_voca/source_filter.h'
os.makedirs(f'{name}/src', exist_ok=True); os.makedirs(os.path.dirname(f'{name}/{H}'), exist_ok=True)
shutil.copy(f'{hdr}/{H}', f'{name}/{H}')
def E(s):
    s = s.replace("constexpr int64_t kBlockFrames = 64;\n", """constexpr int64_t kBlockFrames = 64;

// The transforms of a block run on the calling thread: the framework's default (0) hands
// each block of a few dozen short transforms to a thread pool, which costs more than it saves.
constexpr size_t kFftThreads = 1;
""", 1)
    s = s.replace("spectrum.data());", "spectrum.data(), 1.0F, kFftThreads);")
    s = s.replace("response.data());", "response.data(), 1.0F, kFftThreads);")
    s = s.replace("log_response.data());", "log_response.data(), 1.0F, kFftThreads);")
    s = s.replace("1.0F / static_cast<float>(fft));", "1.0F / static_cast<float>(fft), kFftThreads);")
    assert s.count("kFftThreads") >= 3, s.count("kFftThreads")
    return s
def C(s):
    cur_head = region(s, H0, H1); cur_ctor = region(s, C0, C1)
    return s.replace(cur_head, head_kit).replace(cur_ctor, ctor_kit)
def D(s):
    return s.replace(region(s, P0, P1), pulse_kit)
def G(s):  # final's pulse train with its helpers
    s = s.replace(region(s, P0, P1), region(fin, P0, P1))
    helpers = region(fin, "// Rounding to the nearest integer", "// exp(x) in float")
    return s.replace("}  // namespace\n", helpers + "}  // namespace\n", 1).replace("#include <complex>\n", "#include <complex>\n#include <cstdint>\n")
for e in sys.argv[4:]:
    s = globals()[e](s)
open(f'{name}/src/source_filter.cpp', 'w').write(s)
