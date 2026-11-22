// one FFT variant per TU, like mtmd-audio.cpp in libmtmd: g++ -O3 -DNDEBUG -fPIC -DVAR=v1 -DVARFILE='"variants/v1.inc"'
#include "shim.h"
namespace VAR {
#include VARFILE
}
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
fft_fn CAT(VAR, _fft)  = VAR::mtmd_audio_fft;
fft_fn CAT(VAR, _ifft) = VAR::mtmd_audio_ifft;
