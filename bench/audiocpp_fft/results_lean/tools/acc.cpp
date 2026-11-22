// Accuracy of exp_small / sincos_small (h.inc, the helpers of lean/mid and lean/fast) against double libm.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include "h.inc"
int main() {
  double emax = 0, smax = 0, smax_small = 0;
  for (float x = -14.0F; x <= 6.0F; x = std::nextafter(x, 7.0F)) {
    double e = std::fabs(exp_small(x) - std::exp((double)x)) / std::exp((double)x); emax = std::max(emax, e);
  }
  for (int i = -40000000; i <= 40000000; ++i) {
    const float x = i * (8192.0F / 40000000.0F); float s, c; sincos_small(x, s, c);
    const double es = std::max(std::fabs(s - std::sin((double)x)), std::fabs(c - std::cos((double)x)));
    smax = std::max(smax, es); if (std::fabs(x) < 100) smax_small = std::max(smax_small, es);
    float ls = std::sin(x), lc = std::cos(x);
    (void)ls; (void)lc;
  }
  double lmax = 0;  // libm float for comparison
  for (int i = -4000000; i <= 4000000; ++i) { const float x = i * (8192.0F / 4000000.0F); lmax = std::max(lmax, std::max(std::fabs(std::sin(x) - std::sin((double)x)), std::fabs(std::cos(x) - std::cos((double)x)))); }
  double lemax = 0; for (float x = -14.0F; x <= 6.0F; x = std::nextafter(x, 7.0F)) lemax = std::max(lemax, std::fabs(std::exp(x) - std::exp((double)x)) / std::exp((double)x));
  printf("exp_small max rel error %.3g (= %.2f x 2^-24), std::exp(float) %.3g\nsincos_small max abs error %.3g (|x|<100: %.3g), libm float %.3g\n", emax, emax * 16777216.0, lemax, smax, smax_small, lmax);
}
