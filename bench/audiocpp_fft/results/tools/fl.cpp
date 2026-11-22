#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
// exp(x) in float for -14 <= x <= 6, the range of the clamped log magnitudes. x = n ln 2 + r
// with |r| <= ln 2 / 2 (ln 2 in two parts, so n ln 2 is exact), and the Taylor series of
// exp(r) to r^7: the first term left out is below 6e-9, a tenth of a float ulp. 2^n is
// written straight into the exponent bits. No call and no branch, so a loop of it vectorizes.
inline float exp_bounded(float x) {
    constexpr float kShift = 12582912.0F;  // 1.5 * 2^23: rounds to an integer
    const float n = (x * 1.44269504088896341F + kShift) - kShift;
    const float r = (x - n * 0.693145751953125F) - n * 1.42860676533018704e-6F;
    const float p = 1.0F + r * (1.0F + r * (0.5F + r * (1.0F / 6.0F + r * (1.0F / 24.0F + r * (1.0F / 120.0F +
        r * (1.0F / 720.0F + r * (1.0F / 5040.0F)))))));
    const std::int32_t bits = (static_cast<std::int32_t>(n) + 127) << 23;
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// sin and cos in float for |x| <= kSincosRange. The nearest multiple q of pi / 2 is taken off
// with pi / 2 in three parts (the first two products are exact for |q| < 2^13, Cephes'
// split), which leaves |r| <= pi / 4; the Taylor series stop at r^9 and r^10, whose first
// terms left out are below 2e-9. No call and no branch, so a loop of it vectorizes.
constexpr float kSincosRange = 8192.0F;

inline void sincos_bounded(float x, float & s, float & c) {
    constexpr float kShift = 12582912.0F;  // 1.5 * 2^23: rounds to an integer
    const float q = (x * 0.636619772367581343F + kShift) - kShift;
    const float r = ((x - q * 1.5703125F) - q * 4.837512969970703125e-4F) - q * 7.54978995489188216e-8F;
    const float z = r * r;
    const float sine = r + r * z * (-1.0F / 6.0F + z * (1.0F / 120.0F + z * (-1.0F / 5040.0F + z * (1.0F / 362880.0F))));
    const float cosine = 1.0F + z * (-0.5F + z * (1.0F / 24.0F + z * (-1.0F / 720.0F + z * (1.0F / 40320.0F + z * (-1.0F / 3628800.0F)))));
    // the quarter turns, (s, c) -> (c, -s) for each; q is a float, as integers of another
    // width than the data keep the loop from vectorizing
    const float turn = q - 4.0F * ((0.25F * q + kShift) - kShift - (((0.25F * q + kShift) - kShift) > 0.25F * q ? 1.0F : 0.0F));
    const bool odd = turn == 1.0F || turn == 3.0F;
    const float s1 = odd ? cosine : sine;
    const float c1 = odd ? sine : cosine;
    s = turn >= 2.0F ? -s1 : s1;
    c = turn == 1.0F || turn == 2.0F ? -c1 : c1;
}


int main(){ double emax=0, smax=0, cmax=0, smaxs=0; double lmax=0;
 for(long i=0;i<=20000000;i++){ float x=LO+(HI-LO)*(float)i/20000000.0f; double e=std::exp((double)x); double r=std::fabs(exp_bounded(x)-e)/e; if(r>emax) emax=r; double l=std::fabs(std::exp(x)-e)/e; if(l>lmax) lmax=l;}
 std::mt19937 g(1); std::uniform_real_distribution<float> d(-1,1);
 for(long i=0;i<20000000;i++){ float x = d(g)*(i%2? 8192.0f: 20.0f); float s,c; sincos_bounded(x,s,c); double es=std::fabs(s-std::sin((double)x)), ec=std::fabs(c-std::cos((double)x)); if(i%2==0){ if(es>smaxs) smaxs=es; if(ec>smaxs) smaxs=ec;} if(es>smax) smax=es; if(ec>cmax) cmax=ec;}
 double f=std::pow(2.0,-24); printf("exp_bounded max rel err %.3g (%.2f ulp of 2^-24; std::expf %.2f)\nsincos_bounded max abs err |x|<=20: %.3g, |x|<=8192: sin %.3g cos %.3g\n",emax,emax/f,lmax/f,smaxs,smax,cmax);}
