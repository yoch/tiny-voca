#include <cmath>
#include <cstdio>
#include <random>
constexpr double kPi = 3.14159265358979323846;
// Rounding of 0 <= x < 2^51 to the nearest integer: adding and taking off 2^52 leaves no
// fraction bits. Exact in IEEE double with the default rounding; unlike std::floor and
// std::nearbyint, the compiler vectorizes it without -fno-trapping-math.
inline double round_small(double x) {
    constexpr double kShift = 4503599627370496.0;  // 2^52
    return (x + kShift) - kShift;
}

inline double floor_small(double x) {
    const double r = round_small(x);
    return r > x ? r - 1.0 : r;
}

// sin and cos of 2 pi u for u >= 0 given in cycles, in double. The quarter turn nearest to u
// is taken off exactly, which leaves |x| <= pi / 4, and the Taylor series stop at x^15 and
// x^16: the first term left out is below 5e-17, under the rounding of a double. Written
// without calls or branches so that a loop of them vectorizes; libm's sincos does neither.
inline void sincos_cycles(double u, double & s, double & c) {
    const double quarter = round_small(4.0 * u);
    const double x = (u - 0.25 * quarter) * (2.0 * kPi);
    const double z = x * x;
    const double sine = x + x * z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0 + z * (1.0 / 362880.0 + z * (-1.0 / 39916800.0 +
        z * (1.0 / 6227020800.0 + z * (-1.0 / 1307674368000.0)))))));
    const double cosine = 1.0 + z * (-0.5 + z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0 + z * (-1.0 / 3628800.0 +
        z * (1.0 / 479001600.0 + z * (-1.0 / 87178291200.0 + z * (1.0 / 20922789888000.0))))))));
    // The quarter turns, (s, c) -> (c, -s) for each. Counted in doubles: mixing in integers
    // of another width keeps the loop from vectorizing.
    const double turn = quarter - 4.0 * floor_small(0.25 * quarter);
    const bool odd = turn == 1.0 || turn == 3.0;
    const double s1 = odd ? cosine : sine;
    const double c1 = odd ? sine : cosine;
    s = turn >= 2.0 ? -s1 : s1;
    c = turn == 1.0 || turn == 2.0 ? -c1 : c1;
}


int main(){ std::mt19937_64 g(1); std::uniform_real_distribution<double> d(0,1);
 double es=0, ec=0; long double PI=3.141592653589793238462643383279502884L;
 for(long i=0;i<20000000;i++){ double k=(i%3==0)?0.5:std::floor(d(g)*250)+0.5; double u = k*d(g); double s,c; sincos_cycles(u,s,c);
  long double fu = (long double)u - floorl((long double)u); long double x=2*PI*fu; es=std::fmax(es,std::fabs((double)(s-sinl(x)))); ec=std::fmax(ec,std::fabs((double)(c-cosl(x))));}
 printf("max abs err sin %.3g cos %.3g (vs long double, u up to 250 cycles)\n",es,ec);}
