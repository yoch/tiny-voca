inline float exp_small(float x) {
    const auto n = static_cast<std::int32_t>(x * 1.44269504F + 64.5F) - 64;
    const float r = (x - static_cast<float>(n) * 0.693145751953125F) - static_cast<float>(n) * 1.42860676533018704e-6F;
    const float p = 1.0F + r * (1.0F + r * (0.5F + r * (1.0F / 6.0F + r * (1.0F / 24.0F + r * (1.0F / 120.0F +
        r * (1.0F / 720.0F + r * (1.0F / 5040.0F)))))));
    const std::int32_t bits = (n + 127) * (1 << 23);
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return p * scale;
}

// sin and cos in float for |x| <= kSincosRange. x = q pi / 2 + r with q the integer nearest to
// x 2 / pi and pi / 2 in three parts (Cephes' split, exact products for |q| < 2^13), so |r| <= pi / 4;
// Taylor series to r^9 and r^10 (first terms left out below 2e-9), then q quarter turns. Unlike
// std::sin and std::cos, a loop of it vectorizes.
constexpr float kSincosRange = 8192.0F;
inline void sincos_small(float x, float & s, float & c) {
    const auto q = static_cast<std::int32_t>(x * 0.636619772F + 8192.5F) - 8192;
    const float r = ((x - static_cast<float>(q) * 1.5703125F) - static_cast<float>(q) * 4.837512969970703125e-4F) -
                    static_cast<float>(q) * 7.54978995489188216e-8F;
    const float z = r * r;
    const float sine = r + r * z * (-1.0F / 6.0F + z * (1.0F / 120.0F + z * (-1.0F / 5040.0F + z * (1.0F / 362880.0F))));
    const float cosine = 1.0F + z * (-0.5F + z * (1.0F / 24.0F + z * (-1.0F / 720.0F + z * (1.0F / 40320.0F + z * (-1.0F / 3628800.0F)))));
    // each quarter turn maps (sin, cos) to (cos, -sin)
    const bool odd = (q & 1) != 0;
    const float s1 = odd ? cosine : sine;
    const float c1 = odd ? sine : cosine;
    s = (q & 2) != 0 ? -s1 : s1;
    c = ((q + 1) & 2) != 0 ? -c1 : c1;
}
