#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
typedef void (*synth_fn)(const std::vector<float> &, const std::vector<float> &, std::mt19937 &, std::vector<float> &);
int main(int argc, char ** argv) {
    void * h = dlopen((std::string(argv[1]) + "/libmtmd.so").c_str(), RTLD_NOW);
    synth_fn f = (synth_fn) dlsym(h, argv[2]);
    FILE * fp = fopen(argv[3], "rb"); int64_t nf; if (fread(&nf, 8, 1, fp) != 1) return 1;
    std::vector<float> f0(nf), voc(nf * 134), out; if (fread(f0.data(), 4, nf, fp) != (size_t) nf || fread(voc.data(), 4, voc.size(), fp) != voc.size()) return 1;
    for (int r = 0; r < atoi(argv[4]); r++) { std::mt19937 rng(7); f(voc, f0, rng, out); }
    printf("%zu\n", out.size());
}
