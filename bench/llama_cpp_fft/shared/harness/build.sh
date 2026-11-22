# build.sh OUT "EXTRA_FLAGS" v1 v2 ... : each variant (variants/<v>.inc) in its own TU at the real mtmd flags, plus old
set -e
H=$(cd "$(dirname "$0")" && pwd)
OUT=$1; EXTRA=$2; shift 2
mkdir -p $OUT
FLAGS="-std=gnu++17 -O3 -DNDEBUG -fPIC -Wmissing-declarations -Wmissing-noreturn -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-array-bounds -Wextra-semi -Wno-cast-qual -Wno-maybe-uninitialized -D_USE_MATH_DEFINES $EXTRA"
VARS="old $*"
REG=$OUT/reg.cpp
echo '#include "shim.h"' > $REG
echo 'struct variant { const char * name; fft_fn fft, ifft; };' >> $REG
for v in $VARS; do echo "extern fft_fn ${v}_fft, ${v}_ifft;" >> $REG; done
echo 'variant g_variants[] = {' >> $REG
for v in $VARS; do echo "  { \"$v\", ${v}_fft, ${v}_ifft }," >> $REG; done
echo '}; int g_n_variants = sizeof(g_variants) / sizeof(g_variants[0]);' >> $REG
OBJS=""
for v in $VARS; do
  g++ $FLAGS -I$H -DVAR=$v -DVARFILE="\"variants/$v.inc\"" -c $H/tu.cpp -o $OUT/$v.o 2>&1 | grep -E "warning|error" || true
  OBJS="$OBJS $OUT/$v.o"
done
g++ -std=gnu++17 -O2 -I$H -c $REG -o $OUT/reg.o
g++ -std=gnu++17 -O2 -I$H -c $H/micro.cpp -o $OUT/micro.o
g++ -o $OUT/micro $OUT/micro.o $OUT/reg.o $OBJS
echo "built $OUT/micro: $VARS"
