"""Single-core empirical memory and FP32 matrix ceilings for the E04-B model."""
import json
import os
import time

import numpy as np

rng = np.random.default_rng(42)
clock = time.perf_counter
a = rng.random(32*1024*1024, dtype=np.float32)
b = np.empty_like(a)
bandwidth = []
for _ in range(30):
    start = clock()
    np.copyto(b,a)
    bandwidth.append(2*a.nbytes / (clock()-start) / 1e9)
del a,b
a = rng.random((1024,1024), dtype=np.float32)
b = rng.random((1024,1024), dtype=np.float32)
out = np.empty_like(a)
compute = []
for _ in range(30):
    start = clock()
    np.matmul(a,b,out=out)
    compute.append(2*1024**3 / (clock()-start) / 1e9)
print(json.dumps(dict(affinity=sorted(os.sched_getaffinity(0)), copy_gb_s=bandwidth,
                      matmul_gflop_s=compute, blas=np.show_config(mode='dicts')),indent=2))
