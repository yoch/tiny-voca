"""Compare native SIMD build flags on one CPU using the trained stage outputs."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import onnxruntime as ort

from cpu import DSP, FLOAT_ARRAY


def main(root, reference):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(root/'native/stage.onnx'), options, providers=['CPUExecutionProvider'])
    data = dict(np.load(reference))
    parts = session.run(None,{arg.name:data[arg.name] for arg in session.get_inputs()})
    dsp = DSP(root,parts[0].size)
    results = []
    modes = ('x86-64','x86-64-v3','x86-64-v4')
    libraries = {}
    source = Path(__file__).with_name('dsp.c')
    for mode in modes:
        binary = root / f'dsp-{mode}.so'
        flags = ['-mprefer-vector-width=512'] if mode == 'x86-64-v4' else []
        subprocess.run(['cc','-O3',f'-march={mode}',*flags,'-ffp-contract=off',
                        '-shared','-fPIC',str(source),'-lm','-o',str(binary)],check=True)
        lib = ct.CDLL(str(binary))
        lib.initialize.argtypes = [FLOAT_ARRAY]*5
        lib.initialize.restype = None
        lib.initialize(*dsp.constants)
        lib.filter.argtypes = [ct.c_int]*2 + [FLOAT_ARRAY]*4
        lib.filter.restype = None
        lib.mix.argtypes = [ct.c_int]+[FLOAT_ARRAY]*2
        lib.mix.restype = None
        libraries[mode] = lib
    reference_audio = dsp.late(parts).copy()
    for order in (modes,modes[::-1]):
        for mode in order:
            dsp.lib = libraries[mode]
            for _ in range(5):
                dsp.late(parts)
            timings = []
            for _ in range(30):
                start = time.perf_counter()
                audio = dsp.late(parts)
                timings.append((time.perf_counter()-start)*1000)
            np.testing.assert_allclose(audio,reference_audio,atol=3e-6,rtol=2e-5)
            results.append(dict(march=mode,median_ms=statistics.median(timings),timings_ms=timings))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('reference',type=Path,help='Prior E04 prepared full-dense.npz')
    args = p.parse_args()
    main(args.root,args.reference)
