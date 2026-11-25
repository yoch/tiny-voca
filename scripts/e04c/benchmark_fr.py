"""Time the checked FR late FFT against dense using E04-C's exported graphs."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import onnxruntime as ort

from native import Renderer, constants, digest, load_model, match_frontend, phonemize, session


def benchmark(args):
    if digest(args.checkpoint) != args.checkpoint_sha256 or not match_frontend(args.checkpoint):
        raise ValueError('Checkpoint provenance or frontend mismatch')
    model = load_model(args.checkpoint)
    renderer = Renderer(args.library, constants(model).numpy())
    cases = [
        ('Bonjour ! Comment allez-vous aujourd’hui ?', 0, 1., .08, 1.2, 58880),
        ('Je cherche six chaises sèches. Chut… écoutez le vent.', 1, .9, -.1, 1.4, 114688),
    ]
    fixtures = []
    for text, voice, speed, pitch, spread, samples in cases:
        phones = phonemize(text)
        feeds = dict(tokens=np.array([phones.ids], np.int64), stress=np.array([phones.stress], np.int64),
                     voice=np.array([voice], np.int64), speed=np.array([speed], np.float32),
                     pitch_shift=np.array([pitch], np.float32), pitch_range=np.array([spread], np.float32),
                     noise=np.random.default_rng(42).standard_normal((1, samples), dtype=np.float32))
        fixtures.append((text, model.config.speakers[voice], feeds))

    def render(parts):
        sources, hr, hi, noise, nr, ni, durations = parts
        harmonic = sum(renderer.filter(0, src, real, imag) for src, real, imag in zip(sources, hr, hi))
        return harmonic[None] + renderer.filter(1, noise, nr, ni), durations

    output = []
    graphs = {name: args.graphs/file for name, file in [('dense','dense.onnx'),('fft','compact.onnx')]}
    for order in (('dense','fft'),('fft','dense')):
        results = {}
        for threads in ((1,4) if order[0] == 'dense' else (4,1)):
            for variant in order:
                runtime = session(graphs[variant], threads)
                for index, (text, voice, feeds) in enumerate(fixtures):
                    graph_times, total_times = [], []
                    for run in range(args.warmups + args.runs):
                        start = time.perf_counter_ns()
                        parts = runtime.run(None, feeds)
                        graph_end = time.perf_counter_ns()
                        audio, durations = render(parts) if variant == 'fft' else parts
                        end = time.perf_counter_ns()
                        if run >= args.warmups:
                            graph_times.append((graph_end-start) / 1e6)
                            total_times.append((end-start) / 1e6)
                    results[threads,index,variant] = audio, durations
                    output.append(dict(order=list(order), threads=threads, variant=variant, case=index+1,
                                       text=text, voice=voice, samples=audio.size, graph_ms=graph_times,
                                       total_ms=total_times))
                del runtime
            for index in range(len(fixtures)):
                dense, expected = results[threads,index,'dense']
                fft, actual = results[threads,index,'fft']
                np.testing.assert_array_equal(actual, expected)
                assert fft.shape == dense.shape and np.isfinite(fft).all() and np.isfinite(dense).all()
                reference = dense.astype(np.float64)
                error = reference - fft.astype(np.float64)
                peak = float(np.abs(error).max() / np.abs(reference).max())
                snr = float(10 * np.log10(np.sum(reference**2) / np.sum(error**2)))
                assert peak < 2e-5 and snr > 100
                output.append(dict(order=list(order), threads=threads, case=index+1,
                                   snr_db=snr, normalized_peak_error=peak))
    return dict(checkpoint_sha256=args.checkpoint_sha256, library_sha256=digest(args.library),
                graphs={name:dict(sha256=digest(p),bytes=p.stat().st_size) for name,p in graphs.items()},
                ort=ort.__version__, platform=platform.platform(),
                affinity=sorted(os.sched_getaffinity(0)), runs=args.runs, warmups=args.warmups,
                timing='Warm pre-tokenized ORT inference + late DSP, fixed common noise',
                records=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('checkpoint','graphs','library'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    print(json.dumps(benchmark(args), indent=2, ensure_ascii=False))
