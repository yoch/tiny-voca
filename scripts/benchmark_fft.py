"""Reproducible E04 CPU benchmark; every measurement uses a fresh process.

python scripts/benchmark_fft.py --output /tmp/e04 [--checkpoint weights.pt]
The synthetic model pins durations; a supplied checkpoint is never modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time


def errors(actual, reference):
    import numpy as np
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise ValueError('Output shape changed or audio is non-finite')
    error = actual.astype(np.float64) - reference.astype(np.float64)
    peak = float(np.abs(error).max())
    energy = float(np.sum(error**2))
    signal = float(np.sum(reference.astype(np.float64)**2))
    return dict(max_abs_error=peak,
                normalized_peak_error=peak / max(float(np.abs(reference).max()), 1e-30),
                snr_db=10 * math.log10(signal / energy) if signal and energy else None,
                exact=energy == 0)


def parity(session, data):
    import numpy as np
    result = session.run(None, {i.name: data[i.name] for i in session.get_inputs()})
    metrics = errors(result[0], data['reference'])
    if 'durations' in data:
        np.testing.assert_array_equal(result[1], data['durations'])
        metrics['durations_equal'] = True
    metrics['accepted'] = (metrics['normalized_peak_error'] < 2e-5 and
                           (metrics['exact'] or (metrics['snr_db'] or -float('inf')) > 100))
    return metrics


def worker(args):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.worker), options, providers=["CPUExecutionProvider"])
    data = np.load(args.worker.with_suffix('.npz'))
    feeds = {i.name: data[i.name] for i in session.get_inputs()}
    for _ in range(5):
        session.run(None, feeds)
    elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result = session.run(None, feeds)[0]
        elapsed.append((time.perf_counter() - start) * 1000)
    # Memory belongs to the timed case, before the additional validation inputs.
    peak_rss = int(re.search(r'^VmHWM:\s+(\d+)', Path('/proc/self/status').read_text(), re.M)[1]) / 1024
    metrics = parity(session, data)
    validation = {}
    if args.worker.name.startswith('full-'):
        for path in sorted((args.worker.parent / 'cases').glob('*.npz')):
            validation[path.stem] = parity(session, np.load(path))
    print(json.dumps(dict(model=args.worker.name, threads=args.threads, runs=args.runs,
                         median_ms=statistics.median(elapsed), p95_ms=float(np.percentile(elapsed, 95)),
                         times_ms=elapsed, peak_rss_mib=peak_rss,
                         model_bytes=args.worker.stat().st_size, samples=result.shape[-1],
                         **metrics, validation=validation)))


def prepare(args):
    import numpy as np
    import torch
    import onnx
    from tiny_voca.config import Config
    from fft_candidates import Exportable, export
    from tiny_voca.infer import load_model
    from tiny_voca.model import TinyVoca
    from tiny_voca.text import PHONE_IDS
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.checkpoint:
        model = load_model(args.checkpoint, check_frontend=False)
    else:
        model = TinyVoca(Config()).eval()
        torch.nn.init.zeros_(model.duration.out.weight)
        model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
        torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
        torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    phones = sorted(PHONE_IDS)
    tokens = torch.tensor([[phones[(7*i) % len(phones)] for i in range(24)]])
    one = lambda v: torch.tensor([v], dtype=torch.float32)
    inputs = (tokens, torch.arange(24)[None] % 3, torch.tensor([1]), one(1), one(0), one(1),
              torch.randn(1, model.config.hop * model.config.max_frames))
    class Filters(torch.nn.Module):
        def __init__(self, backend):
            super().__init__()
            self.wrapper = Exportable(model, fft_backend=backend)
        def forward(self, source, hr, hi, nr, ni):
            c = model.config
            return (self.wrapper._filter(source, hr, hi, 'synth', c.synth_fft, c.synth_hop)
                    + self.wrapper._filter(source, nr, ni, 'noise', c.noise_fft, c.noise_hop))
    c = model.config
    samples = c.hop * 188  # 2.005 s at 24 kHz
    source = torch.randn(1, samples)
    responses = []
    for fft, hop in ((c.synth_fft,c.synth_hop),(c.noise_fft,c.noise_hop)):
        shape = (1, samples//hop+3, fft//2+1)
        responses.extend((torch.randn(shape)*0.2, torch.randn(shape)*0.2))
    fi = (source, *responses)
    with torch.no_grad():
        full_reference = Exportable(model, fft_backend='dense')(*inputs)[0].numpy()
        filter_reference = Filters('dense')(*fi).numpy()
        for backend in ('dense','dft','radix2'):
            path = args.output / f'full-{backend}.onnx'
            export(model,path,external_noise=True,fft_backend=backend)
            names = ['tokens','stress','voice','speed','pitch_shift','pitch_range','noise']
            np.savez(path.with_suffix('.npz'), **dict(zip(names,(x.numpy() for x in inputs))), reference=full_reference)
            path = args.output / f'filter-{backend}.onnx'
            names = ['source','hr','hi','nr','ni']
            axes = {n:{1:'samples' if n=='source' else 'frames_'+n[0]} for n in names}
            axes['audio'] = {1:'samples'}
            torch.onnx.export(Filters(backend), fi, str(path), input_names=names, output_names=['audio'],
                              dynamic_axes=axes,opset_version=17,dynamo=False)
            np.savez(path.with_suffix('.npz'), **dict(zip(names,(x.numpy() for x in fi))), reference=filter_reference)
    for path in args.output.glob('*.onnx'):
        onnx.checker.check_model(onnx.load(path), full_check=True)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for kind in ('full', 'filter'):
        data = np.load(args.output / f'{kind}-dense.npz')
        session = ort.InferenceSession(str(args.output / f'{kind}-dense.onnx'), options, providers=['CPUExecutionProvider'])
        feeds = {i.name: data[i.name] for i in session.get_inputs()}
        reference = session.run(None, feeds)[0]
        for path in args.output.glob(f'{kind}-*.npz'):
            old = dict(np.load(path))
            old['reference'] = reference
            if kind == 'full':
                old['durations'] = session.run(None, feeds)[1]
            np.savez(path, **old)
        if kind == 'full':
            cases = args.output / 'cases'
            cases.mkdir(exist_ok=True)
            descriptions = []
            for length, voice, speed, pitch, spread in ((1,0,1.,0.,1.), (5,1,1.25,.1,1.3), (40,0,.8,-.1,.7)):
                tokens = torch.tensor([[phones[(7*i) % len(phones)] for i in range(length)]])
                descriptions.append((f'phones-{length}-voice-{voice}', tokens.numpy(), (torch.arange(length)[None] % 3).numpy(), voice, speed, pitch, spread))
            if args.sentences:
                from tiny_voca.infer import match_frontend
                from tiny_voca.text import phonemize, frontend_identity
                # Text validation must use the checkpoint's matching frontend.
                if not args.checkpoint:
                    raise ValueError('--sentences requires a trained checkpoint')
                match_frontend(args.checkpoint)
                for number, (text, voice, speed, pitch, spread) in enumerate((
                    ('Hello world.', 0, 1., 0., 1.),
                    ('The quick brown fox jumps over the lazy dog.', 1, 1.25, .1, 1.3),
                    ('At the edge of the window, a quiet voice becomes clear.', 0, .8, -.1, .7))):
                    p = phonemize(text)
                    descriptions.append((f'sentence-{number}', np.array([p.ids]), np.array([p.stress]), voice, speed, pitch, spread))
                (args.output/'frontend.json').write_text(json.dumps(frontend_identity(), indent=2)+'\n')
            for name, tokens, stress, voice, speed, pitch, spread in descriptions:
                feeds = dict(tokens=tokens, stress=stress, voice=np.array([voice], dtype=np.int64),
                             speed=one(speed).numpy(), pitch_shift=one(pitch).numpy(), pitch_range=one(spread).numpy(),
                             noise=inputs[-1].numpy())
                audio, durations = session.run(None, feeds)
                np.savez(cases / f'{name}.npz', **feeds, reference=audio, durations=durations)
    # Browser-readable exact inputs and reference, without another dependency.
    for path in args.output.rglob('*.npz'):
        data = np.load(path)
        path.with_suffix('.json').write_text(json.dumps({k:dict(shape=list(v.shape),dtype=str(v.dtype),data=v.flatten().tolist()) for k,v in data.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/tmp/tiny-voca-e04'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--worker', type=Path)
    p.add_argument('--measure-only', action='store_true', help='Reuse prepared graphs and inputs')
    p.add_argument('--sentences', action='store_true', help='Validate three sentences with the checkpoint-matched frontend')
    p.add_argument('--reverse', action='store_true', help='Reverse the complete model/thread order')
    p.add_argument('--result-name', default='cpu-results.json')
    p.add_argument('--threads',type=int, default=1)
    p.add_argument('--runs',type=int, default=30)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.measure_only:
        prepare(args)
    rows = []
    order = [(path, threads) for path in sorted(args.output.glob('*.onnx')) for threads in (1,4)]
    if args.reverse:
        order.reverse()
    for path, threads in order:
        result = subprocess.check_output([sys.executable,__file__,'--worker',str(path),'--threads',str(threads),'--runs',str(args.runs)],text=True)
        row = json.loads(result)
        rows.append(row)
        print(json.dumps(row),flush=True)
    import onnxruntime as ort
    import torch
    report = dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                  python=sys.version, ort=ort.__version__, order='reverse' if args.reverse else 'forward',
                  checkpoint=str(args.checkpoint) if args.checkpoint else 'synthetic seed=0',
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                  loadavg=Path('/proc/loadavg').read_text().strip(),
                  cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(), rows=rows)
    (args.output/args.result_name).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
