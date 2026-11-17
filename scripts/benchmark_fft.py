"""Reproducible E04 CPU benchmark; every measurement uses a fresh process.

python scripts/benchmark_fft.py --output /tmp/e04 [--checkpoint weights.pt]
The synthetic model pins durations; a supplied checkpoint is never modified.
"""
import argparse
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time


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
    reference = data['reference']
    error = result.astype(np.float64) - reference
    print(json.dumps(dict(model=args.worker.name, threads=args.threads, runs=args.runs,
                         median_ms=statistics.median(elapsed), p95_ms=float(np.percentile(elapsed, 95)),
                         peak_rss_mib=int(re.search(r'^VmHWM:\s+(\d+)', Path('/proc/self/status').read_text(), re.M)[1]) / 1024,
                         model_bytes=args.worker.stat().st_size, samples=result.shape[-1],
                         max_abs_error=float(np.abs(error).max()),
                         snr_db=float(10 * np.log10(np.sum(reference.astype(np.float64)**2) / max(np.sum(error**2), 1e-300))))))


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
            np.savez(path, **old)
    # Browser-readable exact inputs and reference, without another dependency.
    for path in args.output.glob('*.npz'):
        data = np.load(path)
        path.with_suffix('.json').write_text(json.dumps({k:dict(shape=list(v.shape),dtype=str(v.dtype),data=v.flatten().tolist()) for k,v in data.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('/tmp/tiny-voca-e04'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--worker', type=Path)
    p.add_argument('--measure-only', action='store_true', help='Reuse prepared graphs and inputs')
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
    for path in sorted(args.output.glob('*.onnx')):
        for threads in (1,4):
            result = subprocess.check_output([sys.executable,__file__,'--worker',str(path),'--threads',str(threads),'--runs',str(args.runs)],text=True)
            row = json.loads(result)
            rows.append(row)
            print(json.dumps(row),flush=True)
    import onnxruntime as ort
    import torch
    report = dict(platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                  ort=ort.__version__, checkpoint=str(args.checkpoint) if args.checkpoint else 'synthetic seed=0', rows=rows)
    (args.output/'cpu-results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
