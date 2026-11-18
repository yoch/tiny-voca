"""E04 staged CPU investigation: ORT neural graph + NumPy native FFT.

Not a production backend. Includes session.run output materialization, Python
boundary, framing, response multiplication, inverse FFT, overlap-add and crop.
Prepare/measure only after the CPU/Web runs, never concurrently with them.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


def native_filter(source, real, imaginary, window, hop):
    import numpy as np
    size = len(window)
    samples = source.shape[-1]
    padded = np.pad(source, ((0, 0), (size//2, size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, size, axis=-1)[:, ::hop] * window
    spectrum = np.fft.rfft(frames, axis=-1)
    product = spectrum * (real + np.complex64(1j) * imaginary)
    wave = np.fft.irfft(product, n=size, axis=-1) * window
    def overlap(x):
        return sum(np.pad(x[..., i*hop:(i+1)*hop].reshape(1, -1), ((0, 0), (i*hop, (3-i)*hop))) for i in range(4))
    envelope = overlap(np.broadcast_to(window*window, frames.shape))
    return overlap(wave)[:, size//2:size//2+samples] / envelope[:, size//2:size//2+samples]


def prepare(args):
    import numpy as np
    import torch
    import onnx
    from tiny_voca.export import Exportable, INPUTS, example_inputs
    from tiny_voca.infer import load_model
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = load_model(args.checkpoint, check_frontend=False)
    class Stage(Exportable):
        def _filter(self, source, real, imaginary, name, fft, hop):
            self.captured.extend((source, real, imaginary))
            return torch.zeros_like(source)
        def forward(self, *inputs):
            self.captured = []
            _, durations = super().forward(*inputs)
            return (*self.captured, durations)
    wrapper = Stage(model, external_noise=True).eval()
    inputs = (*example_inputs(), torch.randn(1, model.config.hop * model.config.max_frames))
    names = [*INPUTS, 'noise']
    outputs = ['pulses', 'hr', 'hi', 'noise_source', 'nr', 'ni', 'durations']
    axes = {'tokens':{1:'tokens'}, 'stress':{1:'tokens'}, 'noise':{1:'noise_samples'}, 'durations':{1:'tokens'}}
    for name in outputs[:-1]:
        axes[name] = {1:'samples' if name in ('pulses', 'noise_source') else 'frames_'+name[0]}
    target = args.output / 'native'
    target.mkdir(exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(target/'stage.onnx'), input_names=names, output_names=outputs,
                          dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(target/'stage.onnx'), full_check=True)
    np.savez(target/'windows.npz', synth=model.vocoder.synth_window.numpy(), noise=model.vocoder.noise_window.numpy())


def worker(args):
    import numpy as np
    import onnxruntime as ort
    from benchmark_fft import errors
    target = args.output / 'native'
    windows = dict(np.load(target/'windows.npz'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target/'stage.onnx'), options, providers=['CPUExecutionProvider']) if args.worker == 'full' else None
    # All arrays reside in RAM before warmup, just like benchmark_fft.py.
    data = dict(np.load(args.output / f'{args.worker}-dense.npz'))
    stage_times, dsp_times = [], []
    boundary_bytes = []
    def run(data, measure=False):
        start = time.perf_counter()
        if session:
            parts = session.run(None, {i.name:data[i.name] for i in session.get_inputs()})
            pulses, hr, hi, noise, nr, ni, durations = parts
            if measure:
                boundary_bytes.append(sum(part.nbytes for part in parts))
        else:
            pulses = noise = data['source']
            hr, hi, nr, ni = (data[n] for n in ('hr','hi','nr','ni'))
            durations = None
        boundary = time.perf_counter()
        audio = native_filter(pulses,hr,hi,windows['synth'],len(windows['synth'])//4) + native_filter(noise,nr,ni,windows['noise'],len(windows['noise'])//4)
        end = time.perf_counter()
        if measure:
            stage_times.append((boundary-start)*1000)
            dsp_times.append((end-boundary)*1000)
        return audio, durations
    for _ in range(5): run(data)
    elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result, durations = run(data, True)
        elapsed.append((time.perf_counter()-start)*1000)
    rss = int(re.search(r'^VmHWM:\s+(\d+)',Path('/proc/self/status').read_text(),re.M)[1])/1024
    def validate(result, durations, data):
        metric = errors(result, data['reference'])
        if durations is not None:
            np.testing.assert_array_equal(durations,data['durations'])
            metric['durations_equal'] = True
        metric['accepted'] = metric['normalized_peak_error'] < 2e-5 and (metric['exact'] or (metric['snr_db'] or -float('inf')) > 100)
        return metric
    validation = {}
    if session:
        for path in sorted((args.output/'cases').glob('*.npz')):
            case = dict(np.load(path))
            validation[path.stem] = validate(*run(case),case)
    row = dict(model=args.worker+'-native', threads=args.threads,runs=args.runs,times_ms=elapsed,
               median_ms=statistics.median(elapsed),p95_ms=float(np.percentile(elapsed,95)),peak_rss_mib=rss,
               stage_times_ms=stage_times,dsp_times_ms=dsp_times,samples=result.shape[-1],
               boundary_output_bytes=boundary_bytes[0] if boundary_bytes else 0,
               stage_model_bytes=(target/'stage.onnx').stat().st_size if session else 0,windows_bytes=(target/'windows.npz').stat().st_size,
               **validate(result,durations,data),validation=validation)
    print(json.dumps(row))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--measure-only',action='store_true')
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--worker',choices=['filter','full'])
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--runs',type=int,default=30)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.measure_only:
        if not args.checkpoint:p.error('--checkpoint required to prepare')
        prepare(args)
    # Fresh dense controls in the same suite avoid comparing only to an earlier run.
    order = [(kind, backend, threads) for kind in ('filter','full') for backend in ('dense','native') for threads in (1,4)]
    if args.reverse:order.reverse()
    rows=[]
    for kind,backend,threads in order:
        if backend == 'native':
            command = [sys.executable,__file__,'--output',str(args.output),'--worker',kind,'--threads',str(threads),'--runs',str(args.runs)]
        else:
            command = [sys.executable,str(Path(__file__).with_name('benchmark_fft.py')),'--worker',str(args.output/f'{kind}-dense.onnx'),'--threads',str(threads),'--runs',str(args.runs)]
        raw=subprocess.check_output(command,text=True)
        row=json.loads(raw);rows.append(row);print(json.dumps(row),flush=True)
    (args.output/('native-reverse.json' if args.reverse else 'native-results.json')).write_text(json.dumps(dict(order='reverse' if args.reverse else 'forward',numpy=__import__('numpy').__version__,rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
