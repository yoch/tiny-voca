"""Sequential E04-B CPU suite. Each variant/thread pair uses a fresh process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_fft import errors
from benchmark_fft_native import native_filter
from tiny_voca.infer import finished, sentences
from tiny_voca.text import phonemize, set_frontend

GRAPHS = dict(dense='dense.onnx', numpy='native/stage.onnx', fused='native/stage.onnx',
              fused2='native/stage.onnx',
              compact='compact.onnx', early='early.onnx')
FLOAT_ARRAY = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


class DSP:
    def __init__(self, root, samples):
        self.lib = ct.CDLL(str(root/'dsp.so'))
        for name, integers, arrays in (('initialize',0,5), ('filter',2,4), ('compact_filter',2,3),
                                      ('oscillator',1,2), ('mix',1,2)):
            fn = getattr(self.lib,name)
            fn.argtypes = [ct.c_int]*integers+[FLOAT_ARRAY]*arrays
            fn.restype = None
        self.constants = [np.array(x,np.float32) for x in json.loads((root/'constants.json').read_text())]
        self.lib.initialize(*self.constants)
        self.audio = np.empty(samples,np.float32)
        self.noise_audio = np.empty_like(self.audio)
        self.pulses = np.empty_like(self.audio)

    def late(self, parts, pool=None):
        pulses, hr, hi, noise, nr, ni, _ = parts
        samples = pulses.size
        noisy = pool.submit(self.lib.filter,1,samples,noise,nr,ni,self.noise_audio) if pool else None
        self.lib.filter(0,samples,pulses,hr,hi,self.audio)
        if noisy is None:
            self.lib.filter(1,samples,noise,nr,ni,self.noise_audio)
        else:
            noisy.result()
        self.lib.mix(samples,self.audio,self.noise_audio)
        return self.audio[:samples]

    def early(self, parts, noise, f0):
        harmonic, noise_head, source, _ = parts
        frames = harmonic.shape[1]
        samples = frames * 256
        if f0:
            self.lib.oscillator(frames,source,self.pulses)
            source = self.pulses
        self.lib.compact_filter(0,frames,source,harmonic,self.audio)
        self.lib.compact_filter(1,frames,noise,noise_head,self.noise_audio)
        self.lib.mix(samples,self.audio,self.noise_audio)
        return self.audio[:samples]


def worker(root, variant, threads, reverse, runs):
    set_frontend('own')
    cases = json.loads((root/'cases.json').read_text())
    noise = np.fromfile(root/'noise.f32',dtype='<f4')
    clock = lambda: time.perf_counter_ns()/1e6
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    start = clock()
    session = ort.InferenceSession(str(root/GRAPHS[variant]),opts,providers=['CPUExecutionProvider'])
    dsp = DSP(root,noise.size) if variant != 'dense' else None
    pool = ThreadPoolExecutor(max_workers=1) if variant == 'fused2' else None
    load = clock()-start
    names = {i.name for i in session.get_inputs()}

    def say(case):
        start = clock()
        text, = sentences(finished(case['text']))
        phones = phonemize(text)
        tokens, stress = phones.ids, phones.stress
        inputs = dict(tokens=np.array([tokens],np.int64), stress=np.array([stress],np.int64),
                      voice=np.array([case['voice']],np.int64),
                      **{key:np.array([case[key]],np.float32) for key in ('speed','pitch_shift','pitch_range')})
        if 'noise' in names:
            inputs['noise'] = noise[None,:case['frames']*256]
        frontend_end = clock()
        parts = session.run(None,inputs)
        ort_end = clock()
        if variant=='dense':
            audio = parts[0].reshape(-1)
        elif variant=='numpy':
            p, hr, hi, n, nr, ni, _ = parts
            audio = (native_filter(p,hr,hi,dsp.constants[0],128)+native_filter(n,nr,ni,dsp.constants[1],64)).reshape(-1)
        elif variant in ('fused','fused2'):
            audio = dsp.late(parts,pool)
        else:
            audio = dsp.early(parts,noise,variant=='early')
        dsp_end = clock()
        published = audio.copy()
        published *= min(1, .99 / max(1e-9, float(np.abs(published).max())))
        end = clock()
        values = dict(total_ms=end-start, first_audio_ms=end-start, frontend_ms=frontend_end-start,
                      ort_ms=ort_end-frontend_end,dsp_ms=dsp_end-ort_end,publish_ms=end-dsp_end)
        return values, audio, parts[-1], sum(x.nbytes for x in parts)

    first, _, _, _ = say(cases[0])
    rows = []
    for case in cases[::-1] if reverse else cases:
        for _ in range(5):
            say(case)
        timings = [say(case)[0] for _ in range(runs)]
        _, audio, durations, boundary = say(case)
        expected = case['frames']*256
        if audio.size != expected or not np.isfinite(audio).all():
            raise ValueError(f'{variant} returned incorrect audio for {case["id"]}')
        base = root/'cpu-audio'
        base.mkdir(exist_ok=True)
        prefix = base/f'{threads}-{case["id"]}'
        path = Path(f'{prefix}-{variant}.npz')
        np.savez(path,audio=audio,durations=durations)
        sf.write(f'{prefix}-{variant}.wav',audio,24000,subtype='FLOAT')
        rows.append(dict(case=case['id'],samples=audio.size,boundary_bytes=boundary,timings=timings))
    rss = int(re.search(r'^VmHWM:\s+(\d+)',Path('/proc/self/status').read_text(),re.M)[1])/1024
    if pool:
        pool.shutdown()
    print(json.dumps(dict(variant=variant,threads=threads,load_ms=load,first_audio_cold=first,
                          peak_rss_mib=rss,rows=rows)))


def suite(args):
    order = [(threads,v) for threads in args.thread_counts for v in args.variants]
    if args.reverse:
        order.reverse()
    all_rows=[]
    for threads, variant in order:
        command = [sys.executable,__file__,str(args.root),'--worker',variant,'--threads',str(threads),'--runs',str(args.runs)]
        if args.reverse:
            command += ['--reverse']
        raw = subprocess.check_output(command,text=True)
        item = json.loads(raw)
        all_rows.append(item)
        print(threads,variant,[round(np.median([t['total_ms'] for t in r['timings']]),2) for r in item['rows']],flush=True)
    for item in all_rows:
        for row in item['rows']:
            prefix = args.root/'cpu-audio'/f'{item["threads"]}-{row["case"]}'
            signal = np.load(f'{prefix}-{item["variant"]}.npz')
            dense = np.load(f'{prefix}-dense.npz')
            np.testing.assert_array_equal(signal['durations'],dense['durations'])
            row['parity'] = dict(durations_equal=True,**errors(signal['audio'],dense['audio']))
    report=dict(ort=ort.__version__,affinity=sorted(os.sched_getaffinity(0)),
                cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),cases=all_rows)
    direction = 'reverse' if args.reverse else 'forward'
    (args.root/f'{args.result_prefix}-{direction}.json').write_text(json.dumps(report,separators=(',',':'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--worker',choices=GRAPHS)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--runs',type=int,default=30)
    p.add_argument('--variants',nargs='+',choices=GRAPHS,default=['dense','numpy','fused','compact'])
    p.add_argument('--thread-counts',nargs='+',type=int,choices=(1,4),default=[1,4])
    p.add_argument('--result-prefix',default='cpu')
    a=p.parse_args()
    if a.worker:
        worker(a.root,a.worker,a.threads,a.reverse,a.runs)
    else:
        suite(a)
