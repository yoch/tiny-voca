"""CPU numerical qualification for English or a compatible Tiny Voca fork."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_bundle import acoustic_config, constants, digest, export_compact
from tiny_voca.export import Exportable, INPUTS, example_inputs, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize


class MixedSourceStage(Exportable):
    """Keep the French fork's frequency-dependent mix in ONNX."""
    def _filter(self, source, real, imaginary, name, fft, hop):
        self.parts.extend((source,real,imaginary))
        return torch.zeros_like(source)

    def forward(self, *inputs):
        self.parts = []
        _, durations = super().forward(*inputs)
        return (*self.parts,durations)


def export_mixed(model,path):
    inputs = (*example_inputs(),torch.randn(1,model.config.max_frames*model.config.hop))
    names = (*INPUTS,'noise')
    outputs = ('sources','hr','hi','noise_source','nr','ni','durations')
    axes = {name:{1:name+'_length'} for name in ('tokens','stress','noise',*outputs)}
    with torch.no_grad():
        torch.onnx.export(MixedSourceStage(model,external_noise=True),inputs,str(path),
                          input_names=names,output_names=outputs,dynamic_axes=axes,
                          opset_version=17,dynamo=False)


class Renderer:
    def __init__(self, library, data):
        # The qualification harness takes a build made for the local CPU. There
        # is no ISA-based selection in the app or in the generic DSP API.
        self.lib = ct.CDLL(str(library.resolve()))
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.lib.tv_context_size.restype = ct.c_size_t
        self.size = self.lib.tv_context_size()
        self.context = ct.create_string_buffer(self.size)
        self.lib.tv_initialize.argtypes = [ct.c_void_p, ct.c_size_t, array, ct.c_size_t]
        self.lib.tv_render.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int] + [array, ct.c_size_t] * 6
        self.lib.tv_filter.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int,
                                      array, ct.c_size_t, array, array, ct.c_size_t, array, ct.c_size_t]
        if not self.lib.tv_initialize(self.context, self.size, data, data.size):
            raise ValueError('Failed to initialize DSP')

    def render(self, pulses, harmonic, noise, noise_head):
        output = np.empty_like(pulses)
        scratch = np.empty_like(output)
        buffers = (pulses, harmonic, noise, noise_head, output, scratch)
        frames = harmonic.shape[1]
        args = [value for array in buffers for value in (array, array.size)]
        if not self.lib.tv_render(self.context, self.size, frames, *args):
            raise ValueError('DSP rejected audio or dimensions')
        return output

    def filter(self, branch, source, real, imag):
        source, real, imag = (np.ascontiguousarray(part) for part in (source,real,imag))
        output = np.empty_like(source)
        if real.shape != imag.shape or not self.lib.tv_filter(self.context,self.size,branch,
                source,source.size,real,imag,real.size,output,output.size):
            raise ValueError('DSP rejected spectral filter inputs')
        return output


def session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])


def qualify(args):
    if digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Unexpected checkpoint hash')
    frontend = match_frontend(args.checkpoint)
    if not frontend:
        raise ValueError('Checkpoint frontend did not match installed frontend')
    model = load_model(args.checkpoint)
    geometry = acoustic_config(model.config)
    args.output.mkdir(parents=True, exist_ok=True)
    dense_path = args.output/'dense.onnx'
    compact_path = args.output/'compact.onnx'
    export(model,dense_path,external_noise=True)
    mixed = hasattr(model.vocoder,'aperiodicity')
    if mixed:
        if args.language != 'fr':
            raise ValueError('Mixed excitation qualification expects the French fork')
        export_mixed(model,compact_path)
    else:
        export_compact(model,compact_path)
    renderer = Renderer(args.library,constants(model).numpy())
    cases = [
        ('Fresh fish and shells. Did you hear that?', 0,.9,.1,1.4),
        ('Wait… then listen to the final soft breath.', 1,1.2,-.1,.8),
        ('We paused to watch the quiet waves and listen to the wind in the trees.' * 3, 0,1.1,0,1),
    ] if args.language == 'en' else [
        ('Bonjour ! Comment allez-vous aujourd’hui ?', 0,1,.08,1.2),
        ('Je cherche six chaises sèches. Chut… écoutez le vent.', 1,.9,-.1,1.4),
        ('Sur le rivage, la brise souffle et les vagues viennent mourir sur les rochers. ' * 3, 1,1.1,0,1),
    ]
    result = []
    for threads in (1,4):
        dense = session(dense_path,threads)
        compact = session(compact_path,threads)
        for text, voice, speed, pitch, pitch_range in cases:
            phones = phonemize(text)
            inputs = dict(tokens=np.array([phones.ids],np.int64),stress=np.array([phones.stress],np.int64),
                          voice=np.array([voice],np.int64),
                          **{name:np.array([value],np.float32) for name, value in
                             (('speed',speed),('pitch_shift',pitch),('pitch_range',pitch_range))})
            rng = np.random.default_rng(42)
            if mixed:
                noise = rng.standard_normal((1,model.config.max_frames*model.config.hop),dtype=np.float32)
                sources, hr, hi, noise_source, nr, ni, durations = compact.run(None,{**inputs,'noise':noise})
                harmonic = sum(renderer.filter(0,source,real,imag) for source,real,imag in zip(sources,hr,hi))
                actual = harmonic[None] + renderer.filter(1,noise_source,nr,ni)
            else:
                harmonic, noise_head, pulses, durations = compact.run(None,inputs)
                noise = rng.standard_normal(pulses.shape,dtype=np.float32)
                actual = renderer.render(pulses,harmonic,noise,noise_head)
            reference, expected_durations = dense.run(None,{**inputs,'noise':noise})
            np.testing.assert_array_equal(durations,expected_durations)
            assert actual.shape == reference.shape and np.isfinite(actual).all()
            signal = reference.astype(np.float64)
            delta = signal - actual
            normalized = np.abs(delta).max()/np.abs(signal).max()
            snr = 10*np.log10(np.sum(signal**2)/np.sum(delta**2))
            assert snr > 100 and normalized < 2e-5, (text,snr,normalized)
            result.append(dict(threads=threads,text=text,voice=model.config.speakers[voice],
                               samples=actual.size,snr_db=snr,normalized_peak_error=normalized))
    return dict(language=args.language,frontend=frontend,mixed_excitation=mixed,geometry=geometry,ort=ort.__version__,
                checkpoint_sha256=args.checkpoint_sha256,library_sha256=digest(args.library),rows=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--language', choices=('en','fr'), default='en')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    print(json.dumps(qualify(args),indent=2,ensure_ascii=False))
