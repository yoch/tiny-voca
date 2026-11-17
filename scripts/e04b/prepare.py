"""Prepare checkpoint-matched E04-B graphs, text cases and deterministic noise."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_fft_native import prepare as prepare_late
from tiny_voca.export import Exportable, INPUTS, example_inputs, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize

CHECKPOINT_SHA = 'ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70'
TEXTS = (
    (0.5, "Look up.", 0, .1, 1.2),
    (1, "Close the window.", 1, -.1, .8),
    (2, "The quiet wind moved across the water.", 0, 0, 1),
    (5, "When the rain finally stopped, we walked across the old wooden bridge and watched the river below.", 1, .06, 1.3),
    (10, "At the edge of the garden, where the path curved past the stone wall and the tall trees cast their shadows over the grass, a little bird sang while we waited for the last light to fade behind the hills.", 0, -.06, .75),
)


class Heads(Exportable):
    def __init__(self, model, keep_pulses):
        super().__init__(model)
        self.keep_pulses = keep_pulses

    def _vocoder(self, hidden, f0, total, noise):
        v = self.model.vocoder
        return v.harmonic(hidden), v.noise(hidden), self._pulses(f0) if self.keep_pulses else f0

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range):
        outputs, durations = super().forward(tokens, stress, voice, speed, pitch_shift, pitch_range)
        return (*outputs, durations)


def prepare(checkpoint, root):
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError('Unexpected public checkpoint SHA256')
    torch.manual_seed(0)
    torch.set_num_threads(1)
    match_frontend(checkpoint)
    model = load_model(checkpoint)
    config = model.config
    expected = (24000, 256, 512, 128, 256, 64, 80, 16, 32, .47)
    actual = tuple(getattr(config, k) for k in ('sample_rate','hop','synth_fft','synth_hop','noise_fft','noise_hop','cep_causal','cep_anti','noise_cep','harmonic_limit'))
    if actual != expected:
        raise ValueError('DSP specializes the public checkpoint configuration')
    root.mkdir(parents=True, exist_ok=True)
    export(model, root/'dense.onnx', external_noise=True)
    prepare_late(SimpleNamespace(checkpoint=checkpoint, output=root))
    for graph, source in (('compact', 'pulses'), ('early', 'f0')):
        path = root / (graph+'.onnx')
        outputs = ('harmonic', 'noise_head', source, 'durations')
        axes = {name: {1: 'tokens' if name in ('tokens','stress','durations') else 'samples' if name == 'pulses' else 'frames'} for name in (*INPUTS[:2], *outputs)}
        with torch.no_grad():
            torch.onnx.export(Heads(model, graph == 'compact').eval(), example_inputs(), str(path),
                              input_names=list(INPUTS), output_names=list(outputs), dynamic_axes=axes,
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(path), full_check=True)
    v = model.vocoder
    constants = [getattr(v,k).numpy() for k in ('synth_window','noise_window','causal_lifter','anti_lifter','noise_lifter')]
    (root/'constants.json').write_text(json.dumps([x.tolist() for x in constants]))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    dense = ort.InferenceSession(str(root/'dense.onnx'), opts, providers=['CPUExecutionProvider'])
    # The noise generation algorithm is out of scope. Reuse exact samples for every backend.
    noise = np.random.default_rng(42).standard_normal(config.hop * config.max_frames, dtype=np.float32)
    noise.astype('<f4').tofile(root/'noise.f32')
    cases = []
    for seconds, sentence, voice, pitch, spread in TEXTS:
        phones = phonemize(sentence)
        data = dict(tokens=np.array([phones.ids]), stress=np.array([phones.stress]), voice=np.array([voice]),
                    speed=np.array([1], np.float32), pitch_shift=np.array([pitch],np.float32),
                    pitch_range=np.array([spread],np.float32), noise=noise[None])
        frames = dense.run(['durations'], data)[0].sum()
        speed = float(np.clip(frames * config.hop / (config.sample_rate * seconds), .25, 4))
        data['speed'][0] = speed
        frames = int(dense.run(['durations'], data)[0].sum())
        cases.append(dict(id=str(seconds), text=sentence, tokens=list(phones.ids), stress=list(phones.stress),
                          voice=voice, speed=speed, pitch_shift=pitch, pitch_range=spread, frames=frames,
                          seconds=frames*config.hop/config.sample_rate))
    (root/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    import dizione
    package = Path(dizione.__file__).parent
    with zipfile.ZipFile(root/'dizione.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in package.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                z.write(path, Path('dizione')/path.relative_to(package))
    print([(c['id'],c['seconds'],len(c['tokens']),c['speed']) for c in cases])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.root)
