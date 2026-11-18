"""Build a matched dense fallback and opt-in Web DSP bundle from a verified checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import onnx
import torch

from tiny_voca.export import Exportable, INPUTS, example_inputs, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import SYMBOL_TO_ID

ACOUSTICS = dict(sample_rate=24000, hop=256, synth_fft=512, synth_hop=128,
                 noise_fft=256, noise_hop=64, cep_causal=80, cep_anti=16, noise_cep=32)
OUTPUTS = ('harmonic', 'noise_head', 'pulses', 'durations')


class CompactExportable(Exportable):
    def _vocoder(self, hidden, f0, total, noise):
        vocoder = self.model.vocoder
        return vocoder.harmonic(hidden), vocoder.noise(hidden), self._pulses(f0)

    def forward(self, tokens, stress, voice, speed, pitch_shift, pitch_range):
        parts, durations = super().forward(tokens, stress, voice, speed, pitch_shift, pitch_range)
        return (*parts, durations)


def acoustic_config(config):
    geometry = {key: getattr(config, key, None) for key in ACOUSTICS}
    if geometry != ACOUSTICS or not 1 <= config.max_frames <= 16384:
        raise ValueError(f'Unsupported vocoder geometry: {geometry}')
    return geometry


def export_compact(model, path, wrapper=CompactExportable):
    """The DSP contract is acoustic geometry; forks may supply their own exporter wrapper."""
    acoustic_config(model.config)
    if wrapper is CompactExportable and hasattr(model.vocoder, 'aperiodicity'):
        raise ValueError('A voice-dependent source requires an explicit exporter adapter')
    axes = {name: {1: 'tokens' if name in ('tokens', 'stress', 'durations')
                  else 'samples' if name == 'pulses' else 'frames'}
            for name in (*INPUTS[:2], *OUTPUTS)}
    with torch.no_grad():
        torch.onnx.export(wrapper(model).eval(), example_inputs(), str(path),
                          input_names=list(INPUTS), output_names=list(OUTPUTS),
                          dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path), full_check=True)


def constants(model):
    v = model.vocoder
    return torch.cat([getattr(v, name).detach().cpu() for name in
                      ('synth_window', 'noise_window', 'causal_lifter', 'anti_lifter', 'noise_lifter')])


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def build(checkpoint, expected_sha, config_file, wasm, output, test_reference=False):
    if not re.fullmatch(r'[a-f0-9]{64}', expected_sha) or digest(checkpoint) != expected_sha:
        raise ValueError('Checkpoint SHA256 does not match the supplied provenance')
    if match_frontend(checkpoint) != 'own':
        raise ValueError('This Web frontend uses dizione and requires a matching checkpoint')
    model = load_model(checkpoint)
    geometry = acoustic_config(model.config)
    config = json.loads(config_file.read_text())
    checkpoint_meta = torch.load(checkpoint, map_location='cpu', weights_only=True)
    frontend_hash = checkpoint_meta['frontend']['fingerprint']
    if (config['model'] != json.loads(json.dumps(model.config.to_dict()))
            or config['phonemizer']['name'] != 'dizione'
            or config['phonemizer']['fingerprint'] != frontend_hash
            or config.get('step') != checkpoint_meta.get('step')
            or config['vocab'] != SYMBOL_TO_ID
            or config['voices'] != list(model.config.speakers)
            or config['hop'] != geometry['hop'] or config['sample_rate'] != geometry['sample_rate']):
        raise ValueError('Browser config does not match checkpoint metadata')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        target = Path(tmp)
        export(model, target/'model.onnx')
        export_compact(model, target/'model-compact.onnx')
        constants(model).numpy().astype('<f4').tofile(target/'dsp-constants.f32')
        shutil.copyfile(wasm, target/'dsp.wasm')
        if test_reference:
            export(model, target/'dense-reference.onnx', external_noise=True)
        names = ('model.onnx', 'model-compact.onnx', 'dsp-constants.f32', 'dsp.wasm')
        config['vocoder_pilot'] = {
            'abi': 1, 'geometry': geometry, 'checkpoint_sha256': expected_sha,
            'frontend_fingerprint': frontend_hash,
            'assets': {name: {'sha256': digest(target/name), 'bytes': (target/name).stat().st_size}
                       for name in names},
        }
        for file in target.iterdir():
            file.replace(output/file.name)
        (target/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        (target/'config.json').replace(output/'config.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-reference', action='store_true', help='Also generate a dense graph with explicit input noise')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    build(args.checkpoint,args.checkpoint_sha256,args.config,args.wasm,args.output,args.test_reference)
