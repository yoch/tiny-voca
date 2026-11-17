"""Test the shared DSP on boundary and waveform cases against the E04 reference."""
import argparse
from pathlib import Path

import numpy as np

from cpu import DSP, native_filter


def verify(root):
    rng = np.random.default_rng(42)
    dsp = DSP(root,24500)
    count = 0
    for branch, fft in enumerate((512,256)):
        hop = fft//4
        for samples in (1, hop-1, hop, hop+1, fft-1, fft, fft+1, 24001):
            response_shape = (1, samples//hop+3, fft//2+1)
            response = [rng.normal(0,.2,size=response_shape).astype(np.float32) for _ in range(2)]
            signal = rng.standard_normal((1,samples),dtype=np.float32)
            identity = [np.ones(response_shape,np.float32),np.zeros(response_shape,np.float32)]
            zeros = np.zeros_like(signal)
            impulse = zeros.copy()
            impulse[0,[0,min(hop-1,samples-1),min(hop,samples-1),samples-1]] = 1
            sine = np.sin(2*np.pi*330*np.arange(samples)/24000).astype(np.float32)[None]
            for x, (real,imag) in ((signal,response),(signal,identity),(zeros,response),
                                   (impulse,response),(sine,response)):
                ref = native_filter(x,real,imag,dsp.constants[branch],hop)
                dsp.lib.filter(branch,samples,x,real,imag,dsp.audio)
                actual = dsp.audio[:samples]
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual,ref[0],atol=3e-6,rtol=2e-5)
                count += 1

    # Diagnostic F0 oscillator: constant 375 Hz is exactly periodic at 24 kHz.
    # The 100 dB end-to-end acceptance test for libm portability is separate.
    pitch = np.full(8,375,np.float32)
    dsp.lib.oscillator(pitch.size,pitch,dsp.pulses)
    np.testing.assert_allclose(dsp.pulses[64:512],dsp.pulses[:448],atol=.002,rtol=.002)
    pitch[2:6] = [380,450,340,340]
    dsp.lib.oscillator(pitch.size,pitch,dsp.audio)
    assert np.isfinite(dsp.audio[:8*256]).all()
    np.testing.assert_array_equal(dsp.pulses[:256],dsp.audio[:256])
    assert np.linalg.norm(dsp.audio[256:8*256]-dsp.pulses[256:8*256]) > 1
    print(f'{count} filter comparisons and constant/transition F0 diagnostics passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    verify(p.parse_args().root)
