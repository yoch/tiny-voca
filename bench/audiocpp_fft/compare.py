"""Parity of two synth outputs (float32 dumps): SNR and 16-bit difference after the same peak scaling."""
import sys
from pathlib import Path
import numpy as np
a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted(a_dir.glob("*.f32")):
    a = np.fromfile(f, np.float32).astype(np.float64); b = np.fromfile(b_dir / f.name, np.float32).astype(np.float64)
    assert a.shape == b.shape
    snr = 10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-300))
    scale = 0.95 / np.abs(a).max()      # same scale for both, as a peak limiter would
    q = lambda x: np.clip(np.round(x * scale * 32767), -32768, 32767).astype(np.int32)
    d = np.abs(q(a) - q(b))
    print(f"{f.stem:9s} samples {a.size:7d}  SNR {snr:6.1f} dB  16-bit max diff {d.max()} LSB, {np.count_nonzero(d)} samples differ")
