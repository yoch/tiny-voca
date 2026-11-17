from dataclasses import asdict, dataclass
import hashlib
import json

CHECKPOINT_FORMAT = 4  # 2-3: byte tokens + learned aligner (rejected explicitly).
ANNOTATION_VERSION = 3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    # Acoustic features (unchanged since feature schema 2; existing files are reused).
    sample_rate: int = 24000
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f0_min: float = 50.0
    f0_max: float = 600.0
    speakers: tuple[str, ...] = ("F1", "M1")
    # Phoneme tokens from tiny_voca.text.SYMBOLS (French front-end version 4).
    vocab_size: int = 54
    # Text encoder and frame-level transformer.
    channels: int = 192
    heads: int = 4
    feedforward: int = 640
    kernel: int = 7
    encoder_layers: int = 4
    prosody_layers: int = 2
    acoustic_layers: int = 5
    dropout: float = 0.1
    predictor_channels: int = 128
    # Source-filter waveform head: harmonic branch and noise branch STFT sizes,
    # causal/anticausal complex-cepstrum orders.
    synth_fft: int = 512
    synth_hop: int = 128
    noise_fft: int = 256
    noise_hop: int = 64
    cep_causal: int = 80
    cep_anti: int = 16
    noise_cep: int = 32
    harmonic_limit: float = 0.47
    max_tokens: int = 512
    max_frames: int = 3000
    mel_mean: float = -5.0
    mel_scale: float = 5.0
    energy_mean: float = -4.0
    energy_scale: float = 2.0
    pitch_reference_hz: float = 200.0

    def __post_init__(self):
        from .text import SYMBOLS
        if self.vocab_size != len(SYMBOLS):
            raise ValueError(f"vocab_size must equal the phoneme inventory size {len(SYMBOLS)}")
        if self.channels % self.heads or (self.channels // self.heads) % 2:
            raise ValueError("channels must be divisible by heads with an even head width")
        if not self.speakers or len(set(self.speakers)) != len(self.speakers):
            raise ValueError("speaker names must be unique and nonempty")
        for fft, hop in ((self.synth_fft, self.synth_hop), (self.noise_fft, self.noise_hop)):
            if fft != 4 * hop or self.hop % hop:
                raise ValueError("synthesis windows need 4x overlap and a hop dividing the acoustic hop")
        if self.cep_causal + self.cep_anti >= self.synth_fft // 2 or self.noise_cep >= self.noise_fft // 2:
            raise ValueError("cepstrum orders must be well below half the synthesis FFT size")
        if min(self.encoder_layers, self.prosody_layers, self.acoustic_layers, self.kernel) < 1 or not self.kernel % 2:
            raise ValueError("layer counts must be positive and the kernel odd")
        if min(self.mel_scale, self.energy_scale) <= 0 or not 0 < self.harmonic_limit < 0.5:
            raise ValueError("invalid normalization scale or harmonic limit")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "speakers": tuple(value["speakers"])})

    @property
    def fingerprint(self):
        return _digest(self.to_dict())

    @property
    def acoustic_config(self):
        """Settings that determine stored wave/mel/prosody tensors. Tokens and
        durations are versioned separately by the annotation layer."""
        fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max")
        return {**{key: getattr(self, key) for key in fields}, "feature_version": 2,
                "pitch_reference_hz": 200, "log_floor": 1e-5}

    @property
    def acoustic_fingerprint(self):
        return _digest(self.acoustic_config)

    @property
    def legacy_feature_fingerprint(self):
        """Fingerprint written by the byte-token preparation (front end 2).

        Feature files carrying it hold compatible acoustics and STALE tokens;
        only the annotation step may read them, and it ignores their tokens.
        """
        fields = ("sample_rate", "hop", "n_fft", "n_mels", "f0_min", "f0_max", "speakers")
        return _digest({**{key: getattr(self, key) for key in fields}, "vocab_size": 259, "feature_version": 2,
                        "frontend_version": 2, "pitch_reference_hz": 200, "log_floor": 1e-5})
