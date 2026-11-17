"""Synthetic fixtures. No real recordings, eSpeak or network are needed."""

import hashlib
import json
import math
from pathlib import Path
import torch

from tiny_voca.config import Config
from tiny_voca.text import SYMBOL_TO_ID, frontend_fingerprint


def small_config():
    return Config(channels=32, heads=2, feedforward=64, kernel=3, encoder_layers=1, prosody_layers=1, acoustic_layers=1,
                  predictor_channels=16, cep_causal=12, cep_anti=4, noise_cep=8, max_frames=400)


SYMBOLS = ("<bos>", "b", "ɔ̃", "ʒ", "u", "<sp>", "t", "u", "s", "a", ".", "<eos>")
STRESS = (0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)


def sample(c, voice=0, durations=(3, 4, 5, 4, 9, 0, 4, 10, 5, 4, 0, 6)):
    frames = sum(durations)
    time = torch.arange(frames * c.hop) / c.sample_rate
    pitch = 0.2 * torch.sin(torch.linspace(0, 3, frames)) - (0.6 if voice else 0.0)
    voiced = torch.ones(frames)
    voiced[:durations[0]] = 0
    voiced[-durations[-1]:] = 0
    return {"tokens": torch.tensor([SYMBOL_TO_ID[s] for s in SYMBOLS]), "stress": torch.tensor(STRESS),
            "durations": torch.tensor(durations), "speaker": voice,
            "wave": 0.1 * torch.sin(2 * math.pi * (120 if voice else 210) * time), "mel": torch.randn(frames, c.n_mels) - 5,
            "prosody": torch.stack((pitch, voiced, torch.zeros(frames)), dim=-1), "pitch_trusted": voiced.clone()}


def write_annotations(root, c, split, count=4, complete=True, annotation="test-annotation", edit=None):
    """Feature files plus an annotation manifest shaped like the real pipeline's."""
    root = Path(root)
    features, annotations = root / "features", root / "annotations"
    (features / split).mkdir(parents=True, exist_ok=True)
    annotations.mkdir(exist_ok=True)
    rows = []
    for i in range(count):
        item = sample(c, voice=i % 2, durations=(3 + i, 4, 5, 4, 9, i % 2, 4, 10, 5, 4, 0, 6))
        text = f"hello world {split} {i // 2}."
        path = f"{split}/{split}{i}.pt"
        torch.save({"wave": item["wave"], "mel": item["mel"], "speaker": item["speaker"],
                    "prosody": torch.stack((item["prosody"][:, 0] * item["prosody"][:, 1], item["prosody"][:, 1],
                                            torch.full((item["mel"].shape[0],), -3.0)), dim=-1),
                    "tokens": torch.tensor([1, 60, 75, 2])}, features / path)   # stale byte tokens: must be ignored
        row = {"id": f"{split}{i}", "split": split, "speaker": c.speakers[i % 2], "text": text,
               "text_hash": hashlib.sha256(text.encode()).hexdigest(), "path": path, "frames": int(item["mel"].shape[0]),
               "symbols": " ".join(SYMBOLS), "tokens": item["tokens"].tolist(), "stress": item["stress"].tolist(),
               "durations": item["durations"].tolist(), "unknown": [], "qc": {}, "provenance": {"source_license": "test-synthetic"},
               "acoustic": c.acoustic_fingerprint, "frontend": frontend_fingerprint(), "annotation": annotation}
        if edit:
            edit(row)
        rows.append(row)
    (annotations / (split + ".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (annotations / "annotations.json").write_text(json.dumps({"fingerprint": annotation, "complete": complete,
                                                              "acoustic_root": str(features.resolve())}))
    return annotations / (split + ".jsonl")
