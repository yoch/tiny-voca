"""Dump real inputs of the waveform head (acoustic hidden states, f0, noise) and the PyTorch wave, plus the head
weights, for the audio.cpp synth benchmark. Run with PYTHONPATH=<tiny-voca>/src.
usage: python dump.py CHECKPOINT OUT_DIR [en|fr]
English: upstream code (github.com/n4ze3m/tiny-voca main) and its public weights (hf n4ze3m/tiny-voca, tiny-voca.pt)."""
import sys
from pathlib import Path
import numpy as np
import torch
from tiny_voca.infer import load_model, speak

ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
lang = sys.argv[3] if len(sys.argv) > 3 else "en"
out.mkdir(parents=True, exist_ok=True)
model = load_model(ckpt)
v = model.vocoder
for name, t in (("hw", v.harmonic.weight), ("hb", v.harmonic.bias), ("nw", v.noise.weight), ("nb", v.noise.bias)):
    t.detach().float().numpy().tofile(out / f"{name}.f32")
caught = []
def hook(module, args, kwargs, result):
    hidden, f0, lengths = args[:3]
    caught.append((hidden[0].detach().float().numpy(), f0[0].detach().float().numpy(), kwargs.get("noise")[0].detach().float().numpy(), result[0].detach().float().numpy()))
v.register_forward_hook(hook, with_kwargs=True)
texts = {"fr": ["Bonjour.", "Le train de onze heures arrive en gare de Lyon.",
               "Il faut parfois du temps pour comprendre ce qui se passe vraiment, et plus encore pour l'expliquer simplement.",
               "En deux mille quatorze, la ville comptait plus de trois cent mille habitants, selon les chiffres officiels publiés au printemps dernier."],
         "en": ["Hello.", "The eleven o'clock train is now arriving on platform two.",
               "Sometimes it takes a while to understand what is really going on, and even longer to explain it simply.",
               "In twenty fourteen, the city had more than three hundred thousand residents, according to official figures published last spring."]}[lang]
cases = []
for i, text in enumerate(texts):
    for voice in model.config.speakers:
        caught.clear()
        speak(model, text, voice)
        for j, (h, f0, noise, wave) in enumerate(caught):
            name = f"u{i}v{voice}s{j}"
            frames = h.shape[0]
            h.tofile(out / f"{name}.hidden"); f0.tofile(out / f"{name}.f0"); noise[: frames * 256].tofile(out / f"{name}.noise"); wave.tofile(out / f"{name}.wave")
            cases.append(f"{name} {frames}")
(out / "cases.txt").write_text("\n".join(cases) + "\n")
print("\n".join(cases))
