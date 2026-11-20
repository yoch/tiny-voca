"""What PyTorch gives for the ten test sentences, for the JavaScript port to be checked against.

    python js/test/reference.py --checkpoint tiny-voca.pt --output reference

Writes ``js/test/sentences.json`` (the texts with their phoneme ids and stress,
from dizione) and, for every sentence and voice, into ``--output``: the frames
per phoneme, the pitch per frame, the noise that was used and the waveform.
"""

import argparse
import json
from pathlib import Path

import torch

from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize

SENTENCES = (
    "Hello! This is tiny voca.",
    "The weather is nice today, and I feel very relaxed.",
    "Did you really pay $42.50 for that on March 3rd, 2024?",
    "She sells seashells by the seashore; the shells she sells are surely seashells.",
    "After a long and difficult journey across the mountains, the travellers finally reached the small village, "
    "where they were welcomed with warm food, a place to sleep and many questions about the road behind them.",
    "No.",
    "What time does the last train leave?",
    "I read the book you gave me, and I will read the next one soon.",
    "The quick brown fox jumps over the lazy dog near the quiet river bank.",
    "It runs in a web page, with no libraries at all: one script and one file.",
)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, default=Path(__file__).with_name("sentences.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, text in enumerate(SENTENCES):
        phones = phonemize(text)
        rows.append({"text": text, "phonemes": " ".join(phones.symbols), "ids": list(phones.ids), "stress": list(phones.stress)})
        tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
        for voice, name in enumerate(model.config.speakers):
            wave, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]))
            samples = int(lengths[0])
            noise = torch.randn(samples, generator=torch.Generator().manual_seed(0))
            stem = str(args.output / f"{number:02d}-{name}")
            Path(stem + ".json").write_text(json.dumps({"durations": details["durations"][0].tolist()}))
            model.frequency(details["prosody"][0, :, 0]).numpy().tofile(stem + ".f0.f32")
            noise.numpy().tofile(stem + ".noise.f32")
            wave[0, :samples].numpy().tofile(stem + ".wave.f32")
            print(f"{number:02d} {name} {len(phones)} phonemes {samples / model.config.sample_rate:.2f} s")
    lines = ",\n".join(" " + json.dumps(row, ensure_ascii=False) for row in rows)
    args.sentences.write_text(f"[\n{lines}\n]\n", encoding="utf-8")

if __name__ == "__main__":
    main()
