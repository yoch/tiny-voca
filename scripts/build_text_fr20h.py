"""Build the extra French train scripts for the 20 h dataset (fr20h), with no overlap with fr4k or the evaluation sets.

Reuses build_text_fr.py (same filters, phonemization check, 20 % templates), but pre-seeds the "seen" set with every
fr4k text (train, validation, test) and every text of the quality-comparison protocol, then writes train rows only.

usage: PYTHONPATH=src python scripts/build_text_fr20h.py --fr4k data/scripts_fr_4k --protocol PROTOCOL.json \
         --collector data/raw_text_fr/sentence-collector.txt --count 8700 --output data/scripts_fr20h_extra/train.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_text_fr as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fr4k", type=Path, required=True)
ap.add_argument("--protocol", type=Path, required=True)
ap.add_argument("--collector", type=Path, required=True)
ap.add_argument("--count", type=int, required=True)
ap.add_argument("--template-share", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=20261121)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()

seen = set()
for split in ("train", "validation", "test"):
    for line in (args.fr4k / f"{split}.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        seen.add(r["text"].lower())
        seen.add(r.get("original_text", r["text"]).lower())
protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
for s in protocol["sets"].values():
    for t in s["texts"]:
        seen.add(t.lower())
excluded = len(seen)

rng = random.Random(args.seed)
lines = args.collector.read_text(encoding="utf-8").splitlines()
tmpl_target = round(args.count * args.template_share)
ext_rows = B.build_rows(lines, args.count - tmpl_target, rng, seen, "common_voice_fr", "https://commonvoice.mozilla.org/", "reading")
tmpl_rows = B.build_rows(B.templates(rng, max(60, (args.count - len(ext_rows)) * 10)), args.count - len(ext_rows), rng, seen,
                         "tiny_voca_fr_templates", "local", "pronunciation")
rows = ext_rows + tmpl_rows
for r in rows:
    r["split"] = "train"
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"excluded {excluded} known texts; wrote {len(rows)} rows ({len(ext_rows)} common voice, {len(tmpl_rows)} templates) to {args.output}")
