"""Evaluate the French phonetic frontend on a verified reference benchmark.

Usage:
    PYTHONPATH=src:tests python scripts/eval_fr_frontend.py [--json out.json] [--show-failures]
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from tiny_voca.text import PHONE_SET, UNK, phonemize

DEFAULT_BENCHMARK = Path(__file__).resolve().parents[1] / "tests" / "french_phoneme_benchmark.json"


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if x == y else 1 + min(prev, dp[j], dp[j - 1])
    return dp[-1]


def load_benchmark(path=DEFAULT_BENCHMARK):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for case in cases:
        cid = case["id"]
        if cid in seen:
            raise ValueError(f"duplicate benchmark id {cid!r}")
        seen.add(cid)
        if not case["variants"]:
            raise ValueError(f"case {cid!r} has no reference variants")
        for variant in case["variants"]:
            phones = variant.split()
            bad = [p for p in phones if p not in PHONE_SET]
            if bad:
                raise ValueError(f"case {cid!r} uses non-inventory phones {bad}")
    return cases


def evaluate(cases):
    by_cat = defaultdict(lambda: {"total": 0, "exact": 0, "edits": 0, "ref_phones": 0, "unknowns": 0})
    failures = []
    for case in cases:
        cat = case["category"]
        result = phonemize(case["text"])
        got_phones = [s for s in result.symbols if s in PHONE_SET or s == UNK]
        got_str = " ".join(got_phones)
        variants = [v.split() for v in case["variants"]]
        exact = got_phones in variants
        best_dist, best_ref = min(((edit_distance(got_phones, ref), ref) for ref in variants), key=lambda x: x[0])

        bucket = by_cat[cat]
        bucket["total"] += 1
        bucket["exact"] += int(exact)
        bucket["edits"] += best_dist
        bucket["ref_phones"] += len(best_ref)
        bucket["unknowns"] += len(result.unknown)

        if not exact:
            failures.append({
                "id": case["id"],
                "category": cat,
                "text": case["text"],
                "got": got_str,
                "expected": case["variants"],
                "edits": best_dist,
                "unknown": list(result.unknown),
            })

    total = sum(b["total"] for b in by_cat.values())
    exact_total = sum(b["exact"] for b in by_cat.values())
    edits_total = sum(b["edits"] for b in by_cat.values())
    ref_total = sum(b["ref_phones"] for b in by_cat.values())
    unk_total = sum(b["unknowns"] for b in by_cat.values())

    categories = {}
    for cat, b in by_cat.items():
        categories[cat] = {
            "total": b["total"],
            "exact": b["exact"],
            "accuracy": round(b["exact"] / b["total"], 4),
            "per": round(b["edits"] / max(b["ref_phones"], 1), 4),
            "unknowns": b["unknowns"],
        }
    return {
        "total": total,
        "exact": exact_total,
        "accuracy": round(exact_total / max(total, 1), 4),
        "per": round(edits_total / max(ref_total, 1), 4),
        "unknowns": unk_total,
        "categories": categories,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    cases = load_benchmark(args.benchmark)
    report = evaluate(cases)
    print(f"Total: {report['exact']}/{report['total']} ({report['accuracy']*100:.1f}%) | PER: {report['per']*100:.2f}% | unknowns: {report['unknowns']}")
    print(f"{'Category':32} {'Exact':>9} {'Acc':>8} {'PER':>8} {'Unk':>5}")
    print("-" * 66)
    for cat, stats in report["categories"].items():
        print(f"{cat:32} {stats['exact']:3d}/{stats['total']:<5d} {stats['accuracy']*100:7.1f}% {stats['per']*100:7.2f}% {stats['unknowns']:5d}")
    if args.show_failures:
        print(f"\nFailures ({len(report['failures'])}):")
        for f in report["failures"]:
            print(f"  [{f['id']}] {f['text']!r}\n    got:      {f['got']}\n    expected: {' | '.join(f['expected'])}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
