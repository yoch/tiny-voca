"""Time real text-to-audio CPU synthesis with fresh sessions."""

import argparse
import json
import resource
import time
from pathlib import Path

from evaluate import cases, feeds, peak_limit, render, session


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def perf_cases(public, output):
    runtime = session(public)
    encoded = list(cases("holdout"))
    measured = []
    for case in encoded:
        audio, _ = render(runtime, feeds(case))
        measured.append((len(audio) / 24000, case))
    # Choose durations before timing candidates, alternating the voices.
    picks = []
    for length, speaker in ((0.5, 0), (1, 1), (2, 0), (5, 1), (10, 0)):
        duration, case = min(
            (m for m in measured if m[1]["voice"] == speaker),
            key=lambda m: abs(m[0] - length),
        )
        picks.append({**case, "label": str(length), "reference_seconds": duration})
    output.write_text(json.dumps(picks, indent=2) + "\n")


def benchmark(args):
    cases_list = json.loads(args.cases.read_text())
    result = {
        "models": [],
        "threads": args.threads,
        "runs": args.runs,
        "warmups": args.warmups,
    }
    # A subprocess per session is used by the caller for clean peak RSS.
    memory_before = rss()
    started = time.perf_counter()
    runtime = session(args.model, args.threads)
    load = (time.perf_counter() - started) * 1000
    list(cases("calibration"))  # initialize the checkpoint-matched frontend

    def request(case):
        t0 = time.perf_counter_ns()
        data = feeds(case, live=True)
        t1 = time.perf_counter_ns()
        audio, durations = render(runtime, data)
        heard = peak_limit(audio).copy()
        t2 = time.perf_counter_ns()
        return {
            "total_ms": (t2 - t0) / 1e6,
            "front_ms": (t1 - t0) / 1e6,
            "samples": len(heard),
            "frames": int(durations.sum()),
        }

    for case in cases_list:
        first = request(case)
        for _ in range(args.warmups):
            request(case)
        values = [request(case) for _ in range(args.runs)]
        result["models"].append(
            {"case": case["label"], "id": case["id"], "first": first, "times": values}
        )
    result.update(
        {
            "load_ms": load,
            "rss_peak_bytes": rss(),
            "rss_before_bytes": memory_before,
            "model": args.model.stem,
        }
    )
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if args.cases:
        benchmark(args)
    else:
        perf_cases(args.model, args.output)
