"""Compare baseline (192ch / FFN 640 / ~4.53M) vs E06 compact (160ch / FFN 384 / ~2.66M) architecture on CPU.

Measures parameter counts, ONNX file size (omit_mask=True), node counts, and CPU single/4-thread latency
without modifying any weights or running GPU training.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tiny_voca.config import Config
from tiny_voca.export import export
from tiny_voca.model import TinyVoca
from tiny_voca.train import TrainingSystem

BENCH_CASES = (
    ("short", 19, 180),
    ("medium", 60, 390),
    ("long", 132, 850),
)


def _measure_variant(name: str, cfg_path: Path, threads: int, warmup: int, runs: int) -> dict:
    cfg = Config.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    torch.manual_seed(0)
    model = TinyVoca(cfg).eval()
    # Pin duration bias so untrained model produces realistic utterance frame counts (~5 frames/token).
    with torch.no_grad():
        model.duration.out.weight.zero_()
        model.duration.out.bias[:] = torch.tensor([1.6, 1.0])

    sys_mod = TrainingSystem(cfg)
    inf_params = sum(p.numel() for p in model.parameters())
    train_only_params = sum(p.numel() for p in sys_mod.mel_head.parameters())

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / f"{name}.onnx"
        export(model, onnx_path, external_noise=True, omit_mask=True)
        size_bytes = onnx_path.stat().st_size
        graph = onnx.load(str(onnx_path))
        node_count = len(graph.graph.node)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        latencies = {}
        noise_buf = np.zeros((1, cfg.hop * cfg.max_frames), dtype=np.float32)
        voice = np.array([0], dtype=np.int64)
        one = np.array([1.0], dtype=np.float32)
        zero = np.array([0.0], dtype=np.float32)
        for label, tok_len, _ in BENCH_CASES:
            tokens = np.full((1, tok_len), 18, dtype=np.int64)
            tokens[0, 0] = 1
            tokens[0, -1] = 2
            stress = np.zeros_like(tokens)
            feeds = {
                "tokens": tokens,
                "stress": stress,
                "voice": voice,
                "speed": one,
                "pitch_shift": zero,
                "pitch_range": one,
                "noise": noise_buf,
            }
            for _ in range(warmup):
                sess.run(None, feeds)
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                out_audio, _ = sess.run(None, feeds)
                times.append((time.perf_counter() - t0) * 1000.0)
            audio_s = out_audio.shape[1] / cfg.sample_rate
            med_ms = statistics.median(times)
            latencies[label] = {
                "tokens": tok_len,
                "audio_s": round(audio_s, 3),
                "median_ms": round(med_ms, 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2),
                "min_ms": round(min(times), 2),
                "rtfx": round(audio_s / (med_ms / 1000.0), 1),
            }

    return {
        "config": cfg.to_dict(),
        "inference_parameters": inf_params,
        "training_mel_head_parameters": train_only_params,
        "onnx_bytes": size_bytes,
        "onnx_nodes": node_count,
        "latencies": latencies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    load_before = os.getloadavg()
    baseline = _measure_variant("fr_baseline_192_640", ROOT / "configs" / "fr_baseline_192_640.json", args.threads, args.warmup, args.runs)
    e06 = _measure_variant("e06_160_384", ROOT / "configs" / "e06_160_384.json", args.threads, args.warmup, args.runs)
    load_after = os.getloadavg()

    payload = {
        "threads": args.threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "baseline_192_640": baseline,
        "e06_160_384": e06,
        "param_ratio": round(e06["inference_parameters"] / baseline["inference_parameters"], 4),
        "onnx_size_ratio": round(e06["onnx_bytes"] / baseline["onnx_bytes"], 4),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
