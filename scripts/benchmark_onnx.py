"""Reproducible CPU profiling, ONNX optimization, and INT8 quantization benchmark (E03).

Usage:
    PYTHONPATH=src /home/yoch/voice-tts/env/bin/python scripts/benchmark_onnx.py \
        --checkpoint /home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt \
        --out-dir /home/yoch/voice-tts/.local-runs/e03 \
        --warmup 6 --runs 35

Each timed variant/thread/length configuration executes in an isolated child
process (--worker) so peak RSS (/proc/self/status VmHWM and resource.getrusage)
and thread-pool states are uncontaminated across variants.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
import onnxruntime as ort
from scipy.spatial.distance import cdist
import torch

from tiny_voca.export import Exportable, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.losses import SpectralLoss
from tiny_voca.model import cepstrum_response, length_mask, pulse_train
from tiny_voca.text import phonemize

DEFAULT_CHECKPOINT = Path("/home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt")
DEFAULT_OUT_DIR = Path("/home/yoch/voice-tts/.local-runs/e03")

BENCHMARK_SENTENCES = (
    ("short", "Bonjour, comment allez-vous ?"),
    ("medium", "As-tu vraiment payé quarante-deux euros et cinquante centimes pour ce beau livre ?"),
    (
        "long",
        "Après un long voyage à travers les montagnes enneigées, les voyageurs fatigués "
        "atteignirent enfin le petit village où ils furent chaleureusement accueillis avec un repas chaud.",
    ),
)

ORT_OPT_LEVELS = {
    "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_peak_rss_kib() -> dict:
    vm_hwm_kib = None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                vm_hwm_kib = int(line.split()[1])
                break
    ru_maxrss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "vm_hwm_kib": vm_hwm_kib,
        "vm_hwm_mib": round(vm_hwm_kib / 1024.0, 2) if vm_hwm_kib is not None else None,
        "ru_maxrss_kib": ru_maxrss_kib,
        "ru_maxrss_mib": round(ru_maxrss_kib / 1024.0, 2),
    }


def read_loadavg() -> str:
    p = Path("/proc/loadavg")
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def cpu_info() -> dict:
    model_name = platform.processor() or "unknown"
    flags = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and model_name in ("unknown", "", "x86_64"):
                model_name = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line.split(":", 1)[1].strip().split()
    interesting = [f for f in ("sse4_1", "sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni") if f in flags]
    return {
        "cpu_model": model_name,
        "logical_cpus": os.cpu_count(),
        "isa_flags": interesting,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "ort_providers": ort.get_available_providers(),
    }


def summarize_times(times_ms: list[float], audio_seconds: float | None = None) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    q25, q75 = np.percentile(arr, [25, 75])
    out = {
        "runs": len(times_ms),
        "median_ms": round(med, 3),
        "p95_ms": round(p95, 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "iqr_ms": round(float(q75 - q25), 3),
    }
    if audio_seconds is not None and med > 0 and p95 > 0:
        out["audio_seconds"] = round(audio_seconds, 4)
        out["rtfx_median"] = round(audio_seconds / (med / 1000.0), 2)
        out["rtfx_p95"] = round(audio_seconds / (p95 / 1000.0), 2)
    return out


def make_session(model_path: Path, threads: int, opt_level_name: str, enable_profiling: bool = False,
                 profile_prefix: Path | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ORT_OPT_LEVELS[opt_level_name]
    if enable_profiling:
        opts.enable_profiling = True
        if profile_prefix is not None:
            opts.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])


def inspect_onnx_graph(model_path: Path) -> dict:
    model = onnx.load(str(model_path))
    inits = {init.name: list(init.dims) for init in model.graph.initializer}
    op_counts = Counter(node.op_type for node in model.graph.node)

    neural_all_linears = []
    neural_transformer_linears = []
    neural_acoustic_linears = []
    dynamic_attention_matmuls = []
    vocoder_cepstrum_matmuls = []

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        has_init = any(inp in inits for inp in node.input)
        if has_init:
            neural_all_linears.append(node.name)
            if any(prefix in node.name for prefix in ("/encoder/", "/prosody_stack/", "/acoustic_stack/")):
                neural_transformer_linears.append(node.name)
            if "/acoustic_stack/" in node.name:
                neural_acoustic_linears.append(node.name)
        elif "/attention/" in node.name:
            dynamic_attention_matmuls.append(node.name)
        else:
            vocoder_cepstrum_matmuls.append(node.name)

    return {
        "total_nodes": len(model.graph.node),
        "op_counts": dict(op_counts.most_common()),
        "neural_all_linears": neural_all_linears,
        "neural_transformer_linears": neural_transformer_linears,
        "neural_acoustic_linears": neural_acoustic_linears,
        "dynamic_attention_matmuls": dynamic_attention_matmuls,
        "vocoder_cepstrum_matmuls": vocoder_cepstrum_matmuls,
        "conv_nodes": [n.name for n in model.graph.node if n.op_type == "Conv"],
        "conv_transpose_nodes": [n.name for n in model.graph.node if n.op_type == "ConvTranspose"],
        "sin_nodes": [n.name for n in model.graph.node if n.op_type == "Sin"],
        "cos_nodes": [n.name for n in model.graph.node if n.op_type == "Cos"],
        "exp_nodes": [n.name for n in model.graph.node if n.op_type == "Exp"],
    }


def verify_quantized_graph(model_path: Path, expected_quant_nodes: list[str], baseline_audit: dict) -> dict:
    model = onnx.load(str(model_path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    names_by_op = defaultdict(set)
    for node in model.graph.node:
        names_by_op[node.op_type].add(node.name)

    # Ensure protected non-neural / DSP operators stayed completely untouched.
    for op_type, baseline_key in (
        ("Conv", "conv_nodes"),
        ("ConvTranspose", "conv_transpose_nodes"),
        ("Sin", "sin_nodes"),
        ("Cos", "cos_nodes"),
        ("Exp", "exp_nodes"),
        ("MatMul", "vocoder_cepstrum_matmuls"),
        ("MatMul", "dynamic_attention_matmuls"),
    ):
        missing = [name for name in baseline_audit[baseline_key] if name not in names_by_op[op_type]]
        if missing:
            raise RuntimeError(f"{model_path.name}: protected {op_type} nodes altered/missing: {missing}")

    quant_matmuls = [n.name for n in model.graph.node if n.op_type == "MatMulInteger"]
    if len(quant_matmuls) != len(expected_quant_nodes):
        raise RuntimeError(
            f"{model_path.name}: expected {len(expected_quant_nodes)} MatMulInteger, found {len(quant_matmuls)}"
        )

    return {
        "total_nodes": len(model.graph.node),
        "MatMulInteger": op_counts.get("MatMulInteger", 0),
        "DynamicQuantizeLinear": op_counts.get("DynamicQuantizeLinear", 0),
        "MatMul_fp32_remaining": op_counts.get("MatMul", 0),
        "Conv": op_counts.get("Conv", 0),
        "ConvTranspose": op_counts.get("ConvTranspose", 0),
        "Sin": op_counts.get("Sin", 0),
        "Cos": op_counts.get("Cos", 0),
        "Exp": op_counts.get("Exp", 0),
    }


def prepare_models(checkpoint: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")

    fp32_rel = out_dir / "tiny-voca-fp32.onnx"
    fp32_ext = out_dir / "tiny-voca-fp32-extnoise.onnx"
    export(model, fp32_rel, external_noise=False, opset=17)
    export(model, fp32_ext, external_noise=True, opset=17)

    audit = inspect_onnx_graph(fp32_ext)
    if len(audit["neural_all_linears"]) != 50 or len(audit["vocoder_cepstrum_matmuls"]) != 4:
        raise RuntimeError(f"unexpected MatMul breakdown: {audit}")

    quant_specs = (
        ("tiny-voca-int8-linear.onnx", audit["neural_all_linears"], False,
         "50 neural nn.Linear MatMul nodes (per-tensor QInt8, reduce_range=True)"),
        ("tiny-voca-int8-linear-pc.onnx", audit["neural_all_linears"], True,
         "50 neural nn.Linear MatMul nodes (per-channel QInt8, reduce_range=True)"),
        ("tiny-voca-int8-transformer.onnx", audit["neural_transformer_linears"], False,
         "44 Transformer stack nn.Linear MatMul nodes (heads kept FP32, per-tensor QInt8, reduce_range=True)"),
        ("tiny-voca-int8-acoustic.onnx", audit["neural_acoustic_linears"], False,
         "20 acoustic_stack nn.Linear MatMul nodes (encoder/prosody/heads/vocoder FP32, per-tensor QInt8, reduce_range=True)"),
    )

    artifacts = {}
    for path, desc in (
        (fp32_rel, "FP32 release export (external_noise=False)"),
        (fp32_ext, "FP32 exact-comparison export (external_noise=True)"),
    ):
        artifacts[path.name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / 1e6, 3),
            "size_mib": round(path.stat().st_size / (1024 * 1024), 3),
            "sha256": sha256_file(path),
            "description": desc,
            "graph_ops": inspect_onnx_graph(path)["op_counts"],
        }

    for filename, target_nodes, per_channel, desc in quant_specs:
        q_path = out_dir / filename
        quantize_dynamic(
            str(fp32_ext),
            str(q_path),
            nodes_to_quantize=target_nodes,
            per_channel=per_channel,
            reduce_range=True,
            weight_type=QuantType.QInt8,
        )
        q_verify = verify_quantized_graph(q_path, target_nodes, audit)
        artifacts[filename] = {
            "path": str(q_path),
            "size_bytes": q_path.stat().st_size,
            "size_mb": round(q_path.stat().st_size / 1e6, 3),
            "size_mib": round(q_path.stat().st_size / (1024 * 1024), 3),
            "sha256": sha256_file(q_path),
            "description": desc,
            "quantized_nodes_count": len(target_nodes),
            "verification": q_verify,
        }

    return {"audit": audit, "artifacts": artifacts}


def dtw_distance_np(mel_a: np.ndarray, mel_b: np.ndarray) -> float:
    """Mean absolute log-mel distance along the best monotonic alignment."""
    cost = cdist(mel_a.T, mel_b.T, metric="cityblock") / mel_a.shape[0]
    m, n = cost.shape
    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, m + 1):
        row = cost[i - 1]
        prev = dp[i - 1]
        cur = dp[i]
        for j in range(1, n + 1):
            cur[j] = row[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
    i, j, steps = m, n, 0
    while i > 0 or j > 0:
        steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag, up, left = dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1
                j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1
    return float(dp[m, n] / steps)


def evaluate_fidelity(checkpoint: Path, out_dir: Path) -> list[dict]:
    torch.set_num_threads(4)
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")
    wrapper = Exportable(model, external_noise=True).eval()
    c = model.config
    mel_loss = SpectralLoss(c.sample_rate, mel=(c.n_fft, c.hop, c.n_mels))

    candidates = [
        ("torch_fp32_exportable", None, None, True),
        ("ort_fp32_ext_opt_off", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_DISABLE_ALL", True),
        ("ort_fp32_ext_opt_basic", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_ENABLE_BASIC", True),
        ("ort_fp32_ext_opt_all", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_ENABLE_ALL", True),
        ("ort_int8_linear", out_dir / "tiny-voca-int8-linear.onnx", "ORT_ENABLE_ALL", True),
        ("ort_int8_linear_pc", out_dir / "tiny-voca-int8-linear-pc.onnx", "ORT_ENABLE_ALL", True),
        ("ort_int8_transformer", out_dir / "tiny-voca-int8-transformer.onnx", "ORT_ENABLE_ALL", True),
        ("ort_int8_acoustic", out_dir / "tiny-voca-int8-acoustic.onnx", "ORT_ENABLE_ALL", True),
        ("ort_fp32_rel_opt_all", out_dir / "tiny-voca-fp32.onnx", "ORT_ENABLE_ALL", False),
    ]

    sessions = {}
    for cand_id, path, opt_name, _ in candidates:
        if path is not None:
            sessions[cand_id] = make_session(path, threads=4, opt_level_name=opt_name)

    rows = []
    for sent_key, text in BENCHMARK_SENTENCES:
        phones = phonemize(text)
        tokens_t = torch.tensor([phones.ids], dtype=torch.long)
        stress_t = torch.tensor([phones.stress], dtype=torch.long)
        lengths_t = torch.tensor([len(phones)], dtype=torch.long)

        for voice_idx, voice_name in enumerate(c.speakers):
            voice_t = torch.tensor([voice_idx], dtype=torch.long)
            ref_t, ref_len_t, details = model.synthesize(tokens_t, stress_t, lengths_t, voice_t, seed=0)
            ref_np = ref_t[0, :int(ref_len_t[0])].numpy()
            ref_dur = details["durations"].numpy()
            ref_mel = mel_loss.log_mel(torch.from_numpy(ref_np)[None])[0].numpy()
            ref_rms = float(np.sqrt(np.mean(ref_np.astype(np.float64) ** 2)))
            ref_peak = float(np.abs(ref_np).max())

            noise_t = torch.randn(len(ref_np), generator=torch.Generator().manual_seed(0))[None]
            padded_noise = np.zeros((1, c.hop * c.max_frames), dtype=np.float32)
            padded_noise[0, :len(ref_np)] = noise_t[0].numpy()

            base_feed = {
                "tokens": np.array([phones.ids], dtype=np.int64),
                "stress": np.array([phones.stress], dtype=np.int64),
                "voice": np.array([voice_idx], dtype=np.int64),
                "speed": np.array([1.0], dtype=np.float32),
                "pitch_shift": np.array([0.0], dtype=np.float32),
                "pitch_range": np.array([1.0], dtype=np.float32),
            }

            for cand_id, path, _, ext_noise in candidates:
                if path is None:
                    one = lambda v: torch.tensor([v], dtype=torch.float32)
                    with torch.inference_mode():
                        out_t, dur_t = wrapper(tokens_t, stress_t, voice_t, one(1.0), one(0.0), one(1.0), noise_t)
                    cand_np = out_t[0].numpy()
                    cand_dur = dur_t.numpy()
                else:
                    feed = {**base_feed, "noise": padded_noise} if ext_noise else base_feed
                    out_arr, cand_dur = sessions[cand_id].run(None, feed)
                    cand_np = out_arr[0]

                finite = bool(np.isfinite(cand_np).all())
                same_dur = bool((cand_dur == ref_dur).all())
                dur_l1_frames = int(np.abs(cand_dur - ref_dur).sum())
                len_diff_frames = (len(cand_np) - len(ref_np)) // c.hop
                cand_rms = float(np.sqrt(np.mean(cand_np.astype(np.float64) ** 2)))
                cand_peak = float(np.abs(cand_np).max())

                rel_rms_err = None
                max_abs_over_peak = None
                if ext_noise and len(cand_np) == len(ref_np):
                    diff = cand_np.astype(np.float64) - ref_np.astype(np.float64)
                    rel_rms_err = float(np.sqrt(np.mean(diff ** 2)) / ref_rms)
                    max_abs_over_peak = float(np.abs(diff).max() / ref_peak)

                cand_mel = mel_loss.log_mel(torch.from_numpy(cand_np)[None])[0].numpy()
                mel_dtw = dtw_distance_np(cand_mel, ref_mel)
                min_frames = min(cand_mel.shape[-1], ref_mel.shape[-1])
                mel_l1_unaligned = float(np.abs(cand_mel[:, :min_frames] - ref_mel[:, :min_frames]).mean())

                rows.append({
                    "candidate": cand_id,
                    "sentence": sent_key,
                    "tokens": len(phones),
                    "voice": voice_name,
                    "external_noise": ext_noise,
                    "finite": finite,
                    "ref_samples": len(ref_np),
                    "cand_samples": len(cand_np),
                    "same_durations": same_dur,
                    "dur_l1_frames": dur_l1_frames,
                    "len_diff_frames": len_diff_frames,
                    "ref_peak": round(ref_peak, 5),
                    "cand_peak": round(cand_peak, 5),
                    "rms_ratio": round(cand_rms / ref_rms, 5),
                    "rel_rms_err": rel_rms_err,
                    "max_abs_over_peak": max_abs_over_peak,
                    "mel_dtw": round(mel_dtw, 6),
                    "mel_l1_prefix": round(mel_l1_unaligned, 6),
                })

    return rows


def _worker_frontend(text: str, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        phonemize(text)
    times_ms = []
    phones = None
    for _ in range(runs):
        t0 = time.perf_counter()
        phones = phonemize(text)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "tokens": len(phones),
        "unknown": list(phones.unknown),
        **summarize_times(times_ms),
        **read_peak_rss_kib(),
    }


def _worker_e2e(variant: str, checkpoint: Path, model_path: Path | None, opt_level: str | None,
                text: str, threads: int, warmup: int, runs: int) -> dict:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    match_frontend(checkpoint)
    phones = phonemize(text)

    if variant == "torch_fp32_synthesize":
        model = load_model(checkpoint, "cpu")
        tokens_t = torch.tensor([phones.ids], dtype=torch.long)
        stress_t = torch.tensor([phones.stress], dtype=torch.long)
        lengths_t = torch.tensor([len(phones)], dtype=torch.long)
        voice_t = torch.tensor([0], dtype=torch.long)
        rate, hop = model.config.sample_rate, model.config.hop
        with torch.inference_mode():
            for _ in range(warmup):
                model.synthesize(tokens_t, stress_t, lengths_t, voice_t)
            times_ms = []
            for _ in range(runs):
                t0 = time.perf_counter()
                wave, out_len, _ = model.synthesize(tokens_t, stress_t, lengths_t, voice_t)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        samples = int(out_len[0])
        frames = samples // hop

    elif variant == "torch_fp32_exportable":
        model = load_model(checkpoint, "cpu")
        wrapper = Exportable(model, external_noise=True).eval()
        tokens_t = torch.tensor([phones.ids], dtype=torch.long)
        stress_t = torch.tensor([phones.stress], dtype=torch.long)
        voice_t = torch.tensor([0], dtype=torch.long)
        one = lambda v: torch.tensor([v], dtype=torch.float32)
        speed_t, shift_t, range_t = one(1.0), one(0.0), one(1.0)
        noise_t = torch.randn(1, model.config.hop * model.config.max_frames, generator=torch.Generator().manual_seed(0))
        rate, hop = model.config.sample_rate, model.config.hop
        with torch.inference_mode():
            for _ in range(warmup):
                wrapper(tokens_t, stress_t, voice_t, speed_t, shift_t, range_t, noise_t)
            times_ms = []
            for _ in range(runs):
                t0 = time.perf_counter()
                wave, durations = wrapper(tokens_t, stress_t, voice_t, speed_t, shift_t, range_t, noise_t)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        samples = int(wave.shape[1])
        frames = int(durations.sum())

    else:
        ckpt_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = ckpt_state["config"]
        rate, hop, max_frames = cfg["sample_rate"], cfg["hop"], cfg["max_frames"]
        del ckpt_state

        session = make_session(model_path, threads=threads, opt_level_name=opt_level)
        one = lambda v: np.array([v], dtype=np.float32)
        feed = {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([0], dtype=np.int64),
            "speed": one(1.0),
            "pitch_shift": one(0.0),
            "pitch_range": one(1.0),
        }
        if any(inp.name == "noise" for inp in session.get_inputs()):
            rng = np.random.default_rng(0)
            feed["noise"] = rng.standard_normal((1, hop * max_frames), dtype=np.float32)

        for _ in range(warmup):
            session.run(None, feed)
        times_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            audio, durations = session.run(None, feed)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        samples = int(audio.shape[1])
        frames = int(durations.sum())

    return {
        "tokens": len(phones),
        "frames": frames,
        "audio_samples": samples,
        **summarize_times(times_ms, audio_seconds=samples / rate),
        **read_peak_rss_kib(),
    }


def _worker_torch_profile(mode: str, checkpoint: Path, text: str, threads: int, warmup: int, runs: int) -> dict:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")
    phones = phonemize(text)
    c = model.config
    v = model.vocoder

    tokens_t = torch.tensor([phones.ids], dtype=torch.long)
    stress_t = torch.tensor([phones.stress], dtype=torch.long)
    lengths_t = torch.tensor([len(phones)], dtype=torch.long)
    voice_t = torch.tensor([0], dtype=torch.long)

    blocks = defaultdict(list)
    e2e_times = []

    if mode == "tinyvoca_native":
        with torch.inference_mode():
            for _ in range(warmup):
                model.synthesize(tokens_t, stress_t, lengths_t, voice_t)

            for _ in range(runs):
                t0 = time.perf_counter()
                hidden, mask = model.encode(tokens_t, stress_t, lengths_t, voice_t)
                t1 = time.perf_counter()
                durations = model.predict_durations(hidden, mask, tokens_t, 1.0)
                t2 = time.perf_counter()
                frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, voice_t)
                t3 = time.perf_counter()
                prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)
                ends = durations.cumsum(1)
                frame_idx = torch.arange(frames.shape[1], device=tokens_t.device)[None]
                owner = torch.searchsorted(ends, frame_idx, right=True).clamp(max=tokens_t.shape[1] - 1)
                vowel = model.vowel_tokens[tokens_t].gather(1, owner) & frame_mask
                prosody = torch.cat(
                    (prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None], prosody[..., 2:]),
                    dim=-1,
                )
                prosody = model.widen_pitch(prosody, frame_mask, 1.0)
                t4 = time.perf_counter()
                acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
                t5 = time.perf_counter()

                # SourceFilter decomposition matching SourceFilter.forward
                n_frames = acoustic.shape[1]
                samples = n_frames * c.hop
                last = (frame_lengths - 1).clamp_min(0)[:, None]
                h = v.harmonic(acoustic)
                split = c.cep_causal + c.cep_anti
                sub = v._subframes(n_frames, frame_lengths, v.harmonic_steps)
                here, following = sub // v.harmonic_steps, torch.minimum((sub + 1) // v.harmonic_steps, last)
                causal = 0.5 * (v._take(h[..., :c.cep_causal], here) + v._take(h[..., :c.cep_causal], following)) * v.causal_lifter
                anti = 0.5 * (v._take(h[..., c.cep_causal:split], here) + v._take(h[..., c.cep_causal:split], following)) * v.anti_lifter
                causal = torch.cat((causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None], causal[..., 1:]), dim=-1)
                harmonic_response = cepstrum_response(causal, anti, c.synth_fft)
                n = v.noise(acoustic)
                sub_n = v._subframes(n_frames, frame_lengths, v.noise_steps)
                shape = v._take(n[..., :c.noise_cep], sub_n // v.noise_steps) * v.noise_lifter
                shape = torch.cat((shape[..., :1] + n[..., c.noise_cep:].flatten(1).gather(1, sub_n)[..., None], shape[..., 1:]), dim=-1)
                noise_response = cepstrum_response(shape, None, c.noise_fft)
                t6 = time.perf_counter()

                f0 = model.frequency(prosody[..., 0])
                audible = length_mask(frame_lengths * c.hop, samples)
                f0_clamped = f0.float().gather(1, torch.minimum(torch.arange(n_frames)[None], last))
                pulses = pulse_train(f0_clamped, c.sample_rate, c.hop, c.harmonic_limit, None) * audible
                t7 = time.perf_counter()

                wave_h = v._shape(pulses, harmonic_response, c.synth_fft, c.synth_hop, v.synth_window)
                t8 = time.perf_counter()

                noise_sig = torch.randn(1, int(frame_lengths[0]) * c.hop)
                wave_n = v._shape(noise_sig * audible, noise_response, c.noise_fft, c.noise_hop, v.noise_window)
                _ = (wave_h + wave_n) * audible
                t9 = time.perf_counter()

                blocks["encode"].append((t1 - t0) * 1000.0)
                blocks["predict_durations"].append((t2 - t1) * 1000.0)
                blocks["expand"].append((t3 - t2) * 1000.0)
                blocks["predict_prosody"].append((t4 - t3) * 1000.0)
                blocks["decode"].append((t5 - t4) * 1000.0)
                blocks["vocoder_cepstrum_response"].append((t6 - t5) * 1000.0)
                blocks["vocoder_pulse_train"].append((t7 - t6) * 1000.0)
                blocks["vocoder_filter_harmonic"].append((t8 - t7) * 1000.0)
                blocks["vocoder_filter_noise"].append((t9 - t8) * 1000.0)
                e2e_times.append((t9 - t0) * 1000.0)

    elif mode == "exportable_extnoise":
        wrapper = Exportable(model, external_noise=True).eval()
        noise_full = torch.randn(1, c.hop * c.max_frames, generator=torch.Generator().manual_seed(0))
        one = lambda val: torch.tensor([val], dtype=torch.float32)
        speed_t, shift_t, range_t = one(1.0), one(0.0), one(1.0)
        with torch.inference_mode():
            for _ in range(warmup):
                wrapper(tokens_t, stress_t, voice_t, speed_t, shift_t, range_t, noise_full)

            for _ in range(runs):
                t0 = time.perf_counter()
                token_mask = torch.ones_like(tokens_t, dtype=torch.bool)
                hidden = model.encoder(model.tokens(tokens_t) + model.stress(stress_t) + model.voices(voice_t)[:, None], token_mask)
                t1 = time.perf_counter()

                output_d = model.duration(hidden, token_mask)
                d_frames = torch.round((torch.exp(output_d[..., 0].clamp(0, math.log1p(400))) - 1) / speed_t).long().clamp_min(1)
                present = model.phone_tokens[tokens_t] | (output_d[..., 1] > 0)
                durations = torch.where(present, d_frames, torch.zeros_like(d_frames))
                t2 = time.perf_counter()

                ends = durations.cumsum(1)
                total = ends[0, -1]
                frame = torch.arange(total)[None]
                index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
                owned = durations.gather(1, index).clamp_min(1)
                offset = frame - (ends.gather(1, index) - durations.gather(1, index))
                relative = (offset.to(hidden.dtype) + 0.5) / owned
                features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
                expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
                expanded = expanded + model.frame_position(features) + model.frame_voices(voice_t)[:, None]
                frame_mask = torch.ones_like(frame, dtype=torch.bool)
                t3 = time.perf_counter()

                prosody_hidden = model.prosody_stack(expanded, frame_mask)
                prosody = model.prosody_head(prosody_hidden)
                vowel = model.vowel_tokens[tokens_t].gather(1, index)
                voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
                voiced = (voicing > 0).to(prosody.dtype)
                mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
                pitch = mean + (prosody[..., 0] - mean) * range_t
                t4 = time.perf_counter()

                condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1)
                acoustic = model.acoustic_stack(prosody_hidden + model.prosody_projection(condition), frame_mask)
                f0 = (c.pitch_reference_hz * torch.exp(pitch + shift_t)).clamp(c.f0_min, c.f0_max)
                t5 = time.perf_counter()

                # Exportable._vocoder decomposition
                last = total - 1
                h = v.harmonic(acoustic)
                split = c.cep_causal + c.cep_anti
                steps = v.harmonic_steps
                sub = torch.minimum(torch.arange(steps * total + 3), steps * total - 1)
                here = torch.div(sub, steps, rounding_mode="floor")
                following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
                envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
                envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
                gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
                envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
                harmonic_real, harmonic_imaginary = wrapper._response(envelope, wrapper.harmonic_real, wrapper.harmonic_imaginary)

                n = v.noise(acoustic)
                steps_n = v.noise_steps
                sub_n = torch.minimum(torch.arange(steps_n * total + 3), steps_n * total - 1)
                shape = n[..., :c.noise_cep].index_select(1, torch.div(sub_n, steps_n, rounding_mode="floor")) * v.noise_lifter
                gain_n = n[..., c.noise_cep:].flatten(1).index_select(1, sub_n)[..., None]
                shape = torch.cat((shape[..., :1] + gain_n, shape[..., 1:]), dim=-1)
                noise_real, noise_imaginary = wrapper._response(shape, wrapper.noise_real, wrapper.noise_imaginary)
                t6 = time.perf_counter()

                pulses = wrapper._pulses(f0)
                t7 = time.perf_counter()

                wave_h = wrapper._filter(pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop)
                t8 = time.perf_counter()

                noise_slice = noise_full.index_select(1, torch.arange(c.hop * total))
                wave_n = wrapper._filter(noise_slice, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop)
                _ = wave_h + wave_n
                t9 = time.perf_counter()

                blocks["encode"].append((t1 - t0) * 1000.0)
                blocks["predict_durations"].append((t2 - t1) * 1000.0)
                blocks["expand"].append((t3 - t2) * 1000.0)
                blocks["predict_prosody"].append((t4 - t3) * 1000.0)
                blocks["decode"].append((t5 - t4) * 1000.0)
                blocks["vocoder_cepstrum_response"].append((t6 - t5) * 1000.0)
                blocks["vocoder_pulse_train"].append((t7 - t6) * 1000.0)
                blocks["vocoder_filter_harmonic"].append((t8 - t7) * 1000.0)
                blocks["vocoder_filter_noise"].append((t9 - t8) * 1000.0)
                e2e_times.append((t9 - t0) * 1000.0)
        n_frames = int(total)
    else:
        raise ValueError(f"unknown torch profile mode: {mode}")

    block_summary = {name: summarize_times(vals) for name, vals in blocks.items()}
    total_median_sum = sum(v["median_ms"] for v in block_summary.values())
    for name, info in block_summary.items():
        info["share_pct"] = round(100.0 * info["median_ms"] / total_median_sum, 2) if total_median_sum > 0 else 0.0

    return {
        "mode": mode,
        "tokens": len(phones),
        "frames": n_frames,
        "audio_seconds": round(n_frames * c.hop / c.sample_rate, 4),
        "blocks": block_summary,
        "sum_blocks_median_ms": round(total_median_sum, 3),
        "e2e_loop": summarize_times(e2e_times, audio_seconds=n_frames * c.hop / c.sample_rate),
        **read_peak_rss_kib(),
    }


def _classify_ort_node(op_name: str, node_name: str) -> str:
    # Vocoder STFT/iSTFT dense convolutions (Conv, Conv_1..3, ConvTranspose, ConvTranspose_1..3)
    if op_name in ("Conv", "ConvTranspose") and node_name.count("/") == 1:
        return "vocoder_stft_conv"
    if op_name == "Conv" and node_name.count("/") > 1:
        return "neural_depthwise_predictor_conv"
    if op_name in ("MatMul", "FusedMatMul") and (
        "/attention/qkv/" in node_name
        or "/attention/out/" in node_name
        or "/expand/" in node_name
        or "/project/" in node_name
        or "/duration/out/" in node_name
        or "/frame_position/" in node_name
        or "/prosody_head/" in node_name
        or "/prosody_projection/" in node_name
        or "/harmonic/" in node_name
        or "/noise/" in node_name
    ):
        return "neural_linear_matmul"
    if "/attention/" in node_name:
        if op_name in ("IsNaN", "Where"):
            return "attention_isnan_where"
        return "attention_qk_softmax"
    if (
        node_name.startswith(("/encoder/", "/duration/", "/frame_position/", "/prosody_stack/",
                              "/prosody_head/", "/prosody_projection/", "/acoustic_stack/",
                              "/harmonic/", "/noise/"))
        or op_name in ("BiasGelu", "Gelu")
        or node_name.startswith(("BiasGelu", "Gelu", "Squeeze", "Transpose"))
    ):
        return "neural_norm_gelu_rotary_misc"
    if node_name in ("/MatMul", "/MatMul_1", "/MatMul_2", "/MatMul_3", "/Sin", "/Sin_1", "/Cos", "/Cos_1", "/Exp_3", "/Exp_4"):
        return "vocoder_response_cepstrum"
    if node_name in ("/Pad", "/Pad_1") or (op_name in ("Conv", "ConvTranspose")):
        return "vocoder_stft_pad"
    return "vocoder_pulses_and_top_level_ops"


def _worker_ort_profile(model_path: Path, out_dir: Path, text: str, threads: int, warmup: int, prof_runs: int) -> dict:
    phones = phonemize(text)
    prefix = out_dir / f"_ort_prof_{os.getpid()}"
    sess = make_session(model_path, threads=threads, opt_level_name="ORT_ENABLE_ALL",
                        enable_profiling=True, profile_prefix=prefix)
    one = lambda v: np.array([v], dtype=np.float32)
    feed = {
        "tokens": np.array([phones.ids], dtype=np.int64),
        "stress": np.array([phones.stress], dtype=np.int64),
        "voice": np.array([0], dtype=np.int64),
        "speed": one(1.0),
        "pitch_shift": one(0.0),
        "pitch_range": one(1.0),
    }
    for _ in range(warmup + prof_runs):
        audio, durations = sess.run(None, feed)
    prof_file = Path(sess.end_profiling())
    events = json.loads(prof_file.read_text(encoding="utf-8"))
    prof_file.unlink(missing_ok=True)

    model_runs = [e for e in events if e.get("name") == "model_run"]
    cutoff_ts = model_runs[warmup]["ts"] if len(model_runs) > warmup else 0
    timed_runs = model_runs[warmup:]
    count = max(len(timed_runs), 1)

    by_group_us = defaultdict(float)
    by_op_us = defaultdict(float)
    stft_conv_nodes_us = defaultdict(float)

    for event in events:
        if event.get("cat") != "Node" or event.get("ts", 0) < cutoff_ts:
            continue
        dur_us = float(event.get("dur", 0.0))
        op_name = event.get("args", {}).get("op_name", "unknown")
        node_name = event.get("name", "").removesuffix("_kernel_time")
        group = _classify_ort_node(op_name, node_name)
        by_group_us[group] += dur_us
        by_op_us[op_name] += dur_us
        if group == "vocoder_stft_conv":
            stft_conv_nodes_us[f"{op_name}:{node_name}"] += dur_us

    total_node_ms = sum(by_group_us.values()) / 1000.0 / count
    avg_run_ms = sum(float(e.get("dur", 0.0)) for e in timed_runs) / 1000.0 / count

    groups_ms = {k: round(v / 1000.0 / count, 3) for k, v in sorted(by_group_us.items(), key=lambda x: -x[1])}
    groups_pct = {k: round(100.0 * (v / 1000.0 / count) / total_node_ms, 2) if total_node_ms > 0 else 0.0
                  for k, v in sorted(by_group_us.items(), key=lambda x: -x[1])}

    return {
        "tokens": len(phones),
        "frames": int(durations.sum()),
        "audio_samples": int(audio.shape[1]),
        "prof_runs": count,
        "avg_model_run_ms": round(avg_run_ms, 3),
        "total_node_kernel_ms": round(total_node_ms, 3),
        "groups_ms": groups_ms,
        "groups_pct": groups_pct,
        "stft_conv_nodes_ms": {k: round(v / 1000.0 / count, 3) for k, v in sorted(stft_conv_nodes_us.items(), key=lambda x: -x[1])},
        "top_ops_ms": {k: round(v / 1000.0 / count, 3) for k, v in sorted(by_op_us.items(), key=lambda x: -x[1])[:12]},
    }


def run_worker_subprocess(payload: dict, threads: int) -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(threads)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker-json", json.dumps(payload)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("WORKER_RESULT:"):
            return json.loads(line.removeprefix("WORKER_RESULT:"))
    raise RuntimeError(f"missing WORKER_RESULT in worker output:\n{proc.stdout}\n{proc.stderr}")


def _run_e2e_matrix(checkpoint: Path, out_dir: Path, threads_list: list[int], warmup: int, runs: int) -> list[dict]:
    e2e_variants = [
        ("torch_fp32_synthesize", None, None),
        ("torch_fp32_exportable", None, None),
        ("ort_fp32_rel_opt_all", out_dir / "tiny-voca-fp32.onnx", "ORT_ENABLE_ALL"),
        ("ort_fp32_ext_opt_off", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_DISABLE_ALL"),
        ("ort_fp32_ext_opt_basic", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_ENABLE_BASIC"),
        ("ort_fp32_ext_opt_all", out_dir / "tiny-voca-fp32-extnoise.onnx", "ORT_ENABLE_ALL"),
        ("ort_int8_linear", out_dir / "tiny-voca-int8-linear.onnx", "ORT_ENABLE_ALL"),
        ("ort_int8_linear_pc", out_dir / "tiny-voca-int8-linear-pc.onnx", "ORT_ENABLE_ALL"),
        ("ort_int8_transformer", out_dir / "tiny-voca-int8-transformer.onnx", "ORT_ENABLE_ALL"),
        ("ort_int8_acoustic", out_dir / "tiny-voca-int8-acoustic.onnx", "ORT_ENABLE_ALL"),
    ]
    e2e_rows = []
    for variant, model_path, opt_level in e2e_variants:
        for threads in threads_list:
            for sent_key, text in BENCHMARK_SENTENCES:
                res = run_worker_subprocess(
                    {
                        "task": "e2e",
                        "variant": variant,
                        "checkpoint": str(checkpoint),
                        "model_path": str(model_path) if model_path is not None else None,
                        "opt_level": opt_level,
                        "text": text,
                        "threads": threads,
                        "warmup": warmup,
                        "runs": runs,
                    },
                    threads=threads,
                )
                e2e_rows.append({"variant": variant, "opt_level": opt_level, "threads": threads, "sentence": sent_key, **res})
                print(
                    f"  e2e {variant:23s} th={threads} {sent_key:6s}: "
                    f"med={res['median_ms']:6.2f}ms p95={res['p95_ms']:6.2f}ms "
                    f"RTFx={res['rtfx_median']:5.1f}x RSS={res['vm_hwm_mib']}MiB"
                )
    return e2e_rows


def run_all(args) -> None:
    threads_list = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
    results = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "warmup": args.warmup,
            "runs": args.runs,
            "threads": threads_list,
            "sentences": [{"key": k, "text": txt, "tokens": len(phonemize(txt))} for k, txt in BENCHMARK_SENTENCES],
            "loadavg_start": read_loadavg(),
            **cpu_info(),
        }
    }

    print("[1/6] Exporting and quantizing candidate ONNX models...")
    prep = prepare_models(args.checkpoint, args.out_dir)
    results["graph_audit"] = prep["audit"]
    results["artifacts"] = prep["artifacts"]

    print("[2/6] Benchmarking front-end (phonemize) in isolated workers...")
    fe_rows = []
    for sent_key, text in BENCHMARK_SENTENCES:
        res = run_worker_subprocess({"task": "frontend", "text": text, "warmup": args.warmup, "runs": args.runs}, threads=1)
        fe_rows.append({"sentence": sent_key, **res})
        print(f"  frontend {sent_key:6s}: tokens={res['tokens']:3d} med={res['median_ms']:6.2f}ms p95={res['p95_ms']:6.2f}ms rss={res['vm_hwm_mib']}MiB")
    results["frontend"] = fe_rows

    print("[3/6] Profiling PyTorch FP32 blocks (TinyVoca & Exportable)...")
    torch_prof_rows = []
    for mode in ("tinyvoca_native", "exportable_extnoise"):
        for threads in threads_list:
            for sent_key, text in BENCHMARK_SENTENCES:
                res = run_worker_subprocess(
                    {"task": "torch_profile", "mode": mode, "checkpoint": str(args.checkpoint),
                     "text": text, "threads": threads, "warmup": args.warmup, "runs": args.runs},
                    threads=threads,
                )
                torch_prof_rows.append({"sentence": sent_key, "threads": threads, **res})
                print(f"  torch_prof {mode:19s} th={threads} {sent_key:6s}: sum={res['sum_blocks_median_ms']:6.2f}ms "
                      f"filt_h={res['blocks']['vocoder_filter_harmonic']['median_ms']:5.2f}ms "
                      f"filt_n={res['blocks']['vocoder_filter_noise']['median_ms']:5.2f}ms")
    results["torch_block_profile"] = torch_prof_rows

    print("[4/6] Profiling ORT FP32 node groups (ORT_ENABLE_ALL)...")
    ort_prof_rows = []
    for threads in threads_list:
        for sent_key, text in BENCHMARK_SENTENCES:
            res = run_worker_subprocess(
                {"task": "ort_profile", "model_path": str(args.out_dir / "tiny-voca-fp32.onnx"),
                 "out_dir": str(args.out_dir), "text": text, "threads": threads, "warmup": 5, "prof_runs": 15},
                threads=threads,
            )
            ort_prof_rows.append({"sentence": sent_key, "threads": threads, **res})
            stft_ms = res["groups_ms"].get("vocoder_stft_conv", 0.0)
            stft_pct = res["groups_pct"].get("vocoder_stft_conv", 0.0)
            print(f"  ort_prof th={threads} {sent_key:6s}: total_kern={res['total_node_kernel_ms']:6.2f}ms "
                  f"stft_conv={stft_ms:5.2f}ms ({stft_pct:5.1f}%)")
    results["ort_node_profile"] = ort_prof_rows

    print("[5/6] Benchmarking end-to-end variants in isolated subprocesses...")
    results["e2e"] = _run_e2e_matrix(args.checkpoint, args.out_dir, threads_list, args.warmup, args.runs)

    print("[6/6] Evaluating numerical & acoustic fidelity vs PyTorch FP32 (F1 & M1)...")
    fidelity_rows = evaluate_fidelity(args.checkpoint, args.out_dir)
    results["fidelity"] = fidelity_rows
    results["meta"]["loadavg_end"] = read_loadavg()

    json_out = args.json_out or (args.out_dir / "benchmark_results.json")
    json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete benchmark results to {json_out}")


def run_e2e_stage(args) -> None:
    threads_list = [int(t.strip()) for t in args.threads.split(",") if t.strip()]
    json_path = args.json_out or (args.out_dir / "benchmark_results.json")
    results = json.loads(json_path.read_text(encoding="utf-8"))
    results["meta"]["e2e_loadavg_start"] = read_loadavg()
    results["e2e"] = _run_e2e_matrix(args.checkpoint, args.out_dir, threads_list, args.warmup, args.runs)
    results["meta"]["e2e_loadavg_end"] = read_loadavg()
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated E2E benchmark results in {json_path}")


def run_worker_entry(payload_json: str) -> None:
    payload = json.loads(payload_json)
    task = payload["task"]
    if task == "frontend":
        res = _worker_frontend(payload["text"], payload["warmup"], payload["runs"])
    elif task == "e2e":
        res = _worker_e2e(
            payload["variant"],
            Path(payload["checkpoint"]),
            Path(payload["model_path"]) if payload["model_path"] else None,
            payload["opt_level"],
            payload["text"],
            payload["threads"],
            payload["warmup"],
            payload["runs"],
        )
    elif task == "torch_profile":
        res = _worker_torch_profile(
            payload["mode"],
            Path(payload["checkpoint"]),
            payload["text"],
            payload["threads"],
            payload["warmup"],
            payload["runs"],
        )
    elif task == "ort_profile":
        res = _worker_ort_profile(
            Path(payload["model_path"]),
            Path(payload["out_dir"]),
            payload["text"],
            payload["threads"],
            payload["warmup"],
            payload["prof_runs"],
        )
    else:
        raise ValueError(f"unknown worker task: {task}")
    print("WORKER_RESULT:" + json.dumps(res, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--runs", type=int, default=35)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--update-e2e", action="store_true", help="Re-run only stage 5 (E2E) into existing JSON")
    parser.add_argument("--worker-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_json is not None:
        run_worker_entry(args.worker_json)
    else:
        if args.warmup < 5:
            parser.error("--warmup must be >= 5")
        if args.runs < 30:
            parser.error("--runs must be >= 30")
        if args.update_e2e:
            run_e2e_stage(args)
        else:
            run_all(args)


if __name__ == "__main__":
    main()
