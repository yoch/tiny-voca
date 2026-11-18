"""Reproducible E04 CPU benchmark: unmasked SDPA, C packed-real FFT DSP, and INT8 acoustic synergy.

Usage:
    PYTHONPATH=src:scripts /home/yoch/voice-tts/env/bin/python scripts/benchmark_e04.py \
        --checkpoint /home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt \
        --out-dir /home/yoch/voice-tts/.local-runs/e04 \
        --warmup 5 --runs 30

Measures each (order, threads, sentence, variant) in an isolated subprocess so
/proc/self/status VmHWM and thread-pool states stay strictly uncontaminated.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
import soundfile as sf
import torch

from benchmark_onnx import (
    BENCHMARK_SENTENCES,
    _classify_ort_node,
    cpu_info,
    dtw_distance_np,
    inspect_onnx_graph,
    make_session,
    read_loadavg,
    read_peak_rss_kib,
    sha256_file,
    summarize_times,
)
from e04.runtime import (
    CDspVocoder,
    compile_dsp,
    export_compact,
    export_fused,
    extract_vocoder_constants,
)
from tiny_voca.export import Exportable, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.losses import SpectralLoss
from tiny_voca.text import phonemize

DEFAULT_CHECKPOINT = Path("/home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt")
DEFAULT_OUT_DIR = Path("/home/yoch/voice-tts/.local-runs/e04")


def _snr_db(ref: np.ndarray, cand: np.ndarray) -> float | None:
    num = float(np.sum(ref.astype(np.float64) ** 2))
    den = float(np.sum((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if den == 0.0:
        return None
    return round(float(10.0 * np.log10(num / den)), 2)


def _quantize_acoustic(src_onnx: Path, dst_onnx: Path) -> tuple[Path, dict]:
    audit = inspect_onnx_graph(src_onnx)
    acoustic_nodes = audit["neural_acoustic_linears"]
    if len(acoustic_nodes) != 20:
        raise RuntimeError(f"{src_onnx.name}: expected 20 acoustic_stack MatMul nodes, got {len(acoustic_nodes)}")
    quantize_dynamic(
        str(src_onnx),
        str(dst_onnx),
        nodes_to_quantize=acoustic_nodes,
        per_channel=False,
        reduce_range=True,
        weight_type=QuantType.QInt8,
    )
    q_audit = inspect_onnx_graph(dst_onnx)
    return dst_onnx, {
        "quantized_nodes": len(acoustic_nodes),
        "MatMulInteger": q_audit["op_counts"].get("MatMulInteger", 0),
        "DynamicQuantizeLinear": q_audit["op_counts"].get("DynamicQuantizeLinear", 0),
    }


def prepare_artifacts(checkpoint: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")
    c = model.config

    so_path = compile_dsp(out_dir / "dsp.so", cc="gcc", march="native")
    consts_path = out_dir / "constants.json"
    consts_path.write_text(json.dumps(extract_vocoder_constants(model)), encoding="utf-8")

    # Noise pool matches checkpoint hop * max_frames cleanly.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(c.hop * c.max_frames, dtype=np.float32)
    (out_dir / "noise.f32").write_bytes(noise.tobytes())

    p_dense_m = export(model, out_dir / "dense-masked.onnx", external_noise=True, omit_mask=False)
    p_dense_u = export(model, out_dir / "dense-unmasked.onnx", external_noise=True, omit_mask=True)
    p_fused_u = export_fused(model, out_dir / "fused-unmasked.onnx", omit_mask=True)
    p_compact_u = export_compact(model, out_dir / "compact-fp32-unmasked.onnx", omit_mask=True)

    p_dense_u_q, q_dense_meta = _quantize_acoustic(p_dense_u, out_dir / "dense-unmasked-int8-acoustic.onnx")
    p_compact_u_q, q_compact_meta = _quantize_acoustic(p_compact_u, out_dir / "compact-int8-acoustic-unmasked.onnx")

    specs = (
        ("dense_masked", p_dense_m, False, "Baseline FP32 ONNX (external_noise=True, omit_mask=False)"),
        ("dense_unmasked", p_dense_u, False, "FP32 ONNX without B=1 attn_mask (0 IsNaN/Where SDPA guards)"),
        ("dense_unmasked_int8_acoustic", p_dense_u_q, False, "Unmasked FP32 + 20 acoustic_stack MatMul QInt8"),
        ("fused_fp32_unmasked", p_fused_u, True, "Unmasked ONNX cepstra+pulses + C packed-real FFT filter()"),
        ("compact_fp32_unmasked", p_compact_u, True, "Unmasked ONNX heads+pulses + C compact_filter()"),
        ("compact_int8_acoustic_unmasked", p_compact_u_q, True, "Compact unmasked + 20 acoustic_stack MatMul QInt8"),
    )

    artifacts = {}
    for vid, path, uses_dsp, desc in specs:
        audit = inspect_onnx_graph(path)
        dsp_bytes = so_path.stat().st_size if uses_dsp else 0
        artifacts[vid] = {
            "onnx_path": str(path),
            "onnx_bytes": path.stat().st_size,
            "onnx_mb": round(path.stat().st_size / 1e6, 3),
            "onnx_sha256": sha256_file(path),
            "dsp_so_bytes": dsp_bytes,
            "total_artifact_bytes": path.stat().st_size + dsp_bytes,
            "total_artifact_mb": round((path.stat().st_size + dsp_bytes) / 1e6, 3),
            "uses_dsp": uses_dsp,
            "description": desc,
            "total_nodes": audit["total_nodes"],
            "op_counts": audit["op_counts"],
        }
        if vid == "dense_unmasked_int8_acoustic":
            artifacts[vid]["quantization"] = q_dense_meta
        elif vid == "compact_int8_acoustic_unmasked":
            artifacts[vid]["quantization"] = q_compact_meta

    artifacts["dsp_so"] = {
        "path": str(so_path),
        "bytes": so_path.stat().st_size,
        "sha256": sha256_file(so_path),
    }
    return artifacts


def profile_ort_comparison(checkpoint: Path, out_dir: Path, threads_list=(1, 4), warmup: int = 5, runs: int = 15) -> dict:
    """Profile ORT kernel distribution for dense_masked, dense_unmasked, and compact_fp32_unmasked."""
    torch.set_num_threads(1)
    match_frontend(checkpoint)
    _, long_text = BENCHMARK_SENTENCES[2]
    phones = phonemize(long_text)
    noise = np.fromfile(out_dir / "noise.f32", dtype="<f4")[None]
    one = lambda v: np.array([v], dtype=np.float32)
    base_feed = {
        "tokens": np.array([phones.ids], dtype=np.int64),
        "stress": np.array([phones.stress], dtype=np.int64),
        "voice": np.array([0], dtype=np.int64),
        "speed": one(1.0),
        "pitch_shift": one(0.0),
        "pitch_range": one(1.0),
    }

    results = {}
    targets = (
        ("dense_masked", "dense-masked.onnx"),
        ("dense_unmasked", "dense-unmasked.onnx"),
        ("compact_fp32_unmasked", "compact-fp32-unmasked.onnx"),
    )
    for vid, model_name in targets:
        for threads in threads_list:
            prefix = out_dir / f"ort_prof_{vid}_{threads}t"
            sess = make_session(out_dir / model_name, threads=threads, opt_level_name="ORT_ENABLE_ALL",
                                enable_profiling=True, profile_prefix=prefix)
            feed = dict(base_feed)
            if any(inp.name == "noise" for inp in sess.get_inputs()):
                feed["noise"] = noise
            for _ in range(warmup + runs):
                sess.run(None, feed)
            prof_file = Path(sess.end_profiling())
            events = json.loads(prof_file.read_text(encoding="utf-8"))
            prof_file.unlink(missing_ok=True)

            model_runs = [e for e in events if e.get("name") == "model_run"]
            cutoff_ts = model_runs[warmup]["ts"] if len(model_runs) > warmup else 0
            timed_runs = model_runs[warmup:]
            count = max(len(timed_runs), 1)
            by_group_us = defaultdict(float)
            for ev in events:
                if ev.get("cat") != "Node" or ev.get("ts", 0) < cutoff_ts:
                    continue
                op_name = ev.get("args", {}).get("op_name", "unknown")
                node_name = ev.get("name", "").removesuffix("_kernel_time")
                by_group_us[_classify_ort_node(op_name, node_name)] += float(ev.get("dur", 0.0))

            mean_groups = {k: round(v / 1000.0 / count, 3) for k, v in sorted(by_group_us.items(), key=lambda x: -x[1])}
            kern_sum = round(sum(mean_groups.values()), 3)
            avg_run_ms = sum(float(e.get("dur", 0.0)) for e in timed_runs) / 1000.0 / count
            results[f"{vid}_{threads}t"] = {
                "model_run_mean_ms": round(avg_run_ms, 3),
                "kern_sum_ms": kern_sum,
                "groups_ms": mean_groups,
                "groups_pct": {k: round(100.0 * v / kern_sum, 2) if kern_sum > 0 else 0.0 for k, v in mean_groups.items()},
            }
    return results


def _run_variant(variant: str, session, dsp: CDspVocoder | None, feed: dict, noise_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    t0 = time.perf_counter()
    outputs = session.run(None, feed)
    t1 = time.perf_counter()
    if variant in ("dense_masked", "dense_unmasked", "dense_unmasked_int8_acoustic"):
        audio, durations = outputs
        dsp_ms = 0.0
    elif variant == "fused_fp32_unmasked":
        audio, durations = dsp.run_fused(outputs)
        dsp_ms = (time.perf_counter() - t1) * 1000.0
    elif variant in ("compact_fp32_unmasked", "compact_int8_acoustic_unmasked"):
        audio, durations = dsp.run_compact(outputs, noise_full)
        dsp_ms = (time.perf_counter() - t1) * 1000.0
    else:
        raise ValueError(f"unknown variant: {variant}")
    ort_ms = (t1 - t0) * 1000.0
    return audio, durations, ort_ms, dsp_ms


def _worker_measure(cfg_json: str) -> None:
    params = json.loads(cfg_json)
    variant = params["variant"]
    model_path = Path(params["model_path"])
    out_dir = Path(params["out_dir"])
    checkpoint = Path(params["checkpoint"])
    text = params["text"]
    threads = int(params["threads"])
    warmup = int(params["warmup"])
    runs = int(params["runs"])

    match_frontend(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    rate, hop, max_frames = cfg["sample_rate"], cfg["hop"], cfg["max_frames"]
    del ckpt

    phones = phonemize(text)
    session = make_session(model_path, threads=threads, opt_level_name="ORT_ENABLE_ALL")
    noise_full = np.fromfile(out_dir / "noise.f32", dtype="<f4")
    dsp = CDspVocoder(out_dir / "dsp.so", out_dir / "constants.json", max_samples=hop * max_frames) if params["uses_dsp"] else None

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
        feed["noise"] = noise_full[None]

    for _ in range(warmup):
        _run_variant(variant, session, dsp, feed, noise_full)

    total_times, ort_times, dsp_times = [], [], []
    audio, durations = None, None
    for _ in range(runs):
        t0 = time.perf_counter()
        audio, durations, ort_ms, dsp_ms = _run_variant(variant, session, dsp, feed, noise_full)
        total_times.append((time.perf_counter() - t0) * 1000.0)
        ort_times.append(ort_ms)
        dsp_times.append(dsp_ms)

    samples = int(audio.shape[1])
    frames = int(durations.sum())
    res = {
        "variant": variant,
        "threads": threads,
        "tokens": len(phones),
        "frames": frames,
        "audio_samples": samples,
        "total": summarize_times(total_times, audio_seconds=samples / rate),
        "ort_median_ms": round(float(np.median(ort_times)), 3),
        "dsp_median_ms": round(float(np.median(dsp_times)), 3),
        **read_peak_rss_kib(),
    }
    print(json.dumps(res))


def run_benchmark_matrix(checkpoint: Path, out_dir: Path, artifacts: dict, warmup: int, runs: int) -> dict:
    variants = [
        "dense_masked",
        "dense_unmasked",
        "dense_unmasked_int8_acoustic",
        "fused_fp32_unmasked",
        "compact_fp32_unmasked",
        "compact_int8_acoustic_unmasked",
    ]
    orders = {}
    for order_name in ("forward", "reverse"):
        seq = list(variants)
        if order_name == "reverse":
            seq.reverse()
        rows = []
        for threads in (1, 4):
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
            env["NUMEXPR_NUM_THREADS"] = str(threads)
            for sent_key, text in BENCHMARK_SENTENCES:
                for vid in seq:
                    spec = artifacts[vid]
                    payload = json.dumps({
                        "variant": vid,
                        "model_path": spec["onnx_path"],
                        "uses_dsp": spec["uses_dsp"],
                        "out_dir": str(out_dir),
                        "checkpoint": str(checkpoint),
                        "text": text,
                        "threads": threads,
                        "warmup": warmup,
                        "runs": runs,
                    })
                    proc = subprocess.run(
                        [sys.executable, __file__, "--worker-json", payload],
                        capture_output=True, text=True, env=env, check=True,
                    )
                    row = json.loads(proc.stdout.strip().splitlines()[-1])
                    row["sentence"] = sent_key
                    row["order"] = order_name
                    rows.append(row)
                    print(
                        f"[{order_name:7s}] {threads}T {sent_key:6s} {vid:30s} "
                        f"med={row['total']['median_ms']:7.2f}ms (ort={row['ort_median_ms']:6.2f}, dsp={row['dsp_median_ms']:5.2f}) "
                        f"rtfx={row['total']['rtfx_median']:5.1f}x rss={row['vm_hwm_mib']}MiB",
                        flush=True,
                    )
        orders[order_name] = rows
    return orders


def evaluate_fidelity(checkpoint: Path, out_dir: Path, artifacts: dict) -> list[dict]:
    torch.set_num_threads(4)
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")
    c = model.config
    mel_loss = SpectralLoss(c.sample_rate, mel=(c.n_fft, c.hop, c.n_mels))
    noise_full = np.fromfile(out_dir / "noise.f32", dtype="<f4")
    dsp = CDspVocoder(out_dir / "dsp.so", out_dir / "constants.json", max_samples=c.hop * c.max_frames)

    variant_ids = [
        "dense_masked",
        "dense_unmasked",
        "dense_unmasked_int8_acoustic",
        "fused_fp32_unmasked",
        "compact_fp32_unmasked",
        "compact_int8_acoustic_unmasked",
    ]
    sessions = {vid: make_session(Path(artifacts[vid]["onnx_path"]), threads=4, opt_level_name="ORT_ENABLE_ALL") for vid in variant_ids}

    wav_dir = out_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sent_key, text in BENCHMARK_SENTENCES:
        phones = phonemize(text)
        tokens_t = torch.tensor([phones.ids], dtype=torch.long)
        stress_t = torch.tensor([phones.stress], dtype=torch.long)
        lengths_t = torch.tensor([len(phones)], dtype=torch.long)
        tokens_np = tokens_t.numpy()
        stress_np = stress_t.numpy()
        one = lambda v: np.array([v], dtype=np.float32)

        for voice_idx, voice_name in enumerate(c.speakers):
            with torch.inference_mode():
                torch_wave_t, torch_len_t, syn_meta = model.synthesize(
                    tokens_t, stress_t, lengths_t, torch.tensor([voice_idx], dtype=torch.long),
                    speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, pitch_jitter=0.0, voiced_vowels=True,
                )
            torch_wave = torch_wave_t[0, :int(torch_len_t[0])].numpy()
            torch_dur = syn_meta["durations"][0].numpy()
            with torch.no_grad():
                torch_mel = mel_loss.log_mel(torch.from_numpy(torch_wave[None]))[0].numpy()

            # Match TinyVoca.synthesize(seed=0) exact per-row random noise stream.
            noise_exact = torch.randn(len(torch_wave), generator=torch.Generator().manual_seed(0)).numpy()
            padded_noise = np.zeros(c.hop * c.max_frames, dtype=np.float32)
            padded_noise[:len(noise_exact)] = noise_exact

            base_feed = {
                "tokens": tokens_np,
                "stress": stress_np,
                "voice": np.array([voice_idx], dtype=np.int64),
                "speed": one(1.0),
                "pitch_shift": one(0.0),
                "pitch_range": one(1.0),
            }
            ref_audio, ref_dur, _, _ = _run_variant(
                "dense_masked", sessions["dense_masked"], None, {**base_feed, "noise": padded_noise[None]}, padded_noise
            )
            ref_wave = ref_audio[0]
            ref_rms = float(np.sqrt(np.mean(ref_wave.astype(np.float64) ** 2)))
            ref_peak = float(np.max(np.abs(ref_wave)))
            with torch.no_grad():
                ref_mel = mel_loss.log_mel(torch.from_numpy(ref_wave[None]))[0].numpy()

            for vid in variant_ids:
                feed = dict(base_feed)
                if any(inp.name == "noise" for inp in sessions[vid].get_inputs()):
                    feed["noise"] = padded_noise[None]
                audio, dur, _, _ = _run_variant(vid, sessions[vid], dsp if artifacts[vid]["uses_dsp"] else None, feed, padded_noise)
                wave = audio[0]
                finite_ok = bool(np.isfinite(wave).all())
                dur_l1 = int(np.abs(dur[0] - ref_dur[0]).sum())
                dur_torch_l1 = int(np.abs(dur[0] - torch_dur).sum())
                samples_diff = int(wave.size - ref_wave.size)

                metrics = {
                    "sentence": sent_key,
                    "voice": voice_name,
                    "variant": vid,
                    "finite": finite_ok,
                    "durations_equal_dense": dur_l1 == 0,
                    "durations_l1_dense": dur_l1,
                    "durations_equal_torch": dur_torch_l1 == 0,
                    "samples_diff": samples_diff,
                }
                if samples_diff == 0 and finite_ok:
                    diff = wave.astype(np.float64) - ref_wave.astype(np.float64)
                    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
                    err_rms = float(np.sqrt(np.mean(diff ** 2)))
                    max_abs = float(np.max(np.abs(diff)))
                    metrics.update({
                        "rms_ratio_dense": round(rms / max(ref_rms, 1e-12), 6),
                        "err_rms_rel_dense": round(err_rms / max(ref_rms, 1e-12), 6),
                        "max_peak_err_dense": round(max_abs / max(ref_peak, 1e-12), 6),
                        "snr_db_dense": _snr_db(ref_wave, wave),
                    })
                if wave.size == torch_wave.size and finite_ok:
                    torch_rms = float(np.sqrt(np.mean(torch_wave.astype(np.float64) ** 2)))
                    diff_t = wave.astype(np.float64) - torch_wave.astype(np.float64)
                    metrics["err_rms_rel_torch"] = round(float(np.sqrt(np.mean(diff_t ** 2))) / max(torch_rms, 1e-12), 6)
                    metrics["snr_db_torch"] = _snr_db(torch_wave, wave)
                with torch.no_grad():
                    cand_mel = mel_loss.log_mel(torch.from_numpy(wave[None]))[0].numpy()
                metrics["mel_dtw_dense"] = round(dtw_distance_np(cand_mel, ref_mel), 5)
                metrics["mel_dtw_torch"] = round(dtw_distance_np(cand_mel, torch_mel), 5)
                rows.append(metrics)

                if (sent_key == "medium" and voice_name == "F1") or (sent_key == "long" and voice_name == "M1"):
                    if vid in ("dense_masked", "dense_unmasked", "compact_fp32_unmasked", "compact_int8_acoustic_unmasked"):
                        sf.write(str(wav_dir / f"{sent_key}_{voice_name}_{vid}.wav"), wave, c.sample_rate)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--fidelity-only", action="store_true")
    parser.add_argument("--worker-json")
    args = parser.parse_args()

    if args.worker_json:
        _worker_measure(args.worker_json)
        return

    meta = {
        "environment": cpu_info(),
        "loadavg_start": read_loadavg(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "warmup": args.warmup,
        "runs": args.runs,
    }
    print("Preparing E04 artifacts...", flush=True)
    artifacts = prepare_artifacts(args.checkpoint, args.out_dir)
    if args.fidelity_only:
        fidelity = evaluate_fidelity(args.checkpoint, args.out_dir, artifacts)
        print(json.dumps(fidelity, indent=2, ensure_ascii=False))
        return
    print("Profiling ORT operator breakdown (dense_masked vs dense_unmasked)...", flush=True)
    ort_profile = profile_ort_comparison(args.checkpoint, args.out_dir)
    print("Running full E04 latency matrix (forward & reverse orders)...", flush=True)
    matrix = run_benchmark_matrix(args.checkpoint, args.out_dir, artifacts, warmup=args.warmup, runs=args.runs)
    print("Evaluating numerical & spectral fidelity...", flush=True)
    fidelity = evaluate_fidelity(args.checkpoint, args.out_dir, artifacts)

    meta["loadavg_end"] = read_loadavg()
    out_json = args.out_dir / "benchmark_e04.json"
    out_json.write_text(
        json.dumps(
            {"meta": meta, "artifacts": artifacts, "ort_profile": ort_profile, "matrix": matrix, "fidelity": fidelity},
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
