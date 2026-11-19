"""Remote Colab L4 runner for P6 paired E06 replay (8k -> 11k checkpoint -> 12k).

Runs the 3 paired arms from the exact same pre-GAN checkpoint e06_20k/step-00008000.pt:
  - C : standard E06 recipe (GAN at 8k, ramp 3000, global clip 10, alpha=1.0)
  - D : identical to C with GAN deferred past 12k (--gan-start 999999 --align-gan-rng)
  - P : identical GAN & ramp to C, protected by --prosody-grad-scale 0.0 --separate-prosody-clip

All 3 arms restore full state from step-00008000.pt (--restore-state-from-init):
generator, critics, both optimizers, sampler, CPU/CUDA RNG, and original LR schedule.
Stage 1 runs all 3 arms 8k -> 11k and saves 11k artifacts BEFORE Stage 2 extends
all 3 arms 11k -> 12k via --resume (allowing a clean budget gate at 11k).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


ARMS = {
    "C": (8000, []),
    "D": (999999, ["--align-gan-rng"]),
    "P": (8000, ["--prosody-grad-scale", "0.0", "--separate-prosody-clip"]),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_cmd(
    train_manifest: Path,
    val_manifest: Path,
    config_path: Path,
    out_dir: Path,
    steps: int,
    gan_start: int,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable, "-m", "tiny_voca.train",
        "--train", str(train_manifest),
        "--validation", str(val_manifest),
        "--config", str(config_path),
        "--output", str(out_dir),
        "--device", "cuda",
        "--steps", str(steps),
        "--batch-frames", "9000",
        "--max-batch", "32",
        "--learning-rate", "0.0003",
        "--warmup", "1000",
        "--lr-half-life", "100000",
        "--grad-clip", "10.0",
        "--gan-start", str(gan_start),
        "--gan-ratio", "0.7",
        "--gan-ramp", "3000",
        "--feature-matching", "30.0",
        "--gan-scale-max", "2000.0",
        "--crop-samples", "8192",
        "--crops", "2",
        "--teacher-hold", "3000",
        "--teacher-ramp", "7000",
        "--teacher-final", "0.0",
        "--validation-size", "32",
        "--save-every", "1000",
        "--keep-every", "1000",
        "--log-every", "50",
        "--seed", "20261113",
        "--max-seconds", "7200",
        *extra,
    ]


def _run_and_log(cmd: list[str], log_path: Path):
    t0 = time.monotonic()
    print(f"=== EXEC: {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== EXEC ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n{' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        rc = proc.wait()
    dt = time.monotonic() - t0
    print(f"=== DONE rc={rc} elapsed={dt:.1f}s ({dt/60:.2f} min) ===", flush=True)
    if rc != 0:
        raise RuntimeError(f"Command failed with rc={rc}: {' '.join(cmd)}")
    return dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1_11k", "stage2_12k", "pack"), required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/content/p6"))
    parser.add_argument("--train", type=Path, default=Path("/content/fr4k/annotations_e01/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("/content/fr4k/annotations_e01/validation.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("/content/repo/configs/e06_160_384.json"))
    parser.add_argument("--ckpt-8k", type=Path, default=Path("/content/p6/step-00008000.pt"))
    parser.add_argument("--arms", nargs="+", default=["D", "C", "P"], choices=("C", "D", "P"))
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    timings_path = args.workdir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    if args.stage == "stage1_11k":
        print(f"Base ckpt SHA-256: {_sha256(args.ckpt_8k)}", flush=True)
        for arm in args.arms:
            out_dir = args.workdir / arm
            gan_start, arm_flags = ARMS[arm]
            extra = ["--init-from", str(args.ckpt_8k), "--restore-state-from-init", *arm_flags]
            cmd = _base_cmd(args.train, args.validation, args.config, out_dir, steps=11000, gan_start=gan_start, extra=extra)
            dt = _run_and_log(cmd, args.workdir / "stage1_11k.log")
            shutil.copy2(out_dir / "inference.pt", out_dir / "inference_11000.pt")
            timings[f"{arm}_8k_to_11k_sec"] = round(dt, 2)
            timings_path.write_text(json.dumps(timings, indent=2) + "\n")

    elif args.stage == "stage2_12k":
        for arm in args.arms:
            out_dir = args.workdir / arm
            gan_start, arm_flags = ARMS[arm]
            extra = ["--resume", str(out_dir / "last.pt"), *arm_flags]
            cmd = _base_cmd(args.train, args.validation, args.config, out_dir, steps=12000, gan_start=gan_start, extra=extra)
            dt = _run_and_log(cmd, args.workdir / "stage2_12k.log")
            shutil.copy2(out_dir / "inference.pt", out_dir / "inference_12000.pt")
            timings[f"{arm}_11k_to_12k_sec"] = round(dt, 2)
            timings_path.write_text(json.dumps(timings, indent=2) + "\n")

    elif args.stage == "pack":
        # Pack ONLY lightweight inference checkpoints, logs, recipes, metrics (NOT 43 MB optimizer last.pt).
        bundle = args.workdir / "p6_artifacts.tar.gz"
        sums = []
        with tarfile.open(bundle, "w:gz") as tf:
            for top_f in ("timings.json", "stage1_11k.log", "stage2_12k.log"):
                p = args.workdir / top_f
                if p.exists():
                    tf.add(p, arcname=top_f)
                    sums.append(f"{_sha256(p)}  {top_f}")
            for arm in ("C", "D", "P"):
                ad = args.workdir / arm
                if not ad.exists():
                    continue
                for fname in ("inference_11000.pt", "inference_12000.pt", "metrics.jsonl", "recipe.json", "config.json"):
                    p = ad / fname
                    if p.exists():
                        rel = f"{arm}/{fname}"
                        tf.add(p, arcname=rel)
                        sums.append(f"{_sha256(p)}  {rel}")
            sum_path = args.workdir / "SHA256SUMS"
            sum_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
            tf.add(sum_path, arcname="SHA256SUMS")
        print(f"Packed {bundle} ({bundle.stat().st_size / (1024*1024):.2f} MB), SHA256={_sha256(bundle)}", flush=True)


if __name__ == "__main__":
    main()
