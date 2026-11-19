"""Reproduce the P4 verdict with full provenance, cross-split intersection audit,
per-recording metrics export, reserved untouched test manifest, and text-level
stratified paired bootstrap CIs (keeping F1/M1 paired per text).

Usage:
    PYTHONPATH=src:scripts ./env/bin/python scripts/reproduce_p4_verdict.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.train import balanced_indices
from eval_p4_ablation import KEY_METRICS, stratified_text_bootstrap_ci
from p0_expressivity import _sentence_category


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01"))
    parser.add_argument("--p0-report", type=Path, default=Path(".local-runs/p0_expressivity/p0_report.json"))
    parser.add_argument("--p4-dev-report", type=Path, default=Path(".local-runs/p4_ablation/p4_final/dev_screen/p4_dev_report.json"))
    parser.add_argument("--p4-final-report", type=Path, default=Path(".local-runs/p4_ablation/final_heldout/p4_final_report.json"))
    parser.add_argument("--p4-final-per-utt", type=Path, default=Path(".local-runs/p4_ablation/final_heldout/p4_final_per_utterance.json"))
    parser.add_argument("--e06-eval-report", type=Path, default=Path(".local-runs/colab/e06_20k/e06_20k_final/eval/report.json"))
    parser.add_argument("--fr35k-eval-report", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/eval/report.json"))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--out-dir", type=Path, default=Path(".local-runs/p5_audit/p4_reproducibility"))
    parser.add_argument("--reserve-out", type=Path, default=Path("docs/experiments/p5_untouched_test_reserve.json"))
    args = parser.parse_args()

    # 1. Full SHA-256 hashes of all P4 checkpoints
    ckpt_map = {
        "base_35k_inference": Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt"),
        "base_35k_last": Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/last.pt"),
        "C_35400": Path(".local-runs/p4_ablation/p4_final/C/inference_35400.pt"),
        "G_35400": Path(".local-runs/p4_ablation/p4_final/G/inference_35400.pt"),
        "F_35400": Path(".local-runs/p4_ablation/p4_final/F/inference_35400.pt"),
        "GF_35400": Path(".local-runs/p4_ablation/p4_final/GF/inference_35400.pt"),
        "C_35900": Path(".local-runs/p4_ablation/p4_final/C/inference_35900.pt"),
        "G_35900": Path(".local-runs/p4_ablation/p4_final/G/inference_35900.pt"),
        "e06_8k": Path(".local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt"),
        "e06_20k_inference": Path(".local-runs/colab/e06_20k/e06_20k_final/inference.pt"),
        "fr_ref_10k": Path(".local-runs/colab/fr_ref_20k_snap10k/snap10k/step-00010000.pt"),
    }
    checkpoint_hashes = {}
    for name, path in ckpt_map.items():
        if path.exists():
            checkpoint_hashes[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    # 2. Load manifests and build id -> text_hash maps
    train_rows = _load_jsonl(args.annotations / "train.jsonl")
    val_rows = _load_jsonl(args.annotations / "validation.jsonl")
    test_rows = _load_jsonl(args.annotations / "test.jsonl")
    all_rows_by_id = {r["id"]: r for r in train_rows + val_rows + test_rows}

    val_ds = AnnotatedDataset(args.annotations / "validation.jsonl", Config(), cache=False)
    monitor_ids = {val_ds.rows[i]["id"] for i in balanced_indices(val_ds, 32)}

    p0_ids = set(json.loads(args.p0_report.read_text(encoding="utf-8"))["selected_ids"])
    p4_dev_ids = set(json.loads(args.p4_dev_report.read_text(encoding="utf-8"))["selected_ids"])
    p4_final_ids = set(json.loads(args.p4_final_report.read_text(encoding="utf-8"))["selected_ids"])

    def eval_report_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()
        rep = json.loads(path.read_text(encoding="utf-8"))
        out = set()
        for s in rep.get("samples", []):
            uid = s.get("id")
            if uid and uid in all_rows_by_id:
                out.add(uid)
        return out

    e06_eval_ids = eval_report_ids(args.e06_eval_report)
    fr35k_eval_ids = eval_report_ids(args.fr35k_eval_report)

    def hashes_of(ids: set[str]) -> set[str]:
        return {all_rows_by_id[i]["text_hash"] for i in ids if i in all_rows_by_id}

    splits_ids = {
        "train": {r["id"] for r in train_rows},
        "monitor_32": monitor_ids,
        "p4_dev": p4_dev_ids,
        "p0_40": p0_ids,
        "p4_final_consulted": p4_final_ids,
        "e06_eval": e06_eval_ids,
        "fr35k_eval": fr35k_eval_ids,
    }
    splits_hashes = {k: hashes_of(v) for k, v in splits_ids.items()}

    intersections = {}
    split_names = list(splits_ids.keys())
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            inter_id = sorted(splits_ids[a] & splits_ids[b])
            inter_th = sorted(splits_hashes[a] & splits_hashes[b])
            intersections[f"{a}__x__{b}"] = {
                "shared_ids": len(inter_id),
                "shared_text_hashes": len(inter_th),
                "example_ids": inter_id[:6],
            }

    # 3. Build strictly untouched test.jsonl reserve manifest (excluding ALL consulted test sets)
    consulted_test_ids = p0_ids | p4_final_ids | (e06_eval_ids & {r["id"] for r in test_rows}) | (fr35k_eval_ids & {r["id"] for r in test_rows})
    consulted_test_hashes = hashes_of(consulted_test_ids)
    by_test_th: dict[str, dict[str, dict]] = {}
    for r in test_rows:
        by_test_th.setdefault(r["text_hash"], {})[r["speaker"]] = r

    untouched_pairs = []
    untouched_by_cat: dict[str, int] = {}
    for th, spks in sorted(by_test_th.items()):
        if th in consulted_test_hashes:
            continue
        if "F1" in spks and "M1" in spks:
            cat = _sentence_category(spks["F1"]["text"])
            untouched_by_cat[cat] = untouched_by_cat.get(cat, 0) + 1
            untouched_pairs.append({
                "text_hash": th,
                "category": cat,
                "ids": [spks["F1"]["id"], spks["M1"]["id"]],
            })

    reserve_payload = {
        "note": "P4-final (80 utt / 40 texts) is now permanently marked CONSULTED alongside P0 (40 utt / 20 texts) and evaluate.py test samples. Use consulted_* as --exclude-manifest for any future final confirmation.",
        "consulted_ids": sorted(consulted_test_ids),
        "consulted_text_hashes": sorted(consulted_test_hashes),
        "p0_text_hashes": sorted(splits_hashes["p0_40"]),
        "p4_dev_text_hashes": sorted(splits_hashes["p4_dev"]),
        "p4_final_text_hashes": sorted(splits_hashes["p4_final_consulted"]),
        "untouched_test_pairs_count": len(untouched_pairs),
        "untouched_test_recordings_count": len(untouched_pairs) * 2,
        "untouched_by_category": untouched_by_cat,
        "untouched_test_hashes": [p["text_hash"] for p in untouched_pairs],
    }
    args.reserve_out.parent.mkdir(parents=True, exist_ok=True)
    args.reserve_out.write_text(json.dumps(reserve_payload, indent=2) + "\n", encoding="utf-8")

    # 4. Attach text_hash to per_utterance rows and run stratified text-level paired bootstrap
    per_utt = json.loads(args.p4_final_per_utt.read_text(encoding="utf-8"))
    for vname, rows in per_utt.items():
        for r in rows:
            if "text_hash" not in r and r["id"] in all_rows_by_id:
                r["text_hash"] = all_rows_by_id[r["id"]]["text_hash"]

    contrasts_spec = {
        # 35,400 2x2 simple effects, factorial main effects, and pure interaction
        "G_minus_C_35400": {"G": 1.0, "C": -1.0},
        "F_minus_C_35400": {"F": 1.0, "C": -1.0},
        "GF_minus_C_35400": {"GF": 1.0, "C": -1.0},
        "interaction_GF_minus_G_minus_F_plus_C_35400": {"GF": 1.0, "G": -1.0, "F": -1.0, "C": 1.0},
        "main_effect_G_35400": {"G": 0.5, "C": -0.5, "GF": 0.5, "F": -0.5},
        "main_effect_F_35400": {"F": 0.5, "C": -0.5, "GF": 0.5, "G": -0.5},
        # vs base_35k
        "C_minus_base35k_35400": {"C": 1.0, "base_35k": -1.0},
        "G_minus_base35k_35400": {"G": 1.0, "base_35k": -1.0},
        "F_minus_base35k_35400": {"F": 1.0, "base_35k": -1.0},
        "GF_minus_base35k_35400": {"GF": 1.0, "base_35k": -1.0},
        # Matched 35,900 contrast
        "G35900_minus_C35900": {"G_35900": 1.0, "C_35900": -1.0},
        "C35900_minus_base35k": {"C_35900": 1.0, "base_35k": -1.0},
        "G35900_minus_base35k": {"G_35900": 1.0, "base_35k": -1.0},
    }

    stratified_results = {}
    focus_metrics = (
        "f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st",
        "gen_f0_local_std_st", "gen_accent_prom_st", "gen_vowel_voiced",
        "mel_dtw", "oracle_dur_stft", "phone_rel_dur_mae", "gen_f0_break_rate",
    )
    for cname, coefs in contrasts_spec.items():
        stratified_results[cname] = {}
        for grp, spk_f, cat_f in (("all", None, None), ("F1", "F1", None), ("M1", "M1", None), ("question", None, "question")):
            stratified_results[cname][grp] = {
                mk: stratified_text_bootstrap_ci(
                    per_utt, coefs, mk, speaker_filter=spk_f, category_filter=cat_f,
                    n_boot=args.n_boot, seed=args.seed,
                )
                for mk in focus_metrics
            }

    # 5. Export per-recording CSV table
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "p4_final_per_recording.csv"
    csv_cols = [
        "variant", "id", "text_hash", "speaker", "category",
        "f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st",
        "gen_f0_std_st", "gen_f0_local_std_st", "gen_accent_prom_st",
        "gen_vowel_voiced", "voicing_agreement", "gen_f0_break_rate",
        "mel_dtw", "oracle_dur_stft", "voice_mel_cosine",
        "phone_mae_ms", "phone_rel_dur_mae", "pre_pausal_ratio", "local_rate_cv",
        "pause_tp", "pause_fp", "pause_fn", "pause_tp_mae_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols)
        w.writeheader()
        for vname, rows in per_utt.items():
            for r in rows:
                w.writerow({"variant": vname, **{c: r.get(c) for c in csv_cols if c != "variant"}})

    out_payload = {
        "checkpoint_sha256": checkpoint_hashes,
        "split_counts": {k: {"ids": len(v), "text_hashes": len(splits_hashes[k])} for k, v in splits_ids.items()},
        "intersections": intersections,
        "selected_text_hashes": {
            "p4_dev": sorted(splits_hashes["p4_dev"]),
            "p4_final_consulted": sorted(splits_hashes["p4_final_consulted"]),
            "p0_40": sorted(splits_hashes["p0_40"]),
        },
        "untouched_test_reserve_summary": {
            "pairs": len(untouched_pairs),
            "recordings": len(untouched_pairs) * 2,
            "by_category": untouched_by_cat,
        },
        "stratified_text_bootstrap_95ci": stratified_results,
        "per_recording_csv": str(csv_path),
    }
    report_path = args.out_dir / "p4_reproducibility_report.json"
    report_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    # Compact stdout summary
    print("=== CHECKPOINT SHA-256 ===")
    for k, v in checkpoint_hashes.items():
        print(f"  {k:20s}: {v['sha256']} ({v['bytes']:,} B)")
    print("\n=== NON-ZERO INTERSECTIONS (by text_hash / id) ===")
    for k, v in intersections.items():
        if v["shared_text_hashes"] > 0 or v["shared_ids"] > 0:
            print(f"  {k:38s}: ids={v['shared_ids']}, text_hashes={v['shared_text_hashes']} {v['example_ids'][:3]}")
    print(f"\n=== UNTOUCHED TEST RESERVE: {len(untouched_pairs)} pairs ({len(untouched_pairs)*2} rec) {untouched_by_cat} ===")
    print("\n=== TEXT-LEVEL STRATIFIED BOOTSTRAP 95% CI (n_boot=5000, all) ===")
    for cname in (
        "G_minus_C_35400", "F_minus_C_35400", "GF_minus_C_35400",
        "interaction_GF_minus_G_minus_F_plus_C_35400",
        "main_effect_G_35400", "main_effect_F_35400",
        "G_minus_base35k_35400", "GF_minus_base35k_35400",
        "G35900_minus_C35900", "G35900_minus_base35k",
    ):
        g = stratified_results[cname]["all"]
        parts = []
        for mk in ("f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st", "mel_dtw", "oracle_dur_stft"):
            ci = g[mk]
            st = "*" if ci["significant"] else ""
            parts.append(f"{mk}={ci['mean_delta']:+.4f}[{ci['ci95_low']:+.4f},{ci['ci95_high']:+.4f}]{st}")
        print(f"  {cname:43s}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
