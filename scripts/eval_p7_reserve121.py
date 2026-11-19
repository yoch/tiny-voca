#!/usr/bin/env python3
"""Single final confirmation of frozen P7 candidate (p7_sp_cal_prosody, sp_thr=0.95)
vs base_35k on the 121 untouched test.jsonl pairs (242 recordings, 0 GPU).

Only executed AFTER P7 won on held-out P4-dev (4 significant wins, 0 regressions)
AND passed human blind A/B listening (7-1 overall; 4-0 real texts, 3-1 pauses,
with P05 confirming calibrated sp_thr=0.95).
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.evaluate import dtw_distance, normalize_for_wer
from tiny_voca.losses import SpectralLoss
from tiny_voca.model import TinyVoca
from tiny_voca.prosody_fr import FrenchProsodyRefiner, load_prosody_refiner, synthesize_with_refiner
from tiny_voca.text import SYMBOL_TO_ID

from eval_p4_ablation import stratified_text_bootstrap_ci
from eval_p5_f0_causal import sha256_file
from p0_expressivity import _sentence_category, compute_intonation_metrics, extract_unbiased_f0

_SP_ID = SYMBOL_TO_ID["<sp>"]
_BOS_ID = SYMBOL_TO_ID["<bos>"]
_EOS_ID = SYMBOL_TO_ID["<eos>"]


def _mean_opt(vals):
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 5) if xs else None


@torch.inference_mode()
def _synth(host: TinyVoca, refiner: FrenchProsodyRefiner | None, row: dict, item: dict, sp_thr: float, seed: int, oracle_dur: bool = False):
    tok = item["tokens"][None]
    st = item["stress"][None]
    tl = torch.tensor([tok.shape[1]], dtype=torch.long)
    spk = torch.tensor([int(item["speaker"])], dtype=torch.long)
    if oracle_dur:
        hidden, mask = host.encode(tok, st, tl, spk)
        durations = item["durations"][None]
        frames, frame_mask, frame_lengths = host.expand(hidden, durations, mask, spk)
        pro_h, host_pro = host.predict_prosody(frames, frame_mask)
        if refiner is not None:
            from tiny_voca.prosody_fr import extract_french_phrasal_features
            ph = extract_french_phrasal_features(item["tokens"], row["text"])[None]
            exp_ph, fpos = FrenchProsodyRefiner.expand_token_features(ph, durations, frame_lengths)
            prosody = refiner.refine_prosody(pro_h.float(), host_pro.float(), exp_ph, fpos, frame_mask, spk).to(host_pro.dtype)
        else:
            prosody = host_pro
        ends = durations.cumsum(1)
        frame = torch.arange(frames.shape[1])[None]
        owner = torch.searchsorted(ends, frame, right=True).clamp(max=tok.shape[1] - 1)
        vowel = host.vowel_tokens[tok].gather(1, owner) & frame_mask
        prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None], prosody[..., 2:]), dim=-1)
        acoustic = host.decode(pro_h, host.hard_prosody(prosody), frame_mask)
        noise = torch.randn(1, int(frame_lengths[0]) * host.config.hop, generator=torch.Generator().manual_seed(seed))
        wav = host.vocoder(acoustic, host.frequency(prosody[..., 0]), frame_lengths, noise=noise)
        return wav[0, : int(frame_lengths[0]) * host.config.hop].numpy(), durations[0].numpy()

    wav, slen, meta = synthesize_with_refiner(
        host, refiner, tok, st, tl, spk, texts=[row["text"]], seed=seed, sp_pause_threshold=sp_thr
    )
    return wav[0, : int(slen[0].item())].numpy(), meta["durations"][0].numpy()


def main():
    t0 = time.time()
    host_path = Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt")
    ref_path = Path(".local-runs/p7_refiner/p7_sp_cal_prosody.pt")
    test_manifest = Path(".local-runs/colab/fr4k/annotations_e01/test.jsonl")
    reserve_json = Path("docs/experiments/p5_untouched_test_reserve.json")
    out_json = Path("docs/experiments/p7_final_reserve121_summary.json")

    # Ensure frozen candidate uses calibrated sp_pause_threshold=0.95
    payload = torch.load(ref_path, map_location="cpu", weights_only=False)
    payload["refiner_state"]["sp_pause_threshold"] = torch.tensor(0.95, dtype=torch.float32)
    payload["sp_pause_threshold"] = 0.95
    torch.save(payload, ref_path)

    host_ckpt = torch.load(host_path, map_location="cpu", weights_only=False)
    cfg = Config(**host_ckpt["config"]) if isinstance(host_ckpt["config"], dict) else host_ckpt["config"]
    host = TinyVoca(cfg).eval()
    host.load_state_dict(host_ckpt["model"])
    refiner = load_prosody_refiner(ref_path)
    spectral = SpectralLoss(cfg.sample_rate)

    reserve = json.loads(reserve_json.read_text(encoding="utf-8"))
    allowed_hashes = set(reserve["untouched_test_hashes"])
    ds = AnnotatedDataset(test_manifest, cfg, voiced_vowels=True)
    id_to_idx = {r["id"]: i for i, r in enumerate(ds.rows)}
    selected = []
    for r in ds.rows:
        if r["text_hash"] in allowed_hashes:
            rr = dict(r)
            rr["category"] = _sentence_category(r["text"])
            selected.append(rr)
    selected.sort(key=lambda x: (x["category"], x["text_hash"], x["speaker"]))
    print(f"Confirming frozen P7 (sp_thr=0.95) vs base_35k on {len(selected)} untouched recordings ({len(selected)//2} texts)...", flush=True)

    phone_mask = host.phone_tokens
    arms = (("base_35k", None, 0.50), ("p7_sp_cal_prosody", refiner, 0.95))
    per_arm: dict[str, list[dict]] = {name: [] for name, _, _ in arms}
    pause_stats = {name: {"sp_tp": 0, "sp_fp": 0, "sp_fn": 0, "pu_tp": 0, "pu_fp": 0, "pu_fn": 0} for name, _, _ in arms}
    asr_batches: dict[str, list[tuple[str, str, str, np.ndarray]]] = {name: [] for name, _, _ in arms}

    for k, row in enumerate(selected):
        uid = row["id"]
        item = ds[id_to_idx[uid]]
        ref_w = item["wave"].numpy()
        ref_d = item["durations"].numpy()
        tok_np = item["tokens"].numpy()
        ref_mel = spectral.log_mel(torch.from_numpy(ref_w)[None])[0].numpy()
        ref_f0 = extract_unbiased_f0(ref_w, cfg, frames=int(ref_d.sum()))

        tok_t = item["tokens"]
        is_ph = phone_mask[tok_t].numpy()
        is_edge = (tok_np == _BOS_ID) | (tok_np == _EOS_ID)
        is_sp = tok_np == _SP_ID
        is_pu = (~is_ph) & (~is_edge) & (~is_sp)
        gt_pos = ref_d > 0

        for name, ref_mod, sp_thr in arms:
            g_free, g_dur = _synth(host, ref_mod, row, item, sp_thr, seed=12000 + k, oracle_dur=False)
            g_ora, _ = _synth(host, ref_mod, row, item, sp_thr, seed=12000 + k, oracle_dur=True)

            g_mel = spectral.log_mel(torch.from_numpy(g_free)[None])[0].numpy()
            mel_dtw = float(dtw_distance(ref_mel, g_mel))
            stft_b, _ = spectral(
                torch.from_numpy(g_ora)[None],
                torch.from_numpy(ref_w)[None],
                torch.tensor([len(ref_w)], dtype=torch.long),
            )
            int_m = compute_intonation_metrics(ref_w, g_free, ref_d, g_dur, tok_np, cfg, ref_f0_info=ref_f0)

            pr_pos = g_dur > 0
            ps = pause_stats[name]
            ps["sp_tp"] += int((pr_pos[is_sp] & gt_pos[is_sp]).sum())
            ps["sp_fp"] += int((pr_pos[is_sp] & ~gt_pos[is_sp]).sum())
            ps["sp_fn"] += int((~pr_pos[is_sp] & gt_pos[is_sp]).sum())
            ps["pu_tp"] += int((pr_pos[is_pu] & gt_pos[is_pu]).sum())
            ps["pu_fp"] += int((pr_pos[is_pu] & ~gt_pos[is_pu]).sum())
            ps["pu_fn"] += int((~pr_pos[is_pu] & gt_pos[is_pu]).sum())

            per_arm[name].append({
                "id": uid,
                "text_hash": row["text_hash"],
                "speaker": row["speaker"],
                "category": row["category"],
                "mel_dtw": round(mel_dtw, 5),
                "oracle_dur_stft": round(float(stft_b.item()), 5),
                "f0_corr": int_m.get("f0_corr"),
                "f0_mae_st": int_m.get("f0_mae_st"),
                "f0_slope_mae_st_s": int_m.get("f0_slope_mae_st_s"),
                "f0_std_ratio": int_m.get("f0_std_ratio"),
            })
            asr_batches[name].append((uid, row["speaker"], row["text"], g_free))

        if (k + 1) % 60 == 0 or k + 1 == len(selected):
            print(f"  acoustic {k + 1}/{len(selected)} ({time.time() - t0:.1f}s)", flush=True)

    # Full ASR on all 242 recordings (faster-whisper small int8 CPU)
    from faster_whisper import WhisperModel
    import jiwer
    import librosa

    print("Running faster-whisper small ASR on all 242 untouched recordings x 2...", flush=True)
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    asr_summary = {}
    for name, _, _ in arms:
        by_spk = defaultdict(list)
        per_u = {}
        for j, (uid, spk, ref_txt, wav) in enumerate(asr_batches[name]):
            wav16 = librosa.resample(wav.astype(np.float32), orig_sr=cfg.sample_rate, target_sr=16000)
            segs, _ = whisper.transcribe(wav16, language="fr", beam_size=5, vad_filter=False)
            hyp = " ".join(s.text.strip() for s in segs).strip()
            rn = normalize_for_wer(ref_txt, language="fr")
            hn = normalize_for_wer(hyp, language="fr")
            per_u[uid] = {"wer": round(float(jiwer.wer([rn], [hn])) if rn else 0.0, 5),
                          "cer": round(float(jiwer.cer([rn], [hn])) if rn else 0.0, 5)}
            by_spk[spk].append((rn, hn))
            by_spk["all"].append((rn, hn))
        asr_summary[name] = {
            g: {
                "samples": len(ps),
                "corpus_wer": round(float(jiwer.wer([p[0] for p in ps], [p[1] for p in ps])), 4),
                "corpus_cer": round(float(jiwer.cer([p[0] for p in ps], [p[1] for p in ps])), 4),
            }
            for g, ps in by_spk.items()
        }
        for r in per_arm[name]:
            r["wer"] = per_u[r["id"]]["wer"]
            r["cer"] = per_u[r["id"]]["cer"]
        print(f"  ASR {name:18s}: WER={asr_summary[name]['all']['corpus_wer']*100:.2f}% CER={asr_summary[name]['all']['corpus_cer']*100:.2f}% ({time.time() - t0:.1f}s)", flush=True)

    pause_summary = {}
    for name, ps in pause_stats.items():
        sp_p = ps["sp_tp"] / max(ps["sp_tp"] + ps["sp_fp"], 1)
        sp_r = ps["sp_tp"] / max(ps["sp_tp"] + ps["sp_fn"], 1)
        pu_p = ps["pu_tp"] / max(ps["pu_tp"] + ps["pu_fp"], 1)
        pu_r = ps["pu_tp"] / max(ps["pu_tp"] + ps["pu_fn"], 1)
        pause_summary[name] = {**ps, "sp_prec": round(sp_p, 4), "sp_rec": round(sp_r, 4),
                               "pu_prec": round(pu_p, 4), "pu_rec": round(pu_r, 4)}

    metrics = ("mel_dtw", "oracle_dur_stft", "f0_corr", "f0_mae_st", "f0_slope_mae_st_s", "f0_std_ratio", "wer", "cer")
    summary = {}
    for name, _, _ in arms:
        summary[name] = {}
        for spk in ("all", "F1", "M1"):
            sub = [r for r in per_arm[name] if spk == "all" or r["speaker"] == spk]
            summary[name][spk] = {"n": len(sub), **{m: _mean_opt([r.get(m) for r in sub]) for m in metrics}}

    contrasts = {}
    for spk in ("all", "F1", "M1"):
        sf = None if spk == "all" else spk
        contrasts[spk] = {
            m: stratified_text_bootstrap_ci(
                per_arm, {"p7_sp_cal_prosody": 1.0, "base_35k": -1.0}, m, speaker_filter=sf, n_boot=5000, seed=20261119
            )
            for m in ("mel_dtw", "oracle_dur_stft", "f0_corr", "f0_mae_st", "f0_slope_mae_st_s", "wer", "cer")
        }

    report = {
        "partition": "reserve121_untouched_test",
        "n_texts": len(selected) // 2,
        "n_recordings": len(selected),
        "host_sha256": sha256_file(host_path),
        "refiner_sha256": sha256_file(ref_path),
        "sp_pause_threshold": 0.95,
        "pause_summary": pause_summary,
        "summary": summary,
        "asr_corpus": asr_summary,
        "paired_bootstrap_95ci_p7_minus_base": contrasts,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved final reserve-121 confirmation to {out_json} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
