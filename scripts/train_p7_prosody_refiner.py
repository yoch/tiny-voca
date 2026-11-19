#!/usr/bin/env python3
"""Train P7 FrenchProsodyRefiner on top of frozen fr_ref_35k (100% local CPU, 0 GPU).

Stages:
  1. Cache frozen fr_ref_35k hidden states & predictions once (token-level + frame-level).
  2. Train Stage A (Rhythm & Pause residual head) + calibrate <sp> threshold on val_cal
     (validation.jsonl excluding P4-dev so P4-dev stays leak-free!).
  3. Train Stage B (Expressive Intonation & Energy residual head) with pitch shape,
     utterance spread, and terminal-inflection losses.
  4. Save self-contained checkpoint (.local-runs/p7_refiner/p7_refiner.pt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_voca.config import Config
from src.tiny_voca.data import VOWEL_IDS, continuous_pitch
from src.tiny_voca.losses import masked_mean, pitch_shape_loss
from src.tiny_voca.model import TinyVoca
from src.tiny_voca.prosody_fr import (
    FrenchProsodyRefiner,
    PHRASAL_DIM,
    extract_french_phrasal_features,
)
from src.tiny_voca.text import PHONE_IDS, SYMBOL_TO_ID

_SP_ID = SYMBOL_TO_ID["<sp>"]
_BOS_ID = SYMBOL_TO_ID["<bos>"]
_EOS_ID = SYMBOL_TO_ID["<eos>"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_p4_dev_ids(p4_dev_report: Path) -> set[str]:
    if not p4_dev_report.exists():
        return set()
    data = json.loads(p4_dev_report.read_text(encoding="utf-8"))
    if isinstance(data.get("selected_ids"), list) and data["selected_ids"]:
        return set(data["selected_ids"])
    for key in ("utterances", "rows", "selected", "samples"):
        if isinstance(data.get(key), list):
            ids = {r["id"] for r in data[key] if isinstance(r, dict) and "id" in r}
            if ids:
                return ids
    return set()


def _load_manifest_rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.inference_mode()
def build_or_load_cache(
    model: TinyVoca,
    cfg: Config,
    manifest: Path,
    acoustic_root: Path,
    cache_path: Path,
    batch_size: int = 32,
) -> list[dict]:
    if cache_path.exists():
        t0 = time.time()
        items = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[cache] Loaded {len(items)} items from {cache_path} in {time.time() - t0:.1f}s")
        return items

    t0 = time.time()
    rows = _load_manifest_rows(manifest)
    items: list[dict] = []

    # Sort by frame length for efficient CPU batching, then restore original order.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["frames"])
    out_by_idx: list[dict | None] = [None] * len(rows)

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        batch_rows = [rows[i] for i in idxs]

        toks_list, str_list, dur_list, phr_list, spk_list = [], [], [], [], []
        tgt_pro_list, trusted_list = [], []
        for r in batch_rows:
            tok = torch.tensor(r["tokens"], dtype=torch.long)
            st = torch.tensor(r["stress"], dtype=torch.long)
            dur = torch.tensor(r["durations"], dtype=torch.long)
            phr = extract_french_phrasal_features(tok, r.get("text", ""))
            spk = cfg.speakers.index(r["speaker"])

            stored = torch.load(acoustic_root / r["path"], map_location="cpu", weights_only=True)
            pitch, trusted = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
            voiced = stored["prosody"][:, 1].clone()
            vowel = torch.repeat_interleave(
                torch.tensor([int(t) in VOWEL_IDS for t in r["tokens"]], dtype=torch.bool),
                dur,
            )
            voiced = torch.maximum(voiced, vowel.to(voiced.dtype))
            energy = (stored["prosody"][:, 2] - cfg.energy_mean) / cfg.energy_scale
            tgt_pro = torch.stack((pitch, voiced, energy), dim=-1)

            toks_list.append(tok)
            str_list.append(st)
            dur_list.append(dur)
            phr_list.append(phr)
            spk_list.append(spk)
            tgt_pro_list.append(tgt_pro)
            trusted_list.append(trusted)

        lengths = torch.tensor([len(x) for x in toks_list], dtype=torch.long)
        speakers = torch.tensor(spk_list, dtype=torch.long)
        tokens_pad = pad_sequence(toks_list, batch_first=True, padding_value=0)
        stress_pad = pad_sequence(str_list, batch_first=True, padding_value=0)
        dur_pad = pad_sequence(dur_list, batch_first=True, padding_value=0)

        enc_hidden, tok_mask = model.encode(tokens_pad, stress_pad, lengths, speakers)
        host_dur_out = model.duration(enc_hidden, tok_mask).float()
        frames, frame_mask, frame_lengths = model.expand(enc_hidden, dur_pad, tok_mask, speakers)
        prosody_hidden, host_prosody = model.predict_prosody(frames, frame_mask)

        for bi, orig_idx in enumerate(idxs):
            n = int(lengths[bi].item())
            t = int(frame_lengths[bi].item())
            r = batch_rows[bi]
            out_by_idx[orig_idx] = {
                "id": r["id"],
                "text": r.get("text", ""),
                "speaker": int(spk_list[bi]),
                "tokens": toks_list[bi].to(torch.int16),
                "durations": dur_list[bi].to(torch.int16),
                "phrasal": phr_list[bi].to(torch.float16),
                "enc_hidden": enc_hidden[bi, :n].to(torch.float16).contiguous(),
                "host_dur_out": host_dur_out[bi, :n].to(torch.float32).contiguous(),
                "prosody_hidden": prosody_hidden[bi, :t].to(torch.float16).contiguous(),
                "host_prosody": host_prosody[bi, :t].to(torch.float32).contiguous(),
                "target_prosody": tgt_pro_list[bi].to(torch.float32).contiguous(),
                "pitch_trusted": trusted_list[bi].to(torch.float16).contiguous(),
                "is_question": "?" in r.get("text", ""),
            }

        if (start // batch_size + 1) % 30 == 0 or start + batch_size >= len(order):
            done = min(start + batch_size, len(order))
            print(f"  [cache {manifest.stem}] {done}/{len(order)} ({time.time() - t0:.1f}s)")

    items = [x for x in out_by_idx if x is not None]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, cache_path)
    print(f"[cache] Saved {len(items)} items to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")
    return items


def _batch_duration(items: list[dict], phone_mask: torch.Tensor):
    bsz = len(items)
    n_max = max(len(x["tokens"]) for x in items)
    c = items[0]["enc_hidden"].shape[-1]
    tokens = torch.zeros((bsz, n_max), dtype=torch.long)
    durations = torch.zeros((bsz, n_max), dtype=torch.long)
    phrasal = torch.zeros((bsz, n_max, PHRASAL_DIM), dtype=torch.float32)
    enc_hidden = torch.zeros((bsz, n_max, c), dtype=torch.float32)
    host_dur = torch.zeros((bsz, n_max, 2), dtype=torch.float32)
    mask = torch.zeros((bsz, n_max), dtype=torch.bool)
    speakers = torch.tensor([x["speaker"] for x in items], dtype=torch.long)

    for i, x in enumerate(items):
        n = len(x["tokens"])
        tokens[i, :n] = x["tokens"].long()
        durations[i, :n] = x["durations"].long()
        phrasal[i, :n] = x["phrasal"].float()
        enc_hidden[i, :n] = x["enc_hidden"].float()
        host_dur[i, :n] = x["host_dur_out"].float()
        mask[i, :n] = True
    return tokens, durations, phrasal, enc_hidden, host_dur, mask, speakers


def _duration_loss(
    pred_dur_out: torch.Tensor,
    host_dur_out: torch.Tensor,
    tokens: torch.Tensor,
    durations: torch.Tensor,
    mask: torch.Tensor,
    phone_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    is_phone = phone_mask[tokens] & mask
    is_edge = ((tokens == _BOS_ID) | (tokens == _EOS_ID)) & mask
    is_sp = (tokens == _SP_ID) & mask
    is_punct = (~is_phone) & (~is_edge) & (~is_sp) & mask

    present = (durations > 0) & mask
    log_tgt = torch.log1p(durations.float())
    reg_err = F.smooth_l1_loss(pred_dur_out[..., 0], log_tgt, beta=0.15, reduction="none")
    reg_w = torch.zeros_like(reg_err)
    reg_w = torch.where(is_phone, torch.ones_like(reg_w), reg_w)
    reg_w = torch.where((is_sp | is_punct) & present, torch.full_like(reg_w, 1.0), reg_w)
    reg_w = torch.where(is_edge & present, torch.full_like(reg_w, 0.3), reg_w)
    loss_reg = masked_mean(reg_err, reg_w)

    # Utterance-level total speech duration preservation (phones + internal pauses).
    internal = (is_phone | ((is_sp | is_punct) & present)).float()
    pred_frames = torch.expm1(pred_dur_out[..., 0].clamp(0.0, math.log1p(400.0)))
    pred_tot = (pred_frames * internal).sum(dim=1)
    tgt_tot = (durations.float() * internal).sum(dim=1).clamp_min(1.0)
    loss_tot = F.smooth_l1_loss((pred_tot - tgt_tot) / tgt_tot, torch.zeros_like(tgt_tot), beta=0.05)

    # Pause presence BCE: separate <sp> (asymmetric FP guard) vs punctuation (balanced).
    pause_tgt = (durations > 0).float()
    bce_raw = F.binary_cross_entropy_with_logits(pred_dur_out[..., 1], pause_tgt, reduction="none")
    sp_w = torch.where(pause_tgt > 0.5, torch.ones_like(bce_raw), torch.full_like(bce_raw, 2.0)) * is_sp.float()
    loss_sp = masked_mean(bce_raw, sp_w)
    loss_punct = masked_mean(bce_raw, is_punct.float())

    # Gentle anchor to frozen host so predictions stay clean out-of-domain.
    delta = pred_dur_out - host_dur_out
    loss_anchor = masked_mean(delta.square().mean(dim=-1), mask.float())

    total = loss_reg + 0.5 * loss_tot + 1.5 * loss_sp + 1.0 * loss_punct + 0.02 * loss_anchor
    return total, {
        "reg": float(loss_reg.item()),
        "tot": float(loss_tot.item()),
        "sp_bce": float(loss_sp.item()),
        "punct_bce": float(loss_punct.item()),
    }


@torch.inference_mode()
def evaluate_stage_a(
    refiner: FrenchProsodyRefiner,
    items: list[dict],
    phone_mask: torch.Tensor,
    sp_threshold: float = 0.50,
    batch_size: int = 64,
) -> dict[str, float]:
    refiner.eval()
    tot_loss = 0.0
    n_batches = 0
    phone_abs_err = 0.0
    phone_cnt = 0
    sp_tp = sp_fp = sp_fn = 0
    pu_tp = pu_fp = pu_fn = 0
    all_sp_probs = []
    all_sp_gt = []

    for s in range(0, len(items), batch_size):
        b = items[s : s + batch_size]
        tokens, durations, phrasal, enc_h, host_dur, mask, speakers = _batch_duration(b, phone_mask)
        pred = refiner.refine_durations(enc_h, host_dur, phrasal, mask, speakers)
        loss, _ = _duration_loss(pred, host_dur, tokens, durations, mask, phone_mask)
        tot_loss += float(loss.item())
        n_batches += 1

        is_phone = phone_mask[tokens] & mask
        is_edge = ((tokens == _BOS_ID) | (tokens == _EOS_ID)) & mask
        is_sp = (tokens == _SP_ID) & mask
        is_punct = (~is_phone) & (~is_edge) & (~is_sp) & mask

        pred_f = torch.round(torch.expm1(pred[..., 0].clamp(0.0, math.log1p(400.0)))).clamp_min(1.0)
        phone_abs_err += float((pred_f - durations.float()).abs()[is_phone].sum().item())
        phone_cnt += int(is_phone.sum().item())

        probs = torch.sigmoid(pred[..., 1])
        sp_p = probs[is_sp]
        sp_g = durations[is_sp] > 0
        all_sp_probs.append(sp_p)
        all_sp_gt.append(sp_g)
        sp_pred = sp_p > sp_threshold
        sp_tp += int((sp_pred & sp_g).sum().item())
        sp_fp += int((sp_pred & ~sp_g).sum().item())
        sp_fn += int((~sp_pred & sp_g).sum().item())

        pu_p = probs[is_punct] > 0.50
        pu_g = durations[is_punct] > 0
        pu_tp += int((pu_p & pu_g).sum().item())
        pu_fp += int((pu_p & ~pu_g).sum().item())
        pu_fn += int((~pu_p & pu_g).sum().item())

    sp_probs_cat = torch.cat(all_sp_probs) if all_sp_probs else torch.empty(0)
    sp_gt_cat = torch.cat(all_sp_gt) if all_sp_gt else torch.empty(0, dtype=torch.bool)
    best_thr, best_f05 = 0.50, -1.0
    for thr_i in range(50, 96, 5):
        thr = thr_i / 100.0
        pr = sp_probs_cat > thr
        tp = int((pr & sp_gt_cat).sum().item())
        fp = int((pr & ~sp_gt_cat).sum().item())
        fn = int((~pr & sp_gt_cat).sum().item())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-8)
        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr

    sp_prec = sp_tp / max(sp_tp + sp_fp, 1)
    sp_rec = sp_tp / max(sp_tp + sp_fn, 1)
    pu_prec = pu_tp / max(pu_tp + pu_fp, 1)
    pu_rec = pu_tp / max(pu_tp + pu_fn, 1)
    return {
        "loss": tot_loss / max(n_batches, 1),
        "phone_mae_ms": (phone_abs_err / max(phone_cnt, 1)) * (256.0 / 24.0),
        "sp_tp": sp_tp,
        "sp_fp": sp_fp,
        "sp_fn": sp_fn,
        "sp_prec": sp_prec,
        "sp_rec": sp_rec,
        "pu_prec": pu_prec,
        "pu_rec": pu_rec,
        "best_sp_thr": best_thr,
        "best_sp_f05": best_f05,
    }


def _batch_prosody(items: list[dict]):
    bsz = len(items)
    n_max = max(len(x["tokens"]) for x in items)
    t_max = max(x["prosody_hidden"].shape[0] for x in items)
    c = items[0]["prosody_hidden"].shape[-1]

    durations = torch.zeros((bsz, n_max), dtype=torch.long)
    phrasal = torch.zeros((bsz, n_max, PHRASAL_DIM), dtype=torch.float32)
    pro_hidden = torch.zeros((bsz, t_max, c), dtype=torch.float32)
    host_pro = torch.zeros((bsz, t_max, 3), dtype=torch.float32)
    tgt_pro = torch.zeros((bsz, t_max, 3), dtype=torch.float32)
    trusted = torch.zeros((bsz, t_max), dtype=torch.float32)
    frame_lengths = torch.zeros((bsz,), dtype=torch.long)
    frame_mask = torch.zeros((bsz, t_max), dtype=torch.bool)
    speakers = torch.tensor([x["speaker"] for x in items], dtype=torch.long)
    is_q = torch.tensor([1.0 if x["is_question"] else 0.0 for x in items], dtype=torch.float32)

    for i, x in enumerate(items):
        n = len(x["tokens"])
        t = x["prosody_hidden"].shape[0]
        durations[i, :n] = x["durations"].long()
        phrasal[i, :n] = x["phrasal"].float()
        pro_hidden[i, :t] = x["prosody_hidden"].float()
        host_pro[i, :t] = x["host_prosody"].float()
        tgt_pro[i, :t] = x["target_prosody"].float()
        trusted[i, :t] = x["pitch_trusted"].float()
        frame_lengths[i] = t
        frame_mask[i, :t] = True

    exp_phrasal, frame_pos = FrenchProsodyRefiner.expand_token_features(phrasal, durations, frame_lengths)
    return pro_hidden, host_pro, tgt_pro, trusted, exp_phrasal, frame_pos, frame_mask, speakers, is_q


def _prosody_loss(
    pred: torch.Tensor,
    host: torch.Tensor,
    tgt: torch.Tensor,
    trusted: torch.Tensor,
    frame_mask: torch.Tensor,
    is_q: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    fm = frame_mask.float()
    rel = (trusted > 0.5) & frame_mask
    rel_f = rel.float()

    # 1. Pitch L1 (trusted voiced frames weighted 1.0, other frames 0.1).
    p_err = F.smooth_l1_loss(pred[..., 0], tgt[..., 0], beta=0.05, reduction="none")
    p_w = torch.where(rel, torch.ones_like(fm), torch.full_like(fm, 0.1)) * fm
    loss_p = masked_mean(p_err, p_w)

    # 2. Local F0 contour shape loss (slopes, curvature, local accent contrast).
    loss_shape = pitch_shape_loss(pred[..., 0], tgt[..., 0], rel_f, energy=tgt[..., 2])

    # 3. Utterance F0 spread (std) matching + terminal inflection matching (last ~25 voiced frames).
    std_errs = []
    term_errs = []
    term_ws = []
    e_std_errs = []
    bsz = pred.shape[0]
    for i in range(bsz):
        idx = rel[i].nonzero(as_tuple=False).squeeze(1)
        if idx.numel() >= 12:
            pv = pred[i, idx, 0]
            tv = tgt[i, idx, 0]
            p_mean = pv.mean()
            t_mean = tv.mean()
            p_std = (pv - p_mean).square().mean().add(1e-6).sqrt()
            t_std = (tv - t_mean).square().mean().add(1e-6).sqrt()
            std_errs.append(F.smooth_l1_loss(p_std, t_std, beta=0.03))

            k_tail = min(25, max(8, idx.numel() // 4))
            p_tail = pv[-k_tail:].mean() - p_mean
            t_tail = tv[-k_tail:].mean() - t_mean
            term_errs.append(F.smooth_l1_loss(p_tail, t_tail, beta=0.04))
            term_ws.append(2.5 if float(is_q[i].item()) > 0.5 else 1.0)

            pe = pred[i, idx, 2]
            te = tgt[i, idx, 2]
            pe_std = (pe - pe.mean()).square().mean().add(1e-6).sqrt()
            te_std = (te - te.mean()).square().mean().add(1e-6).sqrt()
            e_std_errs.append(F.smooth_l1_loss(pe_std, te_std, beta=0.05))

    zero = pred.new_zeros(())
    loss_std = torch.stack(std_errs).mean() if std_errs else zero
    if term_errs:
        tw = torch.tensor(term_ws, device=pred.device, dtype=pred.dtype)
        loss_term = (torch.stack(term_errs) * tw).sum() / tw.sum()
    else:
        loss_term = zero
    loss_estd = torch.stack(e_std_errs).mean() if e_std_errs else zero

    # 4. Voicing BCE & Energy L1.
    loss_v = masked_mean(
        F.binary_cross_entropy_with_logits(pred[..., 1], tgt[..., 1], reduction="none"),
        fm,
    )
    loss_e = masked_mean(
        F.smooth_l1_loss(pred[..., 2], tgt[..., 2], beta=0.1, reduction="none"),
        fm,
    )

    # 5. Residual anchor to frozen host manifold (prevents acoustic_stack OOD artifacts).
    delta = pred - host
    loss_anchor = (
        0.06 * masked_mean(delta[..., 0].square(), fm)
        + 0.04 * masked_mean(delta[..., 1].square(), fm)
        + 0.10 * masked_mean(delta[..., 2].square(), fm)
    )

    total = (
        1.5 * loss_p
        + 1.2 * loss_shape
        + 0.8 * loss_std
        + 1.5 * loss_term
        + 0.3 * loss_v
        + 0.4 * loss_e
        + 0.2 * loss_estd
        + loss_anchor
    )
    return total, {
        "p_l1": float(loss_p.item()),
        "shape": float(loss_shape.item()),
        "p_std": float(loss_std.item()),
        "term": float(loss_term.item()),
        "v_bce": float(loss_v.item()),
        "e_l1": float(loss_e.item()),
    }


@torch.inference_mode()
def evaluate_stage_b(refiner: FrenchProsodyRefiner, items: list[dict], batch_size: int = 32) -> dict[str, float]:
    refiner.eval()
    tot_loss = 0.0
    n_b = 0
    f0_abs_st = 0.0
    f0_cnt = 0
    corrs = []
    q_term_abs_st = []
    std_ratios = []

    for s in range(0, len(items), batch_size):
        b = items[s : s + batch_size]
        pro_h, host_p, tgt_p, trusted, exp_phr, fpos, fmask, speakers, is_q = _batch_prosody(b)
        pred = refiner.refine_prosody(pro_h, host_p, exp_phr, fpos, fmask, speakers)
        loss, _ = _prosody_loss(pred, host_p, tgt_p, trusted, fmask, is_q)
        tot_loss += float(loss.item())
        n_b += 1

        rel = (trusted > 0.5) & fmask
        st_scale = 12.0 / math.log(2.0)
        f0_abs_st += float((pred[..., 0] - tgt_p[..., 0]).abs()[rel].sum().item()) * st_scale
        f0_cnt += int(rel.sum().item())

        for i in range(len(b)):
            idx = rel[i].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() >= 12:
                pv = pred[i, idx, 0]
                tv = tgt_p[i, idx, 0]
                pc = pv - pv.mean()
                tc = tv - tv.mean()
                denom = pc.norm() * tc.norm()
                if float(denom.item()) > 1e-6:
                    corrs.append(float((pc * tc).sum().item() / denom.item()))
                t_std = float(tc.std().item())
                if t_std > 1e-4:
                    std_ratios.append(float(pc.std().item()) / t_std)
                if float(is_q[i].item()) > 0.5:
                    k_tail = min(25, max(8, idx.numel() // 4))
                    q_term_abs_st.append(
                        abs(float((pv[-k_tail:].mean() - pv.mean()) - (tv[-k_tail:].mean() - tv.mean()))) * st_scale
                    )

    return {
        "loss": tot_loss / max(n_b, 1),
        "f0_mae_st": f0_abs_st / max(f0_cnt, 1),
        "f0_corr": sum(corrs) / max(len(corrs), 1),
        "f0_std_ratio": sum(std_ratios) / max(len(std_ratios), 1),
        "q_term_mae_st": sum(q_term_abs_st) / max(len(q_term_abs_st), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt")
    ap.add_argument("--train-manifest", default=".local-runs/colab/fr4k/annotations_e01/train.jsonl")
    ap.add_argument("--val-manifest", default=".local-runs/colab/fr4k/annotations_e01/validation.jsonl")
    ap.add_argument("--acoustic-root", default=".local-runs/colab/fr4k/features")
    ap.add_argument("--p4-dev-report", default=".local-runs/p4_ablation/p4_final/dev_screen/p4_dev_report.json")
    ap.add_argument("--out-dir", default=".local-runs/p7_refiner")
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--epochs-a", type=int, default=18)
    ap.add_argument("--epochs-b", type=int, default=10)
    ap.add_argument("--batch-a", type=int, default=64)
    ap.add_argument("--batch-b", type=int, default=32)
    ap.add_argument("--lr-a", type=float, default=2.0e-3)
    ap.add_argument("--lr-b", type=float, default=1.2e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t_start = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    host_path = Path(args.host)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    ckpt = torch.load(host_path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = TinyVoca(cfg).eval()
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)

    host_sha = _sha256_file(host_path)
    print(f"Loaded frozen host {host_path} (sha256={host_sha[:12]}, params={sum(p.numel() for p in model.parameters()):,})")

    train_items = build_or_load_cache(
        model, cfg, Path(args.train_manifest), Path(args.acoustic_root), cache_dir / "train_cache.pt"
    )
    val_all = build_or_load_cache(
        model, cfg, Path(args.val_manifest), Path(args.acoustic_root), cache_dir / "val_cache.pt"
    )

    p4_dev_ids = _load_p4_dev_ids(Path(args.p4_dev_report))
    val_cal = [x for x in val_all if x["id"] not in p4_dev_ids]
    val_dev = [x for x in val_all if x["id"] in p4_dev_ids]
    print(f"Split check: train={len(train_items)} | val_cal (no P4-dev)={len(val_cal)} | P4-dev held out={len(val_dev)}")

    phone_mask = model.phone_tokens.clone()
    refiner = FrenchProsodyRefiner(
        host_channels=cfg.channels,
        phrasal_dim=PHRASAL_DIM,
        hidden_dim=args.hidden_dim,
        speakers=len(cfg.speakers),
        dropout=0.12,
    )
    n_ref = sum(p.numel() for p in refiner.parameters())
    n_host = sum(p.numel() for p in model.parameters())
    print(f"FrenchProsodyRefiner params={n_ref:,} | Total={n_host + n_ref:,}")

    # Baseline metrics at step 0 (exact identity).
    base_a = evaluate_stage_a(refiner, val_cal, phone_mask, sp_threshold=0.50)
    base_b = evaluate_stage_b(refiner, val_cal)
    print(
        f"[Step 0 / Host Identity on val_cal] "
        f"phone_mae={base_a['phone_mae_ms']:.2f}ms | <sp> TP={base_a['sp_tp']} FP={base_a['sp_fp']} FN={base_a['sp_fn']} "
        f"(P={base_a['sp_prec']:.1%}, R={base_a['sp_rec']:.1%}, best_thr={base_a['best_sp_thr']:.2f}) | "
        f"F0_MAE={base_b['f0_mae_st']:.3f}st r={base_b['f0_corr']:.4f} std_ratio={base_b['f0_std_ratio']:.3f} q_term={base_b['q_term_mae_st']:.3f}st"
    )

    # ---------------- Stage A: Rhythm & Pause ----------------
    dur_params = (
        list(refiner.spk_emb.parameters())
        + list(refiner.dur_in.parameters())
        + list(refiner.dur_conv1.parameters())
        + list(refiner.dur_conv2.parameters())
        + list(refiner.dur_gru.parameters())
        + list(refiner.dur_norm.parameters())
        + list(refiner.dur_head.parameters())
    )
    opt_a = torch.optim.AdamW(dur_params, lr=args.lr_a, weight_decay=1e-2)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=args.epochs_a, eta_min=1e-4)

    best_a_loss = float("inf")
    best_a_state = None
    best_a_metrics = None
    history_a = []

    print(f"\n=== Stage A: Rhythm & Pause Refiner ({args.epochs_a} epochs, batch={args.batch_a}) ===")
    for ep in range(1, args.epochs_a + 1):
        t_ep = time.time()
        refiner.train()
        order = list(range(len(train_items)))
        random.shuffle(order)
        tr_loss = 0.0
        n_b = 0
        for s in range(0, len(order), args.batch_a):
            b = [train_items[i] for i in order[s : s + args.batch_a]]
            tokens, durations, phrasal, enc_h, host_dur, mask, speakers = _batch_duration(b, phone_mask)
            pred = refiner.refine_durations(enc_h, host_dur, phrasal, mask, speakers)
            loss, _ = _duration_loss(pred, host_dur, tokens, durations, mask, phone_mask)
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dur_params, 1.0)
            opt_a.step()
            tr_loss += float(loss.item())
            n_b += 1
        sched_a.step()
        va = evaluate_stage_a(refiner, val_cal, phone_mask, sp_threshold=0.50)
        history_a.append({"epoch": ep, "train_loss": tr_loss / max(n_b, 1), **va})
        mark = ""
        if va["loss"] < best_a_loss:
            best_a_loss = va["loss"]
            best_a_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
            best_a_metrics = va
            mark = " *"
        print(
            f"  ep {ep:02d}/{args.epochs_a} ({time.time() - t_ep:.1f}s) | tr={tr_loss/n_b:.4f} va={va['loss']:.4f} | "
            f"phone_mae={va['phone_mae_ms']:.2f}ms | <sp>@0.50 TP={va['sp_tp']} FP={va['sp_fp']} (P={va['sp_prec']:.1%}, R={va['sp_rec']:.1%}) "
            f"best_thr={va['best_sp_thr']:.2f}{mark}"
        )

    assert best_a_state is not None and best_a_metrics is not None
    refiner.load_state_dict(best_a_state)
    best_sp_thr = float(best_a_metrics["best_sp_thr"])
    refiner.sp_pause_threshold.fill_(best_sp_thr)
    va_cal_thr = evaluate_stage_a(refiner, val_cal, phone_mask, sp_threshold=best_sp_thr)
    print(
        f"-> Best Stage A calibrated @ sp_thr={best_sp_thr:.2f}: "
        f"phone_mae={va_cal_thr['phone_mae_ms']:.2f}ms | <sp> TP={va_cal_thr['sp_tp']} FP={va_cal_thr['sp_fp']} FN={va_cal_thr['sp_fn']} "
        f"(P={va_cal_thr['sp_prec']:.1%}, R={va_cal_thr['sp_rec']:.1%}) | punct P={va_cal_thr['pu_prec']:.1%} R={va_cal_thr['pu_rec']:.1%}"
    )

    # Freeze Stage A (including spk_emb) so Stage B cannot disturb Rhythm/Pause!
    for p in dur_params:
        p.requires_grad_(False)

    # ---------------- Stage B: Expressive Intonation & Energy ----------------
    pro_params = (
        list(refiner.pro_in.parameters())
        + list(refiner.pro_conv1.parameters())
        + list(refiner.pro_conv2.parameters())
        + list(refiner.pro_conv3.parameters())
        + list(refiner.pro_gru.parameters())
        + list(refiner.pro_norm.parameters())
        + list(refiner.pro_head.parameters())
    )
    opt_b = torch.optim.AdamW(pro_params, lr=args.lr_b, weight_decay=1e-2)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs_b, eta_min=1e-4)

    best_b_loss = float("inf")
    best_b_state = None
    best_b_metrics = None
    history_b = []

    print(f"\n=== Stage B: Expressive Intonation & Energy Refiner ({args.epochs_b} epochs, batch={args.batch_b}) ===")
    # Bucket-shuffled batching by length for fast CPU throughput.
    len_order = sorted(range(len(train_items)), key=lambda i: train_items[i]["prosody_hidden"].shape[0])
    for ep in range(1, args.epochs_b + 1):
        t_ep = time.time()
        refiner.train()
        # Mild jitter on length sort so batches vary each epoch while staying tight on padding.
        noisy = [(train_items[i]["prosody_hidden"].shape[0] + random.randint(-40, 40), i) for i in range(len(train_items))]
        noisy.sort(key=lambda pair: pair[0])
        batches = [
            [train_items[idx] for _, idx in noisy[s : s + args.batch_b]]
            for s in range(0, len(noisy), args.batch_b)
        ]
        random.shuffle(batches)
        tr_loss = 0.0
        for b in batches:
            pro_h, host_p, tgt_p, trusted, exp_phr, fpos, fmask, speakers, is_q = _batch_prosody(b)
            pred = refiner.refine_prosody(pro_h, host_p, exp_phr, fpos, fmask, speakers)
            loss, _ = _prosody_loss(pred, host_p, tgt_p, trusted, fmask, is_q)
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pro_params, 1.0)
            opt_b.step()
            tr_loss += float(loss.item())
        sched_b.step()
        vb = evaluate_stage_b(refiner, val_cal)
        history_b.append({"epoch": ep, "train_loss": tr_loss / max(len(batches), 1), **vb})
        mark = ""
        if vb["loss"] < best_b_loss:
            best_b_loss = vb["loss"]
            best_b_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
            best_b_metrics = vb
            mark = " *"
        print(
            f"  ep {ep:02d}/{args.epochs_b} ({time.time() - t_ep:.1f}s) | tr={tr_loss/len(batches):.4f} va={vb['loss']:.4f} | "
            f"F0_MAE={vb['f0_mae_st']:.3f}st r={vb['f0_corr']:.4f} std_ratio={vb['f0_std_ratio']:.3f} q_term={vb['q_term_mae_st']:.3f}st{mark}"
        )

    assert best_b_state is not None and best_b_metrics is not None
    refiner.load_state_dict(best_b_state)

    out_ckpt = out_dir / "p7_refiner.pt"
    payload = {
        "refiner_state": {k: v.cpu() for k, v in refiner.state_dict().items()},
        "refiner_config": {
            "host_channels": cfg.channels,
            "phrasal_dim": PHRASAL_DIM,
            "hidden_dim": args.hidden_dim,
            "speakers": len(cfg.speakers),
            "dropout": 0.12,
        },
        "host_path": str(host_path),
        "host_sha256": host_sha,
        "host_params": n_host,
        "refiner_params": n_ref,
        "sp_pause_threshold": best_sp_thr,
        "host_best_sp_threshold": float(base_a["best_sp_thr"]),
        "val_cal_baseline": {"stage_a": base_a, "stage_b": base_b},
        "val_cal_best": {"stage_a": va_cal_thr, "stage_b": best_b_metrics},
        "history_a": history_a,
        "history_b": history_b,
        "train_seconds": time.time() - t_start,
        "seed": args.seed,
    }
    torch.save(payload, out_ckpt)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {k: v for k, v in payload.items() if k != "refiner_state"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved P7 refiner to {out_ckpt} in {time.time() - t_start:.1f}s total (0.00 u. GPU)!")


if __name__ == "__main__":
    main()
