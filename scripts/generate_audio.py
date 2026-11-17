"""Generate a batch of multi-voice recordings with the teacher TTS. Cloud only.

Uses Qwen3-TTS 1.7B Base cloning the approved VoiceDesign references, batches
of eight (or smaller on OOM), multilingual Whisper screening at 15% normalized
WER, one retry, and peak normalisation to 0.95.

Resumable: every clip is recorded in a SQLite file; rerun the same command.
Output: <root>/audio/<id>.flac and <root>/manifest.jsonl (accepted clips, in the
JSONL form tiny_voca.prepare reads), <root>/review.jsonl (rejected clips).
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import time
import numpy as np
import soundfile as sf
import torch

from tiny_voca.text import PHONE_SET, normalize_for_wer, phonemize

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BATCH = 8


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], prev if x == y else 1 + min(prev, dp[j], dp[j - 1])
    return dp[-1]


def phoneme_error_rate(reference, hypothesis):
    try:
        ref = [s for s in phonemize(reference).symbols if s in PHONE_SET]
        hyp = [s for s in phonemize(hypothesis).symbols if s in PHONE_SET]
    except Exception:
        return 1.0
    if not ref:
        return 1.0
    return edit_distance(ref, hyp) / len(ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True, help="Directory with <VOICE>.wav files")
    parser.add_argument("--voices", type=Path, required=True, help="voices JSON mapping voice_id -> {text, ...}")
    parser.add_argument("--language", default="French", help="Teacher TTS language name (e.g. French, English)")
    parser.add_argument("--asr-model", default="small", help="faster-whisper model for screening")
    parser.add_argument("--asr-language", default="fr", help="Whisper ISO language code (e.g. fr, en)")
    parser.add_argument("--asr-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--max-wer", type=float, default=0.15)
    parser.add_argument("--max-per", type=float, default=0.05, help="Accept homophone spellings when phoneme error rate <= max-per (set <0 to disable)")
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--retry-batch-size", type=int, default=BATCH)
    parser.add_argument("--deadline-file", type=Path)
    parser.add_argument("--deadline-margin", type=float, default=600)
    parser.add_argument("--limit", type=int, help="Only the first N scripts (smoke test)")
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    from jiwer import wer
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly

    audio = args.root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.root / "state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER, payload TEXT)")
    db.commit()
    references = json.loads(args.voices.read_text(encoding="utf-8"))
    voice_ids = tuple(references.keys())
    rows = [json.loads(line) for line in args.scripts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: (r["character_count"], r["id"]))[:args.limit]
    jobs = [{"id": f"{row['id']}_{voice}", "voice": voice, "row": row} for row in rows for voice in voice_ids]
    started = time.time()

    def out_of_time():
        if (args.root / "STOP").exists():
            return True
        if not args.deadline_file:
            return False
        try:
            return time.time() >= json.loads(args.deadline_file.read_text())["deadline"] - args.deadline_margin
        except (OSError, ValueError, KeyError):
            return False

    def report(stage):
        counts = dict(db.execute("SELECT status, count(*) FROM clips GROUP BY status").fetchall())
        hours = sum(json.loads(p)["duration_seconds"] for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted'")) / 3600
        value = {"stage": stage, "target_clips": len(jobs), "counts": counts, "accepted_audio_hours": round(hours, 3),
                 "elapsed_seconds_this_run": round(time.time() - started), "updated": time.time()}
        temporary = args.root / "progress.json.tmp"
        temporary.write_text(json.dumps(value, indent=2))
        os.replace(temporary, args.root / "progress.json")
        print(json.dumps(value), flush=True)

    torch.set_num_threads(6)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa")
    prompts = {
        v: model.create_voice_clone_prompt(
            ref_audio=str(args.references / f"{v}.wav"),
            ref_text=references[v]["text"],
            x_vector_only_mode=False,
        )[0]
        for v in voice_ids
    }
    compute_type = "float16" if args.asr_device == "cuda" else "int8"
    asr = WhisperModel(args.asr_model, device=args.asr_device, compute_type=compute_type, cpu_threads=6)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []

    def analyze(batch, waves, rate, attempt, seed, batch_id, seconds):
        records = []
        for job, wave in zip(batch, waves):
            row, w = job["row"], np.asarray(wave)
            duration, peak, reasons = len(w) / rate, float(np.max(np.abs(w))) if len(w) else 0.0, []
            if not np.isfinite(w).all() or peak < 1e-5:
                reasons.append("invalid_or_silent_audio")
            if duration < 0.5 or duration > 35:
                reasons.append("duration")
            if duration > 0 and not 0.5 <= row["word_count"] / duration <= 7:
                reasons.append("speaking_rate")
            if len(w) and float(np.mean(np.abs(w) < 1e-4)) > 0.8:
                reasons.append("excessive_silence")
            transcript, error, per = "", 1.0, 1.0
            if not reasons:
                factor = int(np.gcd(rate, 16000))
                audio_16k = resample_poly(w.astype("float32"), 16000 // factor, rate // factor)
                segments, _ = asr.transcribe(audio_16k, language=args.asr_language, beam_size=3, condition_on_previous_text=False)
                transcript = " ".join(s.text.strip() for s in segments)
                truth = normalize_for_wer(row["text"], args.asr_language)
                hyp = normalize_for_wer(transcript, args.asr_language) or "<empty>"
                error = wer(truth, hyp)
                per = phoneme_error_rate(row["text"], transcript) if transcript.strip() else 1.0
                if error > args.max_wer and not (args.max_per >= 0 and per <= args.max_per):
                    reasons.append("asr_mismatch")
            if np.isfinite(w).all() and peak > 0:
                sf.write(audio / (job["id"] + ".flac"), w * (0.95 / peak), rate, subtype="PCM_16")
            payload = {
                "id": job["id"], "text_id": row["id"], "voice_id": job["voice"], "text": row["text"], "split": row["split"],
                "category": row["category"], "origin": row["origin"], "source_id": row["source_id"],
                "source_license": row["source_license"], "source_url": row["source_url"], "teacher_model": MODEL,
                "reference_audio": job["voice"] + ".wav", "audio": f"audio/{job['id']}.flac", "duration_seconds": round(duration, 3),
                "generation_seed": seed, "generation_batch": batch_id, "attempt": attempt, "generation_seconds": seconds,
                "asr_transcript": transcript, "normalized_wer": round(float(error), 4), "phoneme_error_rate": round(float(per), 4),
                "review_reasons": reasons,
            }
            records.append((job["id"], "review" if reasons else "accepted", attempt, json.dumps(payload, ensure_ascii=False)))
        return records

    def collect(future):
        records = future.result()
        db.executemany(
            "INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "status=excluded.status, attempts=excluded.attempts, payload=excluded.payload",
            records,
        )
        db.commit()

    def generate(batch, attempt, batch_id):
        seed = args.seed + batch_id + attempt * 100000
        torch.manual_seed(seed)
        start = time.time()
        try:
            waves, rate = model.generate_voice_clone(
                text=[j["row"]["text"] for j in batch],
                language=[args.language] * len(batch),
                voice_clone_prompt=[prompts[j["voice"]] for j in batch],
                max_new_tokens=600,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            generate(batch[:middle], attempt, batch_id)
            generate(batch[middle:], attempt, batch_id + 50000)
            return
        futures.append(pool.submit(analyze, batch, waves, rate, attempt, seed, batch_id, round(time.time() - start, 3)))
        if len(futures) >= 2:
            collect(futures.pop(0))

    stopped = False
    for attempt in (1, 2):
        existing = {uid: (status, n) for uid, status, n in db.execute("SELECT id, status, attempts FROM clips")}
        pending = [
            j for j in jobs
            if (attempt == 1 and j["id"] not in existing)
            or (attempt == 2 and j["id"] in existing and existing[j["id"]][0] == "review" and existing[j["id"]][1] < 2)
        ]
        size = args.batch_size if attempt == 1 else args.retry_batch_size
        for start in range(0, len(pending), size):
            if out_of_time():
                stopped = True
                break
            generate(pending[start:start + size], attempt, start // size if attempt == 1 else start // size + 20000)
            if (start // size) % 16 == 15:
                report("generating")
        while futures:
            collect(futures.pop(0))
        if stopped:
            break
    pool.shutdown(wait=True)
    accepted = [json.loads(p) for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted' ORDER BY id")]
    (args.root / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")
    (args.root / "review.jsonl").write_text("".join(p + "\n" for (p,) in db.execute("SELECT payload FROM clips WHERE status!='accepted' ORDER BY id")), encoding="utf-8")
    report("paused" if stopped else "complete")


if __name__ == "__main__":
    main()
