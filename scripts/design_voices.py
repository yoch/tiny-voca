"""Generate reference voice clips using Qwen3-TTS VoiceDesign. Cloud only."""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
import torch

from tiny_voca.text import normalize_for_wer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", type=Path, default=Path("scripts/voices_fr.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1, help="Generate N seeds per voice; keep all and copy best to <VOICE>.wav")
    parser.add_argument("--asr-model", default="small", help="faster-whisper model for scoring candidates (empty to disable)")
    parser.add_argument("--asr-language", default="fr")
    args = parser.parse_args()
    from qwen_tts import Qwen3TTSModel

    args.output.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.voices.read_text(encoding="utf-8"))
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=dtype, attn_implementation="sdpa")

    asr = None
    if args.asr_model:
        from faster_whisper import WhisperModel
        from jiwer import wer
        from scipy.signal import resample_poly
        asr = WhisperModel(args.asr_model, device="cpu", compute_type="int8", cpu_threads=4)

    manifest = {}
    for voice, spec in specs.items():
        base_seed = int(spec.get("seed", 20261117))
        rows = []
        for k in range(args.candidates):
            seed = base_seed + k
            torch.manual_seed(seed)
            wavs, rate = model.generate_voice_design(
                text=spec["text"],
                language=spec.get("language", "French"),
                instruct=spec["instruct"],
            )
            wave = np.asarray(wavs[0], dtype=np.float32)
            peak = float(np.max(np.abs(wave))) if len(wave) else 0.0
            if peak > 0:
                wave = wave * (0.95 / peak)
            suffix = "" if args.candidates == 1 else f"_cand{k}"
            out_path = args.output / f"{voice}{suffix}.wav"
            sf.write(out_path, wave, rate, subtype="PCM_16")
            duration = round(len(wave) / rate, 3)
            transcript, error = "", None
            if asr is not None:
                factor = int(np.gcd(rate, 16000))
                audio_16k = resample_poly(wave, 16000 // factor, rate // factor)
                segments, _ = asr.transcribe(audio_16k, language=args.asr_language, beam_size=5, condition_on_previous_text=False)
                transcript = " ".join(s.text.strip() for s in segments).strip()
                error = round(float(wer(normalize_for_wer(spec["text"], args.asr_language), normalize_for_wer(transcript, args.asr_language) or "<empty>")), 4)
            info = {"voice": voice, "candidate": k, "seed": seed, "path": str(out_path), "rate": rate,
                    "seconds": duration, "wer": error, "transcript": transcript}
            print(json.dumps(info, ensure_ascii=False), flush=True)
            rows.append(info)
        best = min(rows, key=lambda r: (r["wer"] if r["wer"] is not None else 0.0, abs(r["seconds"] - 8.5)))
        final_path = args.output / f"{voice}.wav"
        if Path(best["path"]) != final_path:
            shutil.copy2(best["path"], final_path)
        manifest[voice] = {**spec, "file": final_path.name, "selected_candidate": best["candidate"],
                           "selected_seed": best["seed"], "sample_rate": best["rate"],
                           "duration_seconds": best["seconds"], "wer": best["wer"], "transcript": best["transcript"]}
    (args.output / "voices.generated.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
