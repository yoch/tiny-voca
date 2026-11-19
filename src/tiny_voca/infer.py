import argparse
import os
from pathlib import Path
import re
import torch
from .config import CHECKPOINT_FORMAT, Config
from .model import TinyVoca, UtteranceTooLong
from .prosody_fr import load_prosody_refiner, prepare_french_structure, prepare_french_tts_text, synthesize_with_refiner
from .text import FRONTEND_VARIABLE, FRONTENDS, NothingToPronounce, frontend_fingerprint, phonemize, set_frontend

# The loudest sample of the audio that ``speak_text`` and the command return.
PEAK = 0.99
# Said when no sentence of a text could be read. The browser demo says the same.
ENGLISH_ONLY = "tiny-voca only reads French for now"
_SPACE = re.compile("[\\s" + chr(0xFEFF) + "]+")    # JavaScript's \s, which the demo uses, takes U+FEFF too
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["'”’)\]»]))\s+(?=["'“‘(\[«]?[A-ZÀ-ÖØ-ÞŒŸ0-9])""")
_NOT_AN_END = re.compile(
    r"(?:^|[\s\"“(\[«])(?:m|mm|mme|mlle|mr|mrs|ms|dr|prof|mgr|sr|jr|st|ste|vs|etc|no|inc|ltd|co|ex|cf|av|ap|bd|fig|vol|chap|art|env|e\.g|i\.e|a\.m|p\.m"
    r"|[A-ZÀ-ÖØ-ÞŒŸ])\.$",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile("[\\s" + chr(0xFEFF) + "]+$")
_CLOSING = "\"'”’)]»"                               # these may stand after the last mark: She said "go!"


def load_model(path, device="cpu", check_frontend=True):
    """``check_frontend=False`` lets a checkpoint speak tokens of the other
    front end (same inventory, slightly different conventions): for listening
    tests of the new phonemizer before the model is retrained on it."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format: byte-token checkpoints (format 2-3) cannot be used with this code")
    if check_frontend and state["frontend"]["fingerprint"] != frontend_fingerprint():
        raise ValueError("checkpoint was trained with a different phoneme inventory/front end")
    model = TinyVoca(Config.from_dict(state["config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def match_frontend(path):
    """Select the front end the checkpoint was trained with."""
    wanted = torch.load(path, map_location="cpu", weights_only=True)["frontend"]["fingerprint"]
    for name in FRONTENDS:
        set_frontend(name)
        try:
            if frontend_fingerprint() == wanted:
                return name
        except ImportError:
            continue
    set_frontend(FRONTENDS[0])
    return None


def sentences(text):
    """The sentences of ``text``, without a break after a title or an initial
    or inside a number. The browser demo splits the same way
    (``web/public/worker.js``); keep the two alike."""
    parts, current = [], ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def finished(text):
    """``text`` with a full stop added when it ends without a mark. Before a
    final mark the voice lengthens the last syllable and lets the sound fall;
    without one the audio breaks off. A face or another sign after the last
    word is no mark ("see you :)"), a comma or a colon is one. The browser
    demo does the same (``web/public/worker.js``); keep the two alike."""
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def shorter(text):
    """``text`` in smaller pieces: cut after commas and the like if it has
    any, else in two between words. ``[]`` for a single word."""
    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


def limit_peak(wave, peak=PEAK):
    """``wave`` scaled down as a whole when its loudest sample is above
    ``peak``; quieter audio is returned as it is. The model sometimes peaks
    above 1.0, and that clips when the audio is saved as 16-bit samples."""
    loudest = float(wave.abs().max()) if wave.numel() else 0.0
    return wave * (peak / loudest) if loudest > peak else wave


def _voice(model, phones, voice, speed, pitch_shift, seed, pitch_range, prosody_refiner=None, text="", sp_pause_threshold=None):
    """Phonemes -> (float waveform tensor, details): one call of the model."""
    device = next(model.parameters()).device
    tokens = torch.tensor([phones.ids], device=device)
    stress = torch.tensor([phones.stress], device=device)
    speaker = torch.tensor([model.config.speakers.index(voice)], device=device)
    lengths_in = torch.tensor([len(phones)], device=device)
    if prosody_refiner is not None or sp_pause_threshold is not None:
        audio, lengths, details = synthesize_with_refiner(
            model, prosody_refiner, tokens, stress, lengths_in, speaker,
            texts=[text], speed=speed, pitch_shift=pitch_shift, seed=seed, pitch_range=pitch_range,
            sp_pause_threshold=sp_pause_threshold,
        )
    else:
        audio, lengths, details = model.synthesize(tokens, stress, lengths_in, speaker, speed, pitch_shift, seed, pitch_range)
    if not torch.isfinite(audio).all():
        raise RuntimeError("model produced non-finite audio")
    return audio[0, :int(lengths[0])].cpu(), details


def speak(model, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, prosody_refiner=None,
          sp_pause_threshold=None, prepare_text=True):
    """One utterance: text -> (float waveform tensor, phonemized text, details).
    Same front end as training. The audio is the model's output as it is and
    can peak above 1.0 (see ``limit_peak``). ``speak_text`` reads text of any
    length."""
    clean = prepare_french_tts_text(text) if prepare_text else text
    phones = phonemize(clean)
    if len(phones) > model.config.max_tokens:
        raise ValueError("text is too long for one utterance; speak_text splits it into sentences")
    if prosody_refiner is None and sp_pause_threshold is None:
        wave, details = _voice(model, phones, voice, speed, pitch_shift, seed, pitch_range)
    else:
        wave, details = _voice(
            model, phones, voice, speed, pitch_shift, seed, pitch_range,
            prosody_refiner=prosody_refiner, text=clean, sp_pause_threshold=sp_pause_threshold,
        )
    return wave, phones, details


def speak_text(model, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, peak=PEAK, prosody_refiner=None,
               sp_pause_threshold=None, prepare_text=True):
    """Text of any length -> (float waveform tensor, phonemized utterances).

    The text is read sentence by sentence and the audio is joined. A text that
    ends without a mark gets a full stop (``finished``). A sentence
    too long for one call of the model (``max_tokens`` phonemes or
    ``max_frames`` frames) is cut after commas, then between words. A sentence
    with nothing to pronounce (another script, a face, signs only) is skipped;
    when no sentence could be read, ``NothingToPronounce`` says so. The audio
    is scaled down as a whole when it peaks above ``peak``, so that it can be
    saved as 16-bit samples without clipping; ``peak=None`` returns the
    model's output as it is."""
    if prepare_text:
        text = prepare_french_structure(text)
    limit = model.config.max_tokens - 12
    waves, spoken = [], []

    def read(piece):
        clean_piece = prepare_french_tts_text(piece) if prepare_text else piece
        try:
            phones = phonemize(clean_piece)
        except NothingToPronounce:
            return
        if len(phones) <= limit:
            try:
                if prosody_refiner is None and sp_pause_threshold is None:
                    wave, _ = _voice(model, phones, voice, speed, pitch_shift, seed + len(waves), pitch_range)
                else:
                    wave, _ = _voice(
                        model, phones, voice, speed, pitch_shift, seed + len(waves), pitch_range,
                        prosody_refiner=prosody_refiner, text=clean_piece, sp_pause_threshold=sp_pause_threshold,
                    )
            except UtteranceTooLong:
                pass
            else:
                waves.append(wave)
                spoken.append(phones)
                return
        pieces = shorter(clean_piece)
        if not pieces:
            raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
        for part in pieces:
            read(part)

    for sentence in sentences(finished(text)):
        read(sentence)
    if not waves:
        raise NothingToPronounce(f"nothing in the text could be read: {ENGLISH_ONLY}")
    wave = torch.cat(waves)
    return (wave if peak is None else limit_peak(wave, peak)), spoken


def main():
    parser = argparse.ArgumentParser(description="Generate speech from a trained tiny-voca inference checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prosody-refiner", type=Path, default=None,
                        help="Optional P7 FrenchProsodyRefiner checkpoint (default: none)")
    parser.add_argument("--sp-pause-threshold", type=float, default=None,
                        help="Optional word-space <sp> pause threshold (e.g. 0.95 or 1.0 to suppress mid-phrase pauses)")
    parser.add_argument("--no-prepare-text", action="store_true",
                        help="Disable inference-time French text cleanup (units, fractions, phone numbers, bullets)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text of any length; it is read sentence by sentence")
    source.add_argument("--text-file", type=Path, help="Read the text from a file (UTF-8)")
    parser.add_argument("--voice", default=None, help="Speaker voice ID (default: model's first voice)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--pitch-shift", type=float, default=0, help="Added to log F0 (0.058 is about one semitone)")
    parser.add_argument("--pitch-range", type=float, default=1, help="Scale pitch movement around the utterance mean (1 = as predicted)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frontend", choices=FRONTENDS, help="Phonemizer (default: TINY_VOCA_FRONTEND, else the one the checkpoint was trained with)")
    parser.add_argument("--allow-frontend-mismatch", action="store_true",
                        help="Speak although the checkpoint was trained on the other front end's tokens")
    args = parser.parse_args()
    import soundfile as sf
    if args.frontend:
        set_frontend(args.frontend)
    elif FRONTEND_VARIABLE not in os.environ:
        match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, args.device, check_frontend=not args.allow_frontend_mismatch)
    refiner = load_prosody_refiner(args.prosody_refiner, args.device) if args.prosody_refiner else None
    voice = args.voice or model.config.speakers[0]
    if voice not in model.config.speakers:
        parser.error(f"unknown voice {voice!r}; choose from {model.config.speakers}")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    try:
        raw, spoken = speak_text(
            model, text, voice, args.speed, args.pitch_shift,
            pitch_range=args.pitch_range, peak=None, prosody_refiner=refiner,
            sp_pause_threshold=args.sp_pause_threshold, prepare_text=not args.no_prepare_text,
        )
    except NothingToPronounce:
        parser.exit(1, f"{ENGLISH_ONLY}. Nothing in the text could be read, so no file was written.\n")
    wave = limit_peak(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Float samples, never above PEAK: a later conversion to 16-bit does not clip.
    sf.write(args.output, wave.numpy(), model.config.sample_rate, subtype="FLOAT")
    for phones in spoken:
        print(" ".join(phones.symbols))
    loudest = float(raw.abs().max())
    scaled = f", scaled down to {PEAK}" if wave is not raw else ""
    print(f"Saved {len(wave)/model.config.sample_rate:.2f}s in {len(spoken)} utterance{'s' if len(spoken) != 1 else ''}; peak={loudest:.4f}{scaled}")


if __name__ == "__main__":
    main()
