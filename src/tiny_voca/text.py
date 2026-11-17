"""French front end: text -> phoneme tokens with a stress feature.

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. Uses eSpeak NG (``fr``) either via the
``espeak-ng`` executable or, if unavailable on PATH, via ``libespeak-ng.so.1``.

* A token is one eSpeak phoneme exactly as eSpeak segments it (``--sep=_``),
  for example ``ʁ``, ``ɑ̃``, ``ɔ̃`` or ``ɲ``. Trailing clitic ``-`` markers
  emitted by French eSpeak (e.g. ``l_e-``, ``d_ə-``) are stripped.
* Lexical/phrase stress is a parallel feature (0 none, 1 primary, 2 secondary),
  not a token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.
"""

import ctypes
import ctypes.util
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import unicodedata

FRONTEND_VERSION = 4
ESPEAK_VOICE = "fr"
SEPARATOR = "_"
FRONTENDS = ("espeak",)
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
CONSONANTS = (
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ",
    "m", "n", "ɲ", "ŋ", "l", "ʁ", "j", "w", "tʃ", "dʒ",
)
VOWELS = (
    "i", "y", "u", "e", "ø", "o", "ə", "ɛ", "œ", "ɔ", "a",
    "ɛ̃", "œ̃", "ɔ̃", "ɑ̃",
)
PHONES = CONSONANTS + VOWELS
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(("p", "t", "k", "f", "s", "ʃ", "tʃ"))
VOWEL_SET = frozenset(VOWELS)
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

# eSpeak phonemes outside the French inventory (foreign loanwords, rare
# allophones) map to the closest inventory phone(s); empty tuple drops silent
# foreign segments such as English ``h``. Anything else becomes <unk>.
FALLBACK = {
    "r": ("ʁ",), "ɹ": ("ʁ",), "ɾ": ("ʁ",), "ɻ": ("ʁ",), "χ": ("ʁ",),
    "ɥ": ("y",), "ʎ": ("j",), "ɫ": ("l",), "ɬ": ("l",),
    "ts": ("t", "s"), "dz": ("d", "z"), "x": ("k",), "ç": ("ʃ",),
    "θ": ("s",), "ð": ("z",), "β": ("v",), "ɣ": ("ɡ",), "ʔ": (), "h": (),
    "ɑ": ("a",), "ɑː": ("a",), "æ": ("a",), "ɐ": ("a",), "ɒ": ("ɔ",),
    "ɪ": ("i",), "ᵻ": ("i",), "ʊ": ("u",), "ʌ": ("œ",), "ɜ": ("œ",), "ɜː": ("œ",), "ɚ": ("œ", "ʁ"),
    "eɪ": ("e", "j"), "aɪ": ("a", "j"), "ɔɪ": ("ɔ", "j"),
    "əʊ": ("o",), "oʊ": ("o",), "aʊ": ("a", "w"),
    "iə": ("i", "j", "ə"), "eə": ("ɛ", "ʁ"),
    "ɛɹ": ("ɛ", "ʁ"), "ɪɹ": ("i", "ʁ"), "ʊɹ": ("u", "ʁ"),
    "ɑːɹ": ("a", "ʁ"), "ɔːɹ": ("ɔ", "ʁ"), "oːɹ": ("ɔ", "ʁ"),
    "aɪə": ("a", "j", "ə"), "aɪɚ": ("a", "j", "œ", "ʁ"),
    "əl": ("ə", "l"), "l̩": ("ə", "l"), "m̩": ("ə", "m"), "n̩": ("ə", "n"),
}

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")",
})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {
    "m", "mme", "mlle", "mr", "mrs", "ms", "dr", "prof", "st", "ste",
    "etc", "ex", "cf", "av", "bd", "fig", "vol", "env",
}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")
_MONSIEUR_ABBR = re.compile(r"(?<![A-ZÀ-ÖØ-ÞŒ]\.\s)\bM\.\s+(?=[A-ZÀ-ÖØ-ÞŒ])")
_ALPHA_WORD_BEFORE_DOT = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿŒœ]+)$")

_ESPEAK_LOCK = threading.Lock()
_ESPEAK_LIB = None


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # eSpeak phonemes that had no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend():
    """Name of the program that produces the phonemes (``"espeak"``)."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def frontend_identity():
    return {
        "frontend_version": FRONTEND_VERSION,
        "voice": ESPEAK_VOICE,
        "separator": SEPARATOR,
        "symbols": list(SYMBOLS),
        "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())},
    }


def frontend_runtime():
    """What an annotation run records about the exact program behind its tokens."""
    return {"espeak_version": espeak_version()}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    text = _MONSIEUR_ABBR.sub("Monsieur ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    contractions, title abbreviations and initialisms stay inside the text piece
    for eSpeak; every other punctuation mark becomes its own token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i - 1] if i else ""
        following = text[i + 1] if i + 1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = _ALPHA_WORD_BEFORE_DOT.search(text[:i])
            if word and word[1].lower() in ABBREVIATIONS:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i + 1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    return shutil.which("espeak-ng")


def _load_libespeak():
    global _ESPEAK_LIB
    if _ESPEAK_LIB is not None:
        return _ESPEAK_LIB
    candidates = [
        ctypes.util.find_library("espeak-ng"),
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
    ]
    lib = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError("Install espeak-ng or libespeak-ng1 before preprocessing/inference")
    lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.espeak_Initialize.restype = ctypes.c_int
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    lib.espeak_SetVoiceByName.restype = ctypes.c_int
    lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
    lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    if lib.espeak_Initialize(2, 0, None, 0) < 0:
        raise RuntimeError("failed to initialize libespeak-ng")
    if lib.espeak_SetVoiceByName(ESPEAK_VOICE.encode("utf-8")) != 0:
        raise RuntimeError(f"failed to select eSpeak voice {ESPEAK_VOICE!r}")
    _ESPEAK_LIB = lib
    return lib


def espeak_version():
    executable = espeak_executable()
    if executable is not None:
        return subprocess.run([executable, "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    _load_libespeak()
    return "libespeak-ng"


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline when calling the CLI: eSpeak NG 1.51
    silently drops the last character of unterminated standard input.
    """
    executable = espeak_executable()
    if executable is not None:
        result = subprocess.run(
            [executable, "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
            input=piece.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
        return parse_espeak(result.stdout)
    with _ESPEAK_LOCK:
        lib = _load_libespeak()
        buf = ctypes.c_char_p(piece.encode("utf-8"))
        ptr = ctypes.c_char_p(buf.value)
        mode = 0x02 | (ord(SEPARATOR) << 8)
        clauses = []
        while ptr.value:
            out = lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, mode)
            if out:
                clauses.append(out.decode("utf-8"))
        return parse_espeak(" ".join(clauses))


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            # Strip format characters, parentheses, and French clitic '-' markers.
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()-")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    no_length = phone.replace("ː", "")
    if no_length != phone:
        if no_length in PHONE_SET:
            return (no_length,)
        if no_length in FALLBACK:
            return FALLBACK[no_length]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            word_symbols, word_stress = [], []
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                if not mapped:
                    continue
                # A fallback may expand to several phones: stress goes to its first vowel.
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    word_symbols.append(item)
                    word_stress.append(level if index == carrier else 0)
            if not word_symbols:
                continue
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            symbols.extend(word_symbols)
            stress.extend(word_stress)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece)))
    return tokenize(pieces)


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]


def normalize_for_wer(text, language="fr"):
    """Lowercase and strip punctuation/hyphens for ASR WER comparisons."""
    text = unicodedata.normalize("NFKC", text.lower()).replace("’", "'").replace("-", " ")
    pattern = r"[^a-z0-9' ]" if language == "en" else r"[^a-zàâäæçéèêëîïôöœùûüÿ0-9' ]"
    return re.sub(r"\s+", " ", re.sub(pattern, " ", text)).strip()
