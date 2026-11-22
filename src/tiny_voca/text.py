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
# Bump whenever normalization, expansion, liaison/elision or post-eSpeak rules change
# without altering the 54-symbol inventory / fallback vocabulary.
FRONTEND_RULES_VERSION = 2
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
    "ʼ": "'", "ʻ": "'", "ʹ": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "–": "—", "―": "—", "‐": "-", "‑": "-", "−": "-",
    "[": "(", "{": "(", "]": ")", "}": ")",
})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {
    "m", "mm", "mme", "mlle", "mgr", "mr", "mrs", "ms", "dr", "prof", "st", "ste",
    "etc", "ex", "cf", "av", "ap", "bd", "fig", "vol", "env", "p", "chap", "art", "tél",
}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")
_MONSIEUR_ABBR = re.compile(r"(?<![A-ZÀ-ÖØ-ÞŒ]\.\s)(?<!\.)\bM\.\s+(?=[A-Za-zÀ-ÖØ-öø-ÿŒœ])")
_ALPHA_WORD_BEFORE_DOT = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿŒœ]+)$")

_MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)
_UNITS_FR = (
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
    "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
)
_TENS_FR = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_VALID = re.compile(r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
_ROMAN_ORDINAL_RE = re.compile(
    r"\b((?:II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|"
    r"XX|XXI|XXII|XXIII|XXIV|XXV|XXVI|XXVII|XXVIII|XXIX|XXX)(?:e|es|ème|èmes)|"
    r"I(?:er|ers|re|res|ère|ères))\b"
)
_ROMAN_CONTEXT_RE = re.compile(
    r"\b(Louis|Henri|Charles|François|Philippe|Napoléon|Jean-Paul|Benoît|Pie|Léon|Clément|"
    r"Innocent|Grégoire|Alexandre|Édouard|George|Georges|Albert|Baudouin|Léopold|"
    r"[Tt]ome|[Cc]hapitre|[Aa]cte|[Ss]cène|[Ll]ivre|[Pp]artie|[Vv]olume|[Aa]rticle|[Ss]iècle)\s+([IVX]+)\b"
)
_ARABIC_ORDINAL_RE = re.compile(r"\b(\d+)(er|ers|re|res|ère|ères|e|es|ème|èmes)\b")
_FRENCH_INITIALISMS = frozenset({
    "UE", "IA", "PME", "PMI", "RSA", "RER", "TER", "IRM", "IVG", "PIB", "CGU", "CDI", "CDD",
    "SPA", "JO", "VIP", "EDF", "GDF", "BTP", "BNP", "QG", "PNB", "CPU", "GPU", "API", "TTS",
    "ASR", "USB", "SSD", "RAM", "SQL", "SSH", "XML", "CSV", "FAQ",
})
_ACRONYM_HINTS = {"INSEE": "ainsé", "ZUP": "zuppe"}
_FRENCH_WORD_HINTS = {
    "annecy": "anneci",
    "nancy": "nanssi",
    "fleury": "fleuri",
    "coud": "cou",
    "couds": "cou",
    "bruns": "brun",
    "vécut": "vécue",
    "traversons": "traverçons",
    "sied": "sié",
    "sieds": "sié",
    "assied": "assié",
    "assieds": "assié",
    "pull": "pulle",
    "pulls": "pulle",
    "wallon": "ouallon",
    "wallons": "ouallons",
    "wallonne": "ouallone",
    "wallonnes": "ouallones",
    "v'là": "vla",
    "zénith": "zénite",
    "zéniths": "zénite",
    "rythme": "ritme",
    "rythmes": "ritme",
    "mythe": "mite",
    "mythes": "mite",
    "luth": "lutte",
    "luths": "lutte",
    "labyrinthe": "labyrinte",
    "labyrinthes": "labyrinte",
    "lithographie": "litografie",
    "lithographies": "litografie",
    "misanthrope": "misantrope",
    "misanthropes": "misantrope",
    "euthanasie": "eutanasie",
    "enthousiasme": "antouziasme",
    "enthousiasmes": "antouziasme",
    "enthousiasmer": "antouziasmer",
    "enthousiasma": "antouziasma",
    "enthousiasmé": "antouziasmé",
    "enthousiasmée": "antouziasmé",
    "enthousiasmés": "antouziasmé",
    "enthousiasmées": "antouziasmé",
    "arithmétique": "aritmétique",
    "arithmétiques": "aritmétique",
    "absinthe": "absinte",
    "absinthes": "absinte",
    "éther": "étère",
    "éthers": "étère",
    "jacinthe": "jacinte",
    "jacinthes": "jacinte",
    "léthifère": "létifère",
    "reims": "rince",
    "amer": "amère",
    "amers": "amère",
    "cuiller": "cuillère",
    "cuillers": "cuillère",
    "esther": "Estère",
    "héroïne": "éroïne",
    "héroïnes": "éroïnes",
    "héroïque": "éroïque",
    "héroïques": "éroïques",
    "héroïsme": "éroïsme",
    "quatre-vingt-onze": "91",
}
_FRENCH_WORD_HINT_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _FRENCH_WORD_HINTS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_DOS_WORD_RE = re.compile(r"\b(?:Dos|dos)\b")
_OUENT_RE = re.compile(r"\b([a-zà-öø-ÿœ'-]*[bcdfghjklmnpqrstvwxzç]ou)(?:es|ent)\b", re.IGNORECASE)
_ELIDED_PREFIX_PHONES = {
    "d": [("d", 0)],
    "l": [("l", 0)],
    "n": [("n", 0)],
    "s": [("s", 0)],
    "m": [("m", 0)],
    "t": [("t", 0)],
    "j": [("ʒ", 0)],
    "c": [("s", 0)],
    "qu": [("k", 0)],
    "jusqu": [("ʒ", 0), ("y", 0), ("s", 0), ("k", 0)],
    "lorsqu": [("l", 0), ("ɔ", 0), ("ʁ", 0), ("s", 0), ("k", 0)],
    "puisqu": [("p", 0), ("y", 0), ("i", 0), ("s", 0), ("k", 0)],
    "quoiqu": [("k", 0), ("w", 0), ("a", 0), ("k", 0)],
}
_ELIDED_WORD_RE = re.compile(
    r"\b(jusqu|lorsqu|puisqu|quoiqu|qu|[dlnsmtjc])'([A-Za-zÀ-ÖØ-öø-ÿŒœ]+(?:['-][A-Za-zÀ-ÖØ-öø-ÿŒœ]+)*)\b",
    re.IGNORECASE,
)
_NO_LIAISON_SPLIT_RE = re.compile(r"(?i)(?:(?<!\bvingt)\s+(?=\bet\b)|\s+(?=\bhameaux?\b))")

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


def vocabulary_identity():
    """Token vocabulary / checkpoint compatibility identity (preserved across E01 rule fixes)."""
    return frontend_identity()


def vocabulary_fingerprint():
    """Compatibility hash governing checkpoint/ONNX token ID compatibility."""
    return frontend_fingerprint()


def frontend_rules_identity():
    """Effective phonetic-rule identity tracked separately from vocabulary compatibility.

    Separating vocabulary compatibility (``frontend_fingerprint()``) from effective
    phonemization rules prevents stale pre-E01 annotations from being reused silently
    while keeping earlier format-4 checkpoints loadable for inference.
    """
    return {
        "vocab_fingerprint": frontend_fingerprint(),
        "rules_version": FRONTEND_RULES_VERSION,
        "abbreviations": sorted(ABBREVIATIONS),
        "initialisms": sorted(_FRENCH_INITIALISMS),
        "acronym_hints": dict(sorted(_ACRONYM_HINTS.items())),
        "word_hints": dict(sorted(_FRENCH_WORD_HINTS.items())),
        "elided_prefixes": sorted(_ELIDED_PREFIX_PHONES),
    }


def frontend_rules_fingerprint():
    return hashlib.sha256(
        json.dumps(frontend_rules_identity(), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _number_fr(n):
    if n < 20:
        return _UNITS_FR[n]
    if n < 70:
        t, u = divmod(n, 10)
        if u == 0:
            return _TENS_FR[t]
        if u == 1:
            return f"{_TENS_FR[t]} et un"
        return f"{_TENS_FR[t]}-{_UNITS_FR[u]}"
    if n < 80:
        u = n - 60
        return "soixante et onze" if u == 11 else f"soixante-{_UNITS_FR[u]}"
    if n < 100:
        u = n - 80
        return "quatre-vingts" if u == 0 else f"quatre-vingt-{_UNITS_FR[u]}"
    if n < 1000:
        h, rest = divmod(n, 100)
        prefix = "cent" if h == 1 else f"{_UNITS_FR[h]} cents" if rest == 0 else f"{_UNITS_FR[h]} cent"
        return prefix if rest == 0 else f"{prefix} {_number_fr(rest)}"
    if n < 1_000_000:
        k, rest = divmod(n, 1000)
        prefix = "mille" if k == 1 else f"{_number_fr(k)} mille"
        return prefix if rest == 0 else f"{prefix} {_number_fr(rest)}"
    return str(n)


def _ordinal_fr(n, feminine=False, plural=False):
    if n == 1:
        base = "première" if feminine else "premier"
    else:
        word = _number_fr(n)
        if word.endswith(("vingts", "cents")):
            word = word[:-1]
        if word.endswith("e"):
            word = word[:-1]
        elif word.endswith("q"):
            word = word + "u"
        elif word.endswith("f"):
            word = word[:-1] + "v"
        base = word + "ième"
    return base + "s" if plural else base


def _roman_to_int(token):
    if not token or not _ROMAN_VALID.match(token):
        return None
    total, prev = 0, 0
    for ch in reversed(token):
        val = _ROMAN_VALUES[ch]
        total = total - val if val < prev else total + val
        prev = val
    return total


def _expand_french_text(text):
    """Expand French numbers, dates, clock times, currencies, units, ordinals,
    abbreviations and eSpeak French pitfalls into speakable French text."""
    # 1. Archaic grand' compounds ("grand'mère", "grand'chose") -> hyphenated form.
    text = re.sub(r"\b([Gg]rand)'(?=[bcdfghjklmnpqrstvwxzçBDFGHJKLMNPQRSTVWXZÇ])", r"\1-", text)
    # 2. Negative numbers ("-5 °C") without touching dates/ranges ("2026-11-17").
    text = re.sub(r"(?<![\w.,-])[-−](?=\d+(?:[.,]\d+)?(?![/-]\d))", "moins ", text)
    # 3. Abbreviations.
    text = re.sub(r"\bMM\.\s*", "Messieurs ", text)
    text = re.sub(r"\bMgr\.?\s+", "Monseigneur ", text)
    text = re.sub(r"\bav\.\s*J\.-?C\.?", "avant Jésus-Christ", text)
    text = re.sub(r"\bap\.\s*J\.-?C\.?", "après Jésus-Christ", text)
    text = re.sub(r"\bpar\s+ex\.", "par exemple", text)
    text = re.sub(r"\bp\.\s*(?=\d)", "page ", text)
    text = re.sub(r"\bfig\.\s*(?=\d)", "figure ", text)
    text = re.sub(r"\bvol\.\s*(?=\d)", "volume ", text)
    text = re.sub(r"\bchap\.\s*(?=\d)", "chapitre ", text)
    text = re.sub(r"\benv\.\s*(?=\d)", "environ ", text)
    text = re.sub(r"\b[nN][°º]\s*", "numéro ", text)

    # 4. Dates (DD/MM/YYYY and YYYY-MM-DD).
    def _date(day, month, year):
        d = "premier" if int(day) == 1 else str(int(day))
        return f"{d} {_MONTHS_FR[int(month) - 1]} {year}"

    text = re.sub(
        r"\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/([12]\d{3})\b",
        lambda m: _date(m[1], m[2], m[3]),
        text,
    )
    text = re.sub(
        r"\b([12]\d{3})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b",
        lambda m: _date(m[3], m[2], m[1]),
        text,
    )

    # 5. Clock times ("14h30", "9h05", "14h00", "8h", "1h30", "00h15").
    def _clock(m):
        hour, mins = int(m[1]), m[2]
        if hour == 1:
            head = "une heure"
        elif hour == 0:
            head = "zéro heure"
        elif hour == 9:
            head = "neuf heures"
        else:
            head = f"{hour} heures"
        if mins is None or mins == "00":
            return head
        return f"{head} {int(mins)}"

    text = re.sub(r"\b(\d{1,3})\s*[hH](?:\s*([0-5]\d))?\b", _clock, text)

    # 6. Currencies ("12,50 €", "1 €", "12 €", "$100", "£50").
    def _euro_cents(m):
        whole, cents = m[1].strip(), m[2]
        unit = "euro" if whole in ("0", "1") else "euros"
        base = f"{'un' if whole == '1' else whole} {unit}"
        return base if cents == "00" else f"{base} {int(cents)}"

    num = r"(?<!\d)(?:\d{1,3}(?:[ .]\d{3})+|\d+)"
    text = re.sub(rf"({num}),(\d{{2}})\s*(?:€|EUR\b)", _euro_cents, text)
    text = re.sub(r"\b1\s*(?:€|EUR\b)", "un euro", text)
    text = re.sub(rf"({num}(?:,\d+)?)\s*(?:€|EUR\b)", r"\1 euros", text)
    text = re.sub(rf"(?:€|EUR\b)\s*({num}),(\d{{2}})\b", lambda m: _euro_cents(m), text)
    text = re.sub(rf"(?:€|EUR\b)\s*({num}(?:,\d+)?)\b", r"\1 euros", text)
    text = text.replace("€", " euros ")
    text = re.sub(rf"\$\s*({num}(?:[.,]\d+)?)\b", r"\1 dollars", text)
    text = re.sub(rf"£\s*({num}(?:[.,]\d+)?)\b", r"\1 livres", text)

    # 7. Temperatures & units.
    text = re.sub(r"°\s*C\b", " degrés Celsius", text)
    text = re.sub(r"°\s*F\b", " degrés Fahrenheit", text)
    text = re.sub(r"\bkm/h\b", "kilomètres par heure", text)
    text = re.sub(r"\bm/s\b", "mètres par seconde", text)
    text = re.sub(r"\bkm²", " kilomètres carrés", text)
    text = re.sub(r"\bm²", " mètres carrés", text)
    text = re.sub(r"\bm³", " mètres cubes", text)
    text = re.sub(r"\b1\s*min\b", "une minute", text)
    text = re.sub(r"\b(\d+(?:,\d+)?)\s*min\b", r"\1 minutes", text)
    # A one-letter unit followed by an apostrophe is an elided pronoun, not a unit: "en 2014 s'est", "les 3 l'ont vu".
    text = re.sub(r"\b(\d+(?:,\d+)?)\s*g\b(?!['’ʼ])", r"\1 grammes", text)
    text = re.sub(r"\b(\d+(?:,\d+)?)\s*m\b(?!['’ʼ])", r"\1 mètres", text)
    text = re.sub(r"\b(\d+(?:,\d+)?)\s*[Ll]\b(?!['’ʼ])", r"\1 litres", text)
    text = re.sub(r"\b(\d+(?:,\d+)?)\s+s\b(?!['’ʼ])", r"\1 secondes", text)

    # 8. Ordinals & contextual Roman numerals.
    def _arabic_ord(m):
        val = int(m[1])
        if val < 1 or val >= 1_000_000:
            return m[0]
        suf = m[2]
        return _ordinal_fr(val, feminine=("re" in suf or "ère" in suf), plural=suf.endswith("s"))

    def _roman_ord(m):
        raw_token = m[0]
        match = re.match(r"^([IVXLCDM]+)(er|ers|re|res|ère|ères|e|es|ème|èmes)$", raw_token)
        val = _roman_to_int(match[1]) if match else None
        if not val:
            return raw_token
        suf = match[2]
        return _ordinal_fr(val, feminine=("re" in suf or "ère" in suf), plural=suf.endswith("s"))

    text = _ARABIC_ORDINAL_RE.sub(_arabic_ord, text)
    text = _ROMAN_ORDINAL_RE.sub(_roman_ord, text)
    text = _ROMAN_CONTEXT_RE.sub(lambda m: f"{m[1]} {_roman_to_int(m[2]) or m[2]}", text)

    # 9. Acronyms / initialisms.
    def _acronym(m):
        word = m[0]
        if word in _ACRONYM_HINTS:
            return _ACRONYM_HINTS[word]
        if word in _FRENCH_INITIALISMS:
            return " ".join(word)
        return word

    text = re.sub(r"\b[A-Z]{2,}\b", _acronym, text)

    # 10. High-precision French homograph disambiguations.
    text = re.sub(
        r"\b(à|de|du|vers|dans|par|pour|sur|en|sous|jusqu'à)\s+(l)'[Ee]st\b",
        r"\1 \2'este",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(l|L)'Est\b", r"\1'este", text)
    text = re.sub(
        r"\btous\s+(deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\b",
        r"tou \1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\ben\s+plus(?=\s+de\b|\s*[,.;:!?—…)]|\s*$)",
        "en plusse",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(une|cette|des)\s+vis\b", r"\1 visse", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(\b(?:je|tu|il|elle|on|nous|vous|ils|elles|qui|ne)\s+)\b(la|les)\s+vis\b|\b(la|les)\s+vis\b",
        lambda m: m[0] if m[1] else f"{m[3]} visse",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(les|des|ces|mes|tes|ses|nos|vos|leurs|quelques|plusieurs)\s+os\b",
        r"\1 eaux",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(mes|tes|ses|nos|vos|leurs)\s+fils\b(?!\s+(?:électriques?|métalliques?|conducteurs?|de\s+(?:fer|soie|coton|laine|cuivre|nylon|acier|plomb)))",
        r"\1 fisse",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(me|te|se|nous|vous)\s+fier\b", r"\1 fié", text, flags=re.IGNORECASE)

    # 11. French words misclassified as (en) or mispronounced by eSpeak NG, and -oues/-ouent.
    text = _DOS_WORD_RE.sub("dô", text)
    text = _FRENCH_WORD_HINT_RE.sub(lambda m: _FRENCH_WORD_HINTS[m[1].lower()], text)
    text = _OUENT_RE.sub(r"\1e", text)
    return re.sub(r"\s+", " ", text).strip()


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
                abbr = word[1].lower()
                if abbr == "etc" or (following != "" and (abbr not in ("art", "vol", "ex") or re.match(r"\s+\d", text[i + 1:]))):
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


def _raw_espeak(piece):
    """Run eSpeak NG on one text piece and return its raw IPA clause output.

    Both the CLI and libespeak-ng require a trailing newline: eSpeak NG 1.51
    can otherwise drop or retain trailing tokens across sequential calls.
    """
    terminated = piece.rstrip("\n") + "\n"
    executable = espeak_executable()
    if executable is not None:
        result = subprocess.run(
            [executable, "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
            input=terminated, text=True, capture_output=True, check=True, timeout=30)
        return result.stdout
    with _ESPEAK_LOCK:
        lib = _load_libespeak()
        encoded = terminated.encode("utf-8")
        buf = ctypes.create_string_buffer(encoded)
        ptr = ctypes.c_char_p(ctypes.addressof(buf))
        mode = 0x02 | (ord(SEPARATOR) << 8)
        clauses = []
        while ptr.value:
            out = lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, mode)
            if out:
                clauses.append(out.decode("utf-8"))
        return " ".join(clauses)


def _espeak_subpiece(sub):
    raw = _raw_espeak(sub)
    if "(en)" in raw:
        for match in _ELIDED_WORD_RE.finditer(sub):
            if "(en)" not in _raw_espeak(match[0]):
                continue
            prefix_phones = _ELIDED_PREFIX_PHONES.get(match[1].lower())
            if prefix_phones is None:
                continue
            left = sub[:match.start()].strip()
            stem_start = match.start() + len(match[1]) + 1
            rest = sub[stem_start:].strip()
            words = _espeak_subpiece(left) if left else []
            if rest:
                rest_words = _espeak_subpiece(rest)
                if rest_words:
                    words.append(list(prefix_phones) + rest_words[0])
                    words.extend(rest_words[1:])
            return words
    return parse_espeak(raw)


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``."""
    words = []
    for sub in _NO_LIAISON_SPLIT_RE.split(piece):
        sub = sub.strip()
        if sub:
            words.extend(_espeak_subpiece(sub))
    for word in words:
        while len(word) >= 2 and word[-1][0] in ("t", "z") and word[-2][0] == word[-1][0]:
            word[-2] = (word[-2][0], max(word[-2][1], word[-1][1]))
            word.pop()
    if re.search(r"-[yY]\b", piece):
        for word in words:
            if len(word) >= 5 and [p for p, _ in word[-5:]] == ["i", "ɡ", "ʁ", "ɛ", "k"]:
                word[-5:] = [("i", max(word[-5][1], word[-2][1]))]
    return words


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
    for punctuation, piece in chunks(_expand_french_text(normalize_text(text))):
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
