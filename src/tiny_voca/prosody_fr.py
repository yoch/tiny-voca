"""French phrasal prosody features and frozen-host Rhythm + Intonation Refiner (P7).

Designed to sit cleanly on top of a frozen ``TinyVoca`` host (e.g. ``fr_ref_35k``)
without touching its acoustic decoder or vocoder weights:
  - Extracts 18 deterministic French phrasal/prosodic features per token (word
    boundaries, word-final accent vowel, function vs content word, clause distance,
    interrogative totale vs partielle, exclamation, geminate boundary) with zero
    external NLP dependencies.
  - ``FrenchProsodyRefiner`` (~360k params) predicts residual durations/pauses and
    residual F0/voicing/energy on top of the frozen host, initialized to exact
    identity so step 0 reproduces the host bit-for-bit.
"""

from __future__ import annotations

import math
import re
import torch
from torch import nn
from torch.nn import functional as F

from .text import PHONE_SET, SYMBOL_TO_ID, SYMBOLS, VOWEL_SET

PHRASAL_DIM = 18

_SP_ID = SYMBOL_TO_ID["<sp>"]
_BOS_ID = SYMBOL_TO_ID["<bos>"]
_EOS_ID = SYMBOL_TO_ID["<eos>"]
_CLAUSE_PUNCT = frozenset((",", ";", ":", "—", "(", ")", ".", "?", "!", "…"))
_INTERNAL_PUNCT = frozenset((",", ";", ":", "—", "(", ")"))

_EST_CE_QUE_RE = re.compile(r"\best[- ]ce\s+qu[e']", re.IGNORECASE)
_WH_RE = re.compile(
    r"(?:^|[\s,;:—(«\"'])"
    r"(?:qu['’]est[- ]ce\s+qu[e']|qui|que|qu['’]|quoi|où|quand|comment|pourquoi|combien|"
    r"quel(?:le)?s?|lequel|laquelle|lesquel(?:le)?s|auquel|auxquel(?:le)?s|duquel|desquel(?:le)?s)"
    r"(?:$|[\s,;:—?»\"'?.!])",
    re.IGNORECASE,
)

# High-frequency French function words / clitics (do not carry nuclear accents;
# unnatural to follow with a mid-phrase <sp> pause).
FRENCH_FUNCTION_WORDS = frozenset((
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
    "à", "en", "dans", "sur", "sous", "par", "pour", "avec", "sans", "chez", "vers",
    "et", "ou", "ni", "mais", "car", "donc", "or", "que", "qu", "qui", "dont", "où", "si", "comme",
    "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "m", "te", "t", "se", "s", "lui", "leur", "y", "ne", "n",
    "ce", "c", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leurs",
    "est", "sont", "suis", "es", "sommes", "êtes", "a", "as", "ont", "ai", "avons", "avez",
    "était", "étaient", "avait", "avaient", "fut", "furent", "sera", "seront",
))
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ0-9]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿŒœ0-9]+)*")


def _is_french_function_word(ortho_word: str) -> bool:
    """Return True if an orthographic French word (possibly elided, e.g. d'un, j'ai) is purely functional."""
    parts = [p for p in re.split(r"['’\-]", ortho_word.lower()) if p]
    return bool(parts) and all(p in FRENCH_FUNCTION_WORDS for p in parts)


def classify_sentence_type(text: str) -> tuple[float, float, float]:
    """Return (is_q_totale, is_q_partielle, is_exclamation) flags for French text."""
    if "?" in text:
        cleaned = _EST_CE_QUE_RE.sub(" ", text)
        if _WH_RE.search(cleaned):
            return 0.0, 1.0, 0.0
        return 1.0, 0.0, 0.0
    if "!" in text:
        return 0.0, 0.0, 1.0
    return 0.0, 0.0, 0.0


def extract_french_phrasal_features(tokens: torch.Tensor, text: str = "") -> torch.Tensor:
    """Compute [N, PHRASAL_DIM] deterministic French prosodic features aligned to ``tokens``."""
    tok_list = [int(t) for t in tokens.tolist()]
    n = len(tok_list)
    feats = torch.zeros((n, PHRASAL_DIM), dtype=torch.float32)
    if n == 0:
        return feats

    syms = [SYMBOLS[t] if 0 <= t < len(SYMBOLS) else "<unk>" for t in tok_list]
    is_q_tot, is_q_part, is_excl = classify_sentence_type(text)

    # Segment token indices into phone-only words separated by non-phone tokens.
    words: list[list[int]] = []
    cur: list[int] = []
    for i, sym in enumerate(syms):
        if sym in PHONE_SET:
            cur.append(i)
        else:
            if cur:
                words.append(cur)
                cur = []
    if cur:
        words.append(cur)

    # Align orthographic words (preserving French elisions/hyphens) to phone words.
    ortho_words = [w.lower() for w in _WORD_RE.findall(text)]
    word_is_func: list[bool] = []
    if ortho_words and words:
        for wi in range(len(words)):
            oi = min(len(ortho_words) - 1, int(round(wi * (len(ortho_words) - 1) / max(len(words) - 1, 1))))
            word_is_func.append(_is_french_function_word(ortho_words[oi]))
    else:
        # Token-only fallback: very short 1-2 phone words without punctuation.
        word_is_func = [len(w) <= 2 for w in words]

    tok_to_word = {}
    for wi, w_idxs in enumerate(words):
        w_len = len(w_idxs)
        last_vowel = None
        for idx in w_idxs:
            tok_to_word[idx] = wi
            if syms[idx] in VOWEL_SET:
                last_vowel = idx
        for pos, idx in enumerate(w_idxs):
            feats[idx, 4] = 1.0 if pos == w_len - 1 else 0.0
            feats[idx, 5] = 1.0 if idx == last_vowel else 0.0
            feats[idx, 6] = (pos + 0.5) / max(w_len, 1)
            feats[idx, 7] = math.log1p(w_len) / 3.0
            feats[idx, 8] = 1.0 if word_is_func[wi] else 0.0

    # Segment into clauses delimited by punctuation or BOS/EOS.
    clause_starts = [0]
    for i, sym in enumerate(syms):
        if sym in _CLAUSE_PUNCT or sym in ("<bos>", "<eos>"):
            clause_starts.append(i)
    clause_starts.append(n)
    clause_bounds = sorted(set(clause_starts))

    for b0, b1 in zip(clause_bounds[:-1], clause_bounds[1:]):
        phone_idxs = [i for i in range(b0, b1) if syms[i] in PHONE_SET]
        c_len = len(phone_idxs)
        if c_len == 0:
            continue
        last_word_in_clause = tok_to_word.get(phone_idxs[-1])
        for p_pos, idx in enumerate(phone_idxs):
            dist_end = c_len - 1 - p_pos
            feats[idx, 10] = (p_pos + 0.5) / c_len
            feats[idx, 11] = math.log1p(dist_end) / 4.0
            if tok_to_word.get(idx) == last_word_in_clause:
                feats[idx, 16] = 1.0

    # Per-token basic & boundary features.
    phone_count = max(sum(1 for s in syms if s in PHONE_SET), 1)
    seen_phones = 0
    for i, sym in enumerate(syms):
        is_ph = sym in PHONE_SET
        feats[i, 0] = 1.0 if is_ph else 0.0
        feats[i, 1] = 1.0 if sym in VOWEL_SET else 0.0
        feats[i, 2] = 1.0 if sym in _INTERNAL_PUNCT else 0.0
        feats[i, 3] = 1.0 if sym == "<sp>" else 0.0
        if is_ph:
            seen_phones += 1
        feats[i, 12] = seen_phones / phone_count
        feats[i, 13] = is_q_tot
        feats[i, 14] = is_q_part
        feats[i, 15] = is_excl

        if sym == "<sp>":
            prev_w = tok_to_word.get(i - 1)
            next_w = tok_to_word.get(i + 1)
            prev_func = bool(prev_w is not None and word_is_func[prev_w])
            next_func = bool(next_w is not None and word_is_func[next_w])
            # Flag spaces immediately after a function word (e.g. "de _", "les _")
            # or linking short phrases into "et"/"ou" where pauses sound unnatural.
            if prev_func or next_func:
                feats[i, 9] = 1.0
            if i > 0 and i + 1 < n and syms[i - 1] == syms[i + 1] and syms[i - 1] in PHONE_SET:
                feats[i, 17] = 1.0

    return feats


class _ResConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 2)
        self.pw2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[..., None].to(x.dtype)
        h = self.dw((x * m).transpose(1, 2)).transpose(1, 2)
        h = self.norm(h)
        h = self.pw2(self.drop(F.gelu(self.pw1(h))))
        return (x + self.drop(h)) * m


class FrenchProsodyRefiner(nn.Module):
    """Lightweight residual Rhythm + Expressive Intonation refiner (~317k params)
    operating on top of a frozen TinyVoca host (total < 4.85M with fr_ref_35k).

    Both output heads are zero-initialized so at initialization the refiner is an
    exact identity transform of the frozen host predictions.
    """

    def __init__(
        self,
        host_channels: int = 192,
        phrasal_dim: int = PHRASAL_DIM,
        hidden_dim: int = 96,
        speakers: int = 2,
        dropout: float = 0.1,
        enable_rhythm: bool = True,
    ):
        super().__init__()
        self.host_channels = host_channels
        self.phrasal_dim = phrasal_dim
        self.hidden_dim = hidden_dim
        self.speakers = speakers
        self.enable_rhythm = enable_rhythm
        self.register_buffer("sp_pause_threshold", torch.tensor(0.50, dtype=torch.float32))
        self.register_buffer("punct_pause_threshold", torch.tensor(0.50, dtype=torch.float32))

        self.spk_emb = nn.Embedding(speakers, hidden_dim)

        # 1. Token-level Rhythm & Pause refiner (BiGRU + multi-scale conv residual)
        if enable_rhythm:
            self.dur_in = nn.Linear(host_channels + phrasal_dim + 2, hidden_dim)
            self.dur_conv1 = _ResConvBlock(hidden_dim, kernel=5, dilation=1, dropout=dropout)
            self.dur_conv2 = _ResConvBlock(hidden_dim, kernel=5, dilation=2, dropout=dropout)
            self.dur_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.dur_norm = nn.LayerNorm(hidden_dim)
            self.dur_head = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.dur_head.weight)
            nn.init.zeros_(self.dur_head.bias)

        # 2. Frame-level Expressive Intonation & Energy refiner
        # Input: host prosody_hidden [C] + expanded phrasal [P] + host prosody pred [3] + relative frame pos [2]
        self.pro_in = nn.Linear(host_channels + phrasal_dim + 5, hidden_dim)
        self.pro_conv1 = _ResConvBlock(hidden_dim, kernel=7, dilation=1, dropout=dropout)
        self.pro_conv2 = _ResConvBlock(hidden_dim, kernel=7, dilation=2, dropout=dropout)
        self.pro_conv3 = _ResConvBlock(hidden_dim, kernel=7, dilation=4, dropout=dropout)
        self.pro_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.pro_norm = nn.LayerNorm(hidden_dim)
        self.pro_head = nn.Linear(hidden_dim, 3)  # residual (Δlog_f0, Δvoicing_logit, Δenergy)
        nn.init.zeros_(self.pro_head.weight)
        nn.init.zeros_(self.pro_head.bias)

    def refine_durations(
        self,
        enc_hidden: torch.Tensor,
        host_dur_out: torch.Tensor,
        phrasal: torch.Tensor,
        token_mask: torch.Tensor,
        speakers: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_rhythm:
            return host_dur_out
        x = torch.cat((enc_hidden, phrasal, host_dur_out), dim=-1)
        h = F.gelu(self.dur_in(x)) + self.spk_emb(speakers)[:, None]
        h = self.dur_conv1(h, token_mask)
        h = self.dur_conv2(h, token_mask)
        g, _ = self.dur_gru(h * token_mask[..., None].to(h.dtype))
        h = self.dur_norm(h + g) * token_mask[..., None].to(h.dtype)
        return host_dur_out + self.dur_head(h) * token_mask[..., None].to(h.dtype)

    def refine_prosody(
        self,
        prosody_hidden: torch.Tensor,
        host_prosody: torch.Tensor,
        expanded_phrasal: torch.Tensor,
        frame_pos: torch.Tensor,
        frame_mask: torch.Tensor,
        speakers: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((prosody_hidden, expanded_phrasal, host_prosody, frame_pos), dim=-1)
        h = F.gelu(self.pro_in(x)) + self.spk_emb(speakers)[:, None]
        h = self.pro_conv1(h, frame_mask)
        h = self.pro_conv2(h, frame_mask)
        h = self.pro_conv3(h, frame_mask)
        g, _ = self.pro_gru(h * frame_mask[..., None].to(h.dtype))
        h = self.pro_norm(h + g) * frame_mask[..., None].to(h.dtype)
        return host_prosody + self.pro_head(h) * frame_mask[..., None].to(h.dtype)

    @staticmethod
    def expand_token_features(
        phrasal: torch.Tensor,
        durations: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand [B, N, P] token phrasal features to [B, T, P] and compute [B, T, 2] frame positions."""
        bsz = durations.shape[0]
        t_max = int(frame_lengths.max().item())
        device = phrasal.device
        ends = durations.cumsum(1)
        frame = torch.arange(t_max, device=device)[None].expand(bsz, t_max).contiguous()
        index = torch.searchsorted(ends, frame, right=True).clamp(max=durations.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        rel_in_tok = (offset.float() + 0.5) / owned.float()
        rel_in_utt = (frame.float() + 0.5) / frame_lengths[:, None].clamp_min(1).float()
        mask = (frame < frame_lengths[:, None]).float()
        exp_p = phrasal.gather(1, index[..., None].expand(-1, -1, phrasal.shape[-1])) * mask[..., None]
        fpos = torch.stack((rel_in_tok, rel_in_utt), dim=-1) * mask[..., None]
        return exp_p, fpos

    @staticmethod
    def decode_durations(
        dur_out: torch.Tensor,
        tokens: torch.Tensor,
        phone_tokens_mask: torch.Tensor,
        token_mask: torch.Tensor,
        speed: float = 1.0,
        sp_pause_threshold: float = 0.75,
        punct_pause_threshold: float = 0.50,
    ) -> torch.Tensor:
        """Convert (log1p_dur, pause_logit) to integer frame durations with separate <sp> threshold."""
        frames = torch.round(torch.expm1(dur_out[..., 0].float().clamp(0, math.log1p(400))) / speed).long().clamp_min(1)
        probs = torch.sigmoid(dur_out[..., 1].float())
        is_sp = tokens == _SP_ID
        thresh = torch.where(is_sp, torch.full_like(probs, sp_pause_threshold), torch.full_like(probs, punct_pause_threshold))
        present = phone_tokens_mask[tokens] | (probs > thresh)
        return torch.where(present, frames, torch.zeros_like(frames)) * token_mask


@torch.inference_mode()
def synthesize_with_refiner(
    model: nn.Module,
    refiner: FrenchProsodyRefiner | None,
    tokens: torch.Tensor,
    stress: torch.Tensor,
    lengths: torch.Tensor,
    speakers: torch.Tensor,
    texts: list[str] | None = None,
    speed: float = 1.0,
    pitch_shift: float = 0.0,
    seed: int = 0,
    pitch_range: float = 1.0,
    pitch_jitter: float = 0.0,
    voiced_vowels: bool = True,
    use_rhythm: bool = True,
    use_prosody: bool = True,
    sp_pause_threshold: float | None = None,
    punct_pause_threshold: float | None = None,
):
    """Run frozen TinyVoca synthesis with an optional FrenchProsodyRefiner.

    When ``refiner`` is freshly zero-initialized (and default thresholds 0.50 are used),
    this produces bit-for-bit identical output to ``model.synthesize(...)``.
    """
    bsz, n_max = tokens.shape
    device = tokens.device
    hidden, mask = model.encode(tokens, stress, lengths, speakers)

    phrasal = None
    if refiner is not None and (use_rhythm or use_prosody):
        ph_rows = []
        for i in range(bsz):
            li = int(lengths[i].item())
            txt = texts[i] if texts is not None and i < len(texts) else ""
            row = extract_french_phrasal_features(tokens[i, :li].cpu(), txt)
            if li < n_max:
                row = F.pad(row, (0, 0, 0, n_max - li))
            ph_rows.append(row)
        phrasal = torch.stack(ph_rows, dim=0).to(device)

    host_dur_out = model.duration(hidden, mask).float()
    if refiner is not None and use_rhythm:
        dur_out = refiner.refine_durations(hidden.float(), host_dur_out, phrasal, mask, speakers)
    else:
        dur_out = host_dur_out

    sp_thr = (
        float(sp_pause_threshold)
        if sp_pause_threshold is not None
        else (float(refiner.sp_pause_threshold.item()) if refiner is not None else 0.50)
    )
    pu_thr = (
        float(punct_pause_threshold)
        if punct_pause_threshold is not None
        else (float(refiner.punct_pause_threshold.item()) if refiner is not None else 0.50)
    )
    durations = FrenchProsodyRefiner.decode_durations(
        dur_out,
        tokens,
        model.phone_tokens,
        mask,
        speed=speed,
        sp_pause_threshold=sp_thr,
        punct_pause_threshold=pu_thr,
    )

    frames, frame_mask, frame_lengths = model.expand(hidden, durations, mask, speakers)
    prosody_hidden, host_prosody = model.predict_prosody(frames, frame_mask)
    if refiner is not None and use_prosody:
        exp_phrasal, frame_pos = FrenchProsodyRefiner.expand_token_features(phrasal, durations, frame_lengths)
        prosody = refiner.refine_prosody(
            prosody_hidden.float(),
            host_prosody.float(),
            exp_phrasal,
            frame_pos,
            frame_mask,
            speakers,
        ).to(host_prosody.dtype)
    else:
        prosody = host_prosody

    if voiced_vowels:
        ends = durations.cumsum(1)
        frame = torch.arange(frames.shape[1], device=device)[None].expand(len(lengths), -1).contiguous()
        owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
        vowel = model.vowel_tokens[tokens].gather(1, owner) & frame_mask
        prosody = torch.cat(
            (
                prosody[..., :1],
                torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                prosody[..., 2:],
            ),
            dim=-1,
        )
    prosody = model.widen_pitch(prosody, frame_mask, pitch_range)
    acoustic = model.decode(prosody_hidden, model.hard_prosody(prosody), frame_mask)
    noise = nn.utils.rnn.pad_sequence(
        [
            torch.randn(int(n) * model.config.hop, generator=torch.Generator().manual_seed(seed + row))
            for row, n in enumerate(frame_lengths)
        ],
        batch_first=True,
    )
    pitch = prosody[..., 0] + pitch_shift
    if pitch_jitter > 0:
        pitch = pitch + model.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
    wave = model.vocoder(acoustic, model.frequency(pitch), frame_lengths, noise=noise.to(hidden.device))
    return wave, frame_lengths * model.config.hop, {
        "durations": durations,
        "prosody": prosody,
        "frame_mask": frame_mask,
        "dur_out": dur_out,
    }


def load_prosody_refiner(path, device="cpu") -> FrenchProsodyRefiner:
    """Load a saved P7 ``FrenchProsodyRefiner`` checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    refiner = FrenchProsodyRefiner(**payload["refiner_config"])
    refiner.load_state_dict(payload["refiner_state"], strict=True)
    return refiner.to(device).eval()


# ---------------------------------------------------------------------------
# Inference-time real-world French text cleaner (0 retraining, preserves
# frontend_fingerprint() and frontend_rules_fingerprint() 100%).
# ---------------------------------------------------------------------------

_FR_PHONE_RE = re.compile(r"(?<!\d)(0[1-9])(?:[\s.\-](\d{2})){4}(?!\d)")
_FR_PHONE_SPLIT_RE = re.compile(r"(?<!\d)(0[1-9])[\s.\-](\d{2})[\s.\-](\d{2})[\s.\-](\d{2})[\s.\-](\d{2})(?!\d)")
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
_URL_RE = re.compile(r"\bhttps?://([A-Za-z0-9.\-]+(?:/[^\s,;)]*[^\s,;).!?])?)")

_UNITS_FR_EXT = (
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*[Kk]o\b"), "kilooctet", "kilooctets"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*Mo\b"), "mégaoctet", "mégaoctets"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*Go\b"), "gigaoctet", "gigaoctets"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*To\b"), "téraoctet", "téraoctets"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*ms\b"), "milliseconde", "millisecondes"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*[µμu]s\b"), "microseconde", "microsecondes"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*ml\b"), "millilitre", "millilitres"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*cl\b"), "centilitre", "centilitres"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*dl\b"), "décilitre", "décilitres"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*mg\b"), "milligramme", "milligrammes"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*kHz\b"), "kilohertz", "kilohertz"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*MHz\b"), "mégahertz", "mégahertz"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*kWh\b"), "kilowattheure", "kilowattheures"),
    (re.compile(r"\b(\d+(?:[.,]\d+)?)\s*MWh\b"), "mégawattheure", "mégawattheures"),
)

# Only override acronyms/initialisms that eSpeak NG mispronounces or that text.py
# previously split with spaces; leave native eSpeak-fluent initialisms (SNCF, TGV,
# RATP, RGPD, CNRS, HLM, CGT, TVA, ADN, PDF, LLM, SMS, GPS, DVD, TV) untouched so
# eSpeak preserves authentic French (secondary..., primary-final) stress.
_FR_ACRONYM_OVERRIDES = {
    "RAM": "rame",
    "EHPAD": "épade",
    "MEDEF": "médèfe",
    "IA": "i a",
    "OMS": "ohèmèsse",
    "CHU": "céachu",
    "AVC": "avécé",
    "FBI": "èfbéi",
    "CIA": "séia",
    "NSA": "ènèssa",
    "IBM": "ibéème",
    "OVH": "ovéache",
    "CPU": "cépéu",
    "GPU": "gépéu",
    "USB": "uèssbé",
    "API": "apéi",
    "PIB": "péibé",
    "UE": "ueuh",
    "EDF": "édéèfe",
    "GDF": "gédéèfe",
    "RSA": "èrèssa",
    "PME": "péèmeuh",
    "PMI": "péèmi",
    "RER": "èreuhèr",
    "TER": "téeuhèr",
    "VTC": "vétécé",
    "KGB": "kagébé",
    "CDG": "sédégé",
    "QCM": "kuséème",
    "CDI": "sédéi",
    "CDD": "sédédé",
    "IVG": "ivégé",
    "IRM": "ièrème",
    "SPA": "èspéa",
    "VIP": "véipé",
    "FAQ": "èfaku",
}
_ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _expand_french_acronym(m: re.Match) -> str:
    word = m.group(0)
    return _FR_ACRONYM_OVERRIDES.get(word, word)


def _unit_plural(num_str: str, singular: str, plural: str) -> str:
    try:
        val = float(num_str.replace(",", "."))
    except ValueError:
        val = 2.0
    unit = singular if abs(val) <= 1.0 else plural
    if num_str == "1" and singular in ("milliseconde", "microseconde"):
        return f"une {unit}"
    return f"{num_str} {unit}"


def prepare_french_structure(text: str) -> str:
    """Pre-splitting structural cleanup (bullet lists & French guillemet spacing).

    Runs BEFORE ``sentences(finished(...))`` so bullet items become separate
    sentences and closing guillemets do not get a stray period, while keeping
    acronyms like ``RGPD.`` intact so ``sentences()`` splits after them instead
    of mistaking spaced letters (``R G P D.``) for a middle initial (``J. K.``).
    """
    if not text or not text.strip():
        return text
    lines = text.splitlines()
    if any(re.match(r"^\s*[-•*]\s+\S", ln) for ln in lines):
        rebuilt: list[str] = []
        for ln in lines:
            m = re.match(r"^\s*[-•*]\s+(.+?)\s*$", ln)
            if m:
                item = m.group(1)
                if item[-1:] not in ".!?…,:;":
                    item = item + "."
                rebuilt.append(item)
            else:
                rebuilt.append(ln)
        text = "\n".join(rebuilt)
    text = re.sub(r"([.!?…])\s+»", r"\1»", text)
    # Expand period-bearing abbreviations before sentence splitting (preserving case)
    # so elided forms like "l'art. 3" never falsely split a sentence at the period.
    text = re.sub(r"\bArt\.\s*(?=\d)", "Article ", text)
    text = re.sub(r"\bart\.\s*(?=\d)", "article ", text)
    text = re.sub(r"\bTél\.\s*", "Téléphone ", text)
    text = re.sub(r"\btél\.\s*", "téléphone ", text)
    return text


def prepare_french_tts_text(text: str) -> str:
    """Clean everyday French text at inference time without touching the frozen
    training frontend fingerprint (computing/metric units, fractions, French phone
    numbers, emails/URLs, bullet lists, abbreviations, acronyms, and guillemet spacing).
    """
    if not text or not text.strip():
        return text

    text = prepare_french_structure(text)

    # 2. Emails & URLs -> speakable French ("arobase", "point", "slash").
    def _email_sub(m: re.Match) -> str:
        user = m.group(1).replace(".", " point ").replace("_", " ").replace("-", " tiret ")
        dom = m.group(2).replace(".", " point ").replace("-", " tiret ")
        return f"{user} arobase {dom}"

    def _url_sub(m: re.Match) -> str:
        raw = m.group(1).rstrip("/")
        if "/" in raw:
            host, path = raw.split("/", 1)
            host_s = host.replace(".", " point ").replace("-", " tiret ")
            path_s = path.replace("/", " slash ").replace(".", " point ").replace("-", " ").replace("_", " ")
            return f"{host_s} slash {path_s}"
        return raw.replace(".", " point ").replace("-", " tiret ")

    text = _EMAIL_RE.sub(_email_sub, text)
    text = _URL_RE.sub(_url_sub, text)

    # 3. French 10-digit phone numbers ("06 12 34 56 78") -> paused digit pairs.
    text = _FR_PHONE_SPLIT_RE.sub(r"\1, \2, \3, \4, \5", text)

    # 4. Smart acronym/initialism detection (period abbreviations handled in prepare_french_structure).
    text = _ALL_CAPS_WORD_RE.sub(_expand_french_acronym, text)

    # 5. Computing & metric units ("725 Ko", "16 Go", "250 ms", "250 ml", "50 cl", "500 mg", "44 kHz").
    for pat, sing, plur in _UNITS_FR_EXT:
        text = pat.sub(lambda m, s=sing, p=plur: _unit_plural(m.group(1), s, p), text)

    # 6. Common fractions & multipliers ("1/2 litre", "3/4 d'heure", "2x plus vite").
    text = re.sub(r"\b1/2\s+heure\b", "une demi-heure", text, flags=re.IGNORECASE)
    text = re.sub(r"\b1/2\b", "un demi", text)
    text = re.sub(r"\b1/3\b", "un tiers", text)
    text = re.sub(r"\b2/3\b", "deux tiers", text)
    text = re.sub(r"\b1/4\b", "un quart", text)
    text = re.sub(r"\b3/4\b", "trois quarts", text)
    text = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*[xX]\b", r"\1 fois", text)

    # 7. Prosodic colons ("Téléphone : 06...", "Ingrédients : ...") -> comma so the
    # frozen host inserts a real ~280 ms pause instead of 0 frames (preserving 14:30).
    text = re.sub(r"(?<!\d)\s*:\s+(?=\S)", ", ", text)

    return text
