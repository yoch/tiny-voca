#!/usr/bin/env python3
"""Semi-automated French TTS defect hunter (100% local CPU, 0 GPU).

Scans a diverse real-world French stress bank (+ validation.jsonl static/ONNX scan)
using three automatic detectors:
  1. Static frontend & IPA detector (<unk>, missed sentence splits, silent -t on
     -gnent, unexpanded units/abbreviations, suspicious acronym phonemization).
  2. ONNX duration detector (0-frame punctuation pauses on ';', ':', '—', ',',
     unexpected <sp> pauses, clipped/non-finite audio).
  3. Round-trip ASR detector (tiny-voca-fr.onnx -> faster-whisper small CER/WER)
     to catch mispronunciations that sound wrong to human ears.

Outputs a concise, deduplicated top-K triage batch (WAVs + suspects.json) ready
for rapid human review in scripts/triage_ui.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import wave

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.evaluate import normalize_for_wer
from tiny_voca.infer import finished, sentences, shorter
from tiny_voca.prosody_fr import prepare_french_structure, prepare_french_tts_text
from tiny_voca.text import NothingToPronounce, phonemize

# Diverse everyday French stress bank: each tuple is (id, category, voice_idx, text).
REAL_WORLD_STRESS_BANK = [
    ("abbr_mme_dr_pr", "abbreviations", 0, "Le Pr Moreau, Me Lefèvre et le Dr Petit se retrouvent av. Victor-Hugo, bât. B, esc. 2."),
    ("abbr_etc_cf_nb", "abbreviations", 1, "N.B. : voir chap. 4, fig. 2, p. 18 et suiv., c.-à-d. environ 300 hab./km²."),
    ("abbr_dates_short", "dates", 0, "Réunion prévue mar. 14 janv. à 8h45, puis ven. 3 févr. à 18h00."),
    ("units_physics_tech", "units", 1, "La batterie délivre 12 V, 2,5 A, soit 30 W, avec un bruit de 42 dB et 500 Mo/s."),
    ("units_area_vol", "units", 0, "Un appartement de 65 m², un jardin de 3 ha, 150 cm de haut et 12 mm d'épaisseur."),
    ("units_medical", "units", 1, "Prenez 500 mg de paracétamol, 5 µg de vitamine D et 10 ml de sirop matin et soir."),
    ("units_currency_big", "currency", 0, "Le budget atteint 2,4 M€, soit 2 400 000 € ou environ 2,6 M$."),
    ("acronyms_admin_fr", "acronyms", 1, "La CPAM, la CNIL, la DGCCRF, le CROUS, le SMIC et le RSA sont concernés."),
    ("acronyms_tech_web", "acronyms", 0, "Envoyez le JSON via HTTP, ouvrez le QR code en Wi-Fi, et vérifiez le SSL et l'IP."),
    ("acronyms_bank_id", "acronyms", 1, "Merci de joindre votre RIB, votre IBAN, votre CNI et un justificatif EDF."),
    ("acronyms_media_trans", "acronyms", 0, "La RATP, le RER A, le TER, la SNCF, l'AFP, le JT et la TNT fonctionnent normalement."),
    ("acronyms_intl", "acronyms", 1, "L'OTAN, l'OCDE, le FMI, la BCE, le G7 et la COP28 se réunissent à Genève."),
    ("homograph_couvent", "homographs", 0, "Les poules du couvent couvent sous le vent d'est."),
    ("homograph_fils_os", "homographs", 1, "Son fils a cassé des fils électriques et trouvé un os, puis plusieurs os."),
    ("homograph_plus", "homographs", 0, "J'en veux plus, mais il n'y en a plus ! De plus, c'est plus cher qu'avant."),
    ("homograph_tous", "homographs", 1, "Ils sont tous venus, tous les deux, et tous ont applaudi."),
    ("homograph_president", "homographs", 0, "Les présidents président la séance et nos portions de gâteau que nous portions sont prêtes."),
    ("homograph_est_soit", "homographs", 1, "À l'est de la ville, soit on tourne à droite, soit il fait demi-tour."),
    ("homograph_sens_vis", "homographs", 0, "Je sens que ça a du sens ; tu vis ici et tu cherches une vis."),
    ("homograph_but_fait", "homographs", 1, "En fait, il a marqué un but au bout du match, le 15 août et le 3 mars."),
    ("numbers_six_huit_dix", "numbers", 0, "Il reste six pommes, huit amis, dix euros, et nous sommes six ou dix."),
    ("numbers_vingt_cent", "numbers", 1, "Vingt personnes, vingt-et-un ans, quatre-vingts pages, quatre-vingt-un livres et deux cents euros."),
    ("numbers_ordinal_roman", "numbers", 0, "Au XXIe siècle, François Ier et Louis XVI figurent au chapitre III, tome II."),
    ("numbers_percent_range", "numbers", 1, "Entre 15 et 20 % des élèves ont obtenu 14/20 ou 18/20 à l'examen."),
    ("numbers_zip_phone", "numbers", 0, "Habite 75011 Paris ou 69002 Lyon ; appelez le +33 6 12 34 56 78."),
    ("liaisons_h_aspire", "liaisons", 1, "Les haricots, les héros, les hiboux, en haut, des hêtres et les hôtels."),
    ("liaisons_tricky", "liaisons", 0, "Un grand homme, neuf ans, neuf heures, un savant anglais et un pied-à-terre."),
    ("silent_letters_tricky", "lexicon", 1, "L'automne, un monsieur, un fusil, un outil, un tabac, un estomac, un porc et un cerf."),
    ("silent_letters_verb", "lexicon", 0, "Les clients gagnent du temps, les médecins soignent bien et les enfants craignent le noir."),
    ("loanwords_everyday", "lexicon", 1, "Un week-end en streaming, un podcast sur smartphone, un e-mail et du feedback en meeting."),
    ("punct_semicolon_dash", "punctuation", 0, "Première idée ; deuxième idée — beaucoup plus importante — et enfin la conclusion."),
    ("punct_parens_quotes", "punctuation", 1, "Le projet (lancé en 2024) s'appelle « Petite Voix » ; qu'en pensez-vous ?"),
    ("punct_ellipsis_interro", "punctuation", 0, "Attends… tu es sûr ?! Oui… enfin, peut-être pas !"),
    ("symbols_math_misc", "symbols", 1, "Note : A & B coûtent 10€ ± 2€, soit ~12€ par personne (#42, 100% garanti)."),
    ("recipe_full", "real_life", 0, "Recette :\n- 250 g de farine\n- 1/4 L de lait\n- 2 c. à s. d'huile\n- 1 pincée de sel"),
    ("weather_bulletin", "real_life", 1, "Météo : -2°C à Strasbourg, 5°C à Lille, 12°C à Bordeaux, rafales à 75 km/h."),
    ("legal_notice", "real_life", 0, "Conformément à l'art. 12 du RGPD, contactez dpo@entreprise.fr avant le 31/12/2026."),
    ("proper_nouns_fr", "proper_nouns", 1, "De Reims à Metz, puis Chamonix, Auxerre, Bruxelles, Caen, Nantes et Saint-Étienne."),
    ("proper_nouns_world", "proper_nouns", 0, "Voyage entre New York, Londres, Tokyo, Pékin, Munich et Barcelone."),
    ("dialogue_natural", "dialogue", 1, "— Tu viens demain ? — Non, j'ai rendez-vous à 10h30 chez le dentiste !"),
]

# Batch 2: focused on proper nouns, homographs, silent letters, liaisons,
# abbreviated dates/ranges, everyday tech/English loanwords, numbered lists & dialogues.
BATCH2_STRESS_BANK = [
    ("b2_cities_fr_1", "proper_nouns", 0, "De Metz à Chamonix, puis Auxerre, Bruxelles, Caen et Reims."),
    ("b2_cities_fr_2", "proper_nouns", 1, "Bourg-en-Bresse, Aix-en-Provence, Laon, Mulhouse, Troyes, Vannes et Avoriaz."),
    ("b2_cities_world", "proper_nouns", 0, "Voyage entre New York, Londres, Tokyo, Pékin, Munich et Barcelone."),
    ("b2_homograph_couvent", "homographs", 1, "Les poules du couvent couvent sous le vent d'est."),
    ("b2_homograph_plus_tous", "homographs", 0, "J'en veux plus, mais il n'y en a plus ! Ils sont tous venus, tous les deux."),
    ("b2_homograph_president", "homographs", 1, "Les présidents président la séance ; nos portions de gâteau que nous portions sont prêtes."),
    ("b2_homograph_fils_os", "homographs", 0, "Son fils a cassé des fils électriques et trouvé un os, puis plusieurs os."),
    ("b2_silent_nouns", "lexicon", 1, "L'automne, un fusil, un outil, un tabac, un estomac, un porc, un cerf et des œufs."),
    ("b2_final_s_tricky", "lexicon", 0, "Jadis, un ours mangeait du maïs, un ananas et du cassis dans le bus."),
    ("b2_gnent_verbs", "lexicon", 1, "Les clients gagnent du temps, les médecins soignent bien et les enfants craignent le noir."),
    ("b2_liaisons_h", "liaisons", 0, "Les haricots, les héros, les hiboux, en haut, des hêtres et les hôtels."),
    ("b2_liaisons_num", "liaisons", 1, "Un grand homme, neuf ans, neuf heures, vingt-et-un ans, quatre-vingt-un livres."),
    ("b2_dates_abbr", "dates", 0, "Réunion lun. 5 janv., mar. 14 févr., mer. 3 avr., ven. 12 sept. et dim. 1er déc."),
    ("b2_ranges_hours", "dates", 1, "Ouvert du lundi au vendredi de 9h à 17h30, période 2020-2026, dossier n° 42."),
    ("b2_loanwords_web", "loanwords", 0, "Un week-end en streaming, un podcast sur smartphone, un e-mail, du feedback et une newsletter."),
    ("b2_loanwords_dev", "loanwords", 1, "Le projet open source sur GitHub utilise Python, JavaScript, le cloud et le Bluetooth."),
    ("b2_numbered_list", "formatting", 0, "Étapes :\n1. Ouvrir l'application\n2. Cliquer sur démarrer\n3. Valider le résultat"),
    ("b2_dialogue_quotes", "dialogue", 1, "— Tu viens demain ? — Non, j'ai rendez-vous ! « Dommage… une prochaine fois ? »"),
]


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _run_frontend(text: str, max_tokens: int = 240):
    structured = prepare_french_structure(text)
    sents = sentences(finished(structured))
    utterances = []
    limit = max_tokens - 12

    def fit(piece: str):
        clean = prepare_french_tts_text(piece)
        try:
            ph = phonemize(clean)
        except NothingToPronounce:
            return
        if len(ph) <= limit:
            utterances.append({"raw": piece, "clean": clean, "ph": ph, "ipa": " ".join(ph.symbols)})
            return
        parts = shorter(clean)
        if not parts:
            return
        for p in parts:
            fit(p)

    for s in sents:
        fit(s)
    return utterances


def _static_flags(text: str, utterances: list[dict]) -> list[str]:
    flags = []
    joined_clean = " ".join(u["clean"] for u in utterances)
    joined_ipa = " ".join(u["ipa"] for u in utterances)

    # 1. Unknown phonemes
    unks = [u["ph"].unknown for u in utterances if u["ph"].unknown]
    if unks:
        flags.append(f"unknown_phones:{unks}")

    # 2. Suspicious leftover silent-t on -gnent verbs (eSpeak bug: gagnent -> g a n j t)
    if re.search(r"n j t\b", joined_ipa):
        flags.append("espeak_gnent_silent_t")

    # 3. Unexpanded symbols (&, ~, ±, #, /, +33, °, M€, M$, c. à s., hab./km², N.B., c.-à-d.)
    for sym, label in (
        (r"&", "unexpanded_ampersand"),
        (r"±", "unexpanded_plusminus"),
        (r"~", "unexpanded_tilde"),
        (r"#\d+", "unexpanded_hash"),
        (r"\b\d+/\d+\b", "unexpanded_slash_fraction"),
        (r"\+\d{2}\s+\d", "unexpanded_intl_phone"),
        (r"\b\d+(?:[.,]\d+)?\s*(?:V|A|W|dB|ha|µg|Mo/s|M€|M\$)\b", "unexpanded_unit"),
        (r"\b(?:Pr|Me|bât\.|esc\.|suiv\.|N\.B\.|c\.-à-d\.|c\.\s*à\s*s\.|mar\.|janv\.|ven\.|févr\.)\b", "unexpanded_abbr"),
    ):
        if re.search(sym, joined_clean):
            flags.append(label)

    # 4. Missed sentence split (". Capital" inside one utterance after cleanup)
    for u in utterances:
        if re.search(r"[.!?]\s+[A-ZÀ-ÖØ-ÞŒ]", u["clean"]):
            flags.append("missed_sentence_split")
            break

    # 5. Suspicious ALL-CAPS acronyms remaining in clean text
    for acr in re.findall(r"\b[A-Z]{2,6}\b", joined_clean):
        if acr not in ("ONU", "OTAN", "NASA", "UNESCO", "SMIC", "CAF", "URSSAF", "CROUS", "SNCF", "RATP", "TGV", "TVA", "RGPD", "CNRS", "ADN", "PDF", "LLM", "SMS", "GPS", "HLM", "CGT", "CFDT", "DVD", "TV", "MM", "XIV", "XVI", "III", "II"):
            flags.append(f"unreviewed_acronym:{acr}")

    return sorted(set(flags))


def _synthesize_onnx(sess: ort.InferenceSession, utterances: list[dict], voice: int, sr: int):
    chunks = []
    dur_flags = []
    gap = np.zeros(int(round(sr * 0.22)), dtype=np.float32)

    for ui, u in enumerate(utterances):
        ph = u["ph"]
        audio, durs = sess.run(None, {
            "tokens": np.array([ph.ids], dtype=np.int64),
            "stress": np.array([ph.stress], dtype=np.int64),
            "voice": np.array([voice], dtype=np.int64),
            "speed": np.array([1.0], dtype=np.float32),
            "pitch_shift": np.array([0.0], dtype=np.float32),
            "pitch_range": np.array([1.0], dtype=np.float32),
        })
        wav = audio[0].astype(np.float32)
        if not np.isfinite(wav).all():
            dur_flags.append("non_finite_audio")
            wav = np.nan_to_num(wav)
        chunks.append(wav)
        if ui + 1 < len(utterances):
            chunks.append(gap)

        # Check punctuation pause drops & unexpected <sp> pauses
        for sym, d in zip(ph.symbols, durs[0]):
            di = int(d)
            if sym in (";", "—", ":") and di == 0:
                dur_flags.append(f"zero_pause_on_{sym}")
            if sym == "<sp>" and di > 0:
                dur_flags.append(f"sp_pause_{di}f")

    full = np.concatenate(chunks) if chunks else np.zeros(sr // 4, dtype=np.float32)
    peak = float(np.max(np.abs(full))) if len(full) else 0.0
    if peak > 0.99:
        full = full * (0.99 / peak)
    return full, sorted(set(dur_flags))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", choices=("batch1", "batch2"), default="batch1")
    ap.add_argument("--onnx", type=Path, default=Path(".local-runs/p7_refiner/onnx/tiny-voca-fr.onnx"))
    ap.add_argument("--val-manifest", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01/validation.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path(".local-runs/fr_hunter/batch_01"))
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-val-scan", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    meta = json.loads(args.onnx.with_suffix(".json").read_text(encoding="utf-8"))
    sr = int(meta.get("sample_rate", 24000))
    max_tokens = int(meta.get("max_tokens", 240))
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    bank = REAL_WORLD_STRESS_BANK if args.bank == "batch1" else BATCH2_STRESS_BANK
    candidates = []
    for cid, cat, voice, text in bank:
        utts = _run_frontend(text, max_tokens=max_tokens)
        sflags = _static_flags(text, utts)
        wav, dflags = _synthesize_onnx(sess, utts, voice, sr)
        candidates.append({
            "id": cid,
            "source": args.bank,
            "category": cat,
            "voice": meta["voices"][voice] if voice < len(meta.get("voices", [])) else str(voice),
            "voice_idx": voice,
            "text": text,
            "cleaned_text": "\n".join(u["clean"] for u in utts),
            "ipa": "\n".join(f"[{i+1}] {u['ipa']}" for i, u in enumerate(utts)),
            "n_sentences": len(utts),
            "static_flags": sflags,
            "dur_flags": dflags,
            "wav": wav,
        })

    # 2. Quick static scan of validation.jsonl (373 real sentences) for any extra corpus bugs
    val_hits = []
    if not args.skip_val_scan and args.val_manifest.exists():
        seen_hash = set()
        for line in args.val_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["text_hash"] in seen_hash:
                continue
            seen_hash.add(r["text_hash"])
            utts = _run_frontend(r["text"], max_tokens=max_tokens)
            sflags = _static_flags(r["text"], utts)
            if sflags:
                voice = 0 if r["speaker"] == "F1" else 1
                wav, dflags = _synthesize_onnx(sess, utts, voice, sr)
                val_hits.append({
                    "id": f"val_{r['id']}",
                    "source": "validation",
                    "category": "corpus_scan",
                    "voice": r["speaker"],
                    "voice_idx": voice,
                    "text": r["text"],
                    "cleaned_text": "\n".join(u["clean"] for u in utts),
                    "ipa": "\n".join(f"[{i+1}] {u['ipa']}" for i, u in enumerate(utts)),
                    "n_sentences": len(utts),
                    "static_flags": sflags,
                    "dur_flags": dflags,
                    "wav": wav,
                })
        print(f"Static corpus scan on validation.jsonl ({len(seen_hash)} unique texts): {len(val_hits)} flagged", flush=True)
        # Keep up to 4 diverse validation hits
        candidates.extend(val_hits[:4])

    # 3. Round-trip ASR with faster-whisper small (CPU int8)
    if not args.skip_asr:
        from faster_whisper import WhisperModel
        import jiwer
        import librosa

        print(f"Running round-trip faster-whisper small ASR on {len(candidates)} candidates...", flush=True)
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        for c in candidates:
            wav16 = librosa.resample(c["wav"], orig_sr=sr, target_sr=16000)
            segs, _ = whisper.transcribe(wav16, language="fr", beam_size=5, vad_filter=False)
            hyp = " ".join(s.text.strip() for s in segs).strip()
            rn = normalize_for_wer(c["cleaned_text"], language="fr")
            hn = normalize_for_wer(hyp, language="fr")
            c["whisper_hyp"] = hyp
            c["cer"] = round(float(jiwer.cer([rn], [hn])) if rn else 0.0, 4)
            c["wer"] = round(float(jiwer.wer([rn], [hn])) if rn else 0.0, 4)
    else:
        for c in candidates:
            c["whisper_hyp"] = ""
            c["cer"] = 0.0
            c["wer"] = 0.0

    # 4. Score & select top-K DIVERSE suspects (max 2 per category so every slot teaches something new!)
    for c in candidates:
        score = (
            1.2 * len(c["static_flags"])
            + 1.5 * len(c["dur_flags"])
            + 4.0 * c["cer"]
            + 1.0 * c["wer"]
        )
        c["suspect_score"] = round(score, 3)

    ranked = sorted(candidates, key=lambda x: x["suspect_score"], reverse=True)
    selected = []
    cat_counts: dict[str, int] = {}
    for c in ranked:
        if len(selected) >= args.top_k:
            break
        cat = c["category"]
        if cat_counts.get(cat, 0) >= 2:
            continue
        selected.append(c)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # 5. Write triage batch (WAVs + suspects.json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_records = []
    for idx, c in enumerate(selected, start=1):
        wav_name = f"S{idx:02d}_{c['id']}.wav"
        _write_wav(args.out_dir / wav_name, c["wav"], sr)
        rec = {k: v for k, v in c.items() if k != "wav"}
        rec["suspect_id"] = f"S{idx:02d}"
        rec["wav_file"] = wav_name
        out_records.append(rec)
        print(
            f"  {rec['suspect_id']} [{rec['category']:13s}] score={rec['suspect_score']:5.2f} "
            f"CER={rec['cer']*100:4.1f}% static={rec['static_flags']} dur={rec['dur_flags']}",
            flush=True,
        )

    manifest = {
        "onnx": str(args.onnx),
        "total_scanned": len(candidates),
        "selected_count": len(out_records),
        "elapsed_seconds": round(time.time() - t0, 1),
        "suspects": out_records,
    }
    (args.out_dir / "suspects.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also save full scan summary for inspection
    (args.out_dir / "all_scanned.json").write_text(
        json.dumps([{k: v for k, v in c.items() if k != "wav"} for c in ranked], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(out_records)} diverse suspects to {args.out_dir} in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
