"""Build the French "gap" scripts: the cases the fr4k + fr20h texts lack (short utterances, ; and :, quotations and
dialogue, yes/no questions, exclamations, sentences over 20 words, numbers), with no overlap with any earlier text or
the comparison protocol. Phonemized with the current front end (no unknown symbol, at most --max-phones tokens).

Sources: the Common Voice sentence collector (CC0) and, for long sentences, the SIWIS prompts (CC BY 4.0,
Yamagishi et al. 2017, doi:10.7488/ds/1705), plus the project templates for numbers.

usage: PYTHONPATH=src python scripts/build_text_fr_gaps.py --collector data/raw_text_fr/sentence-collector.txt \
         --siwis SIWIS/other/all_prompts.txt --seen data/scripts_fr_4k data/scripts_fr20h_extra --protocol PROTOCOL.json \
         --output data/scripts_fr_gaps/train.jsonl
"""
import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tiny_voca.text import NothingToPronounce, phonemize  # noqa: E402

QUOTAS = {"short_turns": 200, "short": 500, "semicolon_colon": 500, "quotes_dialogue": 300, "yes_no_question": 500, "exclamation": 200,
          "long": 700, "numbers": 300}
SENTENCE = re.compile(r"^[«\"—–-]?\s*[A-ZÀ-ÖØ-ÞŒ«\"][A-Za-zÀ-ÖØ-öø-ÿŒœ ,;:'’\-«»\"—–…]+[.?!…»\"]$")
YES_NO = re.compile(r"^(?:Est-ce qu|[A-ZÀ-ÖØ-ÞŒ][a-zà-öø-ÿœ]+-(?:t-)?(?:je|tu|il|elle|on|nous|vous|ils|elles)\b|"
                    r"(?:As|Avez|Es|Êtes|Est|Ont|Sont|Peux|Pouvez|Veux|Voulez|Sais|Savez|Faut|Va|Vas|Allez)\b)")
# Very short everyday turns (1-3 words), rare in the collector.
SHORT_EXTRA = """Oui. | Non. | Oui ! | Non ! | Si. | Merci. | Merci ! | Merci beaucoup. | De rien. | Je vous en prie. | D'accord. |
D'accord ! | Bien sûr. | Bien sûr ! | Pas du tout. | Peut-être. | Sans doute. | Évidemment. | Absolument ! | Exactement. |
Bonjour ! | Bonjour. | Bonsoir ! | Salut ! | Au revoir. | Au revoir ! | À demain ! | À bientôt ! | À plus tard. | Bonne nuit. |
Bonne journée ! | Bonne soirée ! | Bon appétit ! | Bonne chance ! | Félicitations ! | Bravo ! | Joyeux anniversaire ! |
Pardon ? | Pardon. | Excusez-moi. | Désolé. | Désolée. | Je suis désolé. | Pas de problème. | Aucun problème. |
Comment ? | Pourquoi ? | Quand ? | Où ça ? | Qui ça ? | Combien ? | Lequel ? | Laquelle ? | Vraiment ? | Ah bon ? |
Tu es sûr ? | Vous êtes sûre ? | C'est vrai ? | Et alors ? | Et toi ? | Et vous ? | Ça va ? | Ça va. | Ça va bien. |
Très bien. | Pas mal. | Comme ci, comme ça. | Allô ? | Allô, oui ? | Entrez ! | Attendez ! | Attention ! | Stop ! | Au secours ! |
Vite ! | Doucement. | Silence ! | Regarde ! | Écoute. | Écoutez ! | Venez ! | Allons-y ! | C'est parti ! | On y va ? |
Voilà. | Voilà ! | Tiens ! | Tenez. | Et voilà ! | C'est fini. | C'est tout. | C'est ça. | C'est ça ? | Pas encore. |
Pas maintenant. | Plus tard. | Tout de suite. | Jamais ! | Toujours. | Parfois. | Souvent. | Rarement. | Ensuite ? |
Enfin ! | Hélas. | Tant pis. | Tant mieux ! | Quelle chance ! | Quelle horreur ! | Quel dommage. | Super ! | Génial ! |
Formidable ! | Incroyable ! | Pas question ! | Je ne sais pas. | Je sais. | Je comprends. | Compris. | Entendu. | Volontiers. |
Avec plaisir. | Pourquoi pas ? | Bonne idée ! | Mauvaise idée. | Moi aussi. | Moi non plus. | Moi ? | Lui ? | Elle ? |
Nous ? | Personne. | Rien. | Rien du tout. | Tout. | Assez ! | Encore ? | Encore une fois. | Une seconde. | Un instant. |
Deux minutes. | Trois fois. | Le lundi. | Demain matin. | Ce soir. | Hier. | Aujourd'hui. | Maintenant ! | Ici. | Là-bas."""
CAPS = re.compile(r"\b[A-ZÀ-ÖØ-ÞŒ]{2,}\b")


def clean(line):
    t = unicodedata.normalize("NFC", line.strip()).replace("’", "'")
    return re.sub(r"\s+", " ", t)


def category(t):
    words = len(t.split())
    if re.search(r"\d", t):
        return None
    if words > 20:
        return "long"
    if words <= 5:
        return "short"
    if t.endswith("?") and YES_NO.match(t.lstrip("«\"—–- ")):
        return "yes_no_question"
    if ";" in t or ":" in t:
        return "semicolon_colon"
    if "«" in t or '"' in t or t.lstrip()[:1] in "—–-":
        return "quotes_dialogue"
    if t.endswith("!"):
        return "exclamation"
    return None


def usable(t):
    if not SENTENCE.match(t) or CAPS.search(t) or not 2 <= len(t) <= 400:
        return False
    if re.search(r"--|[,;:.!?]\s*[,;:!?]|\s[,.]|\(|\)|\[", t):
        return False
    return t.count("«") == t.count("»") and t.count('"') % 2 == 0   # no quotation fragment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector", type=Path, required=True)
    ap.add_argument("--siwis", type=Path, required=True)
    ap.add_argument("--seen", type=Path, nargs="+", required=True, help="Script folders whose texts are excluded")
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--max-phones", type=int, default=450)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import build_text_fr as B  # templates for numbers (same as the earlier sets)
    seen = set()
    for folder in args.seen:
        for f in folder.glob("*.jsonl"):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                seen.update((r["text"].lower(), r.get("original_text", r["text"]).lower()))
    for s in json.loads(args.protocol.read_text(encoding="utf-8"))["sets"].values():
        seen.update(t.lower() for t in s["texts"])

    rng = random.Random(args.seed)
    pools = {k: [] for k in QUOTAS}
    collector = [clean(l) for l in args.collector.read_text(encoding="utf-8").splitlines() if l.strip()]
    for t in collector:
        c = category(t)
        if c and c != "long" and usable(t):
            pools[c].append((t, "common_voice_fr", "cc0-1.0", "https://commonvoice.mozilla.org/"))
    short_extra = [clean(t) for t in SHORT_EXTRA.split("|") if t.strip()]
    pools["short_turns"] = [(t, "tiny_voca_fr_short_turns", "local", "local") for t in short_extra]
    for line in args.siwis.read_text(encoding="utf-8", errors="replace").splitlines():
        t = clean(line.split("\t", 1)[-1])
        if category(t) == "long" and usable(t):
            pools["long"].append((t, "siwis_prompts", "cc-by-4.0", "https://doi.org/10.7488/ds/1705"))
    pools["numbers"] = [(t, "tiny_voca_fr_templates", "local", "local") for t in B.templates(rng, QUOTAS["numbers"] * 10)]

    rows, report = [], {}
    for cat, quota in QUOTAS.items():
        pool = pools[cat]
        rng.shuffle(pool)
        got = 0
        for t, origin, lic, url in pool:
            if got == quota:
                break
            if t.lower() in seen:
                continue
            try:
                ph = phonemize(t)
            except (NothingToPronounce, ValueError):
                continue
            if ph.unknown or len(ph.symbols) > args.max_phones:
                continue
            seen.add(t.lower())
            uid = hashlib.sha256(t.encode()).hexdigest()[:16]
            rows.append({"id": uid, "text": t, "split": "train", "category": f"gap_{cat}", "origin": origin,
                         "source_id": uid, "source_license": lic, "source_url": url, "word_count": len(t.split()),
                         "character_count": len(t), "audio_status": "not_generated"})
            got += 1
        report[cat] = {"quota": quota, "kept": got, "pool": len(pool)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "categories": report}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
