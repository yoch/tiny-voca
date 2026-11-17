"""Build French train/validation/test script manifests.

Sources:
* Optional plain-text sentence files (e.g. Mozilla Common Voice FR
  ``sentence-collector.txt`` / ``wiki.fr.txt``, CC0-1.0).
* Rich French templates covering numbers in words, prices, clock times, dates,
  names, questions (~25%) and exclamations (~10%).

Every script passes ``tiny_voca.text.phonemize`` with zero unknown phonemes,
contains no unexpanded digits/symbols/acronyms, and splits strictly by sentence
across train, validation, and test with zero overlap.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

from tiny_voca.text import normalize_text, phonemize

CV_FR_BASE = "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/fr"

OK = re.compile(r"^[A-ZÀ-ÖØ-ÞŒ][A-Za-zÀ-ÖØ-öø-ÿŒœ ,;'’\-«»]+[.?!]$")
FIRST = (
    "Claire Thomas Camille Lucas Léa Hugo Chloé Antoine Manon Julien Sarah Maxime "
    "Inès Nicolas Emma Romain Alice Alexandre Zoé Baptiste Jade Mathieu Lina Gabriel "
    "Nora Paul Amélie Samuel Eva Raphaël Yasmine Quentin Sofia Karim Anaïs Olivier"
).split()
LAST = (
    "Martin Bernard Thomas Petit Robert Richard Durand Dubois Moreau Laurent Simon "
    "Michel Lefèvre Leroy Roux David Bertrand Morel Fournier Girard Bonnet Mercier"
).split()
CITIES = (
    "Paris Lyon Marseille Toulouse Bordeaux Lille Nantes Strasbourg Nice Montpellier "
    "Rennes Grenoble Rouen Dijon Angers Annecy"
).split()
PLACES = [
    "à la bibliothèque", "à la gare", "à la pharmacie", "à la boulangerie",
    "au marché", "à l'aéroport", "au musée", "à la mairie", "au café", "à la librairie",
]
ITEMS = [
    "pommes", "cahiers", "billets", "enveloppes", "bougies", "livres", "bouteilles",
    "cartes postales", "croissants", "parapluies",
]
UNITS = (
    "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze "
    "quinze seize dix-sept dix-huit dix-neuf"
).split()
TENS = {2: "vingt", 3: "trente", 4: "quarante", 5: "cinquante", 6: "soixante"}
MONTHS = (
    "janvier février mars avril mai juin juillet août septembre octobre novembre décembre"
).split()
DAYS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
ORDINALS = [
    "premier", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    "douze", "quinze", "vingt", "vingt et un", "vingt-cinq", "trente",
]


def number_fr(n):
    if n < 20:
        return UNITS[n]
    if n < 70:
        t, u = divmod(n, 10)
        if u == 0:
            return TENS[t]
        if u == 1:
            return f"{TENS[t]} et un"
        return f"{TENS[t]}-{UNITS[u]}"
    if n < 80:
        u = n - 60
        return "soixante et onze" if u == 11 else f"soixante-{UNITS[u]}"
    if n < 100:
        u = n - 80
        return "quatre-vingts" if u == 0 else f"quatre-vingt-{UNITS[u]}"
    h, rest = divmod(n, 100)
    prefix = "cent" if h == 1 else f"{UNITS[h]} cents" if rest == 0 else f"{UNITS[h]} cent"
    return prefix if rest == 0 else f"{prefix} {number_fr(rest)}"


def clock_fr(rng):
    hour = rng.randint(1, 23)
    h_word = "une heure" if hour == 1 else f"{number_fr(hour)} heures"
    minute = rng.choice(["pile", "et quart", "et demie", "vingt", "quarante", "dix", "cinq"])
    return f"{h_word} {minute}".strip()


def templates(rng, count):
    name = lambda: rng.choice(FIRST)
    full = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    patterns = [
        lambda: f"Bonjour {name()}, comment s'est passée votre journée ?",
        lambda: f"{name()}, pourriez-vous appeler {full()} avant {clock_fr(rng)} ?",
        lambda: f"La réunion avec {full()} a été déplacée à {rng.choice(DAYS)} {rng.choice(ORDINALS)} {rng.choice(MONTHS)}.",
        lambda: f"Cela coûte {number_fr(rng.randint(2, 500))} euros et {number_fr(rng.randint(10, 95))} centimes, taxes comprises.",
        lambda: f"Il nous faut {number_fr(rng.randint(2, 40))} {rng.choice(ITEMS)}, {number_fr(rng.randint(2, 40))} {rng.choice(ITEMS)} et {number_fr(rng.randint(2, 40))} {rng.choice(ITEMS)}.",
        lambda: f"Est-ce que le train pour {rng.choice(CITIES)} part à {clock_fr(rng)}, ou a-t-il encore du retard ?",
        lambda: f"Je m'appelle {full()} et j'habite à environ {number_fr(rng.randint(2, 90))} kilomètres de {rng.choice(CITIES)}.",
        lambda: f"Est-ce que {name()} a vraiment lu les {number_fr(rng.randint(3, 30))} chapitres en {number_fr(rng.randint(2, 12))} jours ?",
        lambda: f"Merci d'apporter {number_fr(rng.randint(2, 20))} {rng.choice(ITEMS)} {rng.choice(PLACES)} avant {rng.choice(DAYS)}.",
        lambda: f"Le {rng.choice(ORDINALS)} {rng.choice(MONTHS)}, {name()} et {name()} partent en voiture vers {rng.choice(CITIES)}.",
        lambda: f"Cela fait {number_fr(rng.randint(120, 999))} mètres à pied, à quelques pas près !",
        lambda: f"Qui a dit à {name()} que le magasin fermait à {clock_fr(rng)} ?",
        lambda: f"Nous passons d'abord {rng.choice(PLACES)}, puis {rng.choice(PLACES)}, et enfin {rng.choice(PLACES)}.",
        lambda: f"{full()} a obtenu {number_fr(rng.randint(40, 99))} pour cent, soit {number_fr(rng.randint(2, 25))} points de plus que la dernière fois.",
        lambda: f"Attends, veux-tu dire que {name()} est parti il y a {number_fr(rng.randint(2, 50))} minutes ?",
        lambda: f"J'ai demandé à {name()}, {name()} et {name()} de nous retrouver {rng.choice(PLACES)}.",
        lambda: f"Pour cette recette, il faut {number_fr(rng.randint(2, 12))} œufs, du beurre doux et une pincée de sel.",
        lambda: f"Franchement, {name()}, je ne pensais pas voir autant de monde à {rng.choice(CITIES)} aujourd'hui !",
        lambda: f"Peux-tu me rappeler si {full()} préfère venir {rng.choice(DAYS)} ou {rng.choice(DAYS)} ?",
        lambda: f"Il a plu pendant {number_fr(rng.randint(2, 14))} jours à {rng.choice(CITIES)}, alors nous sommes restés lire à la maison.",
    ]
    out = set()
    attempts = 0
    while len(out) < count and attempts < count * 10:
        attempts += 1
        out.add(patterns[rng.randrange(len(patterns))]())
    return sorted(out)


def usable(line):
    line = normalize_text(line)
    if not OK.match(line) or not 25 <= len(line) <= 170 or not 5 <= len(line.split()) <= 30:
        return None
    if re.search(r"\b[A-ZÀ-ÖØ-ÞŒ]{2,}\b|--|\d|[,;:.!?]\s*[,;:.!?]|\s[,;.]", line):
        return None
    return line


def build_rows(candidates, total, rng, seen, origin, url, category):
    pool = sorted({s for s in (usable(l) for l in candidates) if s and s.lower() not in seen})
    rng.shuffle(pool)
    quota = {"?": round(0.25 * total), "!": round(0.10 * total)}
    quota["."] = total - sum(quota.values())
    taken = {"?": 0, "!": 0, ".": 0}
    rows = []
    for text in pool:
        if len(rows) == total:
            break
        end = text[-1]
        if taken[end] >= quota[end]:
            continue
        try:
            phones = phonemize(text)
        except Exception:
            continue
        if phones.unknown or len(phones) > 300:
            continue
        taken[end] += 1
        seen.add(text.lower())
        identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        rows.append({
            "id": identifier, "text": text, "original_text": text, "category": category, "origin": origin,
            "source_id": identifier, "source_url": url,
            "source_license": "cc0-1.0" if origin.startswith("common_voice") else "project-own",
            "group_id": f"{origin}:{identifier}", "split": "train", "llm_model": None, "llm_modified": False,
            "word_count": len(text.split()), "character_count": len(text), "audio_status": "not_generated",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[], help="Optional plain-text files (one sentence per line)")
    parser.add_argument("--fetch-common-voice", action="store_true", help="Fetch CC0 French sentence-collector.txt (and wiki-1.fr.txt with --with-wiki)")
    parser.add_argument("--with-wiki", action="store_true", help="Also fetch wiki-1.fr.txt when --fetch-common-voice is set")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw_text_fr"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--validation-count", type=int, default=250)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--template-share", type=float, default=0.2, help="Fraction of scripts drawn from French templates when --inputs/--fetch-common-voice are given")
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    total = args.train_count + args.validation_count + args.test_count
    ext_lines = []
    if args.fetch_common_voice:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        files = ["sentence-collector.txt"] + (["wiki-1.fr.txt"] if args.with_wiki else [])
        for name in files:
            target = args.cache_dir / name
            if not target.exists():
                urllib.request.urlretrieve(f"{CV_FR_BASE}/{name}", target)
            ext_lines.extend(target.read_text(encoding="utf-8", errors="ignore").splitlines())
    for path in args.inputs:
        ext_lines.extend(path.read_text(encoding="utf-8", errors="ignore").splitlines())

    if ext_lines:
        tmpl_target = round(total * args.template_share)
        ext_target = total - tmpl_target
        ext_rows = build_rows(ext_lines, ext_target, rng, seen, "common_voice_fr", "https://commonvoice.mozilla.org/", "reading")
        tmpl_rows = build_rows(templates(rng, max(60, (total - len(ext_rows)) * 10)), total - len(ext_rows), rng, seen, "tiny_voca_fr_templates", "local", "pronunciation")
        rows = ext_rows + tmpl_rows
    else:
        rows = build_rows(templates(rng, max(60, total * 10)), total, rng, seen, "tiny_voca_fr_templates", "local", "pronunciation")

    rng.shuffle(rows)
    splits = {
        "validation": rows[:args.validation_count],
        "test": rows[args.validation_count:args.validation_count + args.test_count],
        "train": rows[args.validation_count + args.test_count:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, items in splits.items():
        for r in items:
            r["split"] = split
        items.sort(key=lambda r: (r["character_count"], r["id"]))
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items),
            encoding="utf-8",
        )
        summary[split] = len(items)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
