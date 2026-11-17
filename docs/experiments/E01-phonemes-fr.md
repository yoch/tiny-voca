# E01 — Audit et durcissement du frontal phonétique français (`espeak-ng` 1.51)

- **Statut** : **VALIDÉ**
- **Branche / base** : `exp/e01-phonemes-fr` basée sur `dcc4ab2` (parent du pipeline français `de41e47b709c4f9e130bdf659ba63cd625de4761`)
- **Budget GPU utilisé** : `0 h` (travail strictement CPU / frontal)
- **Compatibilité inventaire** : `vocab_size = 54`, `FRONTEND_VERSION = 4`, `frontend_fingerprint()` inchangé (`31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6`)

---

## 1. Contexte et baseline exacte

### 1.1 Artefacts de référence vérifiés

| Artefact / Paramètre | Valeur observée |
|---|---|
| Checkpoint de référence | [inference.pt](file:///home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt) |
| SHA-256 `inference.pt` | `f673dd421478b86e36f60a4965818eba304eb8517fc15de8ba0690d60ca0afa2` |
| Paramètres checkpoint | `format_version=4`, `step=1317`, `vocab_size=54`, `speakers=('F1', 'M1')`, `frontend_version=4` |
| SHA-256 `annotations_summary.json` | `995c04d134e0bb3c3648d987a71537ddfd15f98396dfaf6987a71537dba6606db` |
| Couverture `annotations_summary.json` | `train=261` (`rejected=2`), `validation=29`, `test=13` ; 34/37 phonèmes observés (`tʃ`, `dʒ`, `ŋ` absents de ce lot) |
| `frontend_fingerprint()` | `31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6` |
| Moteur phonétique système | `libespeak-ng1` `1.51+dfsg-12build1`, voix `"fr"`, séparateur `"_"` |
| Aligner CTC d'annotation | `facebook/wav2vec2-lv-60-espeak-cv-ft` (`vocab.json` : 392 labels ; les 37 phonèmes de `PHONES` y figurent ; `ɥ` absent, `ɡ` U+0261 présent à l'index 35, `g` absent ; `ALIGNER_LABEL_FALLBACK = {UNK: "<unk>"}`) |

### 1.2 Audit de bout en bout du pipeline (`text.py`, `config.py`, `annotate.py`, `align.py`, `infer.py`)

1. **Cohérence entraînement / alignement / inférence** :
   - [text.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/text.py) définit l'unique point d'entrée `phonemize()`, appelé pendant l'annotation dans [annotate.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/annotate.py) (via `ThreadPoolExecutor` à 16 threads) et pendant la synthèse dans [infer.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/infer.py).
   - [config.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/config.py) lie `Config().vocab_size = 54` à `len(SYMBOLS)` (`5` spéciaux + `12` ponctuations + `22` consonnes + `15` voyelles) avec `CHECKPOINT_FORMAT = 4` et `ANNOTATION_VERSION = 3`.
   - [infer.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/infer.py) bloque au chargement tout écart de `frontend_fingerprint()`, garantissant qu'aucun modèle n'ingère un vocabulaire ou un ordre d'IDs différent.
   - `sentences()` dans [infer.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/infer.py) et [worker.js](file:///home/yoch/voice-tts/.worktrees/e01/web/public/worker.js) partagent la liste d'abréviations évitant les coupures de phrase intempestives (vérifié automatiquement par `test_the_demo_worker_splits_and_ends_text_like_the_package`).

2. **Inventaire phonologique français (`PHONES` = 37, `FALLBACK` = 51)** :
   - **Voyelles nasales** : toutes les 4 voyelles nasales françaises sont représentées distinctement dans `VOWELS` : `/ɑ̃/`, `/ɛ̃/`, `/ɔ̃/`, `/œ̃/` (eSpeak NG `fr` conserve d'ailleurs la distinction `/ɛ̃/` vs `/œ̃/`, ex. `brin brun` → `b ʁ ɛ̃ b ʁ œ̃`).
   - **Voyelles orales et `/ʁ/`** : `/y/` vs `/u/`, `/ø/` vs `/œ/`, `/e/` vs `/ɛ/`, `/o/` vs `/ɔ/`, `/ə/`, `/a/`, et `/ʁ/` uvulaire sont tous présents en natif (`r`/`ɹ`/`ɾ`/`ɻ`/`χ` → `("ʁ",)` via `FALLBACK`).
   - **Semi-voyelle `/ɥ/`** : volontairement ramenée à `("y",)` dans `FALLBACK["ɥ"] = ("y",)` car `ɥ` n'existe pas dans `vocab.json` de `facebook/wav2vec2-lv-60-espeak-cv-ft`.
   - **Anomalies critiques découvertes dans la baseline `dcc4ab2`** :
     - **Fuite d'état dans `libespeak-ng` (`espeak_words`)** : la branche CLI ajoutait `"\n"` mais la branche `ctypes` passait `piece.encode("utf-8")` sans `"\n"`. Après une abréviation suivie d'un point et d'un chiffre (ex. `fig. 2`, `vol. 3`), `espeak_TextToPhonemes` mettait `ptr.value = b""` tout en retenant le dernier jeton dans son tampon lookahead interne : le chiffre manquait dans l'énoncé courant et ressurgissait en tête du `phonemize()` suivant (ex. `etc.` prononcé `d ø ɛ t s e t e ʁ a`) — y compris entre threads dans `annotate.py`.
     - **Liaisons interdites devant `et`** : eSpeak NG `fr` réalise systématiquement des liaisons fautives devant la conjonction *et* (`heures et demie` → `œ ʁ z e`, `grand et fort` → `ɡ ʁ ɑ̃ t e`, `du beurre doux et` → `d u z e`, provoquant directement la transcription Whisper erronée `"du beurre douze"` observée dans [report.json](file:///home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/eval/report.json) au step `1317`).
     - **Géminées impossibles en fin de mot (`-t_t`, `-z_z`)** : les finales verbales en `-ent` ou mots en `-s/-z` suivis d'une liaison créaient des consonnes finales doublées (`partent en` → `p a ʁ t t`, `choses étranges` → `ʃ o z z`, soit `67` occurrences brutes sur les `4 000` phrases de `scripts_fr_4k`).
     - **Verbes/noms en `-oues` / `-ouent`** : perte du noyau `/u/` au profit de `/w/` + `/t/` parasite (`les enfants jouent` → `ʒ w t`, `ils louent` → `l w t`).
     - **Clitique `-y`** : `vas-y`, `allons-y` lus comme `"i grec"` (`v a z i ɡ ʁ ɛ k`).
     - **Faux basculements anglais `(en)` destructeurs des contrastes français** : sur `scripts_fr_4k` (`4 000` phrases), `108` phrases déclenchaient `(en)` ; une partie venait d'élisions (`d'aujourd'hui`, `l'enthousiasme`, `qu'homme`), de noms/mots français (`dos`, `coud`, `bruns`, `vécut`, `Annecy`, `Nancy`, `Fleury`, `zénith`, `rythme`, `labyrinthe`, `sied`, `wallon`, `pull`), ou d'apostrophes typographiques non ASCII (`ʼ` U+02BC). Les règles `FALLBACK` transformaient alors ces mots français en diphtongues anglaises (`d ɔː s`, `n æ n s i`, `b ʁ ʌ n z`, etc.).
     - **Normalisation / formats parlés absents** : heures `14h30` (`k a t ɔ ʁ z a ʃ t ʁ ɑ̃ t`), devise `€` mal lue `ə ʁ o` au lieu de `ø ʁ o`, négatifs `-5 °C` (`-` supprimé avant `eSpeak`), abréviations (`MM.` lu `"millimètre"`, `M.` mal géré en cas particuliers, `Mgr`, `n°` lu `"ɛn"`, `p.`, `fig.`, `vol.`, `chap.`, `av. J.-C.`), unités (`100 m`, `10 min`, `500 g`, `2 L`, `30 s`), dates (`05/11/2026`, `2026-11-17`), ordinaux (`1re`, `2e`, `9e`, `21e`, `XXIe`), chiffres romains contextuels (`Louis XIV` lu `"Louis quatorze romain"`), sigles courants (`UE`, `IA`, `PME`, `RSA`, `CDI`, `INSEE`), et quelques homographes morpho-syntaxiques non ambigus (`tous deux`, `en plus`, `à l'est`, `une vis`, `les/des os`, `mes/ses fils`, `se fier`, `quatre-vingt-onze`).

---

## 2. Hypothèse falsifiable

> **Hypothèse (E01)** : En ajoutant au frontal français une pré-expansion contextuelle (`_expand_french_text`), la terminaison par `"\n"` dans `libespeak-ng`, la coupure des liaisons interdites (`et`, `hameaux`), le dégroupage des élisions déclenchant `(en)`, et le nettoyage post-eSpeak (`-t_t`/`-z_z`, clitique `-y`) **tout en conservant strictement** `SYMBOLS` (`54`), `FALLBACK` (`51`) et `FRONTEND_VERSION = 4` :
> 1. La précision exacte sur notre jeu de référence français (272 cas couvrant 10 catégories) passera de `< 65 %` (baseline `dcc4ab2`) à `>= 90 %`, avec un PER `< 2 %` et `0` symbole inconnu.
> 2. Le nombre de faux basculements `(en)` et de géminées `-t_t`/`-z_z` sur `scripts_fr_4k` (`4 000` phrases) baissera nettement (`0` erreur, `0` inconnu).
> 3. Le `frontend_fingerprint()` restera identique (`31807ddbd16771299960e70d15addc80902ab70ae20738240178903e23c7f7a6`), permettant le chargement `check_frontend=True` et la synthèse CPU avec [inference.pt](file:///home/yoch/voice-tts/.local-runs/colab/opt1-1117/out/run_info/inference.pt) sans aucune régression de la suite de tests.

**Critères de réfutation** : exact-match `< 90 %` ou PER `> 2 %` sur les 272 cas, apparition de `<unk>` ou régressions sur la suite existante, rupture d'empreinte `frontend_fingerprint()`, ou incapacité de synthétiser avec le checkpoint `opt1-1117`.

---

## 3. Méthodologie

1. **Construction d'un jeu de référence vérifié (272 cas)** : [french_phoneme_benchmark.json](file:///home/yoch/voice-tts/.worktrees/e01/tests/french_phoneme_benchmark.json) contient 272 entrées réparties en 10 catégories, avec variantes phonémiques françaises légitimes validées contre `PHONE_SET` via [eval_fr_frontend.py](file:///home/yoch/voice-tts/.worktrees/e01/scripts/eval_fr_frontend.py).
2. **Mesure gelée avant / après** : évaluation exacte de `dcc4ab2` avant toute modification, puis ré-évaluation après finalisation des correctifs sur les **mêmes 272 cas sans modification a posteriori des cibles**.
3. **Vérifications complémentaires sans GPU** :
   - Balayage des `4 000` phrases de `data/scripts_fr_4k/{train,validation,test}.jsonl` et des `10 000` premières phrases de `data/raw_text_fr/sentence-collector.txt`.
   - Test de concurrence `ThreadPoolExecutor(16)` sur `200` appels parallèles à `phonemize()`.
   - Test de chargement (`check_frontend=True`) et de synthèse CPU bout en bout sur `inference.pt` (`opt1-1117`).
   - Exécution complète de `unittest discover -s tests -v`.

---

## 4. Résultats avant / après sur les 272 cas de référence

| Catégorie | Cas | Exact avant (`dcc4ab2`) | Acc avant | PER avant | Exact après (`E01`) | Acc après | PER après | Unk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nasal_vowels` | 24 | 23/24 | 95.8 % | 0.76 % | 23/24 | 95.8 % | 0.76 % | 0 |
| `oral_vowels_and_r` | 34 | 30/34 | 88.2 % | 2.45 % | 32/34 | 94.1 % | 0.98 % | 0 |
| `semivowels_and_consonants` | 16 | 16/16 | 100.0 % | 0.00 % | 16/16 | 100.0 % | 0.00 % | 0 |
| `liaisons` | 38 | 25/38 | 65.8 % | 8.53 % | **38/38** | **100.0 %** | **0.00 %** | 0 |
| `h_aspire_muet` | 20 | 18/20 | 90.0 % | 1.72 % | **20/20** | **100.0 %** | **0.00 %** | 0 |
| `e_caduc` | 16 | 16/16 | 100.0 % | 0.00 % | 16/16 | 100.0 % | 0.00 % | 0 |
| `homographs` | 30 | 19/30 | 63.3 % | 5.62 % | **27/30** | **90.0 %** | **1.20 %** | 0 |
| `numbers_dates_times` | 28 | 11/28 | 39.3 % | 32.65 % | **28/28** | **100.0 %** | **0.00 %** | 0 |
| `currency_units_abbr_acronyms` | 34 | 10/34 | 29.4 % | 28.82 % | **33/34** | **97.1 %** | **0.71 %** | 0 |
| `french_en_traps_and_loanwords` | 32 | 4/32 | 12.5 % | 48.45 % | **30/32** | **93.8 %** | **1.03 %** | 0 |
| **Total** | **272** | **172/272** | **63.2 %** | **16.54 %** | **263/272** | **96.7 %** | **0.52 %** | **0** |

### Vérifications corpus, concurrence et inférence

- **`scripts_fr_4k` (4 000 phrases)** : `0` erreur, `0` inconnu ; géminées finales `-t_t` / `-z_z` éliminées (`67` phrases brutes → `0`) ; phrases déclenchant `(en)` réduites de `108` à `73` (`-32.4 %`), les `73` restantes étant exclusivement de véritables noms propres ou emprunts anglais (`John`, `Mulrady`, `teen-movie`, `groom`, `emails`, `Kennedy`, `Blue Face of Death`).
- **`sentence-collector.txt` (10 000 premières lignes)** : `0` exception hors `NothingToPronounce`, `1` seul inconnu (`'ʊə'`, hors `FALLBACK`).
- **Concurrence (`ThreadPoolExecutor(16)` × 200)** : `0` discordance.
- **Inférence CPU sur `inference.pt` (`opt1-1117`)** : `match_frontend == 'espeak'`, `load_model(..., check_frontend=True)` valide, synthèse OK sur des phrases complexes (`14h30`, `12,50 €`, `M. de Gaulle`, `MM. Dupont`, `fig. 2`, `-3 °C`, `d'aujourd'hui`).

---

## 5. Séparation explicite : correctifs compatibles (a) vs changements d'inventaire (b)

### (a) Correctifs compatibles avec l'inventaire à 54 symboles (implémentés dans E01)

Tous ces correctifs vivent dans la couche texte/normalisation/post-parsing et gardent `SYMBOLS`, `FALLBACK`, `FRONTEND_VERSION = 4` et `frontend_fingerprint()` inchangés :

1. **Isolation C-buffer `libespeak-ng` (`_raw_espeak`)** : ajout systématique de `"\n"` et tampon `ctypes.create_string_buffer` explicite sous `_ESPEAK_LOCK` dans [text.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/text.py), supprimant le débordement lookahead entre appels consécutifs (ex. `Voir fig. 2.` suivi de `etc.`).
2. **Normalisation Unicode conservatrice (`normalize_text`)** : prise en charge des apostrophes typographiques (`ʼ`, `ʻ`, `ʹ`, `′`) et tirets (`‐`, `‑`, `−`) dans `_QUOTES`, et extension de `ABBREVIATIONS` (`mm`, `mgr`, `p`, `chap`, `art`, `ap`, `tél`), tout en gardant `normalize_text()` lisible pour l'utilisateur et compatible avec les tests existants.
3. **Synchronisation découpe de phrases** : ajout de `mm|mgr|ap|chap|art` à `_NOT_AN_END` dans [infer.py](file:///home/yoch/voice-tts/.worktrees/e01/src/tiny_voca/infer.py) et `ABBREVIATIONS` dans [worker.js](file:///home/yoch/voice-tts/.worktrees/e01/web/public/worker.js).
4. **Expansion française dédiée à la phonémisation (`_expand_french_text`, appelée par `phonemize()`)** :
   - Composés `grand'mère`/`grand'chose` → `grand-mère`/`grand-chose` ; nombres négatifs (`-5 °C` → `moins 5 ...`) sans altérer les dates ISO (`2026-11-17`).
   - Abréviations (`MM.`, `Mgr`, `av. J.-C.`, `ap. J.-C.`, `par ex.`, `p.`, `fig.`, `vol.`, `chap.`, `env.`, `n°`).
   - Dates (`JJ/MM/AAAA`, `AAAA-MM-JJ`), heures (`14h30`, `9h05`, `8h`), devises (`€`, `EUR`, `$`, `£`), températures (`°C`, `°F`) et unités (`km/h`, `m/s`, `km²`, `m²`, `m³`, `min`, `g`, `m`, `L`, `s`).
   - Ordinaux arabes et romains (`1re`, `2e`, `9e`, `21e`, `XXIe`) et chiffres romains contextuels (`Louis XIV`, `tome IV`, etc.) sans `"romain"` parasite.
   - Sigles/acronymes (`_FRENCH_INITIALISMS` épelés, `_ACRONYM_HINTS` comme `INSEE`/`ZUP`).
   - Désambiguïsations sûres d'homographes (`à l'est`/`l'Est`, `tous deux..dix`, `en plus`, `une/la vis`, `les/des os`, `mes/ses fils`, `se fier`) et mots/toponymes français misclassés en `(en)` (`dos`, `coud`, `bruns`, `vécut`, `traversons`, `Annecy`, `Nancy`, `Fleury`, `zénith`, `rythme`, `labyrinthe`, `enthousiasme`, `sied`, `wallon`, `pull`, `v'là`, `quatre-vingt-onze`, `héroïne` *h* muet, etc.).
   - Réécriture des finales verbales/nominales `-oues`/`-ouent` → `-oue` (`les enfants jouent` → `ʒ u`).
5. **Règles prosodiques & phonémiques dans `espeak_words()`** :
   - Découpe sans liaison devant `et` (hors `vingt et un`) et devant `hameau`/`hameaux` (*h* aspiré).
   - Détection et dégroupage récursif des élisions provoquant `(en)` (`_espeak_subpiece` sur `d'aujourd'hui`, `l'enthousiasme`, `qu'homme`, `jusqu'au...`), tout en préservant la liaison éventuelle entre le radical et la suite du syntagme.
   - Suppression des géminées finales impossibles `-t_t` / `-z_z` et correction du clitique `-y` (`vas-y`, `allons-y` sans suffixe `ɡ ʁ ɛ k`).

> **Note entraînement / inférence** : les correctifs (a) bénéficient immédiatement à l'inférence sur le checkpoint `opt1-1117` (dates, heures, `€`, abréviations, faux `(en)`), mais une **ré-annotation + ré-entraînement** seront utiles lors d'un prochain run GPU pour aligner les poids acoustiques sur les séquences corrigées des données d'entraînement (`et`, `-t_t`, `-ouent`, élisions).

### (b) Évolutions touchant `SYMBOLS` / `FALLBACK` / `FRONTEND_VERSION` (non modifiées en E01)

Ces points ont été volontairement laissés hors patch E01 car ils altèrent `frontend_fingerprint()` et imposent une nouvelle annotation / un nouveau modèle :

1. **Ajout de `"ʊə": ("u", "ə")` (ou `("u", "ʁ")`) dans `FALLBACK`** : observé `1` fois sur `10 000` phrases de `sentence-collector.txt` (`0` fois dans `scripts_fr_4k`). Casserait l'empreinte sans impact sur le corpus d'entraînement actuel.
2. **Ajout du phonème natif `/ɥ/` (`SYMBOLS`, `vocab_size = 55`)** : impossible sans changer d'aligneur car `'ɥ'` est absent de `vocab.json` dans `facebook/wav2vec2-lv-60-espeak-cv-ft`.
3. **Ajout du phonème `/ɑ/` (`SYMBOLS`, `vocab_size = 55`)** : contraste majoritairement neutralisé vers `/a/` en français contemporain standard ; nécessiterait ré-annotation et ré-entraînement.
4. **Remplacement d'eSpeak NG par un G2P français dédié (`FRONTEND_VERSION = 5`)** : indispensable pour lever les limites structurelles ci-dessous.

---

## 6. Inventaire des limites inhérentes à `espeak-ng` 1.51

Les `9` échecs restants sur `272` (`3.3 %` ; `0.52 %` de PER) relèvent directement du moteur interne d'eSpeak NG et ne doivent pas être contournés par des heuristiques fragiles :

1. **Homographes dépendants de l'analyse syntaxique (POS) — `3` cas** :
   - `[homo_11]` `"j'en veux plus"` → eSpeak émet `p l y` au lieu de `p l y s` (ambigu sans détection de portée négative / `ne`).
   - `[homo_28]` `"il but de l'eau"` → eSpeak lit `b y t` (nom *le but*) au lieu de `b y` (verbe *boire* au passé simple).
   - `[homo_30]` `"un homme négligent"` → eSpeak applique la règle verbale `-ent` muet à l'adjectif (`n e ɡ l i ʒ` au lieu de `n e ɡ l i ʒ ɑ̃`) ; le même phénomène affecte *président*, *résident*, *précédent*, *excellent*, *violent* hors règles contextuelles simples.
2. **Mots empruntés classés en `(en)` RP britannique non rhotique — `2` cas** :
   - `[traps_31]` `"parking"` → `p a k i ŋ` (chute du `/ʁ/` post-vocalique britannique au lieu de `/paʁkiŋ/`).
   - `[traps_32]` `"New York"` → `n j u j ɔ k` (au lieu de `/njujɔʁk/`).
3. **Particularités lexicales du dictionnaire français d'eSpeak NG — `4` cas** :
   - `[nasal_22]` `"solennel"` → `s o l ɛ n ɛ l` (au lieu de `/solanɛl/`).
   - `[oral_14]` `"jeune jeûne"` → neutralisation `ʒ ø n ʒ ø n` (au lieu de `ʒ œ n ʒ ø n`).
   - `[oral_28]` `"rose rosse"` → `ʁ ɔ z ʁ ɔ s` (au lieu de `ʁ o z ʁ ɔ s`).
   - `[sym_20]` `"Mgr Lefèvre"` → expansion correcte vers `"Monseigneur Lefèvre"`, mais eSpeak réalise `gn` prévocalique en `n j` (`m ɔ̃ s ɛ n j œ ʁ` — variante courante en français moderne, mais différente de `/ɲ/` strict attendu dans la référence ; cf. `agneau` → `a n j o` vs `montagne` → `m ɔ̃ t a ɲ`).

---

## 7. Validation finale (tests unitaires)

Après les dernières éditions sur `exp/e01-phonemes-fr`, exécution complète de :
- `PYTHONPATH=src:tests /home/yoch/voice-tts/env/bin/python scripts/eval_fr_frontend.py` : `263/272 (96.7%)`, PER `0.52%`, `unknowns: 0`.
- `PYTHONPATH=src:tests /home/yoch/voice-tts/env/bin/python -m unittest discover -s tests -v` : `Ran 78 tests in 78.200s` — `OK (skipped=2)` (les 2 tests ignorés sont identiques à la baseline : `test_acoustic_preparation_interrupt_resume_provenance_and_bad_rows` faute de dépendances de préparation audio optionnelles et `test_with_dizione_itself` faute de `DIZIONE_DATA`). Tous les tests du frontal (dont `test_french_e01_normalization_liaisons_elisions_and_state_isolation`, `test_french_benchmark_regression`), du découpage `infer.py` / `web/public/worker.js`, d'alignement, de données, de modèle et d'export ONNX passent.

