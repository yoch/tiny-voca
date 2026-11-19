# P6 — Pré-enregistrement, vérifications pré-GPU et protocole du replay apparié E06 (`8k → 12k`)

- **Statut** : **PRÉ-ENREGISTRÉ AVANT TOUT LANCEMENT GPU** *(solde Colab vérifié avant ouverture VM : `72,34` unités, `0` session active)*
- **Branche / dépôt** : `fr` (`yoch/tiny-voca`) — `main` inchangée, `fr_ref_35k` conservé comme référence active, réglages par défaut inchangés (`--prosody-grad-scale 1.0`, `pitch_shape = 0.0`, `--separate-prosody-clip` inactif par défaut)
- **Checkpoint source commun exact** : `.local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt`
  - `step = 8 000`, architecture `e06_160_384` (`2 657 051` params)
  - SHA-256 : `8308f2d90c657bfe5b458805f2fdd08f531375a37c3490160d1578405b1eefc1`
  - Contient l'état complet : `system`, `critics`, `optimizer`, `critic_optimizer`, `sampler`, `torch_rng`, `cuda_rng`, et les empreintes exactes `datasets` de `fr4k/annotations_e01/{train,validation}.jsonl`.

---

## 1. Vérifications pré-GPU (100 % CPU)

### 1.1 Sonde `AdamW` 1 pas à bruit de validation figé (`val_seed = 999001`) : rampe réelle (`1/3000`) vs pleine intensité (`1,0`)

Exécutée via [audit_p5_adamw_displacement.py](../../scripts/audit_p5_adamw_displacement.py) avec graine de bruit vocodeur **strictement identique** avant/après le pas et entre toutes les variantes (`torch.manual_seed(999001)` avant chaque appel `_eval_batch`) sur `e06_20k/step-00008000.pt` (`step = 8 000`) :

| Variante (`e06_8k`, `step=8 000`) | Rampe réelle (`1/3000`) : `rawG` / `clip` | Rampe réelle : `relΔθ` (`pros` / `pHead` / `voc`) | Rampe réelle : `valΔ f0_st` / `spreadR` / `mel` | Pleine intensité (`ramp=1,0`) : `rawG` / `clip` | Pleine intensité : `relΔθ` (`pros` / `pHead` / `voc`) | Pleine intensité : `valΔ f0_st` / `spreadR` / `mel` |
|---|---|---|---|---|---|---|
| `baseline_gan_on` (**C**) | `15,90` / `0,629` | `1,37e-3` / `8,58e-4` / `2,65e-3` | `+0,0062` / `-0,0036` / `+0,00112` | `3 465,7` / `0,00289` | `1,02e-3` / `6,07e-4` / `2,78e-3` | **`+0,0411`** / `-0,0067` / `+0,00428` |
| `delayed_gan_off` (**D**) | `16,40` / `0,610` | `1,34e-3` / `8,42e-4` / `2,60e-3` | `+0,0066` / `-0,0033` / `+0,00119` | *(sans GAN)* | *(sans GAN)* | *(sans GAN)* |
| `grad_gate_g02` (`α=0,2`) | `15,80` / `0,633` | `9,75e-4` / `8,61e-4` / `2,66e-3` | `+0,0069` / `-0,0043` / `+0,00097` | `3 457,3` / `0,00289` | `8,36e-4` / `6,07e-4` / `2,79e-3` | `+0,0383` / `-0,0041` / `+0,00391` |
| `grad_gate_g00` (`α=0,0`) | `15,80` / `0,633` | `9,57e-4` / `8,61e-4` / `2,66e-3` | `+0,0071` / `-0,0045` / `+0,00088` | `3 457,0` / `0,00289` | `8,38e-4` / `6,07e-4` / `2,79e-3` | `+0,0377` / `-0,0035` / `+0,00383` |
| `sep_clip_prosody` *(seul)* | `15,90` / `p=1,0, a=0,659` | `1,87e-3` / `1,18e-3` / `2,72e-3` | `+0,0093` / `-0,0037` / `+0,00180` | `3 465,7` / `p=0,041, a=0,0029` | **`4,97e-3`** / `6,00e-4` / `2,79e-3` | **`+0,0745`** / **`-0,0314`** / `+0,01013` |
| **`detach_plus_sep_clip` (P)** | `15,80` / `p=1,0, a=0,659` | `1,12e-3` / `1,18e-3` / `2,72e-3` | `+0,0081` / `-0,0050` / `+0,00133` | `3 457,0` / **`p=1,000`**, `a=0,0029` | **`1,12e-3`** / **`1,18e-3`** / `2,79e-3` | **`+0,0081`** / `-0,0050` / **`+0,00337`** |

*Note `fr_ref_35k` (`step = 35 000`, même bruit figé `val_seed = 999001`) : `baseline_gan_on` `valΔ(f0_st, mel) = (-0,0046, +0,00006)` ; `delayed_gan_off` `(-0,0057, -0,00056)` ; `grad_gate_g02` `(-0,0050, +0,00029)` ; `detach_plus_sep_clip` `(-0,0032, +0,00040)`. À la rampe réelle `step = 8 000` (`1/3000`), le GAN est encore à `0,033 %` d'intensité (pas quasi neutre entre bras) ; c'est à pleine intensité (représentative de la fin de rampe `10k–11k`) que le double verrou apparaît et que `P` maintient exactement le même `valΔ f0_st = +0,0081 st` qu'en rampe `1/3000` (zéro choc GAN dans la branche prosodique) tout en donnant le meilleur `valΔ mel = +0,00337`.*

### 1.2 Vérification CPU de reprise exacte et d'alignement RNG (`C`, `D`, `P`)

Implémentée dans [train.py](../../src/tiny_voca/train.py) via trois drapeaux opt-in (tous inactifs par défaut) :
- `--restore-state-from-init` : restaure `system`, `critics`, `optimizer`, `critic_optimizer`, `sampler`, `torch_rng`, `cuda_rng` et conserve le calendrier LR d'origine (`keep_schedule = True`) comme `--resume`, tout en autorisant les surcharges de recette du replay apparié.
- `--separate-prosody-clip` : clippe séparément `(encoder, embeddings, duration, prosody_stack, prosody_head)` et `(acoustic_stack, prosody_projection, vocoder, mel_head)` au seuil `--grad-clip`.
- `--align-gan-rng` : lorsque `step < gan_start` (bras `D`), consomme le même tirage `torch.rand(batch_size, crops, device=...)` que `crops()` afin que les suites aléatoires (dropout, teacher forcing, phase, bruit vocodeur) restent bit-à-bit alignées entre `C`, `D` et `P`.

Vérifié sur le vrai checkpoint `e06_20k/step-00008000.pt` via [verify_p6_cpu_replay.py](../../scripts/verify_p6_cpu_replay.py) (2 pas réels `8 000 → 8 002`) et par les 4 tests unitaires de [test_p6.py](../../tests/test_p6.py) :

| Vérification CPU sur `e06_20k/step-00008000.pt` | Résultat observé |
|---|---|
| Correspondance empreintes `datasets` (`train` + `validation`) | `True` |
| Présence `cuda_rng` dans le checkpoint | `True` |
| `max|θ_system(C) − θ_system(--resume)|` après 2 pas | **`0,0`** *(bit-exact)* |
| `max|θ_critics(C) − θ_critics(--resume)|` après 2 pas | **`0,0`** *(bit-exact)* |
| Identité `torch_rng`, `sampler` et métriques (`C` vs `--resume`) | **`True` / `True` / `True`** |
| Alignement bit-exact `torch_rng`, `batches` et `teacher_fraction` (`D` vs `C`) | **`True` / `True` / `True`** |
| Alignement bit-exact `torch_rng`, `batches` et `teacher_fraction` (`P` vs `C`) | **`True` / `True` / `True`** |
| Pas `8 000` : `grad_norm_prosody` vs `grad_norm_acoustic` dans `P` | `3,30` *(non clippé)* vs `34,47` *(clippé à `10`)* — vs `C` clippé globalement (`34,86`) |

---

## 2. Plan expérimental unique : replay E06 apparié `8k → 11k → 12k`

Tous les trois bras repartent exactement du même `e06_20k/step-00008000.pt` avec `--restore-state-from-init`, mêmes minibatches, même calendrier de teacher forcing (`teacher_hold = 3000`, `teacher_ramp = 7000`, `teacher_final = 0.0`) et mêmes suites aléatoires alignées :

| Bras | Recette entre `8 000` et `12 000` | Contraste causal |
|---|---|---|
| **C** | Recette E06 standard : GAN à `8 000`, rampe `3 000` pas, `alpha = 1,0`, clip global `10` | Témoin commun |
| **D** | Identique à C, mais `--gan-start 999999 --align-gan-rng` (GAN différé après `12k`) | **`D − C`** : effet cumulé du calendrier GAN sur `8k → 12k` |
| **P** | Même GAN et même rampe que C, mais `--prosody-grad-scale 0.0 --separate-prosody-clip` dès `8k` | **`P − C`** : effet de la protection combinée (détachement + clip séparé) |

---

## 3. Critères pré-enregistrés et marges de régression acceptables (figés avant GPU)

Évaluation **100 % hors GPU** sur CPU local via [eval_p6_replay.py](../../scripts/eval_p6_replay.py) à `12k` (ou `11k` si palier budgétaire déclenché) :

### 3.1 Critère principal (`P4-dev`, `40` enregistrements / `20` textes)

- **Protocole** : injecter successivement la seule F0 prédite par `C`, `D` et `P` dans le même hôte figé `fr_ref_35k`, sur les **mêmes durées prédites de `35k`**, avec énergie, voisement et bruit identiques.
- **Métrique principale** : **corrélation F0 mesurée dans le WAV sur masque commun (`common_mask_f0_corr`)**, avec IC 95 % par bootstrap apparié au niveau du texte stratifié par catégorie (`F1`/`M1` groupés par texte, `n_boot = 5 000`), accompagnée de la couverture du masque commun, `common_mask_f0_mae_st` et `common_mask_slope_mae_st_s` pour `D − C` et `P − C`.
- **Condition de gain principal pour `P` vs `C`** :
  1. `IC95_bas(Δ common_mask_f0_corr, P − C) > 0,0` (gain positif statistiquement significatif) ;
  2. `Δ common_mask_f0_mae_st (P − C) <= +0,05 st` (pas de dégradation MAE sur le même masque).

### 3.2 Contrôles et marges de non-régression sur les WAV natifs (`P` vs `C` au même pas)

| Métrique native (`P4-dev` ou `validation.jsonl` questions) | Marge acceptable figée (`P − C`) |
|---|---|
| `Mel-DTW` (free-running, `P4-dev`) | `Δ <= +0,015` **et** `IC95_haut <= +0,025` |
| `STFT` (durées oracle, `P4-dev`) | `Δ <= +0,050` |
| `WER` (`faster-whisper small` FR, `P4-dev`) | `Δ <= +0,020` (`+2,0` pts) |
| `CER` (`faster-whisper small` FR, `P4-dev`) | `Δ <= +0,010` (`+1,0` pt) |
| Questions `totales` (`validation.jsonl`, `n=50`, `F1`/`M1`) | `Δ dirAcc >= -0,050` (`-5,0` pts max) ; inflexion terminale `totales` et `partielles` publiée par voix |

### 3.3 Règle de décision et garde sur les `121` paires vierges (`test.jsonl`)

1. **Si `P` satisfait le critère principal sur `P4-dev` ET toutes les marges natives ci-dessus** → exécuter **une seule confirmation** sur les `121` paires strictement vierges ([p5_untouched_test_reserve.json](p5_untouched_test_reserve.json)).
2. **Si seul `D` progresse contre `C`** (et `P` échoue sur critère principal ou marges natives) → **ne pas toucher aux `121` paires vierges** ; conclure que le calendrier GAN dégrade bien la prosodie sur ce replay et que la protection `P` reste à revoir.
3. **Dans tous les cas** : un résultat sur cette seule graine et largeur `E06` (`160/384`) **ne remplace pas `fr_ref_35k`**.

---

## 4. Discipline budgétaire Colab `L4` (plafond strict `<= 4,5` u.)

- **Estimation (débits E06 publiés, `1,94 u./h`)** : `C + P` (`2 × 4 000` pas à `1,30 sps`) ≈ `3,32 u.`, `D` (`4 000` pas à `9,39 sps`) ≈ `0,23 u.`, transferts/opérations (`~15 min`) ≈ `0,49 u.` → **`4,04 u.` estimées**.
- **Palier obligatoire à `11k`** ([run_p6_colab.py](../../scripts/run_p6_colab.py)) :
  1. Exécuter d'abord `stage1_11k` (`8k → 11k`) pour les 3 bras `D`, `C`, `P` et sauvegarder `inference_11000.pt` + `last.pt` pour tous les trois.
  2. Vérifier la consommation réelle `colab usage` à l'issue de `11k`.
  3. Lancer `stage2_12k` (`11k → 12k`, `~0,89 u.`) **uniquement** si la marge restante sous `4,5 u.` le permet ; sinon retenir `11k` pour les trois bras, rapatrier l'archive et arrêter la VM immédiatement.
- **Évaluation 100 % hors GPU** après arrêt confirmé de la VM (`0` session active).

---

## 5. Kit d'écoute aveugle P5 corrigé (sans clé)

- Fiche corrigée : `REF.wav` identifié comme **audio issu du système enseignant Qwen3-TTS** (`scripts/generate_audio.py`), dépouillement explicitement séparé en **Bloc 1 (`12` comparaisons donneur `8k` / `35k`)** et **Bloc 2 (`4` comparaisons oscillateur / décodeur)**, **`0` jugement humain** recueilli à ce stade.
- Archive prête sans clé (`45` fichiers : fiche + `44` WAV, `0` fichier clé) : `.local-runs/p5_audit/f0_causal/blind_listening_p5_no_key.zip`.
