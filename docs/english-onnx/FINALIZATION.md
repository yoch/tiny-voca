# Finalisation des deux ONNX anglais

La revue est partie de la PR #2 à `a1f1dbe3f443a1b049302fcd8a6c3ff5bb849833`,
sur la base `db851f75c6287a15be4c4e995f931bce3a86b47b`. Le diff complet
comptait 33 fichiers, sans revue, commentaire ou check distant signalé.
Les 20 tests ciblés du code d’origine ont repassé localement.

| Candidat | Décision | Motif |
| --- | --- | --- |
| `exact.onnx` | **READY FOR HF** | FP32, fidélité > 140 dB CPU/WASM et package vérifié. |
| `exact-int8.onnx` | **EXPERIMENTAL** | Candidat compact recommandé, preuves objectives acquises et retour humain A/B en attente. |
| `matmulola-int8` | **REJECT** | Équivalence avec l’INT8 actuel, gain trop irrégulier sur le profil moyen/long. |
| `reconstruct512` et `reconstruct512-int8` | **EXPERIMENTAL** | −5,43 % de paramètres, écart spectral et équivalence perceptuelle insuffisante. Recherche hors paquet. |

## Données difficiles et écoute aveugle

Le jeu additionnel contient douze textes distincts du criblage et du lot
réservé d’origine. Trois textes par groupe couvrent les phrases courtes,
les consonnes, l’intonation et les phrases longues. Les deux voix donnent
24 paires. Les changements de vitesse, de hauteur et d’amplitude de
l’intonation sont fixés dans [le jeu source](release-texts.json).
Les données objectives et les WAV utilisent le bruit commun par cas.
Les WAV d’une même paire reçoivent un gain commun de normalisation.

Sur ces 24 cas, le FP32 et l’INT8 préservent toutes les durées, tous les
nombres d’échantillons et les valeurs finies. Le SNR minimal du FP32 est
de **143,56 dB** sur CPU. L’INT8 donne un RMS log-mel médian de **0,150 dB**
(p95 0,228 dB), et un p05 de cosinus WeSpeaker de **0,99896**. L’ASR
sur ce petit lot passe de 18 à 8 erreurs de mots sur 260 pour l’INT8.
Sur un seul cas long M2, le décodeur répète une partie de la phrase
de référence mais cesse de la répéter aussi avec le FP32 exact. Les
comptes ASR du lot difficile ne démontrent donc pas un gain
d’intelligibilité. Le résultat utile reste l’absence de dégradation
objective observée, y compris sur les 100 textes réservés antérieurs.
[Mesures par rendu](raw/finalization-quality.json.gz)

Le kit anonymise 24 paires et équilibre l’INT8 entre A et B par voix
(six de chaque côté pour chacune). La préférence et la présence d’un
défaut majeur se saisissent séparément. La clé reste hors du ZIP et
hors du dépôt. Une réponse humaine reste nécessaire pour interpréter
l’équivalence perceptuelle. Les divergences majeures propres au
candidat et toute préférence nette pour le public sont à examiner par
voix, groupe et type de défaut.

## Régressions courtes et essai matmulola

La vérification à quatre threads Chromium utilise six rendus courts de
0,64 à 0,99 s, cinq échauffements et 20 chronos par cas et par ordre.
Cinq cas sur six montrent une baisse de médiane dans les deux ordres.
Pour `short1-F2`, la médiane passe de 21,50 à 27,61 ms en aller,
puis de 26,45 à 23,40 ms en retour. Le p95 augmente sur quatre des douze
comparaisons d’ordre et de cas, avec un maximum de +45,4 %. Le gain
sur textes courts dépend donc de la charge et de l’ordre. [Chronos
courts](raw/finalization-short-web.json.gz)

L’essai unique compare `matmulola-int8` à `exact-int8` sur les deux voix
pour 2,53–2,61 s et 5,79–6,34 s. Deux ordres inversés, cinq échauffements
et 15 mesures CPU ou 12 Web sont utilisés par cas et par ordre.
La barre de décision est un gain médian additionnel d’au moins 5 %
pour les quatre cas dans les deux ordres sur un même profil
(runtime et nombre de threads), sans p95 supérieur de plus de 5 %
et avec SNR supérieur à 100 dB à bruit commun.

| Profil | Gain médian incrémental sur les quatre cas et deux ordres |
| --- | ---: |
| CPU, 1 thread | −7,5 à +2,3 % |
| CPU, 4 threads | +1,3 à +11,4 % |
| Chromium, 1 thread | −1,2 à +4,2 % |
| Chromium, 4 threads | −4,0 à +13,4 % |

La fidélité passe le seuil numérique (SNR minimal de 132,96 dB sur
24 cas CPU et 133,08 dB en Web sur les quatre cas à chaque nombre de
threads). Aucun profil ne passe le critère de vitesse. En Web à quatre
threads, trois cas gagnent dans les deux ordres, tandis que
`cons2-F2` inverse son résultat et `long2-M2` gagne seulement 1,8 %
en retour. La piste est arrêtée après cet essai. [Chronos CPU](raw/finalization-ola-cpu.json.gz) · [Chronos Web](raw/finalization-ola-web.json.gz) · [Parité](raw/finalization-ola-parity.json.gz)

## Paquet Hub et suite de recherche

Le paquet assemble les deux seuls ONNX principaux, le manifeste, les
SHA-256 de chaque fichier, le texte de licence Apache-2.0 vérifié
contre la révision source, l’attribution, la model card et un
exemple exécuté sans phonémiseur avec des tokens inclus. Les SHA-256
des deux ONNX correspondent aux graphes déjà mesurés de la PR #2.
Les 22 tests ciblés passent après finalisation, dont deux sur l’écoute
aveugle et les empreintes du manifeste.
[Model card préparée](hf/README.md) · [Manifeste](hf/manifest.json)

Le travail est isolé de `main`, `fr` et de la PR #1. La publication
reste soumise à une autorisation distincte. La réduction réelle des
paramètres fait l’objet d’une [proposition de distillation séparée](DISTILLATION.md).
Le budget consommé est nul et aucun entraînement n’a été lancé.
