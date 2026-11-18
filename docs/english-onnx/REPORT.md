# Tiny Voca anglais autonome

La [finalisation](FINALIZATION.md) contient les décisions courantes, le paquet
Hugging Face et les mesures complémentaires. Le présent document conserve
le protocole de recherche et ses résultats.

## Référence et protocole

La branche part de `main` à `db851f75c6287a15be4c4e995f931bce3a86b47b`.
Elle utilise le checkpoint anglais public, avec **4 537 291 paramètres
appris** et un SHA-256 vérifié. Le graphe ONNX public pèse 21 386 372 octets.
Les graphes de sortie sont des fichiers uniques, utilisent le domaine ONNX
standard et gardent les six entrées et les sorties `audio` et `durations`.
Les exportations sont isolées. L’API d’export habituelle conserve son
comportement par défaut. [Checkpoint et graphe publics](https://huggingface.co/n4ze3m/tiny-voca/tree/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14)

Le classement des canaux et la sélection des variantes utilisent 12
phrases et deux voix. Le lot réservé contient 100 autres phrases, chacune
rendue en F2 et M2. Il couvre cinq groupes de 20 textes (courts, questions,
contrastes phonétiques, noms et nombres, phrases longues) et des variations
de vitesse, de hauteur et d’intonation. Les comparaisons CPU utilisent un
bruit commun dérivé d’une graine par énoncé. Un dérivé de qualification du
graphe public remplace uniquement son générateur aléatoire par ce flux.
Les chronos utilisent les graphes publiables avec bruit interne et incluent
la phonémisation, l’inférence et la copie de la forme d’onde normalisée.

On mesure l’erreur RMS log-mel, le F0 estimé sur l’audio par
autocorrélation, la divergence de voisement et les durées des phones et
pauses. Pour les candidats présélectionnés, Whisper base.en indépendant
décode chaque énoncé en INT8 CPU avec une fenêtre d’au moins 8 s, une
marge de 3 s après le signal et des paramètres communs. Les WER/CER
emploient le normaliseur anglais de Whisper. Les intervalles à 95 % sont
appariés par texte et groupent les deux voix. WeSpeaker ResNet34-LM donne la
similarité cosinus de l’empreinte vocale au rendu dense du même énoncé.
Cette similarité et l’ASR mesurent des propriétés complémentaires à l’écoute.
[Reproduction](REPRODUCE.md)

## Transformations exactes

- La suppression du masque plein à l’export `B=1` reprend le principe du
  port français. Elle élimine les 11 `IsNaN` et leurs `Where` de protection,
  ainsi que les identités associées aux résidus. Les chemins qui utilisent
  des masques explicites gardent leurs opérations.
- Le scan compte les fins de tokens par `ScatterElements(reduction=add)`
  puis `CumSum`. Les durées nulles, y compris les fins multiples à une même
  trame, sont traitées exactement. Le coût du calcul des propriétaires passe
  de O(TF) à O(T+F) pour T tokens et F trames.
- Le chemin spectral concatène réel et imaginaire avant l’analyse et les
  projections cepstrales. Il retire un cadre final qui commence après la
  zone conservée. Les fenêtres à quatre recouvrements donnent un seul
  préfixe de normalisation puis un motif périodique d’un hop. Le graphe
  conserve la division avec cette enveloppe pré-calculée et des longueurs
  dynamiques.
- `matmulola` réalise la synthèse inverse par un `MatMul` et la somme de
  quatre bandes décalées, pour comparer le dispatch `ConvTranspose` et les
  mouvements mémoire. Ce calcul conserve les mêmes coefficients.
- Trois constantes de fréquences RoPE issues de `pow` différaient chacune
  d’un coefficient de 1,86 × 10⁻⁹ sur cet hôte. Un réexport brut donnait un
  SNR minimal de 46,2 dB contre le graphe public malgré des initialisateurs
  identiques. Reprendre les constantes du fichier public rétablit le rendu
  bit à bit du réexport sur le criblage, avant toute autre transformation.
  Cette étape neutralise une dérive de phase due à l’exportateur.

L’attention conserve les opérateurs standards du graphe SDPA et les
réécritures portables. Les expériences DFT et radix-2 d’E04 ne sont pas
reprises. [Expérience française E04](https://github.com/yoch/tiny-voca/blob/0aa5092/docs/experiments/E04-fft-and-sdpa-fr.md)

## Redondance et coût

Les 44 matrices `expand`, `project`, `qkv` et `out` des 11 blocs ont été
analysées en FP64. Dans les cinq blocs acoustiques, `expand` demande des
rangs **173–181** pour préserver 99 % de l’énergie. `project` demande **130–181**.
Le seuil de rentabilité des deux matrices 192 × 640 se trouve au rang
147,7. Le rang 128 conserve 90–94 % de l’énergie de `expand` et 90–99 %
de `project`. Le maximum médian des cosinus absolus entre lignes distinctes
d’`expand` se situe entre 0,30 et 0,44 selon le bloc. La dispersion explique
pourquoi des économies identiques n’ont pas la même conséquence. Dans
l’encodeur, les FFN atteignent 180–183 pour 99 % et les tests de retrait
y modifient les durées. Pour `out` d’attention, une factorisation doit
passer sous le rang 96 pour économiser des paramètres, alors que 134–146
sont nécessaires pour 99 % dans la pile acoustique.

Les MACs des linéaires et convolutions depthwise des piles représentent
1,578 M par token et 2,762 M par trame, en plus de
`2 × 192 × (4T² + 7F²)` MACs pour les produits d’attention. Les
convolutions DFT et inverse contribuent environ 1,581 M par trame
en régime établi, et les projections cepstrales 0,132 M. Les têtes
de sortie, l’oscillateur et la normalisation ajoutent du calcul et du
trafic mémoire. `reconstruct512` économise 245 760 MACs linéaires par
trame acoustique. INT8 réduit la largeur des opérandes et peut accélérer
les kernels, tout en gardant le même compte de MACs logiques et de
coefficients appris. Les tenseurs générés par les nœuds `Constant`
représentent environ 2,9 Mo de valeurs dans le graphe dense.

Le pruning conserve ensemble les lignes d’`expand` et les colonnes de
`project`. Le score est le produit du RMS d’activation après GELU et de
la norme de la colonne de sortie. La variante à budget non uniforme
alloue des blocs de 32 canaux d’après ces scores. La reconstruction
récupère une partie de la contribution des canaux retirés par une
régression ridge fermée sur les canaux conservés et une correction
du biais. Elle utilise seulement les 24 passages de calibration et
coûte zéro heure GPU. Aucun entraînement, backpropagation ou
ajustement des durées, de la prosodie et du générateur de F0 n’a été
effectué pour ces candidats acoustiques.

## Résultats et décision

### Qualité sur 100 textes et 200 rendus

| Variante | Durées identiques | RMS log-mel médian / p95 (dB) | WER / CER (%) | Écart WER et IC 95 % (points) | Cosinus voix médian / p05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public | 200/200 | 0 / 0 | 3,11 / 1,96 | référence | 1 / 1 |
| `exact` | 200/200 | <0,001 / <0,001 | 3,16 / 2,01 | +0,05 (−0,96 à +1,11) | 1 / 1 |
| `matmulola` | 200/200 | <0,001 / <0,001 | — | — | — |
| `exact-int8` | 200/200 | 0,15 / 0,21 | 3,11 / 1,96 | 0,00 (0,00 à 0,00) | 0,99976 / 0,99916 |
| `exact-fp16` | 200/200 | 0,002 / 0,003 | 3,41 / 2,17 | +0,30 (−0,64 à +1,28) | 1 / 1 |
| `reconstruct576` | 200/200 | 0,88 / 1,34 | 2,76 / 1,55 | −0,35 (−1,09 à +0,07) | 0,992 / 0,968 |
| `reconstruct512` | 200/200 | 1,29 / 1,99 | 2,86 / 1,82 | −0,25 (−1,41 à +0,90) | 0,983 / 0,932 |
| `reconstruct512-int8` | 200/200 | 1,30 / 1,98 | 3,41 / 2,18 | +0,30 (−0,65 à +1,30) | 0,983 / 0,932 |

Tous ces rendus ont le même nombre d’échantillons que la référence.
Sur CPU avec bruit commun, le SNR minimal est de **143,42 dB** pour
`exact` et **132,14 dB** pour `matmulola`. Leurs erreurs de crête
normalisées maximales sont de 2,21 × 10⁻⁷ et 7,44 × 10⁻⁷. Les variations
du texte décodé par Whisper sur le graphe exact incluent des doublons
du reconnaisseur et plusieurs variantes de ponctuation. L’INT8 dynamique
préserve les comptes d’erreurs WER et CER de chacune des deux voix.
Son écart de F0 estimé est de 0,005 demi-ton médian et son p95 de
désaccord de voisement de 0,83 %. Pour la combinaison 512 INT8, ces
valeurs passent à 0,043 demi-ton et 2,79 %. Le FP16 de stockage est une
conversion de poids suivie d’un Cast en FP32. En M2, son WER mesuré
augmente de 1,20 point. [Mesures par énoncé](raw/holdout-quality.json.gz)

### Pistes écartées ou limitées

- Ramener les cinq FFN acoustiques de 640 à 160 canaux enlève **20,37 %**
  des paramètres appris. Même après reconstruction, l’écart log-mel médian
  de calibration est de 3,49 dB (maximum 4,64 dB). La réduction de taille
  est réelle, accompagnée d’une modification acoustique trop marquée.
- Réallouer le budget des canaux en fonction de leur importance ne
  domine pas la largeur 512 uniforme. Les deux choix ont le même nombre
  de paramètres et donnent respectivement 1,69 et 1,61 dB sur calibration.
  La reconstruction en largeur 512 ramène cet écart à 0,98 dB.
- Les SVD acoustiques des deux FFN au rang 128 et 96 économisent 3,61 %
  et 9,48 % des paramètres, avec 1,22 et 1,97 dB médians. Factoriser
  seulement `project` au rang 128 économise 1,81 % et donne 0,71 dB.
  Les essais QKV au rang 128 et projection de sortie d’attention au
  rang 80 économisent peu pour leur impact (1,04 et 0,71 dB).
- Les changements prosodiques et de l’encodeur sont plus sensibles.
  Le pruning 576 de l’encodeur change les durées dans 24 passages sur 24.
  Le cumul des trois piles à 512 atteint 11,95 % d’économie et change
  également toutes les durées de calibration.
- La QDQ statique de la pile acoustique donne 0,25 dB d’écart log-mel
  médian sur le lot réservé, contre 0,15 dB pour l’INT8 dynamique. Au
  criblage CPU à un thread, la médiane à environ 2 s est de 25,4 ms
  en QDQ et 22,4 ms en INT8 dynamique. Le FP16 de stockage n’apporte
  aucun gain de calcul FP16.
- Les graphes exacts `mask`, `packed` et `scan` ont été mesurés
  séparément. Le gain de `scan` seul est modeste sur les textes courts.
  Le nombre moindre d’opérateurs de `packed` seul ne donne pas de gain
  robuste à quatre threads au criblage. Les combinaisons retiennent
  la géométrie et les réécritures qui améliorent les parcours réels.

[Tableau complet de présélection](MATRIX.md) · [Spectres des poids et données brutes](raw/)

### Frontière de Pareto mesurée

Les gains sont des réductions de médiane contre le graphe public à 1,97 s,
calculées dans chacun des deux ordres. Les graphes sont qualifiés séparément
sur CPU natif et Chromium/WASM, chacun à un et quatre threads. Le tableau
détaillé contient médiane, p95, RTF, chargement, pic RSS et les passages de
0,68 s et 5,34 s. [Temps et mémoire](LATENCY.md)

| Variante | Paramètres appris | ONNX (Mo) | CPU 1 / 4 threads | Web 1 / 4 threads | Statut |
| --- | ---: | ---: | ---: | ---: | --- |
| Public | 4 537 291 | 21,386 | réf. | réf. | Référence |
| `exact` | 4 537 291 | 21,361 | 6,7–10,6 % / 8,3–9,3 % | 2,7–5,0 % / 8,0–10,2 % | Référence optimisée |
| `matmulola` | 4 537 291 | 21,367 | 8,6–10,3 % / 10,8–11,1 % | 4,1–8,7 % / 14,9–15,5 % | Profil exact moyen/long |
| `exact-fp16` | 4 537 291 | 17,430 | 7,5–10,0 % / 8,2–12,0 % | 2,9–6,4 % / 5,3–8,3 % | Stockage seul |
| `exact-int8` | 4 537 291 | 15,680 | 24,3–26,0 % / 18,6–20,4 % | 15,0–16,5 % / 14,7–14,9 % | Compact à qualifier par écoute |
| `reconstruct512` | 4 290 891 | 20,375 | 11,1–12,9 % / 11,6–16,5 % | 7,6–10,9 % / 7,6–15,6 % | Compression de recherche |
| `reconstruct512-int8` | 4 290 891 | 15,429 | 26,3–26,5 % / 20,3–21,6 % | 18,3–18,6 % / 17,6–19,0 % | Compression de recherche |

Sur le lot Web de 24 passages à chaque nombre de threads, les six variantes
gardent exactement les durées et longueurs. Les minima SNR sont de
**140,53 dB** pour `exact` et de **132,15 dB** pour `matmulola` contre le
graphe public avec le même bruit. L’INT8 et la réduction des canaux
produisent des approximations qui sont évaluées séparément. [Parité Web](raw/web-parity.json.gz)

Pour `exact-int8`, le p95 à 1,97 s passe de 30,37–38,26 à 23,04–23,18 ms
en CPU monothread et de 94,63–104,58 à 81,61–82,94 ms en Web monothread.
Le pic RSS CPU du processus passe d’environ 349 à 329–330 Mio. Les pics
RSS Chromium varient selon l’ordre (723–781 Mio en INT8 à un thread,
759–788 Mio en dense). L’INT8 régresse sur le passage de 0,68 s en
Chromium à quatre threads. `matmulola` et la combinaison 512 INT8
régressent aussi sur ce passage, malgré leur avantage à 5,34 s.
Le bénéfice est établi pour les charges moyennes et longues de ce banc
et pour ce matériel. Les temps d’initialisation Web mesurent le
loopback et incluent Pyodide, Dizione et ORT.

### Recommandation

**`exact.onnx` est le candidat FP32 de référence** pour une future
publication. Il combine un gain mesuré, des durées identiques et une
fidélité numérique supérieure à 140 dB sur les deux runtimes testés.
`matmulola.onnx` convient à l’étude des charges moyennes et longues,
avec un gain CPU à quatre threads de 22,8–23,6 % à 5,34 s.

**`exact-int8.onnx` est le meilleur candidat compact et rapide à
écouter à l’aveugle.** Il réduit la taille de 26,7 % et maintient les
comptes d’erreurs du reconnaisseur sur les deux voix. La compression
structurale combinée réduit réellement les paramètres de **5,43 %**
et la taille de 27,9 %. Son intervalle WER dépasse la marge d’un point
et l’empreinte vocale s’écarte davantage du dense. Le modèle 512 reste
expérimental. Le kit A/B est fourni séparément et aucun résultat
d’écoute humaine n’a été collecté. Dans ce protocole, aucune
réduction de paramètres n’a démontré l’équivalence acoustique
nécessaire pour remplacer le modèle public. [Sommes de contrôle](SHA256SUMS)

Les variantes exportées sont vérifiées par le validateur ONNX et se
chargent dans ONNX Runtime standard. Le jeu de tests ciblé couvre
l’export dynamique, les extrémités spectrales, les durées nulles et
les chemins de synthèse d’origine. **20 tests passent.**
