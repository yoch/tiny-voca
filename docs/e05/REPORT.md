# E05 — Revue des alignements phonétiques français

**Décision.** Conserver −19 ms et la règle actuelle de pause. Le profil −30 ms
reste expérimental. Les WAV indépendants des voix cibles E01 `F1` et `M1` et
leurs frontières de référence n’étaient pas accessibles. La mesure sur ces
voix et la publication de leurs annotations restent en attente de ces données.

## Référence audio et méthode

La branche E05 reprend le frontal E01 et l’aligneur de `fr` à `0aa5092`. E01
emploie 54 symboles et le même inventaire à l’inférence et à l’annotation.
Les 23 extraits Rhapsodie couvrent 94 s. Le développement comporte 11 extraits,
403 phones et quatre enregistrements. Le lot réservé comporte 12 extraits,
498 phones et quatre autres enregistrements. Les coupes suivent les limites
de mots avec jusqu’à 120 ms de silence de bord. Le tiers SAMPA est projeté
dans l’inventaire E01, qui représente /ɥ/ par la même étiquette que /y/.
Le banc utilise eSpeak NG 1.52.0 et un export INT8 du reconnaisseur CTC,
avec une grille d’émission de 20 ms et une trame acoustique de 10,67 ms.
Un passage force les phones du tiers pour isoler le placement. L’autre
force la séquence issue d’E01 et ne mesure les frontières que des phones
appariés à l’identique. [E01](../experiments/E01-phonemes-fr.md)

Seuls les huit WAV originaux, identifiés par SHA-256, entrent dans le
banc. La corrélation MP3–WAV établit un retard de 69,0–69,1 ms dans six
MP3 et un écart nul dans les deux autres. Sur le lot réservé avec phones
de référence, la MAE d’attaque est de 53,2 ms avec la piste MP3 et de
21,3 ms avec la piste WAV. Les textes courts et les repères dérivés du
corpus dans ce dossier sont attribués à Rhapsodie et repris sous
CC BY-NC-SA 3.0 France. [Téléchargements Rhapsodie](https://rhapsodie.modyco.fr/info-telechargement/) · [Mesure du décalage](audio-clock.json)

## Revue des 38 désaccords

E01 reproduit exactement les 38 opérations du premier essai sur les mêmes
WAV, soit 16 insertions, 17 substitutions et 5 suppressions sur 498 phones.
La revue suit chaque opération jusqu’au mot, au tiers publié et au repère
dans le WAV. Les décisions contextuelles s’appuient sur les deux tiers,
la structure phonétique et, dans les cas signalés, les postérieurs CTC
locaux. Une seconde lecture à l’écoute des cas ambigus est encore nécessaire.
[Revue par opération](disagreement-review.json)

| Conclusion | Opérations | Exemples et décision |
| --- | ---: | --- |
| Variantes plausibles | 29 | Schwas facultatifs, liaisons après verbes fléchis, réductions spontanées et timbres vocaliques. Conserver les variantes acoustiques et signaler deux noms propres à vérifier si une prononciation précise est requise. |
| Écarts du frontal | 4 | E01 produit /ʒøn/ pour « jeune », neutralisation déjà documentée. Trois opérations concernent la seule finale de « clubbers » /ɛʁz/, réduite à /e/ par E01. Le tiers et les postérieurs locaux concordent pour /ʁz/. Le lot ne justifie pas de règle lexicale générale. |
| Frontière textuelle perdue | 2 | E01 insère /z/ à « pantalons ensuite » et « mortels on » dans le texte reconstruit sans ponctuation. Le premier cas se situe après 1,11 s de silence annoté. Les deux /z/ disparaissent avec le point restitué. |
| Appariement de la séquence et des tiers | 2 | Dans « je crois », le /ʃ/ du tiers couvre le mot « je » puis déborde sur « crois ». Le rapprochement de [ʃ] avec /ʒə/ produit deux opérations. Le /ʃ/ est compatible avec une réduction dévoisée et la substitution brute /ʃ/ → /ə/ est un effet d’appariement. |
| À départager | 1 | Le tiers omet le /z/ de « des escaliers », attendu en français standard. Le signal CTC local est faible pour /z/ et ne départage pas réalisation et omission dans le tiers. |

Une frontière CTC tardive ne peut créer d’écart dans ce décompte, qui est
calculé sur les séquences avant l’alignement. Une erreur de frontière est
comptée dans la MAE et le p95. La version `rhapsodie-e05-e01-text-v1`
applique seulement les deux corrections ponctuelles au texte de Rhapsodie
avec SHA-256 du texte source. Les WAV, les tiers et les postérieurs
sont identiques d’un passage à l’autre. Le décompte passe à **36/498**
(7,2 %). Avec −19 ms, la MAE de début passe de 21,23 à 21,16 ms et le
rappel des pauses reste à 46,0 %. Corriger le texte ne suffit pas à
détecter le silence de 1,11 s au milieu de l’extrait. [Corrections versionnées](reviewed-text-v1.json) · [Mesure après correction](e01-reviewed-results.json.gz)

## Calibration sur Rhapsodie, avec E01

| Séquence forcée | Décalage | Début MAE / p95 (ms) | Fin MAE / p95 (ms) | Durée MAE / p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Phones du tiers, 498 | −19 ms | 21,32 / 63,90 | 30,41 / 78,09 | 35,19 / 82,78 |
| Phones du tiers, 498 | −30 ms | 19,35 / 58,71 | 28,26 / 67,52 | 33,96 / 82,78 |
| Texte E01, 476 phones appariés | −19 ms | 21,23 / 60,93 | 32,16 / 80,80 | 35,51 / 84,23 |
| Texte E01, 476 phones appariés | −30 ms | 19,15 / 53,96 | 30,22 / 80,67 | 34,61 / 84,23 |

Le gain de **9,8 %** en MAE d’attaque avec E01 se retrouve dans trois
enregistrements sur quatre. Le quatrième régresse de 0,42 ms. Le bootstrap
des quatre enregistrements donne un intervalle de −3,37 à −0,52 ms pour la différence
de MAE d’attaque. Cet intervalle décrit la sensibilité au lot Rhapsodie
et reste borné à quatre groupes. Les phones d’une trame passent de 1 à 3.
[Résultats E01 et écarts par enregistrement](e01-results.json.gz)

| Classe E01 | Phones | Début MAE / p95 (ms), −19 → −30 | Fin MAE / p95 (ms), −19 → −30 |
| --- | ---: | ---: | ---: |
| Occlusives | 92 | 30,3 / 79,7 → 25,2 / 80,9 | 19,7 / 71,5 → 19,6 / 65,6 |
| Fricatives | 72 | 22,6 / 64,5 → 18,1 / 53,8 | 31,3 / 76,1 → 30,8 / 65,6 |
| Voyelles nasales | 30 | 18,9 / 48,9 → 16,6 / 40,8 | 78,3 / 285,0 → 71,4 / 279,1 |
| Voyelles orales | 165 | 18,7 / 52,3 → 18,6 / 49,8 | 36,3 / 85,9 → 33,3 / 84,7 |
| Consonnes nasales | 28 | 21,9 / 62,7 → 18,6 / 52,1 | 25,1 / 101,0 → 25,6 / 97,3 |
| Liquides et semi-voyelles | 82 | 16,3 / 35,6 → 16,5 / 38,9 | 24,9 / 51,5 → 23,4 / 50,4 |
| Schwas | 7 | 12,4 / 23,2 → 7,8 / 16,7 | 20,7 / 42,0 → 13,1 / 31,3 |

Les pauses sont évaluées par trame. Une trame hors des phones du tiers est
une pause de référence. Les scores de bord concernent les deux extrémités,
les scores internes concernent les intervalles entre phones.

| Règle | Précision / rappel global | Précision / rappel interne | Rappel aux bords |
| --- | ---: | ---: | ---: |
| Actuelle, −19 ms et −30 ms | 97,9 % / 46,0 % | 97,8 % / 48,4 % | 24,0 % |
| Îlot contigu, −19 ms | 90,2 % / 8,9 % | 88,4 % / 8,2 % | 15,4 % |
| Îlot avec pont d’une trame, −19 ms | 90,8 % / 9,6 % | 89,2 % / 9,0 % | 15,4 % |

La perte de rappel écarte les deux variantes. Les longues pauses manquées
peuvent gonfler les durées des phones contigus. Cela justifie d’examiner
les fins de groupes, notamment les nasales, avant de changer la règle.

## Voix cibles et annotations E01

Les voix du projet sont `F1` et `M1` (références `F1_cand1` et `M1_cand2`).
Les exemples publics `FR_F1` et `FR_M1` proviennent du checkpoint
DanielSuncost. Leur équivalence avec les références E01 n’a pas été établie.
Le dépôt, les artefacts accessibles et la recherche Drive n’ont fourni
aucun lot WAV indépendant annoté pour les références E01.
[Références des voix](../../scripts/voices_fr.json) · [Protocole E02](../experiments/E02-prosody-calibration.md)

Le rapport publié sur `fr` à `0aa5092` documente une mesure complémentaire
sur les jeux de validation et de test de `fr4k` (373 et 372 énoncés acceptés).
Avec −19 ms, les transitions occlusive → voyelle présentent un biais médian
entre −1,17 et −0,67 ms selon la voix et le split. Ces marqueurs spectraux
fournissent un argument indépendant en faveur du réglage actuel. Le rapport
amont documente aussi la réannotation de 577 énoncés et une baisse de 0,0603 à
0,0369 de la fraction de phones faibles sur ces énoncés. Les manifestes et
features associés sont référencés dans un répertoire local extérieur à cet
espace. Les chiffres amont portent sur les sondes à −19 ms et la réannotation
E01. Le test apparié −19/−30 ms sur les frontières de référence de `F1` et
`M1` reste à faire. [Rapport fr4k publié](../experiments/E05-aligner-qualification.md)

Le harnais cible fige les SHA-256 du lot, des WAV et des exclusions textuelles.
Il compare exactement −19 et −30 ms sur chaque énoncé, à postérieurs et
règle de pause identiques, puis rend les scores par voix et par classe.
Il attend un lot annoté indépendant, avec 40 à 60 extraits, au moins 500
frontières et plusieurs séances par voix. Les annotateurs doivent fixer
la convention sur les fermetures, relâchements, souffles et silences et
arbitrer les variantes acceptables. Avant adoption, viser au moins 10 %
de baisse de MAE d’attaque, un intervalle apparié favorable, aucune hausse
de MAE de fin et aucun recul de p95 supérieur à une trame dans une voix
ou classe prioritaire. Le rappel des pauses doit rester à deux points
de la référence ou mieux, sans hausse de parole prise pour du silence.
Confirmer le résultat en FP32 sur un sous-ensemble et contrôler les
phones d’une trame et la couverture exacte des trames.

Les profils d’annotation `e01-refresh` et `e05-onset-candidate` produisent
des manifestes d’annotation v4 compatibles E01. Le vocabulaire et
l’empreinte des règles E01 sont identiques pour les deux profils. Chaque
profil donne une empreinte de jeu distincte qui inclut les fichiers du
frontal et de l’aligneur, et la reprise refuse
un profil différent. La règle de pause existante s’applique aux deux profils.
Le dépôt contient les corrections Rhapsodie et les tests de réannotation.
Les manifestes des voix cibles restent à générer à partir de leurs
véritables données. [Commandes et format d’entrée](REPRODUCE.md)
