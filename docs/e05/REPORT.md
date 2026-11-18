# E05 — Alignements phonétiques français

Le décalage d’attaque de −30 ms mérite un essai sur les voix cibles. Sur le lot
réservé de Rhapsodie, il réduit de 9,8 % l’erreur moyenne de début des phones
appariés à partir du texte. Les deux variantes de découpage des pauses perdent
trop de rappel. Le corpus d’entraînement français d’origine n’était pas
accessible pour cette mesure.

## Mesure sur CPU

Le banc part du fork français `4cc26ab` et utilise eSpeak NG 1.52.0, l’aligneur
CTC INT8 et l’assemblage de durées du fork. Un premier passage aligne les phones
de référence pour isoler les erreurs temporelles. Un second utilise le texte et
mesure séparément les écarts phonétiques et les frontières des phones appariés
à l’identique. La convention de mesure est de 20 ms pour les émissions CTC
et de 256 échantillons à 24 kHz, soit 10,67 ms, pour les durées acoustiques.

Le lot de développement compte 11 extraits et 403 phones. Le lot réservé
compte 12 extraits et 498 phones. Chacun couvre quatre enregistrements
distincts de l’autre lot. Les 23 extraits totalisent 94 s. Ils sont coupés aux
frontières de mots, avec jusqu’à 120 ms de silence de référence aux extrémités.
La conversion SAMPA suit l’inventaire du fork, qui fait correspondre /ɥ/ à la
même étiquette que /y/. Le choix du décalage et de la politique de pause utilise
uniquement le lot de développement. Les WAV et les pistes phones/mots
proviennent de Rhapsodie, sous CC BY-NC-SA 3.0 France. [Corpus Rhapsodie](https://rhapsodie.modyco.fr/info-telechargement/)

La corrélation WAV–MP3 mesure un retard de 69,0–69,1 ms dans six enregistrements
sur huit. Pour les deux autres, le retard mesuré est nul. Avec les MP3, l’erreur
moyenne d’attaque sur le lot réservé avec phones de référence était de 53,2 ms.
Avec les WAV originaux, elle retombe à 21,3 ms. Les décisions ci-dessous
s’appuient sur les WAV originaux. [Mesures du décalage](audio-clock.json)

## Résultats sur le lot réservé

Les lignes de ce tableau partent du texte et couvrent les 476 phones appariés.
Le F1 des pauses porte sur les trames extérieures aux phones de référence.

| Variante | Début MAE / p95 (ms) | Fin MAE / p95 (ms) | Durée MAE (ms) | Pause F1 / rappel |
| --- | ---: | ---: | ---: | ---: |
| Référence du fork, −19 ms | 21,23 / 60,9 | 32,16 / 80,8 | 35,51 | 0,626 / 46,0 % |
| Décalage choisi sur le développement, −30 ms | 19,15 / 54,0 | 30,22 / 80,7 | 34,61 | 0,626 / 46,0 % |
| Pause contiguë, −19 ms | 25,66 / 65,0 | 33,14 / 81,9 | 40,94 | 0,162 / 8,9 % |
| Pause avec pont d’une trame, −19 ms | 25,50 / 65,0 | 33,14 / 81,9 | 40,78 | 0,174 / 9,6 % |

Avec les phones de référence, −30 ms réduit l’erreur de début de 21,3 à
19,3 ms et celle de fin de 30,4 à 28,3 ms. Au départ du texte, le gain en
début de phone se retrouve dans trois des quatre enregistrements. Le quatrième
passe de 19,4 à 19,8 ms. Les phones d’une seule trame passent de 1 à 3 sur
le lot réservé. Les quatre variantes couvrent la même durée acoustique. À
−19 ms, les occlusives ont la plus forte erreur moyenne de début (30,3 ms sur
92 phones), suivies des fricatives (22,6 ms sur 72). À −30 ms, ces valeurs
tombent à 25,2 et 18,1 ms. [Résultats agrégés et par extrait](wav-results.json.gz)

Le trajet texte → phones présente 38 écarts pour 498 phones de référence
(7,6 %), dont 16 insertions, 17 substitutions et 5 suppressions. Les écarts
les plus fréquents concernent six /ə/ et cinq /z/ supplémentaires, cinq
substitutions /e/ → /ɛ/ et quatre /œ̃/ → /ɛ̃/. Les schwas, liaisons et
distinctions vocaliques peuvent refléter la diction spontanée ou l’accent.
Ces écarts servent à choisir les cas à écouter et à annoter.

Le passage de présélection complet prend environ 12 s sur quatre threads
CPU, dont environ 9 s d’inférence ORT après ouverture de session. Un
changement de paramètres réutilise les postérieurs. Le modèle INT8 sert à
présélectionner des variantes. L’accord avec l’aligneur FP32 d’origine et la
qualité sur les voix d’entraînement restent à mesurer. [Reproduction](REPRODUCE.md)

## Critères avant tout réentraînement

1. **Constituer un petit lot de référence sur les deux voix cibles.** Annoter
   40 à 60 extraits courts, avec au moins 500 frontières dans le lot réservé
   et des comptes par voix et par classe. Séparer les textes et les séances
   entre développement et évaluation. Faire relire un sous-ensemble des
   frontières et consigner les conventions de fermeture des occlusives,
   relâchements, fricatives, respirations et silences. Inclure des liaisons
   obligatoires et interdites, des schwas facultatifs, des nasales et des fins
   de phrase. Annoter explicitement les variantes de prononciation acceptables.
2. **Mesurer les erreurs de prononciation qui peuvent guider le mauvais
   apprentissage.** Comparer eSpeak à la suite phonétique annotée, puis lire
   les cas de liaisons manquantes ou interdites et les consonnes audibles
   omises. Une correction doit réduire ces erreurs sur le lot réservé, sans
   introduire de changements dans les exemples de prononciations acceptables
   ni dans la segmentation des autres classes. Publier les comptes par cas
   et faire arbitrer les écarts ambigus avant de changer les étiquettes.
3. **Confirmer l’avance de 11 ms sur le lot cible.** Comparer d’abord −19 et
   −30 ms avec les mêmes phones et postérieurs. Vérifier ensuite le sens du
   résultat avec l’aligneur FP32 d’origine sur un sous-ensemble stratifié.
   Avant adoption, viser au moins 10 % de baisse de la MAE des débuts, une
   MAE des fins au plus égale à la référence et aucune hausse de p95
   supérieure à une trame acoustique pour une voix ou une classe prioritaire.
   Publier les écarts appariés par enregistrement et leur intervalle
   d’incertitude. La somme exacte des trames, les durées positives des phones
   et l’audit des phones d’une seule trame sont des contrôles obligatoires.
4. **Réexaminer les pauses après annotation des cas audibles.** Distinguer un
   silence de référence d’une fermeture, d’un souffle ou d’une fricative
   faible. Une variante doit améliorer le F1 de pause, limiter la baisse de
   rappel à deux points et ne pas attribuer plus de trames de parole à une
   pause. Rapporter les bords et les pauses internes séparément. Les deux
   variantes contiguës testées ici ne passent pas ce filtre.

Pour **E06**, figer les textes, les WAV, les révisions du checkpoint et du
phonémiseur ainsi que la référence d’alignement par voix. Ces mêmes entrées
permettront de comparer le réseau complet et ses variantes compressées sur
la prononciation, les durées et les pauses. Aucun entraînement ni changement
du chemin de synthèse n’a été effectué pendant ce test.
