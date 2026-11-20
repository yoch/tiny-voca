# Étude séparée — distillation de la pile acoustique

Comparer le checkpoint enseignant 640 et un élève 512 initialisé avec
`reconstruct512`. Les cinq FFN acoustiques gardent leur profondeur, leurs
résidus et le nombre de têtes. Cette réduction retire 246 400 paramètres
du checkpoint (5,43 %). Geler le frontal, l’encodeur, les durées, la pile
prosodique, l’oscillateur et les têtes de cepstre. Entraîner uniquement
les paramètres de la pile acoustique. Conserver la même entrée de trames
et le même masque du côté enseignant et élève.

## Données et pertes

- Élargir la calibration à 2 000 textes anglais, répartis par longueur,
  phonèmes, ponctuation et difficulté. Rendre les deux voix et isoler
  200 textes pour la validation interne. Exclure les 100 textes de
  qualification et les 12 nouveaux textes de l’écoute. Dédupliquer par
  texte normalisé avant de former les deux lots.
- Produire sur CPU et mettre en cache les entrées de la pile acoustique,
  ses sorties enseignantes et les cepstres harmoniques et de bruit après
  les têtes gelées. La loss principale compare les sorties cachées par
  canal, avec une échelle fixée sur la calibration. Une loss secondaire
  sur les cepstres compare les quantités qui alimentent le vocodeur.
  Réutiliser les contrôles et la géométrie d’origine.
- Comparer l’élève 512 initial et distillé sur le lot interne, puis
  effectuer une seule évaluation sur le lot réservé original de 100
  textes et deux voix, sans choisir de checkpoint sur ce lot.

## Budget et arrêt proposés

Budget maximal de **30 minutes GPU sur une seule carte T4 ou équivalente**
et de 1 000 mises à jour, le premier plafond atteint faisant foi. Le
pré-calcul sur CPU est séparé. Évaluer tous les 200 pas. Arrêter si deux
contrôles successifs améliorent de moins de 2 % la loss de validation ou
si les durées divergent. L’export reste autonome et utilise le même
protocole de vérification que le FP32 actuel.

Pour poursuivre après le pilote, viser un écart log-mel médian inférieur
à 0,5 dB sur le lot interne et un p05 de similarité vocale d’au moins
0,98. Sur le lot réservé, conserver les durées et longueurs, vérifier
les limites WER/CER d’un point par voix avec les intervalles appariés et
confirmer le gain de latence. L’écoute aveugle doit couvrir les deux
voix. Réviser la décision en cas de régression manifeste de naturel,
d’intelligibilité ou de timbre. Cette étude peut attendre la décision de
publication des ONNX courants. Dépense et entraînement engagés pour
cette proposition — **zéro**.
