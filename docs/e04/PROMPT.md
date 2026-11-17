Tu reprends E04 dans https://github.com/yoch/tiny-voca, branche
`codex/e04-fft-vocoder`, PR brouillon #1. Travaille sur cette branche ; ne merge pas.

Lis `docs/e04/README.md`, `docs/e04/avx2/REPORT.md`, les JSON bruts et les trois
scripts de benchmark. L'export de production est INCHANGÉ. Aucun réentraînement,
aucune modification des poids ou du DSP n'est autorisé pour fabriquer un gain.

Deux runners ont été mesurés : i7-3770 sans AVX2, puis EPYC 9V74 avec AVX2 et
AVX-512 exposés. DFT et radix-2 restent plus lents que dense sur CPU et Web,
dans les deux ordres de mesure. Les gains entre machines ne sont pas attribuables
à AVX2 seul. Ne recommence pas par supposer que la seule présence d'AVX2 suffit.

Une piste CPU est désormais concrète : `scripts/benchmark_fft_native.py` exporte
la partie neuronale ONNX puis effectue le DSP avec les FFT natives de NumPy.
Les sorties ONNX et le passage Python/NumPy sont inclus dans le temps total.
C'est un prototype isolé, sans implémentation Web et sans validation d'écoute ;
il n'est donc pas promu dans export.py.

1. Reproduis les tests et les benchmarks selon avx2/REPORT.md, avec le checkpoint
   public inchangé, son SHA256 vérifié, du bruit commun déterministe, les deux
   voix et plusieurs longueurs/vitesses/hauteurs. Les phrases doivent employer
   le phonémiseur correspondant au checkpoint, avec ses données publiques déjà
   construites : ne réentraîne rien. Compare les candidats au dense ONNX du même
   runtime. Conserve séparément les écarts CPU/WASM préexistants.
2. Tous les benchmarks doivent être séquentiels, 5 chauffes puis 30 mesures,
   CPU et Chromium/WASM ORT Web 1.20.1 à 1 et 4 threads, filtrage et synthèse
   complète, médiane/p95, taille et mémoire. Répète dans l'ordre inverse et garde
   les temps individuels JSON. VmHWM CPU : processus neuf ; RSS Web : instantané,
   pas pic ni taille du tas WASM. Relève matériel, versions et charge du runner.
3. Cherche une FFT WASM spécialisée hors graphe correspondant au prototype
   natif CPU. Mesure explicitement les transferts JS/WASM, les sorties neuronales,
   le framing, la multiplication complexe, l'overlap-add et le crop. Évalue les
   deux tailles, les frontières, identité, silence et impulsions sans changer le
   DSP. Le prototype NumPy ne constitue pas une preuve de gain Web.
4. Ne promeus une optimisation dans export.py que si les gains de bout en bout
   sont répétables dans le périmètre visé, avec durées/longueurs inchangées et
   les seuils de fidélité du README. Complète une validation acoustique d'écoute.
   Conserve un fallback dense explicite ; un gain CPU ne justifie pas une bascule
   globale Web. Sinon documente le résultat et laisse la production intacte.
5. Mets à jour docs/e04 et la PR SANS merger. Rapporte les mesures, leurs limites,
   les validations et le lien PR. L'accélération E04 reste à livrer.

Tu as autorisation de modifier le repo, installer les dépendances de test,
télécharger les poids/données publics, pousser la branche et mettre à jour la PR.
Aucun merge et aucun réentraînement. Les fichiers /tmp des anciens runners ne
sont pas nécessaires : tout doit rester reproductible depuis ce dépôt.
