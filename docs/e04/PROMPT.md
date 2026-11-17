Tu reprends E04 (accélération du vocoder ONNX) dans le dépôt https://github.com/yoch/tiny-voca,
branche `codex/e04-fft-vocoder`. Travaille sur cette branche et sur sa PR brouillon si elle existe ; ne merge pas.

Lis d'abord `docs/e04/README.md`, les résultats JSON et `scripts/fft_candidates.py`.
L'agent précédent tournait sur un i7-3770 AVX sans AVX2. Vérifie que ton runner expose
AVX2 et consigne le matériel et les versions. Les variantes ONNX DFT et radix-2
étaient plus lentes sur l'ancien runner ; l'absence d'AVX2 n'est pas une explication
établie. L'export de production a donc été laissé INCHANGÉ. Les implémentations
expérimentales sont isolées dans scripts/, avec tests et benchmarks reproductibles.

Objectif : remplacer les convolutions FFT denses de src/tiny_voca/export.py seulement
si un gain réel est démontré, sans réentraînement ni dégradation numérique/acoustique.

1. Reproduis les tests tests/test_export.py et tests/test_fft_candidates.py. Utilise
   le checkpoint public inchangé (URL/SHA et commandes dans le README), du bruit
   déterministe commun et les mêmes entrées pour tous les backends. Vérifie les
   durées, longueurs, deux voix, plusieurs longueurs/vitesses/hauteurs, les limites
   de fenêtres, identité, silence et impulsions. Compare à dense ONNX, pas seulement
   à Torch : les différences préexistantes entre runtimes ne sont pas l'erreur FFT.
2. Exécute scripts/benchmark_fft.py sur CPU 1 et 4 threads, puis
   scripts/benchmark_fft_web.cjs dans Chromium avec ORT Web/WASM 1.20.1. Mesure
   filtrage isolé ET synthèse complète, médiane/p95, mémoire et taille d'artefact.
   Pas de benchmarks simultanés. Répète en variant l'ordre, conserve les JSON bruts.
   Les anciens champs peak_rss_mib « preliminary » sont invalides ; le script corrigé
   utilise VmHWM dans un processus neuf. Le RSS Web est instantané, pas un pic.
3. Si les candidats restent plus lents, cherche une meilleure solution (éventuellement
   FFT WASM spécialisée avec DSP hors graphe et FFT native CPU), en mesurant aussi
   les coûts de transfert. Ne change pas la définition du DSP ni les poids pour
   obtenir artificiellement un gain. Conserve le fallback dense.
4. Ne promeus un candidat dans export.py que si les gains sont répétables sur les
   runtimes ciblés avec les seuils de fidélité du README. Sinon conserve la production,
   documente le résultat négatif et indique que l'accélération reste à réaliser.
5. Mets à jour docs/e04, ajoute les tests nécessaires, vérifie ONNX Runtime CPU et
   Web réellement, et ouvre/mets à jour la PR SANS merger. Rapporte les gains mesurés,
   leurs limites, les validations et le lien PR ; aucune extrapolation de gain AVX2.

Tu as autorisation de modifier le repo, installer les dépendances de test, télécharger
les poids publics, pousser la branche et créer/mettre à jour la PR. Aucun réentraînement,
aucun merge. Les fichiers /tmp de l'ancien runner ne sont pas nécessaires.
