# Reproduire E05

Utiliser la branche `codex/e05-fr-alignment`, basée sur `fr` à `0aa5092`.
Les mesures utilisent Python 3.12, les dépendances Tiny Voca, SciPy, ORT
1.32.0, eSpeak NG 1.52.0 et FFmpeg 7.1.5. Tout s’exécute sur CPU.

## Rhapsodie et revue des désaccords

```sh
export PYTHONPATH=src:scripts/e05 TINY_VOCA_FRONTEND=espeak
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /tmp/e05/model /tmp/e05/rhapsodie
base=https://huggingface.co/onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX/resolve/c69750f5043e5e1f8a71ab95dd3b98338c280c92
curl -fL "$base/onnx/model_int8.onnx" -o /tmp/e05/model/alignment.onnx
curl -fL "$base/vocab.json" -o /tmp/e05/model/vocab.json
echo '74174710e34035bbb7f611601d016c32fc575de7a6f53b1078107dc10a84e7ae  /tmp/e05/model/alignment.onnx' | sha256sum -c -
echo 'd732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0  /tmp/e05/model/vocab.json' | sha256sum -c -
curl -fL https://rhapsodie.modyco.fr/rp/prosodie/finaux/Rhap-proso-all-textgrids.zip -o /tmp/e05/rhapsodie/grids.zip
echo '0cdcbbf3c8cc557d971133812cb62f56e8b6476257280d1ccf957b159279be02  /tmp/e05/rhapsodie/grids.zip' | sha256sum -c -
curl -fL https://rhapsodie.modyco.fr/rp/waves/Rhap-sound-wav.zip -o /tmp/e05/rhapsodie/wavs.zip
unzip -jo /tmp/e05/rhapsodie/wavs.zip Rhap-M0004.wav Rhap-M0008.wav Rhap-M2001.wav Rhap-M0013.wav \
  Rhap-M0010.wav Rhap-M0012.wav Rhap-M0021.wav Rhap-M2005.wav -d /tmp/e05/rhapsodie

python scripts/e05/prepare_rhapsodie.py /tmp/e05/rhapsodie/grids.zip /tmp/e05/rhapsodie /tmp/e05/wav-e01
python scripts/e05/review_disagreements.py --corpus /tmp/e05/wav-e01 \
  --historical docs/e05/wav-results.json.gz --review docs/e05/disagreement-review.json

python scripts/e05/prepare_rhapsodie.py /tmp/e05/rhapsodie/grids.zip /tmp/e05/rhapsodie /tmp/e05/wav-e01-reviewed \
  --reviewed-text docs/e05/reviewed-text-v1.json
for fixture in wav-e01 wav-e01-reviewed; do
  taskset -c 0-3 python scripts/e05/alignment_probe.py \
    --corpus "/tmp/e05/$fixture" --model /tmp/e05/model/alignment.onnx \
    --vocab /tmp/e05/model/vocab.json --cache /tmp/e05/e01-posteriors --output "/tmp/e05/$fixture.json"
done
```

L’archive WAV pèse environ 678 Mo. L’accès HTTP par plages ZIP permet aussi
d’extraire les huit membres en vérifiant leurs CRC. Le préparateur vérifie
ensuite chaque SHA-256 d’origine. Adapter les quatre IDs de `taskset` à la
machine. Un cache neuf inclut le temps d’inférence. Le second passage
réutilise les postérieurs des mêmes échantillons. Les mesures initiales
sous le frontal du fork `4cc26ab` sont conservées dans
[wav-results.json.gz](wav-results.json.gz) comme référence historique.

Pour l’audit d’horloge uniquement, juxtaposer les fichiers `Rhap-*.mp3`
correspondants et lancer `python scripts/e05/audio_clock.py /tmp/e05/rhapsodie`.
Un retard positif signifie que le signal MP3 décodé arrive après le WAV.
[Archives Rhapsodie](https://rhapsodie.modyco.fr/info-telechargement/)

## Lot indépendant des voix E01

Le fichier d’index à fournir contient `dataset_revision`, `voice_revision`,
`annotation_revision` et un tableau `clips`. Chaque clip comporte les champs
suivants.

| Champ | Contenu |
| --- | --- |
| `id` | Identifiant stable et unique du clip |
| `speaker` | `F1` ou `M1` d’E01 |
| `recording` | Séance ou enregistrement original pour le bootstrap |
| `wav`, `sha256` | Chemin du WAV d’origine relatif à `--sources` et son SHA-256 |
| `text` | Transcript relu avec la ponctuation de groupe |
| `reference` | Liste ordonnée des objets `{phone, start, end}`, en secondes à partir du début du WAV |
| `oracle` | Suite de ces phones E01 avec les tokens de début, de fin et de séparation |

Utiliser les voix du projet et dater le lot avant de scorer. Reporter les
lectures alternatives et les désaccords d’annotateurs dans une feuille de
revue, puis fournir les intervalles arbitrés. Les exclusions JSONL, contenant
le champ `text`, doivent couvrir les jeux source d’entraînement, validation
et test déjà vus. La vérification compare des mots normalisés et repère les
doublons audio exacts.

```sh
python scripts/e05/prepare_target.py --index /path/to/targets.json --sources /path/to/original-wavs \
  --exclude-texts /path/to/train.jsonl /path/to/validation.jsonl /path/to/test.jsonl \
  --output /tmp/e05/f1-m1-holdout
python scripts/e05/alignment_probe.py --target --corpus /tmp/e05/f1-m1-holdout \
  --model /tmp/e05/model/alignment.onnx --vocab /tmp/e05/model/vocab.json \
  --cache /tmp/e05/e01-posteriors --output /tmp/e05/f1-m1-results.json
```

Le mode cible compare les deux offsets fixés à l’avance sur les mêmes
postérieurs. Il produit les scores par voix, classe et type de pause et
un intervalle apparié par rééchantillonnage des enregistrements. La première
mesure INT8 sert à la présélection. Pour évaluer la sensibilité à la précision,
utiliser ensuite `--model /tmp/e05/model/alignment-fp32.onnx` avec le même index,
le même vocabulaire et le même cache.

```sh
curl -fL "$base/onnx/model.onnx" -o /tmp/e05/model/alignment-fp32.onnx
echo '93265694093f5f91497181ed9d7791f43bc818d2e23c46caf988fd9b8b1a1fba  /tmp/e05/model/alignment-fp32.onnx' | sha256sum -c -
```

## Réannotation versionnée avec E01

Avec les véritables features et le reconnaisseur PyTorch dans le cache HF,
générer un nouveau jeu en réutilisant le format E01. Mettre la révision de
la source à la place de `fr4k-<revision>`.

```sh
python -m tiny_voca.annotate --device cpu --features /path/to/features \
  --dataset-revision 'fr4k-<revision>' --alignment-profile e01-refresh \
  --output /path/to/annotations-e01-refresh
python -m tiny_voca.annotate --device cpu --features /path/to/features \
  --dataset-revision 'fr4k-<revision>' --alignment-profile e05-onset-candidate \
  --output /path/to/annotations-e05-candidate
```

Le profil candidat sert exclusivement à la comparaison. Les sorties sont
des dossiers séparés au format d’annotation v4 de `fr`. L’identité conserve
l’empreinte du vocabulaire et celle des règles E01, les SHA-256 du code du
frontal et de l’aligneur, la révision du reconnaisseur, l’offset et les
SHA-256 des manifestes source. Corriger les transcriptions source vérifiées
avant cette étape et publier une nouvelle révision du jeu. Conserver les
rejets et le résumé QC à côté des manifestes. La commande sans profil
reste celle d’E01, y compris pour la reprise d’un lot existant.

```sh
python -m pytest -q scripts/e05 tests/test_alignment.py tests/test_data.py tests/test_frontend.py tests/test_e05.py
```
