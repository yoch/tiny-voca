# Matrice de présélection

Les 24 passages de calibration couvrent 12 textes et les deux voix. Les écarts log-mel sont des RMS en dB contre le graphe public, à bruit commun. Les chiffres de latence sont documentés séparément dans le rapport.

| Variante | Paramètres appris | ONNX (Mo) | Durées identiques /24 | Écart log-mel médian / max (dB) |
| --- | ---: | ---: | ---: | ---: |
| `base` | 4,537,291 | 21.386 | 24 | 0.000 / 0.000 |
| `mask` | 4,537,291 | 21.360 | 24 | 0.000 / 0.000 |
| `packed` | 4,537,291 | 21.387 | 24 | 0.000 / 0.000 |
| `scan` | 4,537,291 | 21.387 | 24 | 0.000 / 0.000 |
| `exact` | 4,537,291 | 21.361 | 24 | 0.000 / 0.000 |
| `prune624` | 4,506,491 | 21.237 | 24 | 0.428 / 0.695 |
| `prune576` | 4,414,091 | 20.868 | 24 | 0.989 / 2.620 |
| `prune512` | 4,290,891 | 20.375 | 24 | 1.605 / 3.318 |
| `prune384` | 4,044,491 | 19.389 | 24 | 2.701 / 3.583 |
| `prune160` | 3,613,291 | 17.665 | 24 | 4.762 / 5.743 |
| `weighted512` | 4,290,891 | 20.375 | 24 | 1.689 / 3.573 |
| `solo384first` | 4,438,731 | 20.966 | 24 | 1.078 / 3.763 |
| `solo384last` | 4,438,731 | 20.966 | 24 | 1.014 / 1.214 |
| `svd144` | 4,506,571 | 21.241 | 24 | 0.976 / 3.220 |
| `svd128` | 4,373,451 | 20.708 | 24 | 1.219 / 2.050 |
| `svd96` | 4,107,211 | 19.643 | 24 | 1.972 / 2.663 |
| `svd128out` | 4,455,371 | 21.034 | 24 | 0.710 / 0.928 |
| `prosody576` | 4,488,011 | 21.164 | 24 | 2.565 / 3.161 |
| `encoder576` | 4,438,731 | 20.966 | 0 | 6.646 / 14.079 |
| `combined512` | 3,995,211 | 19.192 | 0 | 8.112 / 15.197 |
| `reconstruct576` | 4,414,091 | 20.868 | 24 | 0.624 / 1.732 |
| `reconstruct512` | 4,290,891 | 20.375 | 24 | 0.977 / 2.618 |
| `reconstruct448` | 4,167,691 | 19.882 | 24 | 1.321 / 2.773 |
| `reconstruct384` | 4,044,491 | 19.389 | 24 | 1.769 / 2.368 |
| `reconstruct160` | 3,613,291 | 17.665 | 24 | 3.487 / 4.639 |
| `exact-int8` | 4,537,291 | 15.680 | 24 | 0.149 / 0.193 |
| `exact-qdq` | 4,537,291 | 15.695 | 24 | 0.250 / 0.393 |
| `exact-fp16` | 4,537,291 | 17.430 | 24 | 0.002 / 0.008 |
| `reconstruct512-int8` | 4,290,891 | 15.429 | 24 | 0.969 / 2.695 |
| `matmulola` | 4,537,291 | 21.367 | 24 | 0.000 / 0.000 |
| `qkv128` | 4,475,851 | 21.116 | 24 | 1.042 / 1.900 |
| `attention80` | 4,506,571 | 21.239 | 24 | 0.714 / 1.984 |

Les SHA-256, MACs par pile, paramètres et opérateurs des graphes figurent dans [l’inventaire JSON](raw/models.json). [Analyses singulières et activations](raw/weights.json.gz)
