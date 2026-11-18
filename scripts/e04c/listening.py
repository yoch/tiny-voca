"""Make an offline blind A/B booklet from E04-B and E04-C audio pairs."""
import argparse
import html
import json
from pathlib import Path
import secrets
import zipfile

import numpy as np
import soundfile as sf


def build(result, previous, output):
    pairs = []
    for name in ('web-0.5s', 'web-1s'):
        pairs.append((previous/f'{name}-dense.wav', previous/f'{name}-compact.wav'))
    for name in ('fricatives','transition','silence-tail','10'):
        pairs.append((result/f'{name}-dense.wav', result/f'{name}-compact.wav'))
    output.mkdir(parents=True, exist_ok=True)
    rows, key = [], {}
    for number, (dense, compact) in enumerate(pairs,1):
        group = f'P{number:02d}'
        a, rate = sf.read(dense,dtype='float32')
        b, other_rate = sf.read(compact,dtype='float32')
        if a.shape != b.shape or rate != other_rate:
            raise ValueError('Pair must have identical shape and sample rate')
        gain = min(1,.99/max(float(np.abs(a).max()),float(np.abs(b).max()),.001))
        kinds = ['dense','compact']
        if secrets.randbelow(2):
            a,b = b,a
            kinds.reverse()
        key[group] = dict(A=kinds[0],B=kinds[1],source=dense.stem)
        for name, audio in zip(('A','B'),(a,b)):
            sf.write(output/f'{group}-{name}.wav',audio*gain,rate,subtype='PCM_16')
        cells = ''.join(f'<td><audio aria-label="{group} {label}" controls preload="none" src="{group}-{label}.wav"></audio></td>'
                        for label in ('A','B'))
        rows.append(f'<tr><th scope="row">{html.escape(group)}</th>{cells}</tr>')
    (output/'index.html').write_text('''<!doctype html>
<meta charset="utf-8"><title>Écoute A/B Tiny Voca</title>
<style>
body {font: 17px system-ui; color: #162334; margin: 2rem auto; max-width: 54rem; padding: 0 1rem}
table {border-collapse: collapse; width: 100%} td,th {border-bottom: 1px solid #ccc; padding: .65rem}
audio {width: min(100%, 21rem)}
</style>
<h1>Écoute A/B Tiny Voca</h1>
<p>Écoutez chaque paire au casque et au même volume. Notez si vous pouvez distinguer les deux versions,
votre préférence éventuelle et l'instant d'un artefact. Gardez la clé fermée jusqu'à la fin.</p>
<table><thead><tr><th>Paire</th><th>A</th><th>B</th></tr></thead><tbody>
'''+'\n'.join(rows)+'</tbody></table>\n',encoding='utf-8')
    (output/'key.json').write_text(json.dumps(key,indent=2)+'\n')
    with zipfile.ZipFile(output/'ecoute-aveugle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(output/'index.html','index.html')
        for group in key:
            for label in ('A','B'):
                name = f'{group}-{label}.wav'
                archive.write(output/name,name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results','previous','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    build(args.results,args.previous,args.output)
