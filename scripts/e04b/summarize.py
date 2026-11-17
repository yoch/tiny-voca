"""Summarize timings with deterministic bootstrap intervals for the median ratio."""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('total_ms','first_audio_ms','frontend_ms','ort_ms','dsp_ms','js_to_wasm_ms','wasm_to_js_ms','publish_ms')


def stats(values):
    a = np.array(values)
    q1, median, q3, p95 = np.percentile(a,[25,50,75,95])
    return dict(median=float(median), p95=float(p95), iqr=float(q3-q1))


def summarize(root):
    rng = np.random.default_rng(42)
    reports = {}
    for platform in ('cpu','web'):
        for order in ('forward','reverse'):
            name = f'{platform}-{order}'
            raw = json.loads((root/(name+'.json')).read_text())
            indexed = {(case['threads'],case['variant'],row['case']): row
                       for case in raw['cases'] for row in case['rows']}
            for (threads,variant,case),row in indexed.items():
                base = indexed[threads,'dense',case]
                values = np.array([t['total_ms'] for t in row['timings']])
                dense = np.array([t['total_ms'] for t in base['timings']])
                reduced = 100*(1 - np.median(values)/np.median(dense))
                if variant == 'dense':
                    interval = [0.,0.]
                else:
                    m = np.median(rng.choice(values,size=(5000,len(values))),axis=1)
                    d = np.median(rng.choice(dense,size=(5000,len(dense))),axis=1)
                    interval = np.percentile(100*(1-m/d),[2.5,97.5]).tolist()
                key = f'{name}:{threads}:{case}:{variant}'
                reports[key] = dict(latency_reduction_pct=float(reduced), bootstrap_95_pct=interval,
                                    metrics={f:stats([t[f] for t in row['timings']]) for f in FIELDS if f in row['timings'][0]},
                                    parity=row['parity'], boundary_bytes=row['boundary_bytes'])
    return reports


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root),separators=(',',':')))
