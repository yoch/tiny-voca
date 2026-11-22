# analyze.py OUT ref build... : medians of synth, net+synth and wall time per config from tts_raw.txt, then WAV parity
import array, collections, math, statistics, sys, wave, os
out, builds = sys.argv[1], sys.argv[2:]
ref = builds[0]
d = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(f"{out}/tts_raw.txt"):
    f = dict(x.split("=", 1) for x in line.split() if "=" in x)
    if "synth_ms" not in f:
        continue
    key = (int(f["threads"]), f["text"], f["voice"], f["frames"])
    d[key][f["build"]].append((float(f["synth_ms"]), float(f["net_ms"]) + float(f["synth_ms"]), float(f["wall_ms"])))
med = lambda l, i: statistics.median(x[i] for x in l)
print("median of %d interleaved runs; ratio = build / %s" % (len(next(iter(d.values()))[ref]), ref))
hdr = f"{'thr':>3} {'text':6} {'voice':5} {'frames':>6} | {ref + ' synth':>10} {'net+synth':>9} {'wall':>6}"
for b in builds[1:]:
    hdr += f" | {b + ' synth':>11} {'(ratio)':>7} {'net+synth':>9} {'wall':>7}"
print(hdr)
for key in sorted(d, key=lambda k: (k[0], ["short", "medium", "long"].index(k[1]), k[2])):
    r = d[key][ref]
    s = f"{key[0]:>3} {key[1]:6} {key[2]:5} {key[3]:>6} | {med(r,0):10.2f} {med(r,1):9.2f} {med(r,2):6.0f}"
    for b in builds[1:]:
        x = d[key][b]
        s += f" | {med(x,0):11.2f} {med(x,0)/med(r,0):7.3f} {med(x,1)/med(r,1):9.3f} {med(x,2)/med(r,2):7.3f}"
    print(s)
print("\nWAV parity vs %s (16-bit samples)" % ref)
def read(p):
    w = wave.open(p); return array.array("h", w.readframes(w.getnframes()))
for i in range(1, 9):
    a = read(f"{out}/wav/u{i}_{ref}.wav")
    s = f"u{i} samples={len(a)}"
    for b in builds[1:]:
        c = read(f"{out}/wav/u{i}_{b}.wav")
        n = min(len(a), len(c))
        e = sum((a[j] - c[j]) ** 2 for j in range(n)); p = sum(x * x for x in a)
        s += f" | {b}: same_len={len(a) == len(c)} max_lsb={max(abs(a[j] - c[j]) for j in range(n))} snr={'inf' if e == 0 else round(10 * math.log10(p / e), 1)}"
    again = f"{out}/wav/u{i}_{builds[1]}_again.wav"
    if os.path.exists(again):
        s += f" | {builds[1]} rerun identical={read(again) == read(f'{out}/wav/u{i}_{builds[1]}.wav')}"
    print(s)
