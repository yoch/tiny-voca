"""Interleaved timing: cmp.py ROUNDS RUNS bin1 bin2 ...  (each round runs every binary once with RUNS inner runs,
pinned to one core). Prints the median over rounds of each case and of the total, in ms and us/frame."""
import re, statistics, subprocess, sys
rounds, runs, bins = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
W = __file__.rsplit("/tools/", 1)[0]
res = {b: {} for b in bins}; frames = {}
for r in range(rounds):
    for b in bins:
        out = subprocess.run(["taskset", "-c", "2", f"{W}/bin/{b}", f"{W}/dumps", runs], capture_output=True, text=True, check=True).stdout
        for m in re.finditer(r"^(\S+)\s+(\d+) frames\s+synth\s+([\d.]+) ms\s+\(pulse_train\s+([\d.]+)", out, re.M):
            res[b].setdefault(m[1], []).append((float(m[3]), float(m[4]))); frames[m[1]] = int(m[2])
total = sum(frames.values())
print(f"{'case':9s} {'frames':>6s} " + " ".join(f"{b:>22s}" for b in bins))
for c in frames:
    print(f"{c:9s} {frames[c]:6d} " + " ".join(f"{statistics.median(x[0] for x in res[b][c]):11.3f} ms ({statistics.median(x[1] for x in res[b][c]):6.3f})" for b in bins))
for b in bins:
    tots = [sum(res[b][c][r][0] for c in frames) for r in range(rounds)]
    ptot = [sum(res[b][c][r][1] for c in frames) for r in range(rounds)]
    print(f"{b:24s} total {statistics.median(tots):8.2f} ms = {1000*statistics.median(tots)/total:6.2f} us/frame (min {1000*min(tots)/total:6.2f}, max {1000*max(tots)/total:6.2f}); pulse_train {1000*statistics.median(ptot)/total:5.2f} us/frame")
base = bins[0]
for b in bins[1:]:
    ratios = sorted(sum(res[b][c][r][0] for c in frames) / sum(res[base][c][r][0] for c in frames) for r in range(rounds))
    q = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]
    print(f"paired ratio {b}/{base}: median {statistics.median(ratios):.3f} (quartiles {q(0.25):.3f} .. {q(0.75):.3f})")
