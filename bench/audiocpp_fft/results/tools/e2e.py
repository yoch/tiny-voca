"""End-to-end audiocpp_cli runs, interleaved: e2e.py ROUNDS OUT_DIR NAME=BIN_DIR NAME=BIN_DIR ...
Each round runs every case with every binary (order alternating); reports median wall time, RTF, peak RSS,
whether every round gave the same WAV, and the WAV parity of each binary against the first one."""
import os, statistics, subprocess, sys, time, json
import numpy as np, soundfile as sf
rounds, out = int(sys.argv[1]), sys.argv[2]
bins = [a.split("=", 1) for a in sys.argv[3:]]
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = f"{W}/acpp/models/tiny-voca-GGUF/tiny-voca-orig.gguf"
texts = {
    "short": "Hello! This is tiny voca.",
    "medium": "The eleven o'clock train is now arriving on platform two, please stand back from the edge.",
    "long": ("In twenty fourteen, the city had more than three hundred thousand residents, according to official figures "
             "published last spring. Most of them live within ten kilometres of the old harbour, where the first houses were "
             "built. Sometimes it takes a while to understand what is really going on, and even longer to explain it simply. "
             "The council has promised new trains, wider pavements and a quieter centre, but nobody yet knows who will pay "
             "for them. For now, the ferries still leave every twenty minutes, and the morning fog still rolls in from the sea."),
}
cases = [(f"{t}_{v}", v, texts[t]) for t in texts for v in ("M2", "F2")]
res = {(b, c[0]): [] for b, _ in bins for c in cases}
hashes = {}
for r in range(rounds):
    for c in cases:
        order = bins if r % 2 == 0 else bins[::-1]
        for name, bdir in order:
            os.makedirs(f"{out}/{name}", exist_ok=True)
            wav = f"{out}/{name}/{c[0]}.wav"
            cmd = [f"{bdir}/audiocpp_cli", "--task", "tts", "--family", "tiny_voca", "--model", model, "--backend", "cpu",
                   "--text", c[2], "--voice-id", c[1], "--seed", "0", "--out", wav]
            t0 = time.perf_counter()
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _, status, ru = os.wait4(p.pid, 0)
            wall = time.perf_counter() - t0
            err = p.stderr.read().decode(errors="replace")
            if status != 0:
                sys.exit(f"{name} {c[0]} failed: {err[-2000:]}")
            data, rate = sf.read(wav, dtype="float64")
            hashes.setdefault((name, c[0]), set()).add(hash(data.tobytes()))
            res[(name, c[0])].append((wall, ru.ru_maxrss / 1024, len(data) / rate))
print(f"{'case':10s} {'audio s':>8s} " + " ".join(f"{n:>34s}" for n, _ in bins))
for c in cases:
    cells = []
    for n, _ in bins:
        v = res[(n, c[0])]
        wall = statistics.median(x[0] for x in v); rss = max(x[1] for x in v); dur = v[0][2]
        cells.append(f"{wall*1000:8.1f} ms RTF {wall/dur:6.4f} RSS {rss:6.1f} MB")
    print(f"{c[0]:10s} {res[(bins[0][0], c[0])][0][2]:8.2f} " + " ".join(f"{x:>34s}" for x in cells))
for n, _ in bins:
    tot = [sum(res[(n, c[0])][r][0] for c in cases) for r in range(rounds)]
    print(f"{n}: total wall median {statistics.median(tot)*1000:.1f} ms over {len(cases)} cases; deterministic across rounds: {all(len(hashes[(n, c[0])]) == 1 for c in cases)}")
base = bins[0][0]
for n, _ in bins[1:]:
    ratios = sorted(sum(res[(n, c[0])][r][0] for c in cases) / sum(res[(base, c[0])][r][0] for c in cases) for r in range(rounds))
    print(f"paired ratio {n}/{base}: median {statistics.median(ratios):.3f} (min {ratios[0]:.3f}, max {ratios[-1]:.3f})")
    for c in cases:
        a, _ = sf.read(f"{out}/{base}/{c[0]}.wav", dtype="float64"); b, _ = sf.read(f"{out}/{n}/{c[0]}.wav", dtype="float64")
        same_len = a.shape == b.shape
        m = min(len(a), len(b)); d = a[:m] - b[:m]
        snr = 10 * np.log10((a[:m] ** 2).sum() / max((d ** 2).sum(), 1e-300))
        info = sf.info(f"{out}/{base}/{c[0]}.wav")
        lsb = np.abs(np.round(a[:m] * 32768) - np.round(b[:m] * 32768)).max() if info.subtype.startswith("PCM_16") else float("nan")
        print(f"  WAV {c[0]:10s} {info.subtype}: same length {same_len} ({len(a)} vs {len(b)}), SNR {snr:6.1f} dB, max diff {lsb:.0f} LSB, differing samples {np.count_nonzero(d)}")
