# E04-C qualification of the opt-in compact Web vocoder

**Decision: PILOT ONLY.** Compact improves the interactive worker latency on
this host in Chromium, Firefox, and WebKit/WPE, with numerical parity against
dense. The existing page remains dense by default. Browser and native ABI
qualification does not replace blind human listening or physical Safari testing.

Starting PR #1 HEAD: `6ddf59e374df6838e273612eef7a28160877c3a6`.
Base: `db851f75c6287a15be4c4e995f931bce3a86b47b` (`main`).
The complete diff applied cleanly, the PR was open and in draft, and the
starting HEAD had no status checks, workflow runs, reviews or inline threads.

## Pilot integration and safety

- Enable explicitly with `?vocoder=compact`. An unflagged page uses dense.
  `&threads=4` requests four ORT threads on a cross-origin isolated page
  with at least four logical CPUs. DSP remains on the synthesis worker.
- The opt-in bundle is generated from an explicitly supplied checkpoint
  SHA-256, matching frontend identity, full checkpoint config, step,
  token inventory and voice list. It contains paired dense and compact graphs.
  The worker checks the ABI, geometry, asset lengths and SHA-256 against the
  same-origin config manifest. This assumes that the origin and manifest are
  trusted. Hashes guard against stale or mixed artifacts.
- Select dense when the flag is absent, the compact bundle is missing or
  incompatible, or WASM initialization, ORT or DSP fails. On rendering errors
  the worker releases the compact session before opening dense and retries
  the current utterance. It never holds two live ONNX sessions. Invalid
  synthesis controls are rejected before running either graph.
- Compact generates length-matched standard normal noise in the reusable WASM
  buffer via Box–Muller. Production dense keeps its ONNX random operator.
  The benchmark measures real random generation on both paths. Numerical
  fidelity uses a separate dense test graph with the identical noise
  samples as compact.
- The C ABI has aligned, caller-owned context storage and no mutable shared
  state or heap allocation. Initialization validates constant lengths and
  finite values and checks the first-hop normalization envelope. Checked
  entry points validate frame ranges, exact input lengths, output capacities
  and writable aliasing. Render rejects non-finite output. Contexts are
  immutable after initialization and concurrent calls need separate output
  buffers. Callers still own pointer accessibility and lifetimes.
- The arena grows geometrically only as needed, reusing it for shorter
  utterances, with an upper bound from the verified config. Each returned
  clip has its own samples because the next utterance can reuse the arena.
  The WASM stack reservation is 64 KiB. The historical E04-B adapter still
  wraps the same core for its old benchmark entry points.

## Whole worker performance

EPYC 9V74 with AVX2/AVX-512, affinity CPUs 0–3 and four CPU equivalents.
Playwright 1.65.1, ORT Web 1.20.1, Pyodide 0.26.4 and the same public
checkpoint as E04-B. All timed runs use the production worker source
with pinned local copies of its CDN packages. The timer includes the worker
message round trip, real phonemization, inference, noise generation,
transfers, normalization and waveform delivery. It excludes audio-device
scheduling.

For each browser and ORT thread count, two fresh contexts per variant ran
in opposite order, with five warmups and 15 timed requests per context.
The same voice, speed and pitch controls produce 48,896 samples (2.037 s).
Times are milliseconds. Ranges are across the two orders. p95 is an estimate
from 15 runs per order and is sensitive to contention.

| Browser / ORT threads | Dense median | Compact median | Dense p95 | Compact p95 | Compact noise median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chromium 159, 1 | 101.38–102.28 | 76.18–76.38 | 104.79–105.41 | 82.13–83.10 | 1.52–1.55 |
| Chromium 159, 4 | 45.25–54.26 | 36.18–36.75 | 65.70–110.95 | 75.00–92.31 | 1.55 |
| Firefox 159, 1 | 99.94–100.00 | 71.20–74.20 | 115.42–128.36 | 73.19–78.17 | 0.80 |
| Firefox 159, 4 | 44.24–44.30 | 34.42–36.82 | 51.12–53.96 | 37.34–45.81 | 0.82–0.84 |
| WebKit 27.2 WPE, 1 | 100.12–100.46 | 75.34–75.64 | 106.85–107.02 | 81.37–88.37 | 0.84–0.86 |
| WebKit 27.2 WPE, 4 | 44.00–44.32 | 36.06–36.44 | 48.31–53.99 | 38.43–41.23 | 0.84 |

Median improves in both orders in all six configurations. Chromium at four
threads has a higher compact p95 in the reverse order (92.31 vs 65.70 ms).
That makes one ORT thread the preferred first pilot configuration.

The compact worker reserves **983,040 bytes** of WASM linear memory for the
timed passage, including the stack/context and **884,712 bytes** of reusable
arena for 191 frames. A long text split at commas into ten chunks reached
**1,900,544 bytes** of WASM memory. E04-B's compact benchmark reserved 469
pages (29.31 MiB) at startup. These numbers measure the DSP memory only.
The browser sampler includes all Chromium processes, using Linux RSS at
100 ms intervals. Chromium one-thread sampled peak was about **747 MiB**
compact vs **767–789 MiB** dense. The four-thread ranges overlap
(791–796 MiB compact, 771–798 MiB dense). Firefox and WPE have exact DSP
arena counts and no validated whole-browser RSS measurement in this run.

Chromium compact initialization was 3.05–3.25 s vs 3.23–4.13 s dense on
localhost. First cold audio after Ready was 122–173 ms compact vs 145–222 ms
dense. Firefox and WPE initialization and first-audio samples are in the raw
records. Cold comparisons cover local assets, not WAN delivery or startup
of the browser process.

The real HTML page produced a playable WAV URL and started two WebAudio
sources in each engine. Worker checks exercised long-text splitting into
ten chunks, changes to speed and pitch, two voices, and independent noise
on repeated requests. Chromium injected a missing WASM and a render failure.
Each produced dense audio after the switch. Normal requests fetched only
the selected graph.

## Fidelity and support decision

With identical seeded samples, eight cases per engine per thread count
passed identical token durations and length, finite output, SNR >100 dB,
and normalized peak error <2e-5. The cases cover roughly 0.5–10 s,
fricatives, pitch changes, pauses and tails. Worst Web SNR is **115.08 dB**;
maximum normalized peak error is **3.06e-6**. Native checks include C
ASan/UBSan, capacity and alias fuzzing, different frame counts, reuse and
concurrent shared immutable contexts. The E04 export suite remains in place.

The French fork's model uses voice- and bin-dependent mixing of harmonic
pulses with reversed noise. A scalar time-domain pre-mix would change that
model. The FR parity adapter keeps the original mix in its ONNX exporter and
uses the shared checked C filter on the three original source/response pairs.
The compact English Web bundle rejects that architecture. The native FR
qualification uses the French fork's pinned code, its existing public
checkpoint and eSpeak, with no new training. French per-case parity is archived alongside the English native results.
Minimum FR SNR is 129.54 dB, peak error at most 8.11e-7.

| Platform | Evidence | Decision |
| --- | --- | --- |
| Chromium Linux, 1 ORT thread | Worker, player, fault recovery, two-order latency and fidelity pass | **PILOT ONLY** |
| Chromium Linux, 4 ORT threads | Median gain and fidelity pass; p95 regresses in one order | **PILOT ONLY**, prefer one thread |
| Firefox Linux, 1/4 ORT threads | Worker, player, latency and fidelity pass in the managed runner; content sandbox had to be disabled for the test instance | **PILOT ONLY** |
| WebKit/WPE Linux, 1/4 ORT threads | Worker, player, latency and fidelity pass using local runtime dependencies and the test instance sandbox disabled | **PILOT ONLY** |
| Safari on macOS/iOS | No Apple hardware or Safari application was available. WPE does not cover Apple audio and process behavior | **REJECT rollout** pending Safari evidence |
| Native x86-64/AVX2 EPYC, ORT 1/4 | Baseline and v3 builds pass numerical checks. No native selection in the Web pilot | **PILOT ONLY** |
| Native x86-64 without AVX2 | Checked C executable passes on QEMU Nehalem. No physical integrated benchmark | **REJECT rollout** |
| Native ARM64 | Checked C executable passes in QEMU AArch64. No physical integrated benchmark | **REJECT rollout** |
| Tiny Voca FR | Native late-boundary parity only. French frontend and mix are outside the Web pilot | **PILOT ONLY** for native evaluation; **REJECT** compact Web |

Blind human scores are still pending. The local listening kit contains six
randomized A/B pairs from E04-B and the new qualification, with equal pair
gain and the key kept outside the listening archive. Numerical equivalence
and successful AudioContext playback do not establish a human listening result.
No platform is marked PROMOTE.

[Setup and reproduction](REPRODUCE.md) · [Chromium](chromium.json.gz) ·
[Firefox](firefox.json.gz) · [WebKit](webkit.json.gz) · [CPU baseline](en-x86-64.json) ·
[CPU AVX2](en-avx2.json) · [Tiny Voca FR](fr.json) · [Environment](environment.json)
