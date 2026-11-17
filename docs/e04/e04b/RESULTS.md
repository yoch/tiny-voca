# E04-B benchmark results

## Two-second case and Pareto comparison

Ranges span both orders. Times are milliseconds. RSS is peak CPU-process RSS or sampled sum of browser processes, with frontend and runtime loaded. The two measurements use different scopes. Bootstrap intervals and all five lengths are in [LATENCY.md](LATENCY.md).

| Runtime / threads | Variant | Median | p95 | RSS MiB | Warm first audio | Cold load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU 1 | dense | 29.87–29.89 | 30.47–42.53 | 423.5–424.4 | 29.87–29.89 | 122.99–127.57 |
| CPU 1 | numpy | 25.21–25.54 | 25.54–34.29 | 437.5–439.0 | 25.21–25.54 | 115.83–129.34 |
| CPU 1 | fused | 24.46–24.56 | 25.05–25.22 | 413.1–413.8 | 24.46–24.56 | 116.90–126.61 |
| CPU 1 | compact | 24.44–25.16 | 26.33–26.37 | 390.1–390.8 | 24.44–25.16 | 104.15–105.50 |
| CPU 4 | dense | 15.75–16.20 | 21.36–22.37 | 421.7–423.4 | 15.75–16.20 | 127.28–130.56 |
| CPU 4 | numpy | 15.83–15.91 | 18.84–25.92 | 436.8–437.2 | 15.83–15.91 | 114.51–114.53 |
| CPU 4 | fused | 14.19–15.76 | 15.18–21.27 | 411.7–414.1 | 14.19–15.76 | 126.81–136.23 |
| CPU 4 | compact | 15.10–15.18 | 15.78–18.36 | 393.0–393.0 | 15.10–15.18 | 105.79–114.18 |
| WEB 1 | dense | 96.73–102.13 | 107.64–108.07 | 754.2–784.6 | 96.70–102.10 | 3105.29–3247.65 |
| WEB 1 | fused | 74.93–76.69 | 80.90–94.22 | 763.9–791.5 | 74.85–76.62 | 3135.96–3205.00 |
| WEB 1 | compact | 72.07–73.42 | 76.34–84.53 | 774.1–816.5 | 72.03–73.37 | 3206.21–3264.00 |
| WEB 4 | dense | 40.84–43.71 | 50.44–52.99 | 760.5–806.8 | 40.81–43.67 | 3258.83–3285.91 |
| WEB 4 | fused | 32.49–34.64 | 42.18–47.08 | 778.7–860.5 | 32.43–34.61 | 3000.69–3271.35 |
| WEB 4 | compact | 32.97–33.09 | 37.78–39.76 | 769.6–832.6 | 32.93–33.05 | 3152.26–3196.10 |

The graphs and native/WASM module sizes are recorded in [environment.json](environment.json). Dense is one graph and the lowest-integration-cost fallback. NumPy adds Python/NumPy and a large tensor boundary. Fused adds a native library or 23 KiB WASM module, two plans, and the same late boundary. Compact shares the C core, adds cepstrum work and lowers that boundary by approximately 85%. F0-only is excluded from the Pareto set due to fidelity.

## Boundary and DSP phases

Two-second case, medians in ms across both orders. ORT includes tensor creation/materialization inside the runtime. CPU passes contiguous ndarray buffers to C. The browser times the explicit host-to-DSP and DSP-to-host copies separately.

| Runtime / threads | Variant | Boundary bytes | Frontend / feeds | ORT | DSP | JS → WASM | WASM → JS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU 1 | dense | 195,856 | 0.356–0.363 | 29.364–29.428 | 0.009–0.009 | — | — |
| CPU 1 | numpy | 1,974,544 | 0.344–0.370 | 21.818–21.969 | 3.005–3.058 | — | — |
| CPU 1 | fused | 1,974,544 | 0.336–0.343 | 21.908–22.053 | 2.050–2.056 | — | — |
| CPU 1 | compact | 298,232 | 0.336–0.359 | 20.273–20.940 | 3.693–3.693 | — | — |
| CPU 4 | dense | 195,856 | 0.338–0.348 | 15.342–15.776 | 0.007–0.007 | — | — |
| CPU 4 | numpy | 1,974,544 | 0.345–0.351 | 12.408–12.586 | 2.991–2.991 | — | — |
| CPU 4 | fused | 1,974,544 | 0.305–0.372 | 11.805–12.989 | 2.023–2.206 | — | — |
| CPU 4 | compact | 298,232 | 0.328–0.336 | 10.971–11.036 | 3.670–3.708 | — | — |
| WEB 1 | dense | 195,856 | 0.820–0.850 | 95.585–100.637 | 0.000–0.000 | 0.000–0.000 | 0.000–0.000 |
| WEB 1 | fused | 1,974,544 | 0.815–0.853 | 70.243–72.425 | 2.650–2.652 | 0.115–0.147 | 0.057–0.060 |
| WEB 1 | compact | 298,232 | 0.833–0.840 | 65.462–66.642 | 5.292–5.310 | 0.022–0.025 | 0.055–0.065 |
| WEB 4 | dense | 195,856 | 0.765–0.838 | 39.530–42.377 | 0.000–0.000 | 0.000–0.000 | 0.000–0.000 |
| WEB 4 | fused | 1,974,544 | 0.740–0.785 | 28.657–30.197 | 2.598–2.603 | 0.110–0.120 | 0.055–0.060 |
| WEB 4 | compact | 298,232 | 0.743–0.778 | 25.867–26.405 | 5.265–5.580 | 0.020–0.020 | 0.060–0.060 |

As a rough upper bound on a late-boundary rewrite, the median ORT-time gap between dense and fused is the time removed from the graph before the replacement DSP and copies. This is an empirical Amdahl estimate. It includes graph/runtime overhead changes and should not be interpreted as a pure kernel trace.

| Runtime / threads | Dense → late ORT-time gap (ms) | Fraction of dense total |
| --- | ---: | ---: |
| CPU 1 | 7.38–7.46 | 24.7–25.0% |
| CPU 4 | 2.35–3.97 | 14.9–24.5% |
| WEB 1 | 23.16–30.39 | 23.9–29.8% |
| WEB 4 | 10.87–12.18 | 26.6–27.9% |

## Fidelity by runtime

All candidates listed here preserved exact per-token durations and sample counts. Raw comparisons use dense from the same runtime and thread count. The maximum normalized peak error and minimum finite SNR cover all five cases and both orders.

| Runtime | Variant | Worst normalized peak | Minimum SNR (dB) |
| --- | --- | ---: | ---: |
| CPU | numpy | 5.05e-07 | 130.68 |
| CPU | fused | 7.93e-07 | 126.62 |
| CPU | compact | 2.4e-06 | 115.28 |
| WEB | fused | 8.69e-07 | 127.05 |
| WEB | compact | 2.69e-06 | 115.56 |

## Parallel CPU DSP probe

Four ORT threads. The pooled variant starts the noise branch on one reusable worker while the caller filters the harmonic branch. The stage above DSP is identical. Each cell gives the range of medians across both orders.

| Variant | 0.5 s | 1 s | 2 s | 5 s | 10 s | DSP at 2 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 4.93–5.11 | 8.56–8.62 | 15.42–15.48 | 39.14–39.85 | 99.18–101.84 | 0.006–0.007 |
| fused | 4.44–4.51 | 7.72–8.72 | 14.88–14.95 | 37.44–38.47 | 96.35–105.00 | 2.059–2.064 |
| fused2 | 4.32–4.48 | 7.87–7.88 | 15.03–17.29 | 38.04–38.14 | 99.70–99.98 | 2.145–2.169 |
