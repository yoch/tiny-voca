---
language:
- en
license: apache-2.0
pipeline_tag: text-to-speech
library_name: onnx
base_model: n4ze3m/tiny-voca
datasets:
- n4ze3m/tiny-voca
tags:
- onnx
- onnxruntime
- int8
- text-to-speech
---

# Tiny Voca English ONNX

Two ONNX variants of [Tiny Voca](https://huggingface.co/n4ze3m/tiny-voca),
Muhammed Nazeem’s English text-to-speech model. Both use the public English
checkpoint and support the original F2 and M2 voices at 24 kHz.

| File | Precision | Size | Learned parameters | Qualification |
| --- | --- | ---: | ---: | --- |
| `exact.onnx` | FP32 | 21.36 MB | 4,537,291 | Numerically qualified reference |
| `exact-int8.onnx` | Dynamic INT8 acoustic projections | 15.68 MB | 4,537,291 | Recommended compact candidate. Human blind listening pending. |

Each file contains the complete synthesis graph, with standard ONNX opset 17,
internal noise generation and the original `audio` and `durations` outputs.
The graph operates on one utterance at a time. The tensor shapes, provenance
and artifact hashes are in [manifest.json](manifest.json).

## Inference

Install `numpy` and `onnxruntime`, then run the bundled example with the
pre-encoded sample.

```sh
sha256sum -c SHA256SUMS
python infer.py exact.onnx speech.wav --voice F2
python infer.py exact-int8.onnx compact.wav --voice M2
```

For custom text, install Dizione at the [pinned source
revision](https://github.com/n4ze3m/dizione/tree/8cd889e4de7be6bd1dd2e6891344a0729fa14dd4)
and set `DIZIONE_DATA` to the directory of the
[upstream phonemizer assets](https://huggingface.co/n4ze3m/tiny-voca/tree/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer).
Then add `--text "Your sentence."` to the example. Callers should split
long passages before synthesis, as the upstream demo does. The duration
output is measured in frames of 256 samples. `voice` is indexed as F2 = 0
and M2 = 1. `speed` and `pitch_range` are multipliers.
`pitch_shift` is in natural-log frequency units.

## Quality and latency

With shared noise, the exact FP32 graph preserved all durations and sample
counts in the native and Chromium tests. Minimum waveform SNR was 143.42 dB
on native ONNX Runtime and 140.53 dB on ONNX Runtime Web. Across 100 English
texts in both voices, the INT8 variant preserved the public graph’s WER and
CER counts per voice (3.11% WER and 1.96% CER overall). It preserved durations
and sample counts and had a median log-mel error of 0.15 dB. An independent
24-render difficult set also passed all duration and length checks. ASR is a
proxy for intelligibility. Human blind listening remains pending for INT8.

The table reports median text-to-audio latency reduction for one 1.97-second
passage on an AMD EPYC 9V74. The ranges come from two reversed orders. These
end-to-end measures include phonemization and waveform delivery.

| Variant | Native 1 thread | Native 4 threads | Chromium 1 thread | Chromium 4 threads |
| --- | ---: | ---: | ---: | ---: |
| FP32 exact | 6.7–10.6% | 8.3–9.3% | 2.7–5.0% | 8.0–10.2% |
| INT8 | 24.3–26.0% | 18.6–20.4% | 15.0–16.5% | 14.7–14.9% |

Tested with ONNX Runtime 1.32.0 on CPU and Chromium 159 with ONNX Runtime Web
1.20.1 WASM on Linux. The short-utterance results at four Chromium threads
are workload-dependent. Mobile, Safari, network startup and other CPU
architectures were outside this benchmark. Dynamic quantization reduces
storage by 26.7% and retains the original number of learned parameters.
See the [benchmark and raw results](https://github.com/yoch/tiny-voca/pull/2).

## License and attribution

The ONNX graphs and example are distributed under Apache-2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). Credit Muhammed Nazeem and cite the
[original Tiny Voca release](https://huggingface.co/n4ze3m/tiny-voca) and
[CITATION.cff](CITATION.cff). The optional Dizione code and language data
come from their upstream repositories, with their own accompanying license
files. See the [upstream phonemizer licenses](https://huggingface.co/n4ze3m/tiny-voca/tree/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer).
