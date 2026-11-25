# tiny-voca

> **This is the `french` branch of a fork.** It turns tiny-voca into a French
> model: an eSpeak NG (`fr`) front end with French rules for liaisons, elisions
> and words eSpeak reads wrongly, a French test suite, multi-GPU training and
> the scripts that built the French data. Its weights are
> [ymelka/tiny-voca-french](https://huggingface.co/ymelka/tiny-voca-french)
> (five voices, 4.5M parameters), with a
> [demo](https://huggingface.co/spaces/ymelka/tiny-voca-french). This branch
> reads French only: for the English model, use
> [n4ze3m/tiny-voca](https://github.com/n4ze3m/tiny-voca).
>
> ```bash
> sudo apt install espeak-ng      # tested with 1.51
> pip install "tiny-voca @ git+https://github.com/yoch/tiny-voca@french-v1"
> hf download ymelka/tiny-voca-french --local-dir tiny-voca-french
> python -m tiny_voca.infer --checkpoint tiny-voca-french/tiny-voca-french.pt \
>   --voice F1 --text "Bonjour ! Ceci est tiny voca, en français." --output bonjour.wav
> ```
>
> The rest of this page is upstream's README, about the English model.

tiny-voca is a small English text-to-speech model. It has two voices, 4.5M
parameters, and runs on a CPU or in the browser.

[Demo](https://tiny-voca.n4ze3m.com) · [Weights](https://huggingface.co/n4ze3m/tiny-voca) · [Dataset](https://huggingface.co/datasets/n4ze3m/tiny-voca)

## Features

- **Small**: 4,537,291 parameters, 18 MB, waveform generator included.
- **Fast**: about 70x real time on four CPU threads.
- **Two voices**: F2 (female) and M2 (male), American English, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, and ONNX Runtime Web.
- **Open**: code, weights, training recipe and dataset. Trained from scratch.

## Install

```bash
git clone https://github.com/n4ze3m/tiny-voca.git
cd tiny-voca
pip install -e ".[own]"
hf download n4ze3m/tiny-voca --local-dir weights
export DIZIONE_DATA=weights/phonemizer
```

## Usage

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca.pt \
  --voice F2 \
  --text "Hello! This is tiny voca." \
  --output hello.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text` |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Raise or lower the pitch |
| `--pitch-range` | 1.0 | More or less pitch movement |

Long text is fine. It is read one sentence at a time and the audio is joined.
A text that ends without a mark gets a full stop, so the last word does not
break off. A sentence with nothing to read, such as one in another script, is
skipped. The output never peaks above 0.99, so it does not clip as a 16-bit
WAV. In Python, `tiny_voca.infer.speak_text` does the same.

## ONNX

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca.pt --output tiny-voca.onnx
```

`--optimized` writes a smaller graph with the same audio, up to float
rounding. In ONNX Runtime on a CPU it is about 10% faster on a 3-second
sentence and 25 to 40% faster on a 10-second one. It needs opset 16 or later.

A ready file is in the weights repository as `onnx/model.onnx`. The
[model card](https://huggingface.co/n4ze3m/tiny-voca) shows how to call it.

## Web demo

The [demo](https://tiny-voca.n4ze3m.com) runs the model in the browser. It
plays each sentence as soon as it is made, and keeps the whole clip in the
player. Its source is in `web/`.

```bash
cd web
bash fetch_assets.sh /path/to/dizione
npx wrangler dev
```

## Plain JavaScript

An experiment in `js/`: the model and the phonemizer in plain scripts, with no
libraries. It reads one 8 MB file and runs in the browser, in Node and in Bun.

```bash
node js/say.mjs "Hello! This is tiny voca." --voice M2 --output hello.wav
```

See [js/README.md](js/README.md).

## Training

```bash
export TINY_VOCA_FRONTEND=own
python -m tiny_voca.prepare --inputs data/train-*.parquet --split train --output features
python -m tiny_voca.annotate --features features --output annotations
python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/NAME --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
python -m tiny_voca.evaluate --checkpoint runs/NAME/inference.pt --annotations annotations --output eval/NAME
```

The released weights took 7.7 hours on one RTX 4090. More about the model and
the data is in [docs/model.md](docs/model.md).

## Tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests
```

## Limits

- English only. Two fixed voices. No voice cloning.
- The F2 voice sounds younger than intended. A more mature voice is planned.
- Foreign names and odd spellings are often read wrong.
- No streaming yet.

## Community

Built by other people on top of tiny-voca.

Languages and voices:

- [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts): German.
- [apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at): Austrian German, with an accent you can turn up or down and a [browser demo](https://tiny-tts.apptor.at).
- [tiny-voca-de-en](https://huggingface.co/Fraunhofer-IEM/tiny-voca-de-en): German and English, with up to 32 voices in each.
- [tiny-voca-nl](https://github.com/tetsuo/tiny-voca-nl): Dutch, one voice, trained from scratch on an M4 Pro in about 5 hours, with a [browser demo](https://huggingface.co/spaces/xxy998/tiny-voca-nl).
- [tiny-voca-it](https://github.com/carbogninalberto/tiny-voca-it): Italian.
- [vocina](https://github.com/OfficinePixel/vocina): Italian, with a browser demo.
- [zerre-tts](https://github.com/erkinemreta1/zerre-tts): Turkish.
- [clarvia-tts-tiny](https://huggingface.co/wite-tech/clarvia-tts-tiny): Turkish, ten voices, with its own [phonemizer](https://huggingface.co/wite-tech/clarvia-phonemizer) for numbers, dates and money.
- [tiny-voca-uzbek-navoiy](https://huggingface.co/Inomjonov/tiny-voca-uzbek-navoiy): Uzbek, one male voice, trained from scratch on 7.7 hours of Navoiy-generated speech.
- [korean-tts](https://github.com/ooodev/korean-tts): Korean, and Korean mixed with English.
- [tiny-voca-ja-tsukuyomi-chan](https://huggingface.co/ayousanz/tiny-voca-ja-tsukuyomi-chan): Japanese, in Tsukuyomi-chan's voice, under that voice's own terms.
- [tiny-voca-ja](https://github.com/gachimoto-code/tiny-voca-ja): Japanese, two voices, trained on synthetic speech only.
- [tiny-voca-zh](https://github.com/xiaoliukuailiu/tiny-voca-zh): Mandarin Chinese, two voices, trained on synthetic speech only.
- [tiny-voca-zh](https://github.com/blue-buff/tiny-voca-zh): Chinese, with male (`M`) and female (`F`) voices and a dedicated Chinese text front end.
- [slyszeglosy](https://github.com/damaon/slyszeglosy): Polish.
- [tiny-voca-ru](https://github.com/yilbegan/tiny-voca-ru): Russian, with a [browser demo](https://huggingface.co/spaces/egnch/tiny-voca-ru-demo).
- [tiny-voca-gr](https://huggingface.co/ComputationalLinguisticsPlatform/tiny-voca-gr): Modern Greek, a male and a female voice.
- [mini-voz](https://github.com/ElMariones/mini-voz): Spanish, with a Castilian accent.
- [tiny-voca-es](https://github.com/stdevMac/tiny-voca): Spanish from Spain, with a browser demo.
- [alpa-es](https://huggingface.co/FranciscoCarlosErra/alpa-es): Rioplatense Spanish.
- [tiny-voca-fr](https://github.com/DanielSuncost/tiny-voca-fr): French, with a browser demo.
- [tiny-voca-ptbr](https://github.com/SemFence/tiny-voca-ptbr): Brazilian Portuguese, with its own phonemizer.
- [tiny-voca-ptbr-kokoro](https://github.com/Haple/tiny-voca-ptbr-kokoro): Brazilian Portuguese, with Kokoro's Brazilian voices as the teacher.
- [tiny-broca](https://huggingface.co/brenorb/tiny-broca): Brazilian Portuguese, two voices, reads letters with no dictionary, with a [browser demo](https://tiny-broca-demo.pages.dev).
- [tiny-voca PT-PT](https://huggingface.co/joserui/tiny-voca-pt-pt): European Portuguese, two voices, trained from scratch on 22.8 hours of synthetic speech, with a rule-based phonemizer and a [browser demo](https://joseruimemory.github.io/tiny-voca-pt-pt/).
- [tiny-voca Arabic](https://github.com/BenkadourKhaled/tiny-voca/tree/arabic-halabi-f2): Modern Standard Arabic, trained on a CPU from the Arabic Speech Corpus.
- [tiny-voca-ar](https://huggingface.co/abdullah/tiny-voca-ar): Modern Standard Arabic, an early version with a [browser demo](https://tiny-voca-ar-site.pages.dev).
- [tiny-voca-arabic](https://huggingface.co/oddadmix/tiny-voca-arabic): Modern Standard Arabic, trained from scratch, with a [browser demo](https://huggingface.co/spaces/oddadmix/tiny-voca-arabic).
- [tiny-voca-he](https://github.com/buzagloidan/tiny-voca-he): Hebrew, one male voice, with a browser demo.
- [pitput](https://huggingface.co/useless-ai/pitput): Hebrew, eight delivery styles and expression tags like laughs and sighs, for research use only.
- [Browser Voice](https://tts.hansda.in/): Bengali, Gujarati, Hindi, Malayalam and Telugu in native script or Roman letters, with a browser demo and its [code](https://github.com/DarvinX/multilingual-browser-tts).
- [tiny-swara](https://huggingface.co/Xeven7/tiny-swara): Bengali, two voices, trained from scratch on free Kaggle GPUs.
- [tiny-voca-hindi](https://huggingface.co/RL10x/tiny-voca-hindi): Hindi, one voice with eight speaking styles, and a C++ engine.
- [tui-voice-mini-ml](https://huggingface.co/Cyrax321/tui-voice-mini-ml): Malayalam, an early preview.
- [tiny-voca-sinhala](https://github.com/Emindu/tiny-voca-sinhala): Sinhala, a male and a female voice, trained from scratch on free Kaggle GPUs, with a [browser demo](https://huggingface.co/spaces/eminduperera/tiny-voca-sinhala).
- [tiny-urdu](https://github.com/Zafar205/tiny-urdu): Roman Urdu, with a browser demo.
- [Kurdish Behdînî Voice](https://behdinan.com/): Kurdish Behdînî, 21 voices, with a browser demo.
- [tiny-ava](https://github.com/mallahyari/tiny-ava): Persian (Farsi), one female voice, trained from scratch on about 20 hours of synthetic speech, with a [browser demo](https://huggingface.co/spaces/mehdi-hf/tiny-ava-demo).
- [tiny-voca Kiswahili](https://huggingface.co/adoamesh/tiny-voca-swahili): Swahili, runs offline on a low-end Android phone.
- [tiny-voca-lg](https://github.com/henryhale/tiny-voca-lg): Luganda, two voices, with a rule-based front end and a [browser demo](https://henryhale.github.io/tiny-voca-lg/).
- [SobbieTTS](https://huggingface.co/PatnaikAshish/SobbieTTS): English, an expressive voice.
- [alpa](https://huggingface.co/77ethers/alpa): Sanskrit, fine-tuned from tiny-voca on a CPU.
- [fiato](https://huggingface.co/cfahlgren1/fiato): English, with sighs, exhales and gasps.

Ports and apps:

- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/tts.md#tiny-voca): runs tiny-voca with `llama-tts`, from [GGUF files](https://huggingface.co/ggml-org/tiny-voca-GGUF).
- [voca](https://github.com/sultson/voca): a rewrite in Rust.
- [tiny-voca-rust](https://github.com/muzapparjan/tiny_voca_rust): another Rust port.
- [audio.cpp](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/tiny_voca.md): runs tiny-voca in C++ with ggml.
- [voca-server](https://github.com/Wereling1992/voca-server): a server with an OpenAI-compatible API.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): runs on an ESP32-S3 microcontroller.
- [tiny-voca-ane](https://github.com/razorback16/tiny-voca-ane): runs on the Apple Neural Engine with Core ML, with a Swift package.
- [tiny-voca-dashboard](https://github.com/kleenpulse/tiny-voca-dashboard): a local dashboard.
- [tinyvoca](https://github.com/spurious-cox/tinyvoca): a Mac app.
- [CopySpeech](https://github.com/ilyaizen/CopySpeech): reads your clipboard aloud, with tiny-voca as the default voice.
- [Tiny Reader](https://github.com/GitThisJoel/tiny-reader): a Firefox add-on that reads web pages and PDFs aloud.
- [tiny-voca GUI](https://github.com/IggyLux/tiny-voca-GUI): a simple local web interface.
- [TinyVocaTTS](https://github.com/RennexTech/TinyVocaTTS): a one-click Windows interface that can also load the community voices.
- [Tiny-Voca-Pinokio](https://github.com/PierrunoYT/Tiny-Voca-Pinokio): a one-click launcher for Pinokio.

Same idea, other sounds:

- [tinygroove](https://github.com/totomanov/tinygroove): a drum machine with no samples.

Write-ups:

- [Three small TTS models](https://ai.satyajit.si/articles/tiny-tts-roundup): how tiny-voca works, measured next to Kitten TTS and Tontaube on one CPU.
- [tiny-voca review](https://andrew.ooo/posts/tiny-voca-review-4-5m-parameter-cpu-text-to-speech/): what it does well, where it breaks and how to run it.
- [AlphaSignal](https://alphasignal.ai/model/tiny-voca-packs-two-ai-voices-into-an-18-mb-model-for-9): a short summary of the model, the data and the limits.

Made something? Open an issue and I will add it.

## Cost

The whole project cost $9.56 of GPU time on one rented RTX 4090. That is the
dataset, the failed tries and the final training run.

![Cost breakdown](docs/cost.png)

## Support

It cost less than a pizza to make. If you like it, you can buy me one:
https://ko-fi.com/n4ze3m

## License

Apache-2.0 for code and weights. CC BY 4.0 for the dataset.

## Citation

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
