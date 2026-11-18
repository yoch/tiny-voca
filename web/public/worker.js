// Runs the phonemizer (dizione, through Pyodide) and the model (ONNX Runtime Web)
// off the main thread. Messages in: {type: "load"} and {type: "speak", id, text,
// voice, speed, pitchShift?, pitchRange?}. Messages out: status, ready, sentence (with its audio, as soon as
// it is made), done (with the whole clip), error (plain: true when the message is
// meant for the reader as it is).

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const DATA = ["lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json"];

importScripts(PYODIDE + "pyodide.js", ORT + "ort.wasm.min.js");

let pyodide, phonemize, session, config, compact, threads;
let ready = false, queue = Promise.resolve();
const flags = new URL(self.location.href).searchParams;
const requestCompact = flags.get("vocoder") === "compact";

const status = (text, progress) => postMessage({ type: "status", text, progress });

// Bump when the files under assets/ change, so browsers fetch them again.
const VERSION = "2";

// Downloaded files are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label, revision = VERSION) {
  const key = `${url}?v=${revision}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("Loading the phonemizer");
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage("numpy");
  const [source, ...files] = await Promise.all([
    download("assets/dizione.zip", "Loading the phonemizer"),
    ...DATA.map((name) => download("assets/phonemizer/" + name, "Loading the dictionary")),
  ]);
  status("Starting the phonemizer");
  pyodide.FS.mkdirTree("/dizione-data");
  DATA.forEach((name, index) => pyodide.FS.writeFile("/dizione-data/" + name, files[index]));
  pyodide.unpackArchive(source.buffer, "zip", { extractDir: "/dizione-src" });
  pyodide.runPython(`
import json, os, sys
os.environ["DIZIONE_DATA"] = "/dizione-data"
sys.path.insert(0, "/dizione-src")
import dizione

def phonemize_json(text):
    try:
        result = dizione.phonemize(text)
    except ValueError:
        # Nothing to pronounce (another script, a face, signs only) is no failure: "null", and the sentence is skipped.
        if any(word for mark, words in dizione.pieces(text) if not mark for word in words):
            raise
        return "null"
    return json.dumps({"ids": list(result.ids), "stress": list(result.stress), "text": str(result)})

phonemize_json("Warm up.")
`);
  phonemize = pyodide.globals.get("phonemize_json");
}

async function verifiedAsset(name, label) {
  const spec = config.vocoder_pilot?.assets?.[name];
  if (!spec || !/^[a-f0-9]{64}$/.test(spec.sha256) ||
      !Number.isSafeInteger(spec.bytes) || spec.bytes < 1) throw Error("Missing pilot asset provenance");
  const bytes = await download("assets/" + name, label, spec.sha256);
  if (bytes.byteLength !== spec.bytes) throw Error("Pilot asset length mismatch");
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, "0")).join("");
  if (hex !== spec.sha256) throw Error("Pilot asset digest mismatch");
  return bytes;
}

async function modelSession(bytes) {
  status("Starting the model");
  const model = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  threads = ort.env.wasm.numThreads;
  return model;
}

async function loadDense() {
  const label = "Loading the model";
  const bytes = config.vocoder_pilot?.assets?.["model.onnx"]
    ? await verifiedAsset("model.onnx", label)
    : await download("assets/model.onnx", label);
  session = await modelSession(bytes);
}

async function loadCompact() {
  importScripts("compact-vocoder.js");
  const pilot = config.vocoder_pilot;
  if (pilot?.abi !== 1 || !/^[a-f0-9]{64}$/.test(pilot.checkpoint_sha256) ||
      pilot.frontend_fingerprint !== config.phonemizer?.fingerprint ||
      config.phonemizer?.name !== "dizione" ||
      config.sample_rate !== COMPACT_GEOMETRY.sample_rate || config.hop !== COMPACT_GEOMETRY.hop ||
      Object.entries(COMPACT_GEOMETRY).some(([key, value]) =>
        pilot.geometry?.[key] !== value || config.model?.[key] !== value)) {
    throw Error("Incompatible compact model manifest");
  }
  const [wasm, constants] = await Promise.all([
    verifiedAsset("dsp.wasm", "Loading the vocoder"),
    verifiedAsset("dsp-constants.f32", "Loading the vocoder"),
  ]);
  const renderer = await CompactVocoder.create(wasm, constants, config.model.max_frames);
  const model = await modelSession(await verifiedAsset("model-compact.onnx", "Loading the model"));
  const outputs = ["harmonic", "noise_head", "pulses", "durations"];
  if (model.outputNames.length !== outputs.length || outputs.some(name => !model.outputNames.includes(name))) {
    await model.release();
    throw Error("Unexpected compact model outputs");
  }
  session = model;
  compact = renderer;
}

async function loadModel() {
  const response = await fetch("assets/config.json");
  if (!response.ok) throw Error("Could not load model configuration");
  config = await response.json();
  ort.env.wasm.wasmPaths = ORT;
  threads = flags.get("threads") === "4" && crossOriginIsolated && navigator.hardwareConcurrency >= 4 ? 4 : 1;
  ort.env.wasm.numThreads = threads;
  if (requestCompact) {
    try {
      await loadCompact();
      return;
    } catch (error) {
      postMessage({ type: "vocoder-fallback", reason: String(error.message || error) });
    }
  }
  await loadDense();
}

async function switchToDense(error) {
  postMessage({ type: "vocoder-fallback", reason: String(error.message || error) });
  compact = null;
  const previous = session;
  session = null;
  await previous.release();
  await loadDense();
}

const ABBREVIATIONS = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;

// Sentences, without breaking after titles, initials or inside numbers.
// tiny_voca.infer.sentences in the Python package splits the same way; keep the two alike.
function sentences(text) {
  const parts = [];
  let current = "";
  for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-Z0-9])/)) {
    current = current ? current + " " + piece : piece;
    if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// A full stop is added when the text ends without a mark: before a final mark the voice lengthens the last
// syllable and lets the sound fall, without one the audio breaks off. A face or another sign after the last word
// is no mark ("see you :)"), a comma or a colon is one. tiny_voca.infer.finished does the same; keep the two alike.
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : body + ".";
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut at commas, then by words.
// A sentence with nothing to pronounce gives no utterance.
function fit(sentence) {
  const result = JSON.parse(phonemize(sentence));
  if (!result) return [];
  if (result.ids.length <= config.model.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:—])\s+/);
  if (pieces.length > 1) return pieces.flatMap(fit);
  const words = sentence.split(" ");
  if (words.length < 2) return [];
  const half = Math.ceil(words.length / 2);
  return [...fit(words.slice(0, half).join(" ")), ...fit(words.slice(half).join(" "))];
}

async function synthesize(phones, voice, speed, pitchShift, pitchRange) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const feeds = {
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(pitchShift), pitch_range: scalar(pitchRange),
  };
  if (compact) {
    try {
      return compact.render(await session.run(feeds), count);
    } catch (error) {
      await switchToDense(error);
    }
  }
  return { audio: (await session.run(feeds)).audio.data };
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// Shown when no sentence of a text could be read. tiny_voca.infer.speak_text says the same.
const ENGLISH_ONLY = "tiny-voca only reads English for now.";

async function speak({ id, text, voice, speed, pitchShift = 0, pitchRange = 1 }) {
  if (!ready || !session) throw Error("Model is not ready");
  if (!Number.isFinite(speed) || speed < 0.25 || speed > 4 ||
      !Number.isFinite(pitchShift) || !Number.isFinite(pitchRange) || pitchRange < 0 || pitchRange > 3 ||
      !Number.isInteger(voice) || voice < 0 || voice >= config.voices.length) throw Error("Invalid synthesis controls");
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0;
  let modelTime = 0, noiseTime = 0, dspTime = 0, wasmBytes = 0, workspaceBytes = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const result = await synthesize(phones, voice, speed, pitchShift, pitchRange);
      const { audio } = result;
      noiseTime += result.noiseTime || 0;
      dspTime += result.dspTime || 0;
      wasmBytes = Math.max(wasmBytes, result.wasmBytes || 0);
      workspaceBytes = Math.max(workspaceBytes, result.workspaceBytes || 0);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      // The page plays this sentence at once, before the whole clip is known: a copy, kept below full scale by itself.
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: ENGLISH_ONLY, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                backend: compact ? "compact" : "dense", threads, noiseTime, dspTime, wasmBytes, workspaceBytes,
                totalTime: performance.now() - started }, [audio.buffer]);
}

// Requests are serialized so a reused arena and its session cannot overlap.
onmessage = ({ data }) => {
  queue = queue.then(() => handle(data));
};

async function handle(data) {
  try {
    if (data.type === "load") {
      if (!ready) {
        const results = await Promise.allSettled([loadPhonemizer(), loadModel()]);
        const failure = results.find(result => result.status === "rejected");
        if (failure) {
          if (session) await session.release();
          session = compact = null;
          throw failure.reason;
        }
        ready = true;
      }
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate,
                    backend: compact ? "compact" : "dense", threads });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
}
