#!/usr/bin/env python3
"""Minimalist local web UI running tiny-voca French ONNX 100% in browser WASM.

Architecture:
  - Browser loads ``onnxruntime-web`` (WASM backend) + local ``tiny-voca-fr.onnx``.
  - ``POST /api/phonemize`` runs the exact Python ``prepare_french_tts_text`` +
    ``tiny_voca.text.phonemize`` frontend and returns sentence token IDs, stress,
    cleaned text, and IPA symbols (so you can inspect every acronym/unit/liaison).
  - Neural synthesis runs 100% inside the browser via ONNX Runtime Web (WASM).

Usage:
    PYTHONPATH=src ./env/bin/python scripts/web_wasm_demo.py \
        --onnx .local-runs/p7_refiner/onnx/tiny-voca-fr.onnx --port 8766
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_voca.infer import finished, sentences, shorter
from tiny_voca.prosody_fr import prepare_french_tts_text
from tiny_voca.text import NothingToPronounce, phonemize


def phonemize_for_wasm(text: str, max_tokens: int = 240, clean_text: bool = True) -> dict:
    cleaned = prepare_french_tts_text(text) if clean_text else text
    limit = max_tokens - 12
    utterances = []

    def fit(piece: str):
        try:
            ph = phonemize(piece)
        except NothingToPronounce:
            return
        if len(ph) <= limit:
            utterances.append({
                "text": piece,
                "ids": list(ph.ids),
                "stress": list(ph.stress),
                "symbols": " ".join(ph.symbols),
            })
            return
        parts = shorter(piece)
        if not parts:
            return
        for part in parts:
            fit(part)

    for sent in sentences(finished(cleaned)):
        fit(sent)

    return {
        "cleaned_text": cleaned,
        "utterances": utterances,
    }


HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tiny-voca FR — Démo ONNX WASM</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #cbd5e1;
    --text: #0f172a;
    --muted: #475569;
    --brand: #2563eb;
    --brand-hover: #1d4ed8;
    --chip-bg: #eff6ff;
    --chip-text: #1e40af;
    --mono-bg: #f1f5f9;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --brand: #3b82f6;
      --brand-hover: #60a5fa;
      --chip-bg: #1e3a5f;
      --chip-text: #bfdbfe;
      --mono-bg: #0b1220;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 1.5rem 1rem 3rem;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }
  .wrap { max-width: 780px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
  h1 { font-size: 1.35rem; margin: 0; }
  .badge {
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.ok { color: #16a34a; border-color: #16a34a; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.15rem;
    margin-bottom: 1rem;
  }
  textarea {
    width: 100%;
    min-height: 110px;
    padding: 0.75rem;
    font: inherit;
    font-size: 1rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg);
    color: var(--text);
    resize: vertical;
  }
  .chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.65rem 0 0.9rem; }
  .chip {
    background: var(--chip-bg);
    color: var(--chip-text);
    border: none;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.82rem;
    cursor: pointer;
  }
  .chip:hover { filter: brightness(0.95); }
  .controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    align-items: end;
    margin-bottom: 0.9rem;
  }
  label { display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 0.25rem; }
  select, input[type="range"] { width: 100%; }
  select {
    padding: 0.45rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font: inherit;
  }
  .toggle-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.88rem; margin-bottom: 0.9rem; }
  .toggle-row input { width: auto; }
  .actions { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; }
  button.primary {
    background: var(--brand);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.25rem;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
  }
  button.primary:hover:not(:disabled) { background: var(--brand-hover); }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.9rem;
    cursor: pointer;
  }
  .metrics { font-size: 0.85rem; color: var(--muted); margin-left: auto; }
  audio { width: 100%; margin-top: 0.8rem; }
  details { margin-top: 0.75rem; }
  summary { cursor: pointer; font-size: 0.85rem; color: var(--muted); }
  pre {
    background: var(--mono-bg);
    padding: 0.65rem;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.8rem;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0.45rem 0 0;
  }
  .err { color: #dc2626; font-size: 0.9rem; margin-top: 0.5rem; }
</style>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.js"></script>
</head>
<body>
<div class="wrap">
  <header>
    <h1>tiny-voca FR — Synthèse ONNX WASM</h1>
    <span id="status" class="badge">Chargement du modèle ONNX WASM…</span>
  </header>

  <div class="card">
    <label for="text">Texte français à lire (Ctrl+Entrée pour lancer)</label>
    <textarea id="text">Le fichier pèse 725 Ko, utilise 16 Go de RAM à 44 kHz, et le rapport de l'ONU et de l'OMS mentionne le RGPD. Tél. : 06 12 34 56 78.</textarea>

    <div class="chips" id="presets"></div>

    <div class="controls">
      <div>
        <label for="voice">Voix</label>
        <select id="voice"></select>
      </div>
      <div>
        <label for="speed">Vitesse : <strong id="speed-val">1.00×</strong></label>
        <input id="speed" type="range" min="0.75" max="1.35" step="0.05" value="1.00">
      </div>
      <div>
        <label for="pitch-range">Expressivité (pitch range) : <strong id="pr-val">1.00</strong></label>
        <input id="pitch-range" type="range" min="0.80" max="1.40" step="0.05" value="1.00">
      </div>
    </div>

    <label class="toggle-row">
      <input id="clean-toggle" type="checkbox" checked>
      Nettoyage français actif (unités Ko/Go/ml/ms, sigles RAM/ONU/OMS/RGPD, fractions, téléphone, listes)
    </label>

    <div class="actions">
      <button id="speak-btn" class="primary" disabled>Lire en WASM</button>
      <button id="dl-btn" class="secondary" hidden>Télécharger WAV</button>
      <span id="metrics" class="metrics"></span>
    </div>

    <div id="err" class="err"></div>
    <audio id="player" controls hidden></audio>

    <details open>
      <summary>Inspection texte nettoyé & phonèmes IPA</summary>
      <pre id="inspect">—</pre>
    </details>
  </div>
</div>

<script>
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const PRESETS = [
  ["Tech & sigles", "Le fichier pèse 725 Ko, utilise 16 Go de RAM à 44 kHz, et le rapport de l'ONU et de l'OMS mentionne le RGPD. Tél. : 06 12 34 56 78."],
  ["Sigles divers", "Le CNRS, le CHU, la NASA, l'UNESCO, le MEDEF, un EHPAD, un AVC et le FBI sont cités dans cet article."],
  ["Recette à puces", "Ingrédients :\\n- 250 ml de lait\\n- 50 cl d'eau\\n- 500 mg de sel\\n- 1/2 litre d'huile"],
  ["Fractions & email", "Attendez 3/4 d'heure, 2x plus vite, puis écrivez à bonjour@exemple.fr ou visitez https://example.org/docs."],
  ["Pauses <sp>", "Peux-tu me rappeler si Karim Martin préfère venir vendredi ou lundi ? J'ai demandé à Lina, Claire et Jade de nous retrouver au café."],
  ["Questions FR", "Est-ce que tu viens demain soir ? Et pourquoi ne m'as-tu rien dit avant ?"],
];

let session = null;
let meta = null;
let lastWavUrl = null;

const $ = (id) => document.getElementById(id);

function initPresets() {
  const box = $("presets");
  PRESETS.forEach(([label, txt]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = label;
    b.onclick = () => { $("text").value = txt; };
    box.appendChild(b);
  });
}

function encodeWav(samples, sampleRate) {
  const numSamples = samples.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);
  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, Math.round(s * 32767), true);
    off += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function init() {
  initPresets();
  $("speed").oninput = () => { $("speed-val").textContent = Number($("speed").value).toFixed(2) + "×"; };
  $("pitch-range").oninput = () => { $("pr-val").textContent = Number($("pitch-range").value).toFixed(2); };
  $("text").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("speak-btn").disabled) speak();
  });
  $("speak-btn").onclick = speak;

  try {
    const t0 = performance.now();
    meta = await (await fetch("/assets/model.json")).json();
    const sel = $("voice");
    (meta.voices || ["F1", "M1"]).forEach((v, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = v;
      sel.appendChild(opt);
    });
    $("status").textContent = "Téléchargement ONNX…";
    const resp = await fetch("/assets/model.onnx");
    const bytes = new Uint8Array(await resp.arrayBuffer());
    $("status").textContent = "Initialisation ONNX Runtime WASM…";
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    const sp = meta.sp_pause_threshold != null ? ` · sp_thr=${meta.sp_pause_threshold}` : "";
    $("status").textContent = `WASM prêt (${(bytes.byteLength / 1e6).toFixed(1)} Mo en ${dt}s${sp})`;
    $("status").classList.add("ok");
    $("speak-btn").disabled = false;
  } catch (err) {
    $("status").textContent = "Erreur chargement WASM";
    $("err").textContent = String(err);
  }
}

async function speak() {
  $("err").textContent = "";
  $("speak-btn").disabled = true;
  $("metrics").textContent = "Synthèse WASM en cours…";
  try {
    const text = $("text").value;
    const voiceIdx = Number($("voice").value);
    const speed = Number($("speed").value);
    const pitchRange = Number($("pitch-range").value);
    const cleanText = $("clean-toggle").checked;

    const tPh0 = performance.now();
    const phRes = await fetch("/api/phonemize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, clean_text: cleanText }),
    });
    const phData = await phRes.json();
    if (!phRes.ok) throw new Error(phData.error || "Erreur phonémisation");
    const tPh = performance.now() - tPh0;

    if (!phData.utterances || phData.utterances.length === 0) {
      throw new Error("Rien à prononcer dans ce texte.");
    }

    $("inspect").textContent =
      "Texte nettoyé :\\n" + phData.cleaned_text + "\\n\\n" +
      phData.utterances.map((u, i) => `[${i + 1}] ${u.symbols}`).join("\\n");

    const chunks = [];
    let totalLen = 0;
    const tWasm0 = performance.now();
    for (const u of phData.utterances) {
      const n = u.ids.length;
      const out = await session.run({
        tokens: new ort.Tensor("int64", BigInt64Array.from(u.ids.map(BigInt)), [1, n]),
        stress: new ort.Tensor("int64", BigInt64Array.from(u.stress.map(BigInt)), [1, n]),
        voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voiceIdx)]), [1]),
        speed: new ort.Tensor("float32", Float32Array.from([speed]), [1]),
        pitch_shift: new ort.Tensor("float32", Float32Array.from([0.0]), [1]),
        pitch_range: new ort.Tensor("float32", Float32Array.from([pitchRange]), [1]),
      });
      const data = out.audio.data;
      chunks.push(data);
      totalLen += data.length;
    }
    const tWasm = performance.now() - tWasm0;

    const sr = meta.sample_rate || 24000;
    const gapSamples = Math.round(sr * 0.22);
    const totalWithGaps = totalLen + Math.max(0, chunks.length - 1) * gapSamples;
    const full = new Float32Array(totalWithGaps);
    let offset = 0;
    let peak = 0;
    for (let ci = 0; ci < chunks.length; ci++) {
      const c = chunks[ci];
      full.set(c, offset);
      offset += c.length;
      if (ci + 1 < chunks.length) offset += gapSamples;
      for (let i = 0; i < c.length; i++) {
        const a = Math.abs(c[i]);
        if (a > peak) peak = a;
      }
    }
    if (peak > 0.99) {
      const scale = 0.99 / peak;
      for (let i = 0; i < full.length; i++) full[i] *= scale;
    }

    const audioSec = full.length / sr;
    const rtf = (tWasm / 1000) / Math.max(audioSec, 0.01);
    $("metrics").textContent =
      `Audio ${audioSec.toFixed(2)}s · WASM ${(tWasm).toFixed(0)} ms (RTF ${rtf.toFixed(2)}×) · Frontend ${tPh.toFixed(0)} ms`;

    const blob = encodeWav(full, sr);
    if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
    lastWavUrl = URL.createObjectURL(blob);
    const player = $("player");
    player.src = lastWavUrl;
    player.hidden = false;
    player.play().catch(() => {});

    const dl = $("dl-btn");
    dl.hidden = false;
    dl.onclick = () => {
      const a = document.createElement("a");
      a.href = lastWavUrl;
      a.download = "tiny-voca-fr.wav";
      a.click();
    };
  } catch (err) {
    $("err").textContent = String(err);
    $("metrics").textContent = "";
  } finally {
    $("speak-btn").disabled = false;
  }
}

init();
</script>
</body>
</html>
"""


def make_handler(onnx_path: Path, meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    max_tokens = int(meta.get("max_tokens", 240))
    onnx_bytes = onnx_path.read_bytes()
    meta_bytes = meta_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/assets/model.onnx":
                self._send(HTTPStatus.OK, onnx_bytes, "application/octet-stream")
            elif path == "/assets/model.json":
                self._send(HTTPStatus.OK, meta_bytes, "application/json; charset=utf-8")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/phonemize":
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(payload.get("text", ""))
                clean_text = bool(payload.get("clean_text", True))
                result = phonemize_for_wasm(text, max_tokens=max_tokens, clean_text=clean_text)
                self._send(HTTPStatus.OK, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            except Exception as exc:
                err = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.BAD_REQUEST, err, "application/json; charset=utf-8")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, default=Path(".local-runs/p7_refiner/onnx/tiny-voca-fr.onnx"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    meta_path = args.onnx.with_suffix(".json")
    if not args.onnx.exists() or not meta_path.exists():
        raise SystemExit(f"Missing ONNX or metadata JSON: {args.onnx}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.onnx, meta_path))
    print(f"tiny-voca FR WASM demo ready on http://{args.host}:{args.port} ({args.onnx.name}, {args.onnx.stat().st_size/1e6:.1f} MB)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
