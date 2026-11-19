#!/usr/bin/env python3
"""Minimalist local web UI for rapid human triage of suspected French TTS defects.

Displays one suspect at a time (original text, cleaned text, IPA, automatic flags,
Whisper round-trip transcription & CER, audio player) and auto-saves your verdicts
(OK / Bug mineur / Vrai bug + note) to ``triage_decisions.json``.

Usage:
    ./env/bin/python scripts/triage_ui.py --dir .local-runs/fr_hunter/batch_01 --port 8767
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Triage défauts FR — tiny-voca</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f8fafc; --card: #ffffff; --border: #cbd5e1;
    --text: #0f172a; --muted: #475569; --mono: #f1f5f9;
    --ok: #16a34a; --minor: #d97706; --bug: #dc2626; --brand: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8; --mono: #0b1220;
      --ok: #22c55e; --minor: #f59e0b; --bug: #ef4444; --brand: #60a5fa;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 1.5rem 1rem 3rem; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.45; }
  .wrap { max-width: 780px; margin: 0 auto; }
  header { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
  h1 { font-size: 1.25rem; margin: 0; }
  .progress { font-size: 0.9rem; color: var(--muted); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; }
  .meta-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.7rem; font-size: 0.8rem; }
  .tag { padding: 0.18rem 0.55rem; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); }
  .tag.flag { border-color: var(--minor); color: var(--minor); }
  .prompt { font-size: 1.1rem; font-weight: 600; margin: 0.4rem 0 0.8rem; white-space: pre-wrap; }
  audio { width: 100%; margin: 0.6rem 0 0.9rem; }
  .verdicts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-bottom: 0.8rem; }
  .vbtn {
    padding: 0.7rem 0.5rem; border-radius: 8px; border: 2px solid var(--border);
    background: transparent; color: var(--text); font-weight: 600; font-size: 0.92rem; cursor: pointer;
  }
  .vbtn.sel-ok { border-color: var(--ok); background: color-mix(in srgb, var(--ok) 15%, transparent); }
  .vbtn.sel-minor { border-color: var(--minor); background: color-mix(in srgb, var(--minor) 15%, transparent); }
  .vbtn.sel-bug { border-color: var(--bug); background: color-mix(in srgb, var(--bug) 15%, transparent); }
  textarea {
    width: 100%; min-height: 64px; padding: 0.6rem; font: inherit; font-size: 0.92rem;
    border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text);
  }
  .nav { display: flex; justify-content: space-between; align-items: center; margin-top: 0.9rem; gap: 0.5rem; }
  .nav button { padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid var(--border); background: transparent; color: var(--text); cursor: pointer; font-weight: 600; }
  .hint { font-size: 0.8rem; color: var(--muted); }
  details { margin-top: 0.85rem; }
  summary { cursor: pointer; font-size: 0.85rem; color: var(--muted); }
  pre { background: var(--mono); padding: 0.6rem; border-radius: 8px; font-size: 0.78rem; white-space: pre-wrap; word-break: break-word; margin: 0.4rem 0 0; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Triage semi-automatisé des défauts FR</h1>
    <span id="prog" class="progress">…</span>
  </header>
  <div class="card">
    <div id="tags" class="meta-row"></div>
    <div id="prompt" class="prompt"></div>
    <audio id="player" controls></audio>
    <div class="verdicts">
      <button id="btn-ok" class="vbtn" type="button">OK / Faux positif (o)</button>
      <button id="btn-minor" class="vbtn" type="button">Bug mineur (m)</button>
      <button id="btn-bug" class="vbtn" type="button">Vrai bug à corriger (b)</button>
    </div>
    <textarea id="note" placeholder="Note rapide (ex. « 12 V dit douze vé », « pas de pause sur ; », « coupure bizarre »)…"></textarea>
    <div class="nav">
      <button id="prev" type="button">← Précédent</button>
      <span class="hint">Espace = lire/pause · o / m / b = verdict · ← / → = naviguer</span>
      <button id="next" type="button">Suivant →</button>
    </div>
    <details open>
      <summary>Détails automatiques (texte nettoyé, Whisper, IPA)</summary>
      <pre id="details"></pre>
    </details>
  </div>
</div>

<script>
let suspects = [];
let decisions = {};
let idx = 0;
let saveTimer = null;
const $ = (id) => document.getElementById(id);

async function load() {
  const res = await fetch("/api/suspects");
  const data = await res.json();
  suspects = data.suspects || [];
  decisions = data.decisions || {};
  render();
}

function render() {
  if (!suspects.length) return;
  const s = suspects[idx];
  const dec = decisions[s.suspect_id] || { verdict: "", note: "" };
  const doneCount = Object.values(decisions).filter((d) => d.verdict).length;
  $("prog").textContent = `${s.suspect_id} (${idx + 1}/${suspects.length}) · ${doneCount}/${suspects.length} triés`;

  const tags = $("tags");
  tags.innerHTML = "";
  const addTag = (txt, isFlag) => {
    const el = document.createElement("span");
    el.className = "tag" + (isFlag ? " flag" : "");
    el.textContent = txt;
    tags.appendChild(el);
  };
  addTag(s.voice, false);
  addTag(s.category, false);
  addTag(`CER ${(s.cer * 100).toFixed(1)}%`, s.cer >= 0.15);
  (s.static_flags || []).forEach((f) => addTag(f, true));
  (s.dur_flags || []).forEach((f) => addTag(f, true));

  $("prompt").textContent = s.text;
  $("player").src = "/audio/" + encodeURIComponent(s.wav_file);
  $("note").value = dec.note || "";

  $("btn-ok").className = "vbtn" + (dec.verdict === "ok" ? " sel-ok" : "");
  $("btn-minor").className = "vbtn" + (dec.verdict === "minor" ? " sel-minor" : "");
  $("btn-bug").className = "vbtn" + (dec.verdict === "bug" ? " sel-bug" : "");

  $("details").textContent =
    `Texte nettoyé :\\n${s.cleaned_text}\\n\\n` +
    `Whisper (ce qu'il a compris) :\\n${s.whisper_hyp || "—"}\\n\\n` +
    `IPA :\\n${s.ipa}`;
}

async function saveCurrent(verdictOverride) {
  const s = suspects[idx];
  if (!s) return;
  const prev = decisions[s.suspect_id] || { verdict: "", note: "" };
  const updated = {
    verdict: verdictOverride !== undefined ? verdictOverride : prev.verdict,
    note: $("note").value,
  };
  decisions[s.suspect_id] = updated;
  render();
  await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suspect_id: s.suspect_id, id: s.id, ...updated }),
  });
}

$("btn-ok").onclick = () => saveCurrent("ok");
$("btn-minor").onclick = () => saveCurrent("minor");
$("btn-bug").onclick = () => saveCurrent("bug");
$("note").oninput = () => {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveCurrent(undefined), 350);
};
$("prev").onclick = () => { idx = (idx - 1 + suspects.length) % suspects.length; render(); };
$("next").onclick = () => { idx = (idx + 1) % suspects.length; render(); };

document.addEventListener("keydown", (e) => {
  if (document.activeElement === $("note")) return;
  if (e.code === "Space") {
    e.preventDefault();
    const p = $("player");
    if (p.paused) p.play(); else p.pause();
  } else if (e.key === "o") saveCurrent("ok");
  else if (e.key === "m") saveCurrent("minor");
  else if (e.key === "b") saveCurrent("bug");
  else if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
});

load();
</script>
</body>
</html>
"""


def make_handler(batch_dir: Path):
    suspects_path = batch_dir / "suspects.json"
    decisions_path = batch_dir / "triage_decisions.json"
    data = json.loads(suspects_path.read_text(encoding="utf-8"))
    suspects = data["suspects"]
    decisions: dict[str, dict] = {}
    if decisions_path.exists():
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/suspects":
                body = json.dumps({"suspects": suspects, "decisions": decisions}, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
            elif path.startswith("/audio/"):
                fname = Path(unquote(path[len("/audio/"):])).name
                fpath = batch_dir / fname
                if fpath.exists() and fpath.suffix.lower() == ".wav":
                    self._send(HTTPStatus.OK, fpath.read_bytes(), "audio/wav")
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/decision":
                self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            sid = str(payload["suspect_id"])
            decisions[sid] = {
                "id": payload.get("id", ""),
                "verdict": payload.get("verdict", ""),
                "note": payload.get("note", ""),
            }
            decisions_path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
            self._send(HTTPStatus.OK, b'{"ok":true}', "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path(".local-runs/fr_hunter/batch_01"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.dir))
    print(f"French defect triage UI ready on http://{args.host}:{args.port} ({args.dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
