"""Minimalist local web UI for comfortable pair-by-pair blind A/B audio listening.

Zero external dependencies (Python stdlib http.server + accessible HTML/CSS/JS).
Auto-discovers `*_A.wav` / `*_B.wav` (and optional `*_REF.wav`) pairs in any blind
listening folder, parses text/speaker/category from any local markdown sheet, and
STRICTLY refuses to read or serve any key/secret files.

Usage:
    ./env/bin/python scripts/blind_ab_ui.py --dir .local-runs/p6_replay/listening_p6/blind_p6
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

_MD_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|")


def _is_forbidden(name: str) -> bool:
    low = name.lower()
    return "key" in low or "secret" in low


def discover_blind_pairs(folder: Path) -> list[dict]:
    """Discover sorted *_A.wav / *_B.wav pairs and optional metadata from markdown sheets."""
    meta_by_id: dict[str, dict] = {}
    for md in sorted(folder.glob("*.md")):
        if _is_forbidden(md.name):
            continue
        for line in md.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = _MD_ROW_RE.match(line.strip())
            if m:
                pid, spk, cat, text = (g.strip() for g in m.groups())
                meta_by_id[pid] = {"speaker": spk, "category": cat, "text": text}

    pairs: list[dict] = []
    for wav_a in sorted(folder.glob("*_A.wav")):
        pid = wav_a.name[:-6]
        wav_b = folder / f"{pid}_B.wav"
        if not wav_b.exists():
            continue
        # Match optional REF either `<pid>_REF.wav` or `<Pxx>_<uid>_REF.wav`.
        ref_name = None
        cand_ref = folder / f"{pid}_REF.wav"
        if cand_ref.exists():
            ref_name = cand_ref.name
        else:
            prefix = pid.split("_", 1)[0]
            matches = sorted(folder.glob(f"{prefix}_*_REF.wav"))
            if matches:
                ref_name = matches[0].name
        info = meta_by_id.get(pid, {})
        pairs.append({
            "pair_id": pid,
            "a_file": wav_a.name,
            "b_file": wav_b.name,
            "ref_file": ref_name,
            "speaker": info.get("speaker", ""),
            "category": info.get("category", ""),
            "text": info.get("text", ""),
        })
    return pairs


HTML_PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Écoute aveugle A/B</title>
<style>
  :root {
    color-scheme: light dark;
    --brand: #2563eb;
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #cbd5e1;
    --text: #0f172a;
    --muted: #475569;
    --active-bg: #dbeafe;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --brand: #60a5fa;
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --active-bg: #1e3a5f;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    accent-color: var(--brand);
    line-height: 1.45;
  }
  main {
    max-width: 720px;
    margin: 1.5rem auto;
    padding: 0 1rem 3rem;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  h1 { font-size: 1.15rem; margin: 0; }
  .progress-wrap {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-size: .9rem;
    color: var(--muted);
  }
  progress { width: 120px; height: 10px; }
  .pills {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    margin-bottom: 1rem;
  }
  .pill {
    min-inline-size: 32px;
    min-block-size: 30px;
    padding: .2rem .5rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
    cursor: pointer;
    font-size: .85rem;
  }
  .pill[data-done="1"] { border-color: var(--brand); font-weight: 600; }
  .pill[aria-current="true"] { background: var(--brand); color: #fff; border-color: var(--brand); }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
  }
  .meta {
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
    font-size: .82rem;
    color: var(--muted);
    margin-bottom: .5rem;
  }
  .badge {
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: .1rem .55rem;
  }
  .phrase {
    font-size: 1.12rem;
    font-weight: 500;
    margin: .25rem 0 1rem;
  }
  .players {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .75rem;
    margin-bottom: 1rem;
  }
  @media (max-width: 560px) { .players { grid-template-columns: 1fr; } }
  .player-box {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .75rem;
    display: flex;
    flex-direction: column;
    gap: .5rem;
  }
  .player-box[data-playing="1"] {
    border-color: var(--brand);
    background: var(--active-bg);
  }
  .play-btn {
    min-block-size: 40px;
    padding: .45rem .8rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: .95rem;
    font-weight: 600;
    cursor: pointer;
    text-align: left;
  }
  .play-btn:hover { border-color: var(--brand); }
  audio { width: 100%; height: 34px; }
  .ref-details {
    margin-bottom: 1rem;
    font-size: .88rem;
    color: var(--muted);
  }
  .ref-details audio { margin-top: .4rem; }
  fieldset.vote {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .75rem;
    margin: 0 0 1rem;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .5rem;
  }
  fieldset.vote legend { font-size: .85rem; color: var(--muted); padding: 0 .35rem; }
  .vote label {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .4rem;
    min-block-size: 42px;
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    padding: .35rem .5rem;
  }
  .vote label:has(input:checked) {
    border-color: var(--brand);
    background: var(--active-bg);
  }
  .remark-label { display: block; font-size: .85rem; color: var(--muted); margin-bottom: .3rem; }
  textarea {
    width: 100%;
    min-height: 64px;
    padding: .6rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font: inherit;
    resize: vertical;
  }
  .nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    gap: .5rem;
    flex-wrap: wrap;
  }
  button.btn {
    min-inline-size: 44px;
    min-block-size: 38px;
    padding: .45rem .9rem;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
    font: inherit;
    font-size: .92rem;
    cursor: pointer;
  }
  button.btn.primary { background: var(--brand); color: #fff; border-color: var(--brand); }
  .export-bar {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    gap: .6rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .copy-wrap { display: inline-flex; align-items: center; gap: .5rem; }
  .tip { font-size: .85rem; color: var(--muted); opacity: 0; transition: opacity .15s; }
  .tip[data-show="1"] { opacity: 1; }
  .shortcuts { font-size: .8rem; color: var(--muted); margin-top: .75rem; }
  kbd {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0 .3rem;
    font-size: .78rem;
  }
  .sr-only {
    position: absolute; clip-path: inset(50%); overflow: hidden;
    width: 1px; height: 1px; margin: -1px; padding: 0; border: 0; white-space: nowrap;
  }
</style>
</head>
<body>
<main>
  <header>
    <h1>Écoute aveugle A / B</h1>
    <div class="progress-wrap">
      <span id="prog-text">0 / 0</span>
      <progress id="prog-bar" max="1" value="0"></progress>
    </div>
  </header>

  <nav id="pills" class="pills" aria-label="Paires"></nav>

  <section class="card" aria-live="off">
    <div class="meta">
      <span class="badge" id="m-id"></span>
      <span class="badge" id="m-spk"></span>
      <span class="badge" id="m-cat"></span>
    </div>
    <p class="phrase" id="m-text"></p>

    <div class="players">
      <div class="player-box" id="box-a">
        <button type="button" class="play-btn" id="btn-a">▶ Écouter A <kbd>1</kbd></button>
        <audio id="audio-a" controls preload="metadata"></audio>
      </div>
      <div class="player-box" id="box-b">
        <button type="button" class="play-btn" id="btn-b">▶ Écouter B <kbd>2</kbd></button>
        <audio id="audio-b" controls preload="metadata"></audio>
      </div>
    </div>

    <details class="ref-details" id="ref-wrap">
      <summary>Afficher la référence optionnelle (Qwen3-TTS)</summary>
      <audio id="audio-ref" controls preload="none"></audio>
    </details>

    <fieldset class="vote">
      <legend>Préférence (naturel / expressivité)</legend>
      <label><input type="radio" name="choice" value="A"> <span>A meilleur <kbd>a</kbd></span></label>
      <label><input type="radio" name="choice" value="tie"> <span>Égalité <kbd>e</kbd></span></label>
      <label><input type="radio" name="choice" value="B"> <span>B meilleur <kbd>b</kbd></span></label>
    </fieldset>

    <label class="remark-label" for="remark">Remarque libre (intonation, rythme, pause, timbre…)</label>
    <textarea id="remark" placeholder="Optionnel : ex. A plus naturel sur la fin, B a une pause étrange…"></textarea>

    <div class="nav">
      <button type="button" class="btn" id="prev-btn">← Précédent</button>
      <span id="save-status" class="tip" role="status"></span>
      <button type="button" class="btn primary" id="next-btn">Suivant →</button>
    </div>

    <div class="export-bar">
      <span class="copy-wrap">
        <button type="button" class="btn" id="copy-btn" command="--copy" commandfor="summary-field">Copier le résumé</button>
        <span class="tip" id="copy-tip" role="status"></span>
      </span>
      <button type="button" class="btn" id="dl-btn">Télécharger JSON</button>
      <textarea id="summary-field" class="sr-only" readonly aria-hidden="true"></textarea>
    </div>
    <p class="shortcuts">Raccourcis (hors champ texte) : <kbd>1</kbd> A · <kbd>2</kbd> B · <kbd>Espace</kbd> pause · <kbd>a</kbd>/<kbd>e</kbd>/<kbd>b</kbd> voter · <kbd>←</kbd>/<kbd>→</kbd> naviguer.</p>
  </section>
</main>

<script>
let pairs = [];
let idx = 0;
let answers = {};
let saveTimer = null;
let copyTimer = null;
let lastMsg = '';

const audioA = document.getElementById('audio-a');
const audioB = document.getElementById('audio-b');
const audioRef = document.getElementById('audio-ref');
const boxA = document.getElementById('box-a');
const boxB = document.getElementById('box-b');
const remarkEl = document.getElementById('remark');
const summaryEl = document.getElementById('summary-field');

function stopAll(except) {
  [audioA, audioB, audioRef].forEach((el) => {
    if (el !== except) { el.pause(); el.currentTime = 0; }
  });
}

function togglePlay(el) {
  if (!el.paused) { el.pause(); return; }
  stopAll(el);
  el.currentTime = 0;
  el.play().catch(() => {});
}

audioA.addEventListener('play', () => { stopAll(audioA); boxA.dataset.playing = '1'; boxB.dataset.playing = '0'; });
audioA.addEventListener('pause', () => { boxA.dataset.playing = '0'; });
audioA.addEventListener('ended', () => { boxA.dataset.playing = '0'; });
audioB.addEventListener('play', () => { stopAll(audioB); boxB.dataset.playing = '1'; boxA.dataset.playing = '0'; });
audioB.addEventListener('pause', () => { boxB.dataset.playing = '0'; });
audioB.addEventListener('ended', () => { boxB.dataset.playing = '0'; });
audioRef.addEventListener('play', () => stopAll(audioRef));

document.getElementById('btn-a').addEventListener('click', () => togglePlay(audioA));
document.getElementById('btn-b').addEventListener('click', () => togglePlay(audioB));

function buildSummaryText() {
  return pairs.map((p, i) => {
    const a = answers[p.pair_id] || {};
    const ch = a.choice === 'tie' ? 'égalité' : (a.choice || '?');
    const rem = a.comment ? ` (${a.comment.trim()})` : '';
    return `${i + 1}: ${ch}${rem}`;
  }).join('; ');
}

function updateSummaryField() {
  summaryEl.value = buildSummaryText();
}

function renderPills() {
  const nav = document.getElementById('pills');
  nav.innerHTML = '';
  let done = 0;
  pairs.forEach((p, i) => {
    const has = Boolean(answers[p.pair_id] && answers[p.pair_id].choice);
    if (has) done++;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pill';
    btn.textContent = String(i + 1);
    btn.dataset.done = has ? '1' : '0';
    if (i === idx) btn.setAttribute('aria-current', 'true');
    btn.addEventListener('click', () => { stopAll(null); idx = i; renderPair(); });
    nav.appendChild(btn);
  });
  document.getElementById('prog-text').textContent = `${done} / ${pairs.length}`;
  const pb = document.getElementById('prog-bar');
  pb.max = Math.max(pairs.length, 1);
  pb.value = done;
  updateSummaryField();
}

function renderPair() {
  const p = pairs[idx];
  if (!p) return;
  document.getElementById('m-id').textContent = `#${idx + 1} · ${p.pair_id}`;
  document.getElementById('m-spk').textContent = p.speaker || 'Voix';
  document.getElementById('m-cat').textContent = p.category || 'Paire';
  document.getElementById('m-text').textContent = p.text || p.pair_id;

  audioA.src = `/audio/${encodeURIComponent(p.a_file)}`;
  audioB.src = `/audio/${encodeURIComponent(p.b_file)}`;
  const refWrap = document.getElementById('ref-wrap');
  if (p.ref_file) {
    refWrap.hidden = false;
    refWrap.open = false;
    audioRef.src = `/audio/${encodeURIComponent(p.ref_file)}`;
  } else {
    refWrap.hidden = true;
    audioRef.removeAttribute('src');
  }

  const saved = answers[p.pair_id] || {};
  document.querySelectorAll('input[name="choice"]').forEach((inp) => {
    inp.checked = inp.value === saved.choice;
  });
  remarkEl.value = saved.comment || '';
  document.getElementById('prev-btn').disabled = idx === 0;
  document.getElementById('next-btn').disabled = idx === pairs.length - 1;
  renderPills();
}

function persist() {
  const p = pairs[idx];
  if (!p) return;
  const checked = document.querySelector('input[name="choice"]:checked');
  answers[p.pair_id] = {
    pair_id: p.pair_id,
    choice: checked ? checked.value : '',
    comment: remarkEl.value.trim(),
  };
  try { localStorage.setItem('blind_ab_' + location.port, JSON.stringify(answers)); } catch {}
  renderPills();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const st = document.getElementById('save-status');
    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pairs: pairs.map(x => answers[x.pair_id] || {pair_id: x.pair_id, choice: '', comment: ''})}),
      });
      if (res.ok) {
        st.textContent = 'Enregistré';
        st.dataset.show = '1';
        setTimeout(() => { st.dataset.show = '0'; st.textContent = ''; }, 1400);
      }
    } catch {}
  }, 200);
}

document.querySelectorAll('input[name="choice"]').forEach((inp) => inp.addEventListener('change', persist));
remarkEl.addEventListener('input', persist);
document.getElementById('prev-btn').addEventListener('click', () => { if (idx > 0) { stopAll(null); idx--; renderPair(); } });
document.getElementById('next-btn').addEventListener('click', () => { if (idx < pairs.length - 1) { stopAll(null); idx++; renderPair(); } });

function reportCopy(ok) {
  const tip = document.getElementById('copy-tip');
  clearTimeout(copyTimer);
  let msg = ok ? 'Résumé copié !' : 'Échec copie';
  if (msg === lastMsg) msg += '\u00A0';
  lastMsg = msg;
  tip.textContent = msg;
  tip.dataset.show = '1';
  copyTimer = setTimeout(() => { tip.dataset.show = '0'; tip.textContent = ''; }, 2000);
}

async function copySummary() {
  updateSummaryField();
  const txt = summaryEl.value;
  if (!txt) return;
  try {
    await navigator.clipboard.writeText(txt);
    reportCopy(true);
  } catch {
    reportCopy(false);
  }
}

if ('CommandEvent' in window) {
  document.addEventListener('command', (ev) => {
    if (ev.command === '--copy') copySummary();
  }, true);
} else {
  document.getElementById('copy-btn').addEventListener('click', copySummary);
}

document.getElementById('dl-btn').addEventListener('click', () => {
  const payload = {
    summary: buildSummaryText(),
    judgments: pairs.map((p, i) => ({idx: i + 1, pair_id: p.pair_id, ...(answers[p.pair_id] || {choice: '', comment: ''})})),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const a = Object.assign(document.createElement('a'), {href: URL.createObjectURL(blob), download: 'judgments_blind.json'});
  a.click();
  URL.revokeObjectURL(a.href);
});

document.addEventListener('keydown', (e) => {
  if (document.activeElement === remarkEl) return;
  if (e.key === '1') { e.preventDefault(); togglePlay(audioA); }
  else if (e.key === '2') { e.preventDefault(); togglePlay(audioB); }
  else if (e.key === ' ') { e.preventDefault(); stopAll(null); }
  else if (e.key === 'a' || e.key === 'A') { document.querySelector('input[value="A"]').click(); }
  else if (e.key === 'e' || e.key === 'E') { document.querySelector('input[value="tie"]').click(); }
  else if (e.key === 'b' || e.key === 'B') { document.querySelector('input[value="B"]').click(); }
  else if (e.key === 'ArrowRight' && idx < pairs.length - 1) { stopAll(null); idx++; renderPair(); }
  else if (e.key === 'ArrowLeft' && idx > 0) { stopAll(null); idx--; renderPair(); }
});

(async () => {
  const res = await fetch('/api/pairs');
  const data = await res.json();
  pairs = data.pairs || [];
  (data.saved || []).forEach((r) => { if (r.pair_id) answers[r.pair_id] = r; });
  renderPair();
})();
</script>
</body>
</html>
"""


def make_handler(folder: Path, save_path: Path):
    pairs = discover_blind_pairs(folder)
    allowed_wavs = {p["a_file"] for p in pairs} | {p["b_file"] for p in pairs} | {p["ref_file"] for p in pairs if p["ref_file"]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj: dict, status: int = HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/pairs":
                saved = []
                if save_path.exists():
                    try:
                        saved = json.loads(save_path.read_text(encoding="utf-8")).get("pairs", [])
                    except (OSError, ValueError):
                        saved = []
                self._send_json({"pairs": pairs, "saved": saved})
                return
            if parsed.path.startswith("/audio/"):
                name = Path(unquote(parsed.path[len("/audio/"):])).name
                if _is_forbidden(name) or name not in allowed_wavs:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                wav_path = (folder / name).resolve()
                if not wav_path.is_relative_to(folder.resolve()) or not wav_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                data = wav_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(min(length, 1 << 20))
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self._send_json({"ok": True})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="Folder containing *_A.wav / *_B.wav blind pairs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--save", type=Path, help="Path to auto-save JSON judgments (default: <dir>/judgments_blind.json)")
    args = parser.parse_args()

    folder = args.dir.resolve()
    pairs = discover_blind_pairs(folder)
    if not pairs:
        raise SystemExit(f"No *_A.wav / *_B.wav pairs found in {folder}")
    save_path = args.save or (folder / "judgments_blind.json")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(folder, save_path))
    print(f"Blind A/B UI ready on http://{args.host}:{args.port} ({len(pairs)} pairs from {folder})", flush=True)
    print(f"Auto-saving judgments to {save_path} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
