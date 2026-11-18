/* Real Chromium, checkpoint-matched dizione and standard ONNX Runtime WASM. */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');

const [models, fixture, output, ...variants] = process.argv.slice(2);
const fixed = process.env.EN_ONNX_FIXED === '1';
const runs = Number(process.env.EN_ONNX_RUNS || 25);
const cases = JSON.parse(fs.readFileSync(fixture));
const roots = {
  ort: path.join(path.dirname(path.dirname(require.resolve('onnxruntime-web'))), 'dist'),
  py: path.dirname(require.resolve('pyodide/package.json')),
  dict: process.env.DIZIONE_DATA,
  models,
  source: path.dirname(process.env.DIZIONE_ZIP),
};
const server = http.createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<html><script src="/ort/ort.wasm.min.js"></script><script src="/py/pyodide.js"></script></html>');
    return;
  }
  const [, folder, name, extra] = req.url.split('/');
  if (!Object.hasOwn(roots, folder) || !name || extra || name === '.' || name === '..') {
    res.writeHead(404).end();
    return;
  }
  const file = path.join(roots[folder], name);
  res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm'
    : /\.m?js$/.test(file) ? 'text/javascript' : 'application/octet-stream');
  fs.createReadStream(file).on('error', () => res.writeHead(404).end()).pipe(res);
});

function processTreeRss(root) {
  const rows = execFileSync('ps', ['-e', '-o', 'pid=,ppid=,rss='], {encoding: 'utf8'})
    .trim().split('\n').map(row => row.trim().split(/\s+/).map(Number));
  function visit(pid) {
    const own = rows.find(row => row[0] === pid)?.[2] || 0;
    return rows.filter(row => row[1] === pid)
      .reduce((sum, child) => sum + visit(child[0]), own);
  }
  return visit(root) * 1024;
}

async function setup({threads, graph, zip, fixed}) {
  const start = performance.now();
  if (!crossOriginIsolated) throw Error('Cross-origin isolation required');
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = threads;
  const beginModel = async () => {
    const started = performance.now();
    const runtime = await ort.InferenceSession.create('/models/' + graph + '.onnx', {executionProviders: ['wasm']});
    return [runtime, performance.now() - started];
  };
  const beginFrontend = async () => {
    const started = performance.now();
    const py = await loadPyodide({indexURL: '/py/'});
    await py.loadPackage('numpy');
    const names = ['lexicon.tsv', 'align.json', 'g2p.json', 'g2p.npz', 'homographs.json'];
    const bytes = async url => {
      const response = await fetch(url);
      if (!response.ok) throw Error('Could not fetch ' + url);
      return new Uint8Array(await response.arrayBuffer());
    };
    const [source, ...data] = await Promise.all([bytes('/source/' + zip), ...names.map(n => bytes('/dict/' + n))]);
    py.FS.mkdirTree('/dizione-data');
    names.forEach((name, i) => py.FS.writeFile('/dizione-data/' + name, data[i]));
    py.unpackArchive(source, 'zip', {extractDir: '/dizione-src'});
    py.runPython(`
import json, os, sys
os.environ["DIZIONE_DATA"] = "/dizione-data"
sys.path.insert(0, "/dizione-src")
import dizione
def phones(text):
    p = dizione.phonemize(text)
    return json.dumps([list(p.ids), list(p.stress)])
phones("Warm up.")
`);
    return [py.globals.get('phones'), performance.now() - started];
  };
  const [[runtime, model_ms], [phones, front_ms]] = await Promise.all([beginModel(), beginFrontend()]);
  const load_ms = performance.now() - start;
  let noise;
  if (fixed) {
    // Same deterministic Gaussian source for both graphs. JS and Python
    // comparisons use separate noise generators and same-runtime references.
    noise = new Float32Array(786432);
    let seed = 2026;
    const uniform = () => {
      seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
      return ((seed >>> 0) + 1) / 4294967297;
    };
    for (let i = 0; i < noise.length; i += 2) {
      const radius = Math.sqrt(-2 * Math.log(uniform()));
      const angle = 2 * Math.PI * uniform();
      noise[i] = radius * Math.cos(angle);
      noise[i+1] = radius * Math.sin(angle);
    }
  }
  function inputs(test) {
    const [ids, stress] = JSON.parse(phones(test.text));
    if (test.tokens.join() !== ids.join() || test.stress.join() !== stress.join()) {
      throw Error('Phonemizer mismatch for ' + test.id);
    }
    const ints = (a, dims) => new ort.Tensor('int64', BigInt64Array.from(a, BigInt), dims);
    const scalar = v => new ort.Tensor('float32', Float32Array.of(v), [1]);
    const feed = {
      tokens: ints(ids, [1, ids.length]), stress: ints(stress, [1, ids.length]), voice: ints([test.voice], [1]),
      speed: scalar(test.speed), pitch_shift: scalar(test.pitch_shift), pitch_range: scalar(test.pitch_range),
    };
    if (noise) feed.noise = new ort.Tensor('float32', noise, [1, noise.length]);
    return feed;
  }
  window.requestAudio = async test => {
    const start = performance.now();
    const feed = inputs(test);
    const front = performance.now();
    const result = await runtime.run(feed);
    const output = result.audio.data.slice();
    const durations = Array.from(result.durations.data, Number);
    if (output.length !== durations.reduce((a,b) => a + b, 0) * 256) throw Error('output shape');
    let peak = 0;
    for (const sample of output) {
      if (!Number.isFinite(sample)) throw Error('non-finite output');
      peak = Math.max(peak, Math.abs(sample));
    }
    if (peak > .99) for (let i=0; i<output.length; ++i) output[i] *= .99 / peak;
    return {total_ms: performance.now() - start, front_ms: front - start, samples: output.length};
  };
  window.compareAudio = async (reference, candidates, testCases) => {
    const graphs = await Promise.all(candidates.map(name =>
      ort.InferenceSession.create('/models/' + name + '.onnx', {executionProviders: ['wasm']})));
    const records = {};
    for (const test of testCases) {
      const feed = inputs(test);
      const expected = await runtime.run(feed);
      records[test.id] = {};
      for (const [i, model] of graphs.entries()) {
        const actual = await model.run(feed);
        const ref = expected.audio.data;
        const wave = actual.audio.data;
        const length_equal = wave.length === ref.length;
        const duration_equal = expected.durations.data.every((x, k) => x === actual.durations.data[k]);
        let error = 0, signal = 0, peak = 0, max = 0;
        if (length_equal) for (let k=0; k<ref.length; ++k) {
          if (!Number.isFinite(wave[k])) throw Error('non-finite output');
          error += (ref[k] - wave[k]) ** 2;
          signal += ref[k] ** 2;
          peak = Math.max(peak, Math.abs(ref[k]));
          max = Math.max(max, Math.abs(ref[k] - wave[k]));
        }
        records[test.id][candidates[i]] = {
          duration_equal, length_equal, samples: wave.length,
          snr_db: length_equal && error ? 10*Math.log10(signal/error) : null,
          normalized_peak: length_equal ? max / Math.max(peak, 1e-12) : null,
        };
      }
    }
    return records;
  };
  return {load_ms, model_ms, front_ms, threads};
}

(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = 'http://127.0.0.1:' + server.address().port;
  const results = [];
  const orders = fixed ? ['forward'] : ['forward', 'reverse'];
  try {
    for (const order of orders) {
      const sequence = [1, 4].flatMap(threads => variants.map(graph => ({threads, graph})));
      if (order === 'reverse') sequence.reverse();
      for (const options of fixed ? sequence.filter(x => x.graph === variants[0]) : sequence) {
        const browserServer = await chromium.launchServer({headless: true, args: ['--no-sandbox']});
        const browser = await chromium.connect(browserServer.wsEndpoint());
        const pid = browserServer.process().pid;
        let rss_peak_bytes = processTreeRss(pid);
        const poll = setInterval(() => { rss_peak_bytes = Math.max(rss_peak_bytes, processTreeRss(pid)); }, 250);
        try {
          const page = await browser.newPage();
          await page.goto(url);
          const stats = await page.evaluate(setup, {...options, fixed, zip: path.basename(process.env.DIZIONE_ZIP)});
          const record = {...options, ...stats, order, browser: browser.version()};
          if (fixed) {
            record.parity = await page.evaluate(
              ({variant, candidates, cases}) => window.compareAudio(variant, candidates, cases),
              {variant: options.graph, candidates: variants.slice(1), cases});
          } else {
            record.cases = await page.evaluate(async ({cases, runs}) => {
              const measures = [];
              for (const test of cases) {
                const first = await window.requestAudio(test);
                for (let i=0; i<5; ++i) await window.requestAudio(test);
                const times = [];
                for (let i=0; i<runs; ++i) times.push(await window.requestAudio(test));
                measures.push({case: test.label, id: test.id, first, times});
              }
              return measures;
            }, {cases, runs});
          }
          results.push({...record, rss_peak_bytes});
          console.log(order, options.threads, options.graph, stats.load_ms);
        } catch (error) {
          results.push({...options, order, error: String(error)});
          console.error(order, options.threads, options.graph, String(error));
        } finally {
          clearInterval(poll);
          await browser.close();
          await browserServer.close();
        }
        fs.writeFileSync(output, JSON.stringify(results) + '\n');
      }
    }
  } finally {
    server.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
