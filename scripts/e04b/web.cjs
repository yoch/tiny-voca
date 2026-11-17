/* Real Chromium, same dizione/Pyodide and ORT WASM versions as worker.js. */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('playwright');
const root = path.resolve(process.argv[2]);
const phonemizer = path.resolve(process.env.DIZIONE_DATA);
const ort = path.join(path.dirname(path.dirname(require.resolve('onnxruntime-web'))), 'dist');
const pyodide = path.dirname(require.resolve('pyodide/package.json'));
const variants = (process.env.E04B_VARIANTS || 'dense,fused,compact').split(',');
const reverse = process.argv.includes('--reverse');
const runs = Number(process.env.E04B_RUNS || 30);
const cases = JSON.parse(fs.readFileSync(path.join(root,'cases.json')));
const routes = { ort, pyodide, phonemizer, native: path.join(root,'native') };
const server = http.createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<!doctype html><script src="/ort/ort.wasm.min.js"></script><script src="/pyodide/pyodide.js"></script>');
    return;
  }
  const [, folder] = req.url.split('/');
  const file = path.join(routes[folder] ?? root, path.basename(req.url));
  res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' :
    /\.m?js$/.test(file) ? 'text/javascript' : 'application/octet-stream');
  fs.createReadStream(file).on('error', () => { res.statusCode = 404; res.end(); }).pipe(res);
});

async function setup({variant, threads}) {
  const start = performance.now();
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = threads;
  if (!crossOriginIsolated) throw Error('WASM threads require cross-origin isolation');

  const loadFrontend = async () => {
    const py = await loadPyodide({indexURL:'/pyodide/'});
    await py.loadPackage('numpy');
    const names = ['lexicon.tsv','align.json','g2p.json','g2p.npz','homographs.json'];
    const bytes = async filename => new Uint8Array(await (await fetch(filename)).arrayBuffer());
    py.FS.mkdirTree('/dizione-data');
    const [source, ...files] = await Promise.all([bytes('/dizione.zip'), ...names.map(x => bytes('/phonemizer/'+x))]);
    names.forEach((name,i) => py.FS.writeFile('/dizione-data/'+name,files[i]));
    py.unpackArchive(source,'zip',{extractDir:'/dizione-src'});
    py.runPython(`
import json, os, sys
os.environ['DIZIONE_DATA'] = '/dizione-data'
sys.path.insert(0,'/dizione-src')
import dizione

def phonemize_json(text):
    result = dizione.phonemize(text)
    return json.dumps([list(result.ids), list(result.stress)])

phonemize_json("Warm up.")
`);
    return py.globals.get('phonemize_json');
  };

  const graph = variant === 'fused' ? 'native/stage' : variant;
  const [frontend, model, noiseBuffer, wasm, constants] = await Promise.all([
    loadFrontend(),
    ort.InferenceSession.create('/'+graph+'.onnx',{executionProviders:['wasm']}),
    fetch('/noise.f32').then(r => r.arrayBuffer()),
    variant === 'dense' ? null : WebAssembly.instantiateStreaming(fetch('/dsp.wasm')),
    variant === 'dense' ? null : fetch('/constants.json').then(r => r.json()),
  ]);
  const noise = new Float32Array(noiseBuffer);
  let heap, e, buffers;
  if (wasm) {
    e = wasm.instance.exports;
    e._initialize();
    let next = e.__heap_base.value;
    const allocate = size => {
      const addr = next;
      next += size * 4;
      return addr;
    };
    const constantPtrs = constants.map(values => allocate(values.length));
    const harmonic = Math.floor(noise.length/128)+3;
    const noisy = Math.floor(noise.length/64)+3;
    const late = variant === 'fused';
    const frames = noise.length / 256;
    const sizes = late ? {pulses:noise.length, hr:harmonic*257, hi:harmonic*257, noise_source:noise.length, nr:noisy*129, ni:noisy*129} :
      {pulses:noise.length, noise_source:noise.length, harmonic:frames*98, noise_head:frames*36, f0:frames};
    sizes.audio = sizes.secondary = noise.length;
    buffers = Object.fromEntries(Object.entries(sizes).map(([name,size]) => [name,allocate(size)]));
    e.memory.grow(Math.max(0, Math.ceil((next - e.memory.buffer.byteLength)/65536)));
    heap = new Float32Array(e.memory.buffer);
    constants.forEach((values,i) => heap.set(values,constantPtrs[i]/4));
    e.initialize(...constantPtrs);
    if (!late) heap.set(noise,buffers.noise_source/4);
  }

  const load_ms = performance.now()-start;
  window.runCase = async (test, retain = false) => {
    const start = performance.now();
    const [ids, stress] = JSON.parse(frontend(test.text));
    if (retain && (ids.join() !== test.tokens.join() || stress.join() !== test.stress.join())) throw Error('Phonemizer parity');
    const ints = (a, dims) => new ort.Tensor('int64',BigInt64Array.from(a,BigInt),dims);
    const scalar = v => new ort.Tensor('float32',Float32Array.of(v),[1]);
    const inputs = {
      tokens:ints(ids,[1,ids.length]), stress:ints(stress,[1,ids.length]), voice:ints([test.voice],[1]),
      speed:scalar(test.speed), pitch_shift:scalar(test.pitch_shift), pitch_range:scalar(test.pitch_range),
    };
    const samples = test.frames * 256;
    if (variant === 'dense' || variant === 'fused') {
      inputs.noise = new ort.Tensor('float32',noise.subarray(0,samples),[1,samples]);
    }
    const frontendEnd = performance.now();
    const parts = await model.run(inputs);
    const ortEnd = performance.now();
    let out, inputCopyEnd = ortEnd, dspEnd = ortEnd, outputCopyEnd = ortEnd;
    if (variant === 'dense') {
      out = parts.audio.data;
    } else {
      for (const [key,value] of Object.entries(parts)) {
        if (key !== 'durations') heap.set(value.data,buffers[key]/4);
      }
      inputCopyEnd = performance.now();
      const b = buffers;
      if (variant === 'fused') {
        e.filter(0,samples,b.pulses,b.hr,b.hi,b.audio);
        e.filter(1,samples,b.noise_source,b.nr,b.ni,b.secondary);
      } else {
        if (variant === 'early') e.oscillator(test.frames,b.f0,b.pulses);
        e.compact_filter(0,test.frames,b.pulses,b.harmonic,b.audio);
        e.compact_filter(1,test.frames,b.noise_source,b.noise_head,b.secondary);
      }
      e.mix(samples,b.audio,b.secondary);
      dspEnd = performance.now();
      out = heap.slice(b.audio/4,b.audio/4+samples);
      outputCopyEnd = performance.now();
    }
    const heard = out.slice();
    let peak = 0;
    for (const x of heard) peak = Math.max(peak,Math.abs(x));
    if (peak > .99) for (let i=0;i<heard.length;++i) heard[i] *= .99/peak;
    const first = performance.now();
    const final = heard.slice();
    const end = performance.now();
    const timing = {
      frontend_ms:frontendEnd-start, ort_ms:ortEnd-frontendEnd, js_to_wasm_ms:inputCopyEnd-ortEnd,
      dsp_ms:dspEnd-inputCopyEnd, wasm_to_js_ms:outputCopyEnd-dspEnd, first_audio_ms:first-start,
      publish_ms:end-outputCopyEnd, total_ms:end-start,
    };
    if (!retain) return timing;
    const durations = Array.from(parts.durations.data,Number);
    if (durations.reduce((a,b)=>a+b,0)*256 !== samples || out.length !== samples || out.some(x=>!Number.isFinite(x))) {
      throw Error('Audio length, durations or finite check failed');
    }
    return {timing, audio:Array.from(out), durations, samples:final.length,
      boundary_bytes:Object.values(parts).reduce((sum,x)=>sum+x.data.byteLength,0)};
  };
  return { load_ms, wasm_pages:e ? e.memory.buffer.byteLength/65536 : 0 };
}

function wav(file, audio) {
  const b = Buffer.alloc(44+4*audio.length);
  b.write('RIFF'); b.writeUInt32LE(b.length-8,4); b.write('WAVEfmt ',8);
  b.writeUInt32LE(16,16); b.writeUInt16LE(3,20); b.writeUInt16LE(1,22);
  b.writeUInt32LE(24000,24); b.writeUInt32LE(96000,28);
  b.writeUInt16LE(4,32); b.writeUInt16LE(32,34);
  b.write('data',36); b.writeUInt32LE(b.length-44,40);
  audio.forEach((value,i) => b.writeFloatLE(value,44+i*4));
  fs.writeFileSync(file,b);
}

function parity(audio, reference) {
  let signal=0, error=0, peak=0, max=0;
  for (let i=0;i<audio.length;i++) {
    signal += reference[i]**2;
    error += (audio[i]-reference[i])**2;
    peak = Math.max(peak,Math.abs(reference[i]));
    max = Math.max(max,Math.abs(audio[i]-reference[i]));
  }
  return {max_abs_error:max, normalized_peak_error:max/peak, snr_db:error ? 10*Math.log10(signal/error) : null, exact:!error};
}

(async () => {
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const all = [];
  const output = path.join(root,'web-audio');
  fs.mkdirSync(output,{recursive:true});
  const order = [1,4].flatMap(threads => variants.map(variant => ({threads,variant})));
  if (reverse) order.reverse();
  try {
    for (const options of order) {
      const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
      let poll;
      try {
        const page = await browser.newPage();
        await page.goto('http://127.0.0.1:'+server.address().port);
        const cdp = await browser.newBrowserCDPSession();
        let peak_rss_mib = 0, measuring = false;
        const rss = async () => {
          if (measuring) return;
          measuring = true;
          try {
            const {processInfo} = await cdp.send('SystemInfo.getProcessInfo');
            const mib = processInfo.reduce((total,proc) => {
              try {
                const data = fs.readFileSync('/proc/'+proc.id+'/status','utf8');
                return total+Number(data.match(/^VmRSS:\s+(\d+)/m)[1])/1024;
              } catch {return total;}
            },0);
            peak_rss_mib = Math.max(mib,peak_rss_mib);
          } finally {measuring=false;}
        };
        poll = setInterval(() => rss().catch(() => {}),100);
        const load = await page.evaluate(setup,options);
        const cold = await page.evaluate(test => window.runCase(test),cases[0]);
        const rows = [];
        for (const test of reverse ? [...cases].reverse() : cases) {
          const result = await page.evaluate(async ({test,runs}) => {
            for(let i=0;i<5;i++) await window.runCase(test);
            const timings = [];
            for(let i=0;i<runs;i++) timings.push(await window.runCase(test));
            return {timings,...await window.runCase(test,true)};
          },{test,runs});
          const {audio,durations,timing,...row} = result;
          const name = path.join(output,[options.threads,test.id,options.variant].join('-'));
          fs.writeFileSync(name+'.json',JSON.stringify({audio,durations}));
          wav(name+'.wav',audio);
          rows.push({case:test.id,...row});
        }
        await rss();
        all.push({...options,...load,peak_rss_mib,first_audio_cold:cold,browser:browser.version(),rows});
        console.log(options, rows.map(r => r.timings.map(t=>t.total_ms).sort((a,b)=>a-b)[Math.floor(runs/2)]));
      } finally {
        clearInterval(poll);
        await browser.close();
      }
    }
    for (const item of all) for (const row of item.rows) {
      const read = v => JSON.parse(fs.readFileSync(path.join(output,[item.threads,row.case,v].join('-')+'.json')));
      const {audio,durations} = read(item.variant), ref = read('dense');
      if (durations.join() !== ref.durations.join()) throw Error('Durations changed');
      row.parity = {durations_equal:true,...parity(audio,ref.audio)};
    }
    fs.writeFileSync(path.join(root,reverse?'web-reverse.json':'web-forward.json'),JSON.stringify({ort:'1.20.1',pyodide:'0.26.4',cases:all})+'\n');
  } finally {server.close();}
})().catch(error => {console.error(error);process.exitCode=1;});
