/* Run the production worker/player against local, pinned qualification assets. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const http = require('node:http');
const { chromium, firefox, webkit } = require('playwright');

const directory = path.resolve(process.argv[2]);
const assets = path.join(directory, 'assets');
const web = path.resolve(__dirname, '../../web/public');
const ort = path.join(path.dirname(path.dirname(require.resolve('onnxruntime-web'))), 'dist');
const pyodide = path.dirname(require.resolve('pyodide/package.json'));
const reports = [];
const cases = JSON.parse(fs.readFileSync(path.join(directory, 'cases.json')));
const requested = (process.env.E04C_BROWSERS || 'chromium,firefox,webkit').split(',');
const rounds = Number(process.env.E04C_ROUNDS || 2);
const runs = Number(process.env.E04C_RUNS || 15);
let fault, requests = [];
const mime = name => name.endsWith('.wasm') ? 'application/wasm'
  : name.endsWith('.html') ? 'text/html'
  : /\.m?js$/.test(name) ? 'text/javascript'
  : name.endsWith('.json') ? 'application/json' : 'application/octet-stream';

const server = http.createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  const url = new URL(req.url, 'http://127.0.0.1');
  const parts = url.pathname.split('/').slice(1);
  const name = parts.at(-1) || 'index.html';
  if (name === 'qualification.html') {
    res.setHeader('Content-Type','text/html');
    return res.end('<!doctype html><script src="/ort/ort.wasm.min.js"></script><script src="/compact-vocoder.js"></script>');
  }
  const roots = { ort, pyodide, assets, phonemizer: process.env.DIZIONE_DATA };
  const root = parts[0] === 'assets' && parts[1] === 'phonemizer' ? roots.phonemizer : roots[parts[0]] || web;
  requests.push(url.pathname);
  if (fault === 'missing' && name === 'dsp.wasm') {
    res.statusCode = 404;
    return res.end();
  }
  res.setHeader('Content-Type', mime(name));
  const file = path.join(root, name);
  if (name === 'worker.js') {
    // Change dependency URLs only. Synthesis, fallback, messages and timing use the production source.
    return res.end(fs.readFileSync(file,'utf8')
      .replace('https://cdn.jsdelivr.net/pyodide/v0.26.4/full/', '/pyodide/')
      .replace('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/', '/ort/'));
  }
  if (name === 'compact-vocoder.js' && fault === 'render') {
    return res.end(fs.readFileSync(file,'utf8') + '\nCompactVocoder.prototype.render = () => {throw Error("injected test failure")};');
  }
  const stream = fs.createReadStream(file);
  stream.on('error', () => {res.statusCode = 404; res.end();});
  stream.pipe(res);
});

const mid = cases.find(c => c.id === '2');
const long = Array(10).fill('The fresh sea breeze swept softly through the trees as we watched the quiet waves from the shore').join(', ') + '.';
const hard = 'She sells seashells by the seashore. Wait… Did you hear that? Yes!';
const clean = message => ({...message, audio: undefined});

async function boot({variant, threads}) {
  const q = new URLSearchParams({threads});
  if (variant === 'compact') q.set('vocoder','compact');
  const worker = new Worker('/worker.js?' + q);
  window.pilotWorker = worker;
  window.pilotFaults = [];
  worker.addEventListener('message', event => {
    if (event.data.type === 'vocoder-fallback') window.pilotFaults.push(event.data.reason);
  });
  const ready = new Promise((resolve, reject) => {
    worker.onerror = event => reject(Error(event.message));
    worker.onmessage = event => {
      if (event.data.type === 'error') reject(Error(event.data.message));
      if (event.data.type === 'ready') resolve(event.data);
    };
  });
  const start = performance.now();
  worker.postMessage({type:'load'});
  return { ...await ready, loadMs: performance.now()-start };
}

async function speak({text, voice, speed, pitchShift, pitchRange}) {
  const worker = window.pilotWorker;
  const started = performance.now();
  return await new Promise((resolve,reject) => {
    let first, count = 0, length = 0;
    function receive({data}) {
      if (data.type === 'sentence') {
        first ??= performance.now() - started;
        ++count;
        length += data.audio.length;
      }
      if (data.type === 'done' || data.type === 'error') {
        worker.removeEventListener('message',receive);
        if (data.type === 'error') return reject(Error(data.message));
        const {audio, ...metrics} = data;
        if (!audio.length || length !== audio.length || audio.some(x => !Number.isFinite(x))) {
          return reject(Error('Bad output audio'));
        }
        // A short fingerprint tests that independent requests produce distinct noise.
        let checksum = 0;
        for (let i = 0; i < audio.length; i += 19) checksum += audio[i];
        resolve({...metrics, samples:audio.length, count, firstMs:first,
          roundTripMs:performance.now()-started, checksum, faults:[...window.pilotFaults]});
      }
    }
    worker.addEventListener('message',receive);
    worker.postMessage({type:'speak',id:1,text,voice,speed,pitchShift,pitchRange});
  });
}

async function ui(page, base) {
  await page.addInitScript(() => {
    const AudioSource = globalThis.AudioBufferSourceNode;
    window.startedAudio = 0;
    if (AudioSource) {
      const start = AudioSource.prototype.start;
      AudioSource.prototype.start = function (...args) {
        window.startedAudio++;
        return start.apply(this,args);
      };
    }
  });
  await page.goto(base + '/?vocoder=compact');
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Ready.');
  await page.locator('#text').fill('She sees fish. Did you hear that?');
  await page.locator('#speak').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Done.');
  await page.waitForFunction(() => document.querySelector('#audio').duration > 0);
  return page.evaluate(() => ({
    audioStarts: window.startedAudio, duration: document.querySelector('#audio').duration,
    hasBlob: document.querySelector('#audio').src.startsWith('blob:'),
    phonemes: document.querySelector('#phonemes').textContent,
  }));
}

async function parity({cases, threads, saveAudio}) {
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = threads;
  const get = async name => new Uint8Array(await (await fetch('/assets/'+name)).arrayBuffer());
  const dense = await ort.InferenceSession.create(await get('dense-reference.onnx'),{executionProviders:['wasm']});
  const ref = [];
  const feeds = c => {
    const integers = (v, dims) => new ort.Tensor('int64', BigInt64Array.from(v,BigInt),dims);
    const scalar = n => new ort.Tensor('float32',Float32Array.of(n),[1]);
    return {tokens:integers(c.tokens,[1,c.tokens.length]),stress:integers(c.stress,[1,c.stress.length]),
      voice:integers([c.voice],[1]),speed:scalar(c.speed),pitch_shift:scalar(c.pitch_shift),pitch_range:scalar(c.pitch_range)};
  };
  // Identical Box–Muller samples. The deterministic generator is confined to qualification.
  function random() {
    let state = 42;
    return () => ((state = (Math.imul(state,1664525)+1013904223) >>> 0) / 4294967296);
  }
  const noise = new Float32Array(3000*256);
  fillGaussian(noise, random());
  for (const c of cases) {
    const out = await dense.run({...feeds(c),noise:new ort.Tensor('float32',noise,[1,noise.length])});
    ref.push({audio:out.audio.data, durations:out.durations.data});
  }
  await dense.release();
  const model = await ort.InferenceSession.create(await get('model-compact.onnx'),{executionProviders:['wasm']});
  const dsp = await CompactVocoder.create(await get('dsp.wasm'),await get('dsp-constants.f32'),3000);
  const rows = [];
  for (const [i,c] of cases.entries()) {
    const candidate = await model.run(feeds(c));
    const {audio} = dsp.render(candidate,c.tokens.length,random());
    const reference = ref[i].audio;
    if (audio.length !== reference.length || reference.some(x => !Number.isFinite(x)) ||
        String(candidate.durations.data) !== String(ref[i].durations)) throw Error('Parity length or durations');
    let max=0, peak=0, signal=0, error=0;
    for (let j=0;j<audio.length;j++) {
      max=Math.max(max,Math.abs(audio[j]-reference[j]));
      peak=Math.max(peak,Math.abs(reference[j]));
      signal+=reference[j]**2;
      error+=(audio[j]-reference[j])**2;
    }
    const snr = error ? 10*Math.log10(signal/error) : Infinity;
    const normalized = max / peak;
    if (!(snr > 100 && normalized < 2e-5)) throw Error(`Parity threshold failed: ${snr} dB / ${normalized}`);
    rows.push({id:c.id,samples:audio.length,snr_db:snr,normalized_peak:normalized,
      ...(saveAudio ? {dense:Array.from(reference),compact:Array.from(audio)} : {})});
  }
  await model.release();
  return rows;
}

function wav(file, audio) {
  const bytes = Buffer.alloc(44+4*audio.length);
  bytes.write('RIFF'); bytes.writeUInt32LE(bytes.length-8,4); bytes.write('WAVEfmt ',8);
  bytes.writeUInt32LE(16,16); bytes.writeUInt16LE(3,20); bytes.writeUInt16LE(1,22);
  bytes.writeUInt32LE(24000,24); bytes.writeUInt32LE(96000,28);
  bytes.writeUInt16LE(4,32); bytes.writeUInt16LE(32,34);
  bytes.write('data',36); bytes.writeUInt32LE(bytes.length-44,40);
  audio.forEach((x,i) => bytes.writeFloatLE(x,44+i*4));
  fs.writeFileSync(file,bytes);
}

async function rssSampler(browser, name) {
  if (name !== 'chromium' || process.platform !== 'linux') return { finish:async () => null };
  const cdp = await browser.newBrowserCDPSession();
  let peak = 0, pending;
  async function measure() {
    const {processInfo} = await cdp.send('SystemInfo.getProcessInfo');
    const sum = processInfo.reduce((bytes,p) => {
      try {
        const content = fs.readFileSync('/proc/'+p.id+'/status','utf8');
        return bytes + Number(content.match(/^VmRSS:\s+(\d+)/m)[1]) * 1024;
      } catch {return bytes;}
    },0);
    peak = Math.max(peak,sum);
  }
  const timer = setInterval(() => {
    if (!pending) pending = measure().catch(() => {}).finally(() => {pending = null;});
  },100);
  return { finish:async () => { clearInterval(timer); await pending; await measure(); await cdp.detach(); return peak; } };
}

async function run() {
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const base = 'http://127.0.0.1:'+server.address().port;
  const output = path.join(directory, process.env.E04C_REPORT || 'browser-results.json');
  const save = () => fs.writeFileSync(output, JSON.stringify(reports,null,2)+'\n');
  try {
    for (const [name, engine] of Object.entries({chromium,firefox,webkit})) {
      if (!requested.includes(name)) continue;
      const record = {engine:name, runs:[], parity:[]};
      reports.push(record);
      let browser;
      try {
        browser = await engine.launch({
          headless:true,args:name==='chromium'?['--no-sandbox']:[],
          executablePath:name==='webkit'?process.env.E04C_WEBKIT_EXECUTABLE:undefined,
        });
        record.version = browser.version();
        for (const threads of [1,4]) {
          for (let round = 0; round < rounds; round++) {
            const order = round%2 ? ['compact','dense'] : ['dense','compact'];
            for (const variant of order) {
              requests = [];
              const context = await browser.newContext();
              const page = await context.newPage();
              const rss = await rssSampler(browser,name);
              try {
                await page.goto(base+'/qualification.html');
                const ready = await page.evaluate(boot,{variant,threads});
                assert.equal(ready.backend,variant);
                assert.equal(ready.threads,threads);
                assert.equal(requests.filter(x=>/\/model(?:-compact)?\.onnx$/.test(x)).length,1);
                const input = {text:mid.text,voice:mid.voice,speed:mid.speed,pitchShift:.08,pitchRange:1.25};
                const first = await page.evaluate(speak,input);
                for (let i=0;i<5;i++) await page.evaluate(speak,input);
                const measurements = [];
                for (let i=0;i<runs;i++) measurements.push(await page.evaluate(speak,input));
                if (measurements.length) assert.notEqual(first.checksum,measurements[0].checksum);
                assert.ok(measurements.every(m=>m.backend===variant && m.faults.length===0));
                const scenario = round===0 && variant==='compact' && threads===1
                  ? {long:await page.evaluate(speak,{...input,text:long,speed:1.5}),
                    hard:await page.evaluate(speak,{...input,text:hard,voice:1,speed:1.2,pitchShift:-.1})} : null;
                if (scenario) assert.ok(scenario.long.count>1 && scenario.hard.count>1);
                record.runs.push({threads,variant,round,ready,first,measurements,scenario,
                  sampledPeakRssBytes:await rss.finish()});
                console.log(name,threads,round,variant,Math.round(first.roundTripMs),'ms',scenario?scenario.long.count+' segments':'');
              } catch (error) {
                await rss.finish().catch(()=>{});
                record.runs.push({threads,variant,round,error:String(error)});
                console.error(name,threads,variant,String(error));
              } finally {await context.close();save();}
            }
          }
          const context = await browser.newContext();
          try {
            const page = await context.newPage();
            await page.goto(base+'/qualification.html');
            const saveAudio = name==='chromium' && threads===1;
            const rows = await page.evaluate(parity,{cases,threads,saveAudio});
            for (const {id,dense,compact,...values} of rows) {
              if (saveAudio) {
                wav(path.join(directory,id+'-dense.wav'),dense);
                wav(path.join(directory,id+'-compact.wav'),compact);
              }
              record.parity.push({threads,id,...values});
            }
          } catch (error) {record.parity.push({threads,error:String(error)});}
          finally {await context.close();save();}
        }
        const context = await browser.newContext();
        try {record.ui = await ui(await context.newPage(),base);}
        catch (error) {record.ui = {error:String(error)};}
        finally {await context.close();}
        if (name==='chromium') {
          for (const mode of ['missing','render']) {
            fault=mode;requests=[];
            const context = await browser.newContext();
            try {
              const page=await context.newPage();
              await page.goto(base+'/qualification.html');
              const ready=await page.evaluate(boot,{variant:'compact',threads:1});
              assert.equal(ready.backend,mode==='missing'?'dense':'compact');
              const done=await page.evaluate(speak,{text:'Yes, please.',voice:0,speed:1});
              assert.equal(done.backend,'dense');
              assert.equal(done.faults.length,1);
              assert.equal(requests.filter(x=>x==='/assets/model.onnx').length,1);
              (record.fallback ??= []).push({mode,done:clean(done)});
            } finally {fault=null;await context.close();}
          }
        }
      } catch (error) {record.error=String(error);console.error(name,String(error));}
      finally {await browser?.close();save();}
    }
    if (!reports.length || reports.some(record => record.error || record.ui?.error ||
        record.runs.some(row => row.error) || record.parity.some(row => row.error))) {
      process.exitCode = 1;
    }
  } finally {server.close();}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
