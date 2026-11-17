/* Sequential Chromium + ORT Web/WASM; fresh browser per timed case.
   NODE_PATH=/tmp/e04-web/node_modules node scripts/benchmark_fft_web.cjs /tmp/e04
   Optional: --reverse --result-name=web-reverse.json */
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const root = path.resolve(process.argv[2]);
const reverse = process.argv.includes('--reverse');
const resultName = process.argv.find(x=>x.startsWith('--result-name='))?.split('=')[1] || 'web-results.json';
const ortRoot = path.dirname(path.dirname(require.resolve('onnxruntime-web')));
const ortVersion = JSON.parse(fs.readFileSync(path.join(ortRoot,'package.json'))).version;
if (ortVersion !== '1.20.1') throw Error(`Expected ORT Web 1.20.1, got ${ortVersion}`);
const server = http.createServer((req,res) => {
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  const url = new URL(req.url,'http://localhost');
  if (url.pathname === '/') {res.setHeader('Content-Type','text/html');res.end('<!doctype html><script src="/ort/ort.wasm.min.js"></script>');return;}
  const file = url.pathname.startsWith('/ort/') ? path.join(ortRoot,'dist',path.basename(url.pathname)) :
    path.join(root,url.pathname.startsWith('/cases/')?'cases':'',path.basename(url.pathname));
  res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':file.endsWith('.js')||file.endsWith('.mjs')?'text/javascript':'application/octet-stream');
  fs.createReadStream(file).on('error',()=>{res.statusCode=404;res.end();}).pipe(res);
});
const launch = ()=>chromium.launch({headless:true,args:['--no-sandbox']});
async function open(browser) {
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.addScriptTag({content:`
    function feedsFor(session,data) {
      return Object.fromEntries(session.inputNames.map(name=>{
        const v=data[name];
        return [name,new ort.Tensor(v.dtype==='int64'?'int64':'float32',v.dtype==='int64'?BigInt64Array.from(v.data,BigInt):Float32Array.from(v.data),v.shape)];
      }));
    }
    function metrics(result,ref) {
      const audio=result.audio.data;
      if(JSON.stringify(result.audio.dims)!==JSON.stringify(ref.shape))throw Error('Output shape changed');
      let peak=0,referencePeak=0,signal=0,error=0;
      for(let i=0;i<audio.length;i++){
        if(!Number.isFinite(audio[i]))throw Error('Non-finite audio');
        const delta=audio[i]-ref.data[i];peak=Math.max(peak,Math.abs(delta));referencePeak=Math.max(referencePeak,Math.abs(ref.data[i]));
        signal+=ref.data[i]**2;error+=delta**2;
      }
      const exact=error===0, snr=signal&&error?10*Math.log10(signal/error):null;
      let durationsEqual;
      if(ref.durations){durationsEqual=ref.durations.every((x,i)=>BigInt(x)===result.durations.data[i])&&ref.durations.length===result.durations.data.length;if(!durationsEqual)throw Error('Durations changed');}
      const normalized=peak/Math.max(referencePeak,1e-30);
      return {max_abs_error:peak,normalized_peak_error:normalized,snr_db:snr,exact,durations_equal:durationsEqual,accepted:normalized<2e-5&&(exact||snr>100)};
    }
    function cpuReference(data){return {...data.reference,durations:data.durations?.data};}
  `});
  return page;
}
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const rows=[];
  const cases=fs.existsSync(path.join(root,'cases'))?fs.readdirSync(path.join(root,'cases')).filter(x=>x.endsWith('.json')).map(x=>`cases/${x}`):[];
  try {
    // Dense references come from separate browsers, before any timed work.
    // This keeps their sessions and buffers out of the measured process RSS.
    for(const threads of [1,4]) {
      const browser=await launch();
      try {
        const page=await open(browser);
        const references=await page.evaluate(async({threads,cases})=>{
          ort.env.wasm.wasmPaths='/ort/';ort.env.wasm.numThreads=threads;
          if(threads===4&&!crossOriginIsolated)throw Error('Missing cross-origin isolation');
          const refs={};
          for(const kind of ['filter','full']) {
            const session=await ort.InferenceSession.create('/'+kind+'-dense.onnx',{executionProviders:['wasm']});
            for(const name of [kind+'-dense.json',...(kind==='full'?cases:[])]) {
              const data=await(await fetch('/'+name)).json();const out=await session.run(feedsFor(session,data));
              refs[name]={shape:out.audio.dims,data:Array.from(out.audio.data),durations:out.durations?Array.from(out.durations.data,Number):undefined};
            }
            await session.release();
          }
          return refs;
        },{threads,cases});
        fs.writeFileSync(path.join(root,`web-reference-${threads}.json`),JSON.stringify(references));
      } finally {await browser.close();}
    }
    const order=['filter-dense','filter-dft','filter-radix2','full-dense','full-dft','full-radix2'].flatMap(model=>[1,4].map(threads=>({model,threads})));
    if(reverse)order.reverse();
    for (const {model,threads} of order) {
      const browser=await launch();
      try {
        const page=await open(browser);
        const row=await page.evaluate(async({model,threads})=>{
          ort.env.wasm.wasmPaths='/ort/';ort.env.wasm.numThreads=threads;
          if(threads===4&&!crossOriginIsolated)throw Error('Missing cross-origin isolation');
          const session=await ort.InferenceSession.create('/'+model+'.onnx',{executionProviders:['wasm']});
          const data=await(await fetch('/'+model+'.json')).json();const feeds=feedsFor(session,data);
          const refs=await(await fetch('/web-reference-'+threads+'.json')).json();
          for(let i=0;i<5;i++)await session.run(feeds);
          const times=[];let result;
          for(let i=0;i<30;i++){const start=performance.now();result=await session.run(feeds);times.push(performance.now()-start);}
          const ordered=[...times].sort((a,b)=>a-b), rank=.95*(ordered.length-1),lo=Math.floor(rank);
          // Keep this session alive for validation after the RSS sample.
          window.benchmarkSession=session;
          return {model,threads,runs:30,times_ms:times,median_ms:(ordered[14]+ordered[15])/2,p95_ms:ordered[lo]+(ordered[lo+1]-ordered[lo])*(rank-lo),
            samples:result.audio.data.length,...metrics(result,refs[model.split('-')[0]+'-dense.json']),cpu_reference:metrics(result,cpuReference(data))};
        },{model,threads});
        const cdp=await browser.newBrowserCDPSession();
        const {processInfo}=await cdp.send('SystemInfo.getProcessInfo');
        let rss=0;
        for(const p of processInfo.filter(p=>p.type!=='browser')){
          const status=fs.readFileSync(`/proc/${p.id}/status`,'utf8');rss+=Number(status.match(/^VmRSS:\s+(\d+)/m)[1])/1024;
        }
        row.browser_process_rss_mib=rss;
        row.validation=await page.evaluate(async({model,threads,cases})=>{
          if(!model.startsWith('full-'))return {};
          const refs=await(await fetch('/web-reference-'+threads+'.json')).json();const validation={};
          for(const name of cases){const data=await(await fetch('/'+name)).json();const out=await window.benchmarkSession.run(feedsFor(window.benchmarkSession,data));validation[name]={...metrics(out,refs[name]),cpu_reference:metrics(out,cpuReference(data))};}
          return validation;
        },{model,threads,cases});
        row.model_bytes=fs.statSync(path.join(root,model+'.onnx')).size;
        row.browser=await browser.version();rows.push(row);console.log(JSON.stringify(row));
        fs.writeFileSync(path.join(root,resultName),JSON.stringify({ort:ortVersion,order:reverse?'reverse':'forward',reference:'same-runtime dense ONNX (CPU cross-runtime differences reported separately)',rows},null,2)+'\n');
      } finally {await browser.close();}
    }
  } finally {server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
