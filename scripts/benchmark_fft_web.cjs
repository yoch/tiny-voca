/* Real Chromium + ONNX Runtime Web WASM, isolated browser process per case.
   NODE_PATH=/tmp/e04-web/node_modules node scripts/benchmark_fft_web.cjs /tmp/e04
   Install playwright and onnxruntime-web@1.20.1 in that prefix first. */
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
const root = path.resolve(process.argv[2]);
const ortRoot = path.dirname(path.dirname(require.resolve('onnxruntime-web')));
const server = http.createServer((req,res) => {
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  const url = new URL(req.url,'http://localhost');
  if (url.pathname === '/') {res.setHeader('Content-Type','text/html');res.end('<!doctype html><script src="/ort/ort.wasm.min.js"></script>');return;}
  const file = url.pathname.startsWith('/ort/') ? path.join(ortRoot,'dist',path.basename(url.pathname)) : path.join(root,path.basename(url.pathname));
  res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':file.endsWith('.js')||file.endsWith('.mjs')?'text/javascript':'application/octet-stream');
  fs.createReadStream(file).on('error',()=>{res.statusCode=404;res.end();}).pipe(res);
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const rows=[];
  try {
    for (const model of ['filter-dense','filter-dft','filter-radix2','full-dense','full-dft','full-radix2']) for(const threads of [1,4]) {
      const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
      try {
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${server.address().port}`);
        const row = await page.evaluate(async({model,threads})=>{
          ort.env.wasm.wasmPaths='/ort/'; ort.env.wasm.numThreads=threads;
          if(threads===4&&!crossOriginIsolated) throw Error('Missing cross-origin isolation');
          const session=await ort.InferenceSession.create(`/${model}.onnx`,{executionProviders:['wasm']});
          const data=await (await fetch(`/${model}.json`)).json();
          const feeds={};
          for(const name of session.inputNames){const v=data[name]; feeds[name]=new ort.Tensor(v.dtype==='int64'?'int64':'float32',v.dtype==='int64'?BigInt64Array.from(v.data,BigInt):Float32Array.from(v.data),v.shape);}
          for(let i=0;i<5;i++) await session.run(feeds);
          const times=[]; let result;
          for(let i=0;i<30;i++){const start=performance.now();result=await session.run(feeds);times.push(performance.now()-start);}
          const audio=result.audio.data,ref=data.reference.data;
          if(audio.length!==ref.length) throw Error('Output length changed');
          let max=0,signal=0,error=0;
          for(let i=0;i<audio.length;i++){const delta=audio[i]-ref[i];max=Math.max(max,Math.abs(delta));signal+=ref[i]**2;error+=delta**2;if(!Number.isFinite(audio[i]))throw Error('Non-finite audio');}
          times.sort((a,b)=>a-b);
          return {model,threads,runs:30,median_ms:(times[14]+times[15])/2,p95_ms:times[28],max_abs_error:max,snr_db:10*Math.log10(signal/error),samples:audio.length};
        },{model,threads});
        // Chromium's process metrics: include all renderer/worker processes, exclude the browser host.
        const cdp=await browser.newBrowserCDPSession();
        const {processInfo}=await cdp.send('SystemInfo.getProcessInfo');
        let rss=0;
        for(const p of processInfo.filter(p=>p.type!=='browser')){
          try{const text=fs.readFileSync(`/proc/${p.id}/status`,'utf8');rss+=Number(text.match(/^VmRSS:\s+(\d+)/m)[1])/1024;}catch{}
        }
        row.browser_process_rss_mib=rss;
        row.browser=await browser.version(); rows.push(row);console.log(JSON.stringify(row));
      } finally {await browser.close();}
    }
    fs.writeFileSync(path.join(root,'web-results.json'),JSON.stringify({ort:'1.20.1',rows},null,2)+'\n');
  } finally {server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
