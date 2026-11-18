const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

const script = fs.readFileSync(path.join(__dirname,'../../web/public/compact-vocoder.js'),'utf8');
const {CompactVocoder, fillGaussian} = vm.runInThisContext(
  `(() => {${script}\nreturn {CompactVocoder, fillGaussian};})()`);
const wasm = fs.readFileSync(process.env.TINY_VOCA_TEST_WASM);
function random() {
  let state = 42;
  return () => ((state = (Math.imul(state,1664525)+1013904223) >>> 0) / 4294967296);
}
const constants = new Float32Array(896);
for (const [offset,n] of [[0,512],[512,256]]) {
  for (let i=0;i<n;i++) constants[offset+i] = .5-.5*Math.cos(2*Math.PI*i/n);
}
constants.fill(1,768);
const bytes = new Uint8Array(constants.buffer);
function input(frames) {
  const tensor = (data,dims,type) => ({data,dims,type});
  const float = (...dims) => tensor(new Float32Array(dims.reduce((a,b)=>a*b,1)),dims,'float32');
  return {pulses:float(1,frames*256),harmonic:float(1,frames,98),noise_head:float(1,frames,36),
    durations:tensor(BigInt64Array.of(BigInt(frames)),[1,1],'int64')};
}

test('compact WASM validates shapes, grows on demand and preserves identity with zero cepstra',async () => {
  const dsp = await CompactVocoder.create(wasm,bytes,3000);
  assert.ok(dsp.heap.byteLength <= 2*65536);
  for (const frames of [1,5,16,37,256,1024,1]) {
    const out = dsp.render(input(frames),1,random());
    const expected = new Float32Array(frames*256);
    fillGaussian(expected,random());
    assert.equal(out.audio.length,expected.length);
    for (let i=0;i<expected.length;i++) assert.ok(Math.abs(out.audio[i]-expected[i]) < 4e-6);
    assert.equal(out.workspaceBytes,dsp.framesCapacity*1158*4);
    assert.ok(out.wasmBytes < out.workspaceBytes+3*65536);
  }
  assert.throws(()=>dsp.render(input(3001),1),/utterance size/);
  assert.throws(()=>dsp.render(input(0),1),/utterance size/);
  const malformed = input(4);
  malformed.durations.data[0] = 5n;
  assert.throws(()=>dsp.render(malformed,1),/dimensions/);
  const badSignal = input(4);
  badSignal.harmonic.data[0] = Infinity;
  assert.throws(()=>dsp.render(badSignal,1),/rejected/);
  const first = dsp.render(input(1),1).audio;
  const second = dsp.render(input(1),1).audio;
  assert.notDeepEqual(first,second);
});

test('compact WASM rejects invalid constants and render capacity',async () => {
  await assert.rejects(()=>CompactVocoder.create(wasm,new Uint8Array(896*4),3000),/constants/);
  await assert.rejects(()=>CompactVocoder.create(wasm,bytes,20000),/configuration/);
});
