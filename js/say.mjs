import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { readGGUF, TinyVoca } from "./tiny-voca.js";
import { fromGGUF } from "./dizione/engine.js";
import { utterancesOf, limited, wav } from "./text.js";

const options = { voice: "M2", output: "tiny-voca.wav", file: "" };
const words = [];
for (let args = process.argv.slice(2); args.length; ) {
  const arg = args.shift();
  if (arg.startsWith("--") && arg.slice(2) in options) options[arg.slice(2)] = args.shift();
  else words.push(arg);
}
const text = words.join(" ");
if (!text) {
  console.log('node say.mjs "Hello! This is tiny voca." [--voice M2] [--output tiny-voca.wav] [--file tiny-voca.gguf]');
  process.exit(1);
}

const local = fileURLToPath(new URL("tiny-voca.gguf", import.meta.url));
const source = options.file || (existsSync(local) ? local : "https://huggingface.co/n4ze3m/tiny-voca/resolve/main/gguf/tiny-voca-js.gguf");
const started = performance.now();
const buffer = /^https?:/.test(source) ? await (await fetch(source)).arrayBuffer() : new Uint8Array(readFileSync(source)).buffer;
const fetched = performance.now();
const file = readGGUF(buffer);
const voca = new TinyVoca(file);
const phonemizer = await fromGGUF(file);
const ready = performance.now();

const chunks = utterancesOf(text, phonemizer, voca.config.max_tokens - 12)
  .map(({ ids, stress }, seed) => voca.synthesize(ids, stress, { voice: options.voice, seed }).audio);
const whole = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
chunks.reduce((at, chunk) => (whole.set(chunk, at), at + chunk.length), 0);
const done = performance.now();
writeFileSync(options.output, new Uint8Array(wav(limited(whole), voca.config.sample_rate)));

const seconds = whole.length / voca.config.sample_rate, work = (done - ready) / 1000;
console.log(`${options.output}: ${seconds.toFixed(2)} s of audio in ${work.toFixed(2)} s, ${(seconds / work).toFixed(1)} times real time`);
console.log(`file read in ${(fetched - started).toFixed(0)} ms, model and phonemizer ready in ${(ready - fetched).toFixed(0)} ms`);
