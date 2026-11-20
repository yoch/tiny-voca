import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync, brotliCompressSync } from "node:zlib";
import { readGGUF } from "../tiny-voca.js";
import { fromGGUF } from "../dizione/engine.js";
import { Model } from "../dizione/g2p.js";
import { toText } from "../dizione/convert.js";
import { packLexicon, unpackLexicon, gunzip } from "../dizione/lexicon.js";

const [path, folder, output = "dictionary.json"] = process.argv.slice(2);
if (!folder) {
  console.error("usage: node dictionary.mjs tiny-voca.gguf VECTORS_FOLDER [dictionary.json]");
  process.exit(1);
}
const file = readGGUF(new Uint8Array(readFileSync(path)).buffer), phonemizer = await fromGGUF(file);
const entries = [...phonemizer.lexicon], raw = new TextEncoder().encode(entries.map(([word, text]) => `${word}\t${text}\n`).join(""));
const gzip = (bytes) => gzipSync(bytes, { level: 9 }).length, brotli = (bytes) => brotliCompressSync(bytes).length;
const report = {};
const say = (name, value) => { report[name] = value; console.log(`${name}: ${value}`); writeFileSync(output, JSON.stringify(report, null, 1) + "\n"); };
const same = (map) => map.size === entries.length && entries.every(([word, text]) => map.get(word) === text);

say("words", entries.length);
say("dictionary as it is, bytes", raw.length);
say("a. gzip, bytes", gzip(raw));
say("a. brotli, bytes", brotli(raw));
const packed = packLexicon(entries);
say("b. packed, bytes", packed.length);
say("b. packed + gzip, bytes", gzip(packed));
say("b. packed + brotli, bytes", brotli(packed));
let started = performance.now();
const back = unpackLexicon(await gunzip(gzipSync(packed, { level: 9 })));
say("b. unpack in ms", (performance.now() - started).toFixed(0));
say("b. every word and pronunciation comes back the same", same(back));

const names = Object.keys(file.tensors).filter((name) => name.startsWith("g2p/")), float32 = phonemizer.model;
const bytes = { 32: 0, 16: 0, 8: 0 }, weights = { 16: {}, 8: {} };
for (const name of names) {
  const { shape, data } = file.tensors[name], half = new Float16Array(data), width = data.length / shape[0];
  const small = shape.length > 1 ? new Float32Array(data.length) : null;
  for (let row = 0; small && row < shape[0]; row++) {
    const values = data.subarray(row * width, (row + 1) * width), scale = values.reduce((top, w) => Math.max(top, Math.abs(w)), 0) / 127 || 1;
    values.forEach((w, i) => { small[row * width + i] = Math.round(w / scale) * scale; });
  }
  weights[16][name] = { shape, data: new Float32Array(half) };
  weights[8][name] = small ? { shape, data: small } : weights[16][name];
  bytes[32] += data.byteLength;
  bytes[16] += half.byteLength;
  bytes[8] += small ? data.length + 4 * shape[0] : half.byteLength;
}
const models = { 16: new Model(float32.settings, (name) => weights[16]["g2p/" + name]), 8: new Model(float32.settings, (name) => weights[8]["g2p/" + name]) };
const vectors = readdirSync(folder).filter((name) => name.endsWith(".tsv")).flatMap((name) => {
  const lines = readFileSync(join(folder, name), "utf8").split("\n");
  return lines.slice(lines.indexOf("") + 1).filter(Boolean).map((line) => line.split("\t"));
});
function differing() {
  let count = 0;
  for (const [text, ids, stress] of vectors) {
    phonemizer.cache.clear();
    let got = "-\t-";
    try {
      const result = phonemizer.phonemize(text);
      got = result.ids.map((id) => id.toString(16).padStart(2, "0")).join("") + "\t" + result.stress.join("");
    } catch (error) {
      if (error.name !== "ValueError") throw error;
    }
    if (got !== ids + "\t" + stress) count++;
  }
  return count;
}
say("unknown-word model, float32, bytes", bytes[32]);
say(`float32 model: test texts that differ, of ${vectors.length}`, differing());
for (const bits of [16, 8]) {
  say(`unknown-word model, ${bits}-bit, bytes`, bytes[bits]);
  phonemizer.model = models[bits];
  say(`${bits}-bit model: test texts that differ, of ${vectors.length}`, differing());
}
phonemizer.model = float32;
phonemizer.cache.clear();

const memo = new Map(), predict = float32.predict.bind(float32);
float32.predict = (word) => {
  if (!memo.has(word)) memo.set(word, predict(word));
  return [...memo.get(word)];
};
started = performance.now();
const guesses = [], exceptions = [], different = { 16: 0, 8: 0 };
entries.forEach(([word, text], i) => {
  const symbols = float32.predict(word).join(" ");
  for (const bits of [16, 8]) if (models[bits].predict(word).join(" ") !== symbols) different[bits]++;
  const guess = toText(phonemizer.guess(word));
  memo.clear();
  guesses.push(guess);
  if (guess !== text) exceptions.push([word, text]);
  if (i % 10000 === 0) console.log(`  ${i} words, ${((performance.now() - started) / 1000).toFixed(0)} s`);
});
const modelSeconds = (performance.now() - started) / 1000;
say("16-bit model: dictionary words with other phonemes", different[16]);
say("8-bit model: dictionary words with other phonemes", different[8]);
say("c. words the model already gets right", entries.length - exceptions.length);
say("c. entries left as exceptions", exceptions.length);
const reduced = Buffer.concat([packLexicon(entries.map(([word, text], i) => [word, guesses[i] === text ? "ə" : "ɐ"])), packLexicon(exceptions)]);
say("c. all words (to know what is a word) + exceptions, packed + gzip, bytes", gzip(reduced));
say("c. all words + exceptions, packed + brotli, bytes", brotli(reduced));
say("c. exceptions alone, packed + gzip, bytes", gzip(packLexicon(exceptions)));
const kept = new Map(exceptions);
say("c. every word still comes out the same", entries.every(([word, text], i) => (kept.get(word) ?? guesses[i]) === text));
started = performance.now();
for (const [word] of entries) phonemizer.lexicon.get(word);
say("c. a lookup in the full dictionary, microseconds", (1000 * (performance.now() - started) / entries.length).toFixed(2));
say("c. a lookup that needs the model, microseconds", (1e6 * modelSeconds / entries.length / 3).toFixed(0));
