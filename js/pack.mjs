import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { readGGUF } from "./tiny-voca.js";
import { PACKED, packLexicon } from "./dizione/lexicon.js";

const [input, tables, output] = process.argv.slice(2);
if (!output) {
  console.error("usage: node pack.mjs tiny-voca-orig.gguf DIZIONE_DATA_FOLDER tiny-voca.gguf");
  process.exit(1);
}
const file = readGGUF(new Uint8Array(readFileSync(input)).buffer);

const encoder = new TextEncoder();
const u32 = (n) => new Uint8Array(new Uint32Array([n]).buffer);
const u64 = (n) => new Uint8Array(new BigUint64Array([BigInt(n)]).buffer);
const string = (text) => { const bytes = encoder.encode(text); return [u64(bytes.length), bytes]; };
const padded = (size) => Math.ceil(size / 32) * 32;

function eightBit(data) {
  const bytes = new Uint8Array(Math.ceil(data.length / 32) * 34), view = new DataView(bytes.buffer);
  for (let block = 0; block * 32 < data.length; block++) {
    const values = data.subarray(block * 32, block * 32 + 32);
    const scale = Math.fround(values.reduce((top, w) => Math.max(top, Math.abs(w)), 0) / 127), inverse = scale ? Math.fround(1 / scale) : 0;
    view.setFloat16(block * 34, scale, true);
    values.forEach((w, i) => { const code = Math.fround(w * inverse); view.setInt8(block * 34 + 2 + i, Math.sign(code) * Math.round(Math.abs(code))); });
  }
  return bytes;
}

const words = new TextDecoder().decode(file.files["phonemizer/lexicon.tsv"]).split("\n").filter(Boolean).map((line) => line.split("\t"));
const files = { ...file.files, ["phonemizer/" + PACKED]: gzipSync(packLexicon(words), { level: 9 }) };
delete files["phonemizer/lexicon.tsv"];
for (const name of readdirSync(tables).filter((name) => name.endsWith(".txt")).sort()) files["phonemizer/" + name] = readFileSync(join(tables, name));
const names = Object.keys(files), offsets = names.reduce((list, name) => [...list, list.at(-1) + files[name].length], [0]);

const infos = [], blobs = [];
let offset = 0, matrices = 0;
for (const [name, { shape, type, data }] of Object.entries(file.tensors)) {
  if (type !== 0) throw new Error("the input is packed already: start from the float32 file");
  const large = shape.length === 2 && name.includes(".blocks.");
  const bytes = large ? eightBit(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  matrices += large;
  infos.push(...string(name), u32(shape.length), ...[...shape].reverse().map(u64), u32(large ? 8 : 0), u64(offset));
  blobs.push(bytes, new Uint8Array(padded(bytes.length) - bytes.length));
  offset += padded(bytes.length);
}
const kept = Object.entries(file.keys).filter(([key]) => !key.startsWith("audiocpp.embedded_files.")).map(([, bytes]) => bytes);
const header = [u32(0x46554747), u32(3), u64(Object.keys(file.tensors).length), u64(kept.length + 3), ...kept,
                ...string("audiocpp.embedded_files.names"), u32(9), u32(8), u64(names.length), ...names.flatMap(string),
                ...string("audiocpp.embedded_files.offsets"), u32(9), u32(10), u64(offsets.length), ...offsets.map(u64),
                ...string("audiocpp.embedded_files.data"), u32(9), u32(0), u64(offsets.at(-1)), ...names.map((name) => files[name]), ...infos];
const size = header.reduce((sum, part) => sum + part.length, 0);
const out = Buffer.concat([...header, new Uint8Array(padded(size) - size), ...blobs]);
writeFileSync(output, out);
console.log(`${output}: ${out.length} bytes, ${gzipSync(out, { level: 9 }).length} gzipped. ${matrices} matrices at 8 bits, ${names.length} files inside.`);
