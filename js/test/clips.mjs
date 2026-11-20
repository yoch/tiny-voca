import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { readGGUF, TinyVoca } from "../tiny-voca.js";

const [file, folder, label, ...picked] = process.argv.slice(2);
const sentences = JSON.parse(readFileSync(new URL("sentences.json", import.meta.url), "utf8"));
const voca = new TinyVoca(readGGUF(new Uint8Array(readFileSync(file)).buffer));
mkdirSync(folder, { recursive: true });

function wav(audio, rate) {
  const loudest = audio.reduce((top, sample) => Math.max(top, Math.abs(sample)), 0), gain = loudest > 0.99 ? 0.99 / loudest : 1;
  const bytes = Buffer.alloc(44 + 2 * audio.length);
  bytes.write("RIFF", 0); bytes.writeUInt32LE(36 + 2 * audio.length, 4); bytes.write("WAVEfmt ", 8);
  bytes.writeUInt32LE(16, 16); bytes.writeUInt16LE(1, 20); bytes.writeUInt16LE(1, 22); bytes.writeUInt32LE(rate, 24);
  bytes.writeUInt32LE(2 * rate, 28); bytes.writeUInt16LE(2, 32); bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36); bytes.writeUInt32LE(2 * audio.length, 40);
  audio.forEach((sample, n) => bytes.writeInt16LE(Math.round(sample * gain * 32767), 44 + 2 * n));
  return bytes;
}

for (const number of picked.map(Number)) {
  for (const voice of voca.voices) {
    const { ids, stress } = sentences[number];
    let audio;
    try {
      ({ audio } = voca.synthesize(ids, stress, { voice }));
    } catch (error) {
      console.log(`sentence ${number} ${voice} ${label}: ${error.message}`);
      continue;
    }
    const name = `s${number}-${voice}-${label}.wav`;
    writeFileSync(`${folder}/${name}`, wav(audio, voca.config.sample_rate));
    console.log(`${name}: ${(audio.length / voca.config.sample_rate).toFixed(2)} s`);
  }
}
