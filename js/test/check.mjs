import { readFileSync } from "node:fs";
import { readGGUF, TinyVoca, spent } from "../tiny-voca.js";

const [file, reference, runs = "1"] = process.argv.slice(2);
const floats = (path) => new Float32Array(new Uint8Array(readFileSync(path)).buffer);
const sentences = JSON.parse(readFileSync(new URL("sentences.json", import.meta.url), "utf8"));

let started = performance.now();
const voca = new TinyVoca(readGGUF(new Uint8Array(readFileSync(file)).buffer));
console.log(`loaded ${file} in ${(performance.now() - started).toFixed(0)} ms`);

const loadTime = performance.now() - started;
const rate = voca.config.sample_rate, rows = [], apart = [];
let audioSeconds = 0, workSeconds = 0, phonemes = 0;

function pitchOfEach(f0, durations) {
  let t = 0;
  return Array.from(durations, (frames) => {
    let sum = 0;
    for (let k = 0; k < frames; k++) sum += Math.log2(f0[t++]);
    return frames ? sum / frames : NaN;
  });
}
for (const [number, { ids, stress }] of sentences.entries()) {
  for (const voice of voca.voices) {
    const stem = `${reference}/${String(number).padStart(2, "0")}-${voice}`;
    const expected = { durations: JSON.parse(readFileSync(stem + ".json", "utf8")).durations, f0: floats(stem + ".f0.f32"),
                       wave: floats(stem + ".wave.f32") };
    const noise = floats(stem + ".noise.f32");
    let result, best = Infinity;
    for (let run = 0; run < Number(runs); run++) {
      started = performance.now();
      result = voca.synthesize(ids, stress, { voice, noise: noise.length === expected.wave.length ? noise : null });
      best = Math.min(best, (performance.now() - started) / 1000);
    }
    const other = expected.durations.filter((frames, n) => frames !== result.durations[n]).length;
    const same = other === 0 && result.audio.length === expected.wave.length;
    let peak = 0, worst = 0, cents = 0;
    if (same) {
      for (let n = 0; n < result.audio.length; n++) {
        peak = Math.max(peak, Math.abs(expected.wave[n]));
        worst = Math.max(worst, Math.abs(expected.wave[n] - result.audio[n]));
      }
      for (let t = 0; t < result.f0.length; t++) cents = Math.max(cents, Math.abs(1200 * Math.log2(result.f0[t] / expected.f0[t])));
    }
    const ours = pitchOfEach(result.f0, result.durations), theirs = pitchOfEach(expected.f0, expected.durations);
    ours.forEach((pitch, n) => { if (pitch === pitch && theirs[n] === theirs[n]) apart.push(Math.abs(1200 * (pitch - theirs[n]))); });
    phonemes += ids.length;
    const seconds = result.audio.length / rate;
    audioSeconds += seconds;
    workSeconds += best;
    rows.push({ sentence: number, voice, phonemes: ids.length, seconds: seconds.toFixed(2), "durations equal": same,
                "other durations": other, "pitch apart (cents)": same ? cents.toExponential(1) : "-",
                "wave error (% of peak)": same ? (100 * worst / peak).toFixed(3) : "-", "x real time": (seconds / best).toFixed(1) });
  }
}
console.table(rows);
const equal = rows.filter((row) => row["durations equal"]).length;
const errors = rows.filter((row) => row["durations equal"]).map((row) => Number(row["wave error (% of peak)"])).sort((a, b) => a - b);
console.log(`durations equal: ${equal} of ${rows.length}`);
if (errors.length) console.log(`wave error, % of peak: median ${errors[errors.length >> 1]}, largest ${errors[errors.length - 1]}`);
console.log(`speed: ${audioSeconds.toFixed(1)} s of audio in ${workSeconds.toFixed(2)} s = ${(audioSeconds / workSeconds).toFixed(1)} x real time`);
const total = Object.values(spent).reduce((a, b) => a + b, 0);
console.log("where the time goes: " + Object.entries(spent).sort((a, b) => b[1] - a[1])
  .map(([name, ms]) => `${name} ${(100 * ms / total).toFixed(0)}%`).join(", "));
apart.sort((a, b) => a - b);
console.log("summary " + JSON.stringify({
  file, "durations equal": `${equal} of ${rows.length}`,
  "phonemes with another duration": `${rows.reduce((sum, row) => sum + row["other durations"], 0)} of ${phonemes}`,
  "pitch apart, cents (median phoneme)": apart[apart.length >> 1].toFixed(2), "pitch apart, cents (9 in 10 below)": apart[Math.floor(apart.length * 0.9)].toFixed(1),
  "x real time": (audioSeconds / workSeconds).toFixed(1), "load ms": loadTime.toFixed(0) }));
