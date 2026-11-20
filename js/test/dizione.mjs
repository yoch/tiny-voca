import { readFileSync, writeFileSync } from "node:fs";
import { readGGUF } from "../tiny-voca.js";
import { fromGGUF } from "../dizione/engine.js";

const [kind, file, texts, output] = process.argv.slice(2);
if (!output) {
  console.error("usage: node dizione.mjs ids|units|words model.gguf texts.txt out.txt");
  process.exit(1);
}
const started = performance.now();
const phonemizer = await fromGGUF(readGGUF(new Uint8Array(readFileSync(file)).buffer));
const loaded = performance.now();
const show = (phones) => (phones ? phones.map(([phone, stress]) => `${phone}/${stress}`).join(" ") : "-");
const answer = {
  ids(text) {
    const result = phonemizer.phonemize(text);
    return result.ids.join(" ") + "\t" + result.stress.join(" ");
  },
  units(text) {
    return phonemizer.analyse(text).flatMap(([kind, value]) => (kind === "punct" ? ["P:" + value]
      : value.map((unit) => `U:${unit.word}|${unit.kind}|${Number(unit.attach)}|${unit.source}|${show(unit.phones)}`))).join("\x1f");
  },
  words(word) {
    const [phones, source] = phonemizer.word(word);
    return `${source}\t${show(phones)}`;
  },
}[kind];
const lines = readFileSync(texts, "utf8").split("\n");
if (lines.at(-1) === "") lines.pop();
const out = lines.map((line) => {
  try {
    return answer(line);
  } catch (error) {
    return `ERROR ${error.name}`;
  }
});
writeFileSync(output, out.join("\n") + "\n");
console.log(`${kind}: ${lines.length} lines, load ${(loaded - started).toFixed(0)} ms, ${(lines.length / (performance.now() - loaded) * 1000).toFixed(0)} lines/s`);
