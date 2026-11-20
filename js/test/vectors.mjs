import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { readGGUF } from "../tiny-voca.js";
import { fromGGUF } from "../dizione/engine.js";

const [file, folder, show = 10] = process.argv.slice(2);
if (!folder) {
  console.error("usage: node vectors.mjs model.gguf FOLDER [how many differing texts to show]");
  process.exit(1);
}
const phonemizer = await fromGGUF(readGGUF(new Uint8Array(readFileSync(file)).buffer));
const hex = (ids) => ids.map((id) => id.toString(16).padStart(2, "0")).join("");

function read(text) {
  phonemizer.cache.clear();
  try {
    const result = phonemizer.phonemize(text);
    return [hex(result.ids), result.stress.join("")];
  } catch (error) {
    if (error.name === "ValueError") return ["-", "-"];
    return [`${error.name}: ${error.message}`, ""];
  }
}

let texts = 0, differ = 0;
for (const name of readdirSync(folder).filter((name) => name.endsWith(".tsv")).sort()) {
  const lines = readFileSync(join(folder, name), "utf8").split("\n");
  let count = 0, wrong = 0;
  for (const line of lines.slice(lines.indexOf("") + 1)) {
    if (!line) continue;
    const [text, ids, stress] = line.split("\t"), [gotIds, gotStress] = read(text);
    count++;
    if (gotIds === ids && gotStress === stress) continue;
    if (differ + wrong++ < show) console.log(`${name}: ${JSON.stringify(text)}\n  expected ${ids} ${stress}\n  got      ${gotIds} ${gotStress}`);
  }
  console.log(`${name}: ${count} texts, ${wrong} differ`);
  texts += count;
  differ += wrong;
}
console.log(`all: ${texts} texts, ${differ} differ`);
process.exit(differ ? 1 : 0);
