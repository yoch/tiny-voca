"""The demo plays each sentence as soon as it is made, and skips what it cannot read.

The demo's own JavaScript is run with node: the player with a stand-in for the browser's audio clock, the
worker's speak() with a stand-in for the model. Those tests are skipped when node is not installed. The few
lines of Python the worker runs in the browser are run here as they are.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import types
import unittest

from tiny_voca.config import Config

try:
    import dizione
    from dizione.engine import default_directory
    HAVE_DIZIONE = (default_directory() / "lexicon.tsv").exists()
except ImportError:
    dizione, HAVE_DIZIONE = None, False

PUBLIC = Path(__file__).resolve().parents[1] / "web" / "public"
ENGLISH_ONLY = "tiny-voca only reads French for now."

CLOCK = r"""
const fs = require("fs");
const livePlayer = new Function(fs.readFileSync(process.argv[1], "utf8") + "\nreturn livePlayer;")();
const RATE = 24000;

// In place of the browser's AudioContext: it notes what is started and when, and the test moves its clock.
// "refused": the browser does not let the page make sound. "slow": the sound has not started yet.
function fakeContext(kind = "running") {
  const context = {
    currentTime: 0, state: kind === "running" ? "running" : "suspended", destination: {}, started: [], woken: 0,
    resume() { context.woken += 1; return kind === "refused" ? Promise.reject(new Error("not allowed")) : new Promise(() => {}); },
    createBuffer(channels, length, sampleRate) {
      const data = new Float32Array(length);
      return { length, sampleRate, getChannelData: () => data };
    },
    createBufferSource() {
      const source = { buffer: null, onended: null, to: null, when: null, over: false, cut: false,
        connect(node) { source.to = node; },
        start(when) { source.when = when; context.started.push(source); },
        stop() { if (!source.over) { source.over = true; source.cut = true; if (source.onended) source.onended(); } } };
      return source;
    },
    pass(seconds) {
      context.currentTime += seconds;
      for (const source of context.started) {
        if (!source.over && source.when + source.buffer.length / source.buffer.sampleRate <= context.currentTime) {
          source.over = true;
          if (source.onended) source.onended();
        }
      }
    },
  };
  return context;
}
const tone = (length, level) => Float32Array.from({ length }, (_, i) => level * Math.sin(i / 7));
const starts = (context) => context.started.map((source) => source.when);
const facts = {};
"""

PLAYER = CLOCK + r"""
const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
const later = () => new Promise((resolve) => setTimeout(resolve, 1));

(async () => {
{ // Sentences that are made faster than they are spoken.
  const context = fakeContext();
  context.currentTime = 3.25;
  let changes = 0;
  const player = livePlayer(context, () => { changes += 1; });
  const idle = player.playing;
  player.begin();
  const waiting = player.playing;
  const clips = [tone(24000, 0.1), tone(36000, 0.2), tone(12000, 0.3)];
  player.add(clips[0], RATE); context.pass(0.2);
  player.add(clips[1], RATE); context.pass(0.1);
  player.add(clips[2], RATE);
  const played = player.end();
  const before = changes;
  context.pass(2.4);
  const nearTheEnd = player.playing;
  context.pass(0.4);
  facts.early = { idle, waiting, starts: starts(context), played, nearTheEnd, after: player.playing, told: changes > before,
                  intact: context.started.every((source, i) => same(source.buffer.getChannelData(0), clips[i])),
                  rates: context.started.map((source) => source.buffer.sampleRate),
                  toTheSpeakers: context.started.every((source) => source.to === context.destination) };
}
{ // A sentence that is made after the one before it has ended.
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.add(tone(24000, 0.1), RATE); context.pass(1.5);
  const dry = player.playing;
  player.add(tone(24000, 0.1), RATE);
  player.add(tone(0, 0.1), RATE);
  facts.late = { starts: starts(context), dry };
}
{ // Stop, more sentences of the stopped request, then a new request.
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.add(tone(24000, 0.1), RATE); player.add(tone(24000, 0.1), RATE); context.pass(0.5);
  player.stop();
  const cut = context.started.map((source) => source.cut);
  const stopped = player.playing;
  player.add(tone(24000, 0.1), RATE);
  const ignored = context.started.length;
  const played = player.end();
  context.pass(4);
  player.begin();
  player.add(tone(24000, 0.1), RATE);
  facts.stop = { cut, stopped, ignored, played, again: starts(context).slice(2), playing: player.playing };
}
{ // Stop before the first sentence is made.
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.stop();
  player.add(tone(24000, 0.1), RATE);
  facts.soon = { played: player.end(), started: context.started.length, playing: player.playing };
}
{ // A new request while the last one is still being spoken.
  const context = fakeContext();
  const player = livePlayer(context);
  player.begin();
  player.add(tone(48000, 0.1), RATE); context.pass(0.5);
  player.begin();
  player.add(tone(24000, 0.1), RATE);
  facts.again = { cut: context.started.map((source) => source.cut), starts: starts(context) };
}
{ // The browser refuses the sound; and a request that brought no audio.
  const context = fakeContext("refused");
  const player = livePlayer(context);
  player.begin();
  await later();
  player.add(tone(24000, 0.1), RATE); player.add(tone(24000, 0.1), RATE);
  const played = player.end();
  const quiet = livePlayer(fakeContext());
  quiet.begin();
  facts.off = { woken: context.woken, played, cut: context.started.map((source) => source.cut), playing: player.playing,
                nothing: quiet.end() };
}
{ // The sound is slow to start: a short text is done before the browser's clock runs.
  const context = fakeContext("slow");
  const player = livePlayer(context);
  player.begin();
  await later();
  player.add(tone(24000, 0.1), RATE);
  const played = player.end();
  const waiting = player.playing;
  const cut = context.started.map((source) => source.cut);
  context.state = "running";
  context.pass(1.2);
  facts.slow = { played, waiting, cut, starts: starts(context), after: player.playing };
}
console.log(JSON.stringify(facts));
})();
"""

PAGE = CLOCK + r"""
const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
// The voice buttons as the page has them. The checked one is the voice of a page that nobody has touched.
const voices = [...html.matchAll(/<input type="radio" name="voice" value="(\d+)"( checked)?><span>(\w+)<\/span>/g)]
  .map(([, value, checked, name]) => ({ value, name, checked: Boolean(checked) }));
facts.voices = voices;

// In place of the browser: elements that only hold what the page sets on them, and a worker that notes what it
// is sent. The page's own script runs as it is. `sound` says whether this browser has Web Audio.
function open(sound) {
  const elements = {};
  const element = () => ({ value: "", textContent: "", hidden: false, disabled: false, firstElementChild: { style: {} }, plays: 0, pauses: 0,
    append() {}, focus() {}, addEventListener() {},
    play() { this.plays += 1; if (this.onplay) this.onplay(); return Promise.resolve(); }, pause() { this.pauses += 1; } });
  const document = { getElementById: (id) => (elements[id] = elements[id] || element()), createElement: element,
                     querySelector: () => voices.find((voice) => voice.checked) };
  document.getElementById("result").hidden = true;
  const contexts = [];
  const window = sound ? { AudioContext: function (options) { const context = fakeContext(); context.asked = options; contexts.push(context); return context; } } : {};
  const workers = [];
  function Worker() { this.sent = []; this.postMessage = (message) => this.sent.push(message); workers.push(this); }
  new Function("document", "window", "Worker", "livePlayer", script)(document, window, Worker, livePlayer);
  const say = (data) => workers[0].onmessage({ data });
  say({ type: "ready", sampleRate: RATE });
  elements.text.value = "Some text.";
  elements.text.oninput();
  const look = () => ({ status: elements.status.textContent, button: elements.speak.textContent, disabled: elements.speak.disabled,
                        plays: elements.audio.plays, pauses: elements.audio.pauses, shown: !elements.result.hidden,
                        starts: contexts.length ? starts(contexts[0]) : [], cut: contexts.length ? contexts[0].started.map((source) => source.cut) : [] });
  return { click: () => elements.speak.onclick(), player: elements.audio, sent: workers[0].sent, say, look, context: () => contexts[0] };
}
const sentence = (id, index, total) => ({ type: "sentence", id, index, total, phonemes: "<bos> t <eos>", audio: tone(RATE, 0.1), sampleRate: RATE });
const done = (id, seconds) => ({ type: "done", id, audio: tone(RATE * seconds, 0.1), sampleRate: RATE, totalTime: 500, phonemeTime: 20, modelTime: 400 });

{ // Nothing in the text could be read; then a real failure.
  const page = open(true);
  page.click();
  const asked = page.sent[page.sent.length - 1];
  page.say({ type: "error", id: 1, message: "tiny-voca only reads French for now.", plain: true });
  const plain = page.look();
  page.click();
  page.say({ type: "error", id: 2, message: "the model broke" });
  facts.messages = { asked: { type: asked.type, id: asked.id, text: asked.text, voice: asked.voice }, plain, failure: page.look() };
}
{ // The sentences play as they come; the clip waits in the player.
  const page = open(true);
  page.click();
  const clicked = page.look();
  page.say(sentence(1, 0, 2)); page.say(sentence(1, 1, 2));
  const speaking = page.look();
  page.say(done(1, 2));
  const finished = page.look();
  page.context().pass(3);
  facts.live = { asked: page.context().asked, clicked, speaking, finished, after: page.look() };
}
{ // A browser without Web Audio: the clip plays in the player when it is ready, as it used to.
  const page = open(false);
  page.click();
  page.say(sentence(1, 0, 1));
  const speaking = page.look();
  page.say(done(1, 1));
  facts.old = { speaking, finished: page.look() };
}
{ // Stop while the sentences are made; a new request; the player is started while its sentences play.
  const page = open(true);
  page.click();
  page.say(sentence(1, 0, 3));
  page.click();
  const stopped = page.look();
  page.say(sentence(1, 1, 3)); page.say(sentence(1, 2, 3)); page.say(done(1, 3));
  const finished = page.look();
  page.click();
  page.say(sentence(2, 0, 1)); page.say(done(2, 1));
  const again = page.look();
  page.player.play();
  facts.stop = { stopped, finished, again, player: page.look(), requests: page.sent.filter((message) => message.type === "speak").length };
}
console.log(JSON.stringify(facts));
"""

WORKER = r"""
const fs = require("fs");
const source = fs.readFileSync(process.argv[1], "utf8");
const take = (pattern) => source.match(pattern)[0];
const code = [take(/const ABBREVIATIONS = .*;/), take(/function sentences\(text\) \{[\s\S]*?\n\}/),
              take(/function finished\(text\) \{[\s\S]*?\n\}/), take(/function fit\(sentence\) \{[\s\S]*?\n\}/),
              take(/const PEAK = .*;/), take(/function limitPeak\(samples\) \{[\s\S]*?\n\}/),
              take(/const ENGLISH_ONLY = .*;/), take(/async function speak\(\{[\s\S]*?\n\}/), "return speak;"].join("\n");

// In place of the phonemizer and the model. The phonemizer answers the way the worker's own does: the phonemes of
// a sentence, or "null" when there is nothing in it to pronounce (here "XD" and other scripts). The model gives
// a tone, loud when the sentence says so. Messages are passed on the way the browser does it: what is handed
// over is gone here.
const wave = (text) => Float32Array.from({ length: 400 + 100 * text.length },
                                         (_, i) => (text.includes("loud") ? 1.2 : 0.5) * Math.sin(i / 5 + 1.5));
const order = [];
const messages = [];
const phonemize = (sentence) => (/[A-Za-z]/.test(sentence.replace(/XD/g, ""))
                                 ? JSON.stringify({ text: sentence, ids: [1], stress: [0] }) : "null");
const synthesize = async (phones) => { order.push("made " + phones.text); return wave(phones.text); };
const postMessage = (message, transfer = []) => {
  order.push(message.type === "sentence" ? "sent " + message.phonemes : message.type);
  messages.push(structuredClone(message, { transfer }));
};
const config = { sample_rate: 24000, model: { max_tokens: 512 } };
const speak = new Function("phonemize", "synthesize", "postMessage", "config", code)(phonemize, synthesize, postMessage, config);
const peak = (samples) => samples.reduce((most, value) => Math.max(most, Math.abs(value)), 0);

speak({ id: 7, text: process.argv[2], voice: 0, speed: 1 }).then(() => {
  const heard = messages.filter((message) => message.type === "sentence");
  const last = messages[messages.length - 1];
  const raw = heard.map((message) => wave(message.phonemes));
  const joined = Float32Array.from(raw.flatMap((samples) => Array.from(samples)));
  const scale = 0.99 / peak(joined);
  console.log(JSON.stringify({
    order,
    sentences: heard.map((message, i) => ({ id: message.id, index: message.index, total: message.total, rate: message.sampleRate,
                                            length: message.audio.length, peak: peak(message.audio),
                                            untouched: message.audio.every((value, j) => value === raw[i][j]) })),
    last: last.type !== "done" ? last :
      { type: last.type, id: last.id, rate: last.sampleRate, length: last.audio.length, peak: peak(last.audio),
        farthest: last.audio.reduce((most, value, i) => Math.max(most, Math.abs(value - joined[i] * scale)), 0) },
  }));
});
"""


class Said:
    """What the stand-in phonemizer below gives back, shaped like dizione's result."""

    def __init__(self, words):
        self.ids, self.stress, self.words = [1] + [5] * len(words) + [2], [0] * (len(words) + 2), words

    def __str__(self):
        return " ".join(["<bos>", *self.words, "<eos>"])


def stand_in():
    """In place of dizione: it reads words of plain letters. "XD", other scripts and signs give nothing to
    pronounce, and it says so in words of its own. The word "boom" breaks it."""
    def pieces(text):
        words = [[] if word.lower() == "xd" else [[(word, 0)]] for word in re.findall(r"[A-Za-z]+", text)]
        return [(False, word) for word in words] + [(True, ".")]

    def phonemize(text):
        words = [word[0][0][0] for mark, word in pieces(text) if not mark and word]
        if "boom" in words:
            raise ValueError("the dictionary is broken")
        if not words:
            raise ValueError("no word in it")
        return Said(words)

    return types.SimpleNamespace(pieces=pieces, phonemize=phonemize)


def glue(phonemizer):
    """``phonemize_json`` as the worker defines it for the Python that runs in the browser, with ``phonemizer`` as
    its ``dizione``."""
    source = (PUBLIC / "worker.js").read_text(encoding="utf-8")
    code = re.search(r"^def phonemize_json\(text\):\n(?:(?: {4}.*)?\n)+", source, re.MULTILINE).group(0)
    names = {"json": json, "dizione": phonemizer}
    exec(code, names)
    return names["phonemize_json"]


def node(script, *arguments):
    done = subprocess.run(["node", "-e", script, *arguments], capture_output=True, text=True, encoding="utf-8")
    if done.returncode:
        raise AssertionError(done.stderr)
    return json.loads(done.stdout)


class Times(unittest.TestCase):
    def assertTimes(self, times, expected):
        self.assertEqual(len(times), len(expected))
        for time, wanted in zip(times, expected):
            self.assertAlmostEqual(time, wanted, places=9)


@unittest.skipUnless(shutil.which("node"), "needs node to run the demo's own code")
class LivePlayerTests(Times):
    @classmethod
    def setUpClass(cls):
        cls.facts = node(PLAYER, str(PUBLIC / "player.js"))

    def test_each_sentence_starts_where_the_one_before_it_ends(self):
        early = self.facts["early"]
        # The clock stands at 3.25 s. The first sentence starts 0.05 s from then, and the 1.0 s and 1.5 s of audio
        # that follow leave no gap and do not overlap.
        self.assertTimes(early["starts"], [3.3, 4.3, 5.8])
        self.assertTrue(early["intact"] and early["toTheSpeakers"])
        self.assertEqual(early["rates"], [24000, 24000, 24000])

    def test_it_plays_from_the_request_to_the_end_of_the_last_sentence(self):
        early = self.facts["early"]
        self.assertFalse(early["idle"])
        self.assertTrue(early["waiting"])                               # before the first sentence is made
        self.assertTrue(early["played"])
        self.assertTrue(early["nearTheEnd"])                            # the worker is done, the voice is not
        self.assertFalse(early["after"])
        self.assertTrue(early["told"])                                  # the page hears about it and resets its button

    def test_a_late_sentence_starts_right_away(self):
        # Made at 1.5 s, when the first one ended at 1.05 s: it does not start in the past. Empty audio is left out.
        self.assertTimes(self.facts["late"]["starts"], [0.05, 1.55])
        self.assertTrue(self.facts["late"]["dry"])

    def test_stop_silences_the_request_and_what_it_still_sends(self):
        stop = self.facts["stop"]
        self.assertEqual(stop["cut"], [True, True])
        self.assertFalse(stop["stopped"])
        self.assertEqual(stop["ignored"], 2)
        self.assertTrue(stop["played"])                                 # the page must not start the whole clip instead
        self.assertTimes(stop["again"], [4.55])
        self.assertTrue(stop["playing"])

    def test_stop_before_the_first_sentence_is_a_stop_too(self):
        self.assertEqual(self.facts["soon"], {"played": True, "started": 0, "playing": False})

    def test_a_new_request_cuts_the_one_before_it(self):
        self.assertEqual(self.facts["again"]["cut"], [True, False])
        self.assertTimes(self.facts["again"]["starts"], [0.05, 0.55])    # not after the 2 s that were cut

    def test_the_page_is_told_when_there_is_nothing_to_hear(self):
        off = self.facts["off"]
        self.assertEqual(off["woken"], 1)
        self.assertFalse(off["played"])                                 # the page then plays the whole clip, as it used to
        self.assertEqual(off["cut"], [True, True])                      # and the queue cannot start by itself later
        self.assertFalse(off["playing"])
        self.assertFalse(off["nothing"])

    def test_a_slow_start_is_not_a_refusal(self):
        slow = self.facts["slow"]
        self.assertTrue(slow["played"])                                 # the text was done before the sound was running
        self.assertTrue(slow["waiting"])
        self.assertEqual(slow["cut"], [False])
        self.assertTimes(slow["starts"], [0.05])                        # and it is heard as soon as the clock runs
        self.assertFalse(slow["after"])


@unittest.skipUnless(shutil.which("node"), "needs node to run the demo's own code")
class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker = str(PUBLIC / "worker.js")
        cls.facts = node(WORKER, worker, "One. Two is loud. Three")
        cls.mixed = node(WORKER, worker, "XD. こんにちは. One. XD. Three.")
        cls.unreadable = {text: node(WORKER, worker, text) for text in ("こんにちは、世界。", "XD", "XD. こんにちは. XD", "!!! ... ???")}

    def test_each_sentence_is_sent_before_the_next_one_is_made(self):
        self.assertEqual(self.facts["order"], ["made One.", "sent One.", "made Two is loud.", "sent Two is loud.",
                                               "made Three.", "sent Three.", "done"])

    def test_each_sentence_comes_with_its_audio_below_full_scale(self):
        sentences = self.facts["sentences"]
        self.assertEqual([(s["id"], s["index"], s["total"], s["rate"]) for s in sentences],
                         [(7, 0, 3, 24000), (7, 1, 3, 24000), (7, 2, 3, 24000)])
        self.assertEqual([s["length"] for s in sentences], [800, 1600, 1000])
        self.assertEqual([s["untouched"] for s in sentences], [True, False, True])
        self.assertAlmostEqual(sentences[1]["peak"], 0.99, places=5)     # 1.2 as the model made it

    def test_the_whole_clip_still_comes_at_the_end(self):
        done = self.facts["last"]
        self.assertEqual((done["type"], done["id"], done["rate"], done["length"]), ("done", 7, 24000, 3400))
        self.assertAlmostEqual(done["peak"], 0.99, places=5)
        self.assertLess(done["farthest"], 1e-6)                          # every sentence scaled by the same factor

    def test_a_sentence_with_nothing_to_pronounce_is_skipped(self):
        # Four sentences: "XD. こんにちは." and "XD." have nothing to pronounce, "One." and "Three." are read.
        self.assertEqual(self.mixed["order"], ["made One.", "sent One.", "made Three.", "sent Three.", "done"])
        self.assertEqual([(s["index"], s["total"]) for s in self.mixed["sentences"]], [(1, 4), (3, 4)])
        self.assertEqual((self.mixed["last"]["type"], self.mixed["last"]["length"]), ("done", 800 + 1000))

    def test_a_text_with_nothing_to_pronounce_gets_a_plain_message_and_no_clip(self):
        for text, facts in self.unreadable.items():
            self.assertEqual(facts["order"], ["error"], text)            # no sentence, no empty clip
            self.assertEqual(facts["last"], {"type": "error", "id": 7, "message": ENGLISH_ONLY, "plain": True}, text)


@unittest.skipUnless(shutil.which("node"), "needs node to run the demo's own code")
class PageTests(Times):
    """The page's own script, with stand-ins for the browser and the worker."""

    @classmethod
    def setUpClass(cls):
        cls.facts = node(PAGE, str(PUBLIC / "player.js"), str(PUBLIC / "index.html"))

    def test_the_page_opens_with_the_m2_voice(self):
        # F2 is still there and still the first button. M2 is the one chosen when the page opens, and its value is
        # M2's place among the model's voices.
        self.assertEqual(self.facts["voices"], [{"value": "0", "name": "F1", "checked": False},
                                                {"value": "1", "name": "M1", "checked": True}])
        self.assertEqual(Config().speakers, ("F1", "M1"))
        self.assertEqual(self.facts["messages"]["asked"]["voice"], 1)    # what an untouched page asks the worker for

    def test_a_text_it_cannot_read_gets_the_plain_message(self):
        messages = self.facts["messages"]
        self.assertEqual(messages["asked"], {"type": "speak", "id": 1, "text": "Some text.", "voice": 1})
        plain = messages["plain"]
        self.assertEqual(plain["status"], ENGLISH_ONLY)                  # as it is: no "Something went wrong"
        self.assertEqual((plain["button"], plain["disabled"], plain["plays"]), ("Speak", False, 0))
        failure = messages["failure"]
        self.assertEqual(failure["status"], "Something went wrong: the model broke")
        self.assertEqual((failure["button"], failure["disabled"]), ("Speak", False))

    def test_the_sentences_play_as_they_come_and_the_clip_waits_in_the_player(self):
        live = self.facts["live"]
        self.assertEqual(live["asked"], {"sampleRate": 24000})          # the sound runs at the model's rate
        self.assertEqual((live["clicked"]["button"], live["clicked"]["disabled"]), ("Stop", False))
        self.assertTimes(live["speaking"]["starts"], [0.05, 1.05])
        self.assertEqual(live["speaking"]["status"], "Speaking sentence 2 of 2")
        finished = live["finished"]
        self.assertEqual((finished["status"], finished["shown"], finished["plays"]), ("Done.", True, 0))
        self.assertEqual((finished["button"], finished["cut"]), ("Stop", [False, False]))      # still speaking
        self.assertEqual((live["after"]["button"], live["after"]["disabled"]), ("Speak", False))

    def test_without_web_audio_the_clip_plays_when_it_is_ready(self):
        old = self.facts["old"]
        self.assertEqual((old["speaking"]["starts"], old["speaking"]["button"], old["speaking"]["disabled"]), ([], "Speak", True))
        finished = old["finished"]
        self.assertEqual((finished["status"], finished["plays"], finished["button"], finished["disabled"]), ("Done.", 1, "Speak", False))

    def test_stop_and_the_player_silence_the_sentences(self):
        stop = self.facts["stop"]
        stopped = stop["stopped"]
        self.assertEqual((stopped["button"], stopped["disabled"], stopped["cut"]), ("Speak", True, [True]))    # the clip is still being made
        finished = stop["finished"]
        self.assertEqual(len(finished["starts"]), 1)                     # the later sentences stayed silent
        self.assertEqual((finished["status"], finished["plays"], finished["button"], finished["disabled"]), ("Done.", 0, "Speak", False))
        again = stop["again"]
        self.assertEqual((again["button"], again["pauses"], again["cut"]), ("Stop", 2, [True, False]))         # each request pauses the player
        player = stop["player"]
        self.assertEqual((player["plays"], player["cut"], player["button"]), (1, [True, True], "Speak"))
        self.assertEqual(stop["requests"], 2)


class PhonemizerInTheBrowserTests(unittest.TestCase):
    def test_nothing_to_pronounce_is_null_and_not_an_error(self):
        phonemize_json = glue(stand_in())
        for text in ("XD", "こんにちは、世界。", "!!! ...", "XD :) XD"):
            self.assertEqual(phonemize_json(text), "null", text)
        self.assertEqual(json.loads(phonemize_json("Hello there. XD")),
                         {"ids": [1, 5, 5, 2], "stress": [0, 0, 0, 0], "text": "<bos> Hello there <eos>"})

    def test_another_failure_is_still_a_failure(self):
        with self.assertRaisesRegex(ValueError, "the dictionary is broken"):
            glue(stand_in())("boom goes the text")

    @unittest.skipUnless(HAVE_DIZIONE, "needs dizione and its built data (DIZIONE_DATA)")
    def test_with_dizione_itself(self):
        phonemize_json = glue(dizione)
        for text in ("XD", ":D", "こんにちは、世界。", "你好，世界！", "안녕하세요", "مرحبا بالعالم", "Привет мир.", "!!! ...", "\U0001F642"):
            self.assertEqual(phonemize_json(text), "null", text)
        expected = dizione.phonemize("Hello there.")
        self.assertEqual(json.loads(phonemize_json("Hello there.")),
                         {"ids": list(expected.ids), "stress": list(expected.stress), "text": str(expected)})
        self.assertNotEqual(phonemize_json("Hello there こんにちは my friend."), "null")


if __name__ == "__main__":
    unittest.main()
