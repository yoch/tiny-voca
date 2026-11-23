import collections
import json
import subprocess
import sys
import wave
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LISTENING = ROOT / "scripts" / "english_onnx" / "listening.py"


def write_wave(path):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16)


def run_listening(*args, check=True):
    return subprocess.run(
        [sys.executable, str(LISTENING), *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
    )


def test_blind_release_kit_and_response(tmp_path):
    items = [
        {"id": f"{i}-{name}", "text": f"Case {i}", "voice": voice, "group": "short"}
        for i in range(12)
        for voice, name in enumerate(("F2", "M2"))
    ]
    for case in items:
        for name in ("original", "exact-int8"):
            write_wave(tmp_path / f"{name}-{case['id']}.wav")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(items))
    kit = tmp_path / "listening.zip"
    key = tmp_path / "private.json"
    run_listening(
        "--fixture",
        fixture,
        "--wavs",
        tmp_path,
        "--output",
        kit,
        "--key",
        key,
        "exact-int8",
    )
    assignment = json.loads(key.read_text())
    pairs = assignment["pairs"]
    assert len(pairs) == 24
    assert collections.Counter(
        v["voice"] for v in pairs.values() if v["A"] == "exact-int8"
    ) == {0: 6, 1: 6}
    with ZipFile(kit) as bundle:
        assert bundle.testzip() is None
        assert len(bundle.namelist()) == 49
        assert all(
            "original" not in name and "int8" not in name for name in bundle.namelist()
        )
        page = bundle.read("index.html")
        assert b"exact-int8" not in page and b"private.json" not in page
        assert assignment["study"].encode() in page

    response = {"study": assignment["study"]}
    for i, (label, row) in enumerate(pairs.items()):
        side = "A" if row["A"] == "exact-int8" else "B"
        response[label] = side if i == 0 else "same"
        response[f"defect-{label}"] = side if i == 0 else "none"
    ratings = tmp_path / "responses.json"
    ratings.write_text(json.dumps(response))
    result = json.loads(run_listening("--key", key, "--score", ratings).stdout)
    assert result["totals"]["major_issue:exact-int8"] == 1
    assert result["totals"]["preference:same"] == 23
    response["study"] = "unrelated"
    ratings.write_text(json.dumps(response))
    rejected = run_listening("--key", key, "--score", ratings, check=False)
    assert rejected.returncode != 0
    assert "different listening kit" in rejected.stderr


def test_hf_manifest_preserves_the_two_candidate_hashes():
    docs = ROOT / "docs" / "english-onnx"
    release = json.loads((docs / "hf" / "manifest.json").read_text())
    expected = {
        name.removeprefix("./"): value
        for value, name in (
            line.split() for line in (docs / "SHA256SUMS").read_text().splitlines()
        )
    }
    assert set(release["variants"]) == {"exact.onnx", "exact-int8.onnx"}
    assert {name: row["sha256"] for name, row in release["variants"].items()} == {
        name: expected[name] for name in release["variants"]
    }
    assert all(
        row["learned_parameters"] == 4_537_291 for row in release["variants"].values()
    )
