"""Assemble the two verified ONNX variants offline into a Hub-ready directory."""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from build import DOCS, inspect_graph, sha

LICENSE_SHA = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def validate_interface(path, manifest):
    info = inspect_graph(path, expected_noise=False)
    expected = manifest["variants"][path.name]
    if (info["sha256"], info["bytes"]) != (expected["sha256"], expected["bytes"]):
        raise ValueError(f"Artifact differs from the verified benchmark: {path}")
    model = onnx.load(path, load_external_data=False)
    if [(op.domain, op.version) for op in model.opset_import] != [
        ("", manifest["export"]["opset"])
    ]:
        raise ValueError(f"Unexpected opset domains: {path}")
    for label, declarations in (
        ("inputs", model.graph.input),
        ("outputs", model.graph.output),
    ):
        observed = {io.name: io.type.tensor_type for io in declarations}
        if observed.keys() != manifest[label].keys():
            raise ValueError(f"Unexpected {label}: {path}")
        for name, value in observed.items():
            declared = manifest[label][name]
            dtype = {
                "int64": onnx.TensorProto.INT64,
                "float32": onnx.TensorProto.FLOAT,
            }[declared["dtype"]]
            dims = value.shape.dim
            if (
                value.elem_type != dtype
                or len(dims) != len(declared["shape"])
                or any(
                    dim.dim_value != expected
                    if isinstance(expected, int)
                    else not dim.dim_param
                    for dim, expected in zip(dims, declared["shape"], strict=True)
                )
            ):
                raise ValueError(f"Unexpected type or shape for {name}: {path}")


def stage(graphs, output):
    templates = DOCS / "hf"
    manifest = json.loads((templates / "manifest.json").read_text(encoding="utf-8"))
    root = DOCS.parents[1]
    if sha(root / "LICENSE") != LICENSE_SHA:
        raise ValueError("License differs from pinned upstream Apache-2.0 license")
    for name in manifest["variants"]:
        validate_interface(graphs / name, manifest)
    output.mkdir()
    assets = [
        templates / name
        for name in ("README.md", "NOTICE", "manifest.json", "infer.py", "sample.json")
    ]
    assets.extend(root / name for name in ("LICENSE", "CITATION.cff"))
    assets.extend(graphs / name for name in manifest["variants"])
    for asset in assets:
        shutil.copyfile(asset, output / asset.name)
    checksums = "".join(
        f"{sha(asset)}  {asset.name}\n" for asset in sorted(output.iterdir())
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage(args.graphs, args.output)
