"""Selective acoustic-stack storage experiments using standard ONNX ops."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from build import inspect_graph
from evaluate import add_noise_input, cases, feeds
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


class Samples(CalibrationDataReader):
    def __init__(self):
        self.data = iter(feeds(case) for case in cases("calibration"))

    def get_next(self):
        return next(self.data, None)


def targets(graph):
    initializers = {x.name for x in graph.graph.initializer}
    return [
        node
        for node in graph.graph.node
        if node.op_type == "MatMul"
        and node.name.startswith("/acoustic_stack/")
        and node.input[1] in initializers
    ]


def cast_weights(source, output, weights):
    # Web kernels still see float32. This experiment changes storage precision only.
    graph = onnx.load(source)
    casts = []
    for weight in graph.graph.initializer:
        if weight.name not in weights:
            continue
        name = weight.name
        packed = name + "_fp16"
        weight.CopyFrom(
            numpy_helper.from_array(
                numpy_helper.to_array(weight).astype(np.float16), packed
            )
        )
        casts.append(helper.make_node("Cast", [packed], [name], to=TensorProto.FLOAT))
    old = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(casts + old)
    onnx.save(graph, output)


def static_fixed(release, output):
    """Use the identical calibrated graph with explicit noise for validation."""
    graph = onnx.load(release)
    add_noise_input(graph)
    onnx.save(graph, output)


def quantize(args):
    manifest_path = args.root / "models.json"
    manifest = json.loads(manifest_path.read_text())
    record = dict(manifest[args.source])
    record["graphs"] = {}
    record["storage_transform"] = args.method
    model_name = args.source + "-" + args.method
    for fixed in (False, True):
        suffix = "-fixed" if fixed else ""
        source = args.root / (args.source + suffix + ".onnx")
        output = args.root / (model_name + suffix + ".onnx")
        graph = onnx.load(source)
        matches = targets(graph)
        if not matches:
            raise ValueError("no acoustic projection matched")
        names = [node.name for node in matches]
        if args.method == "int8":
            quantize_dynamic(
                source,
                output,
                nodes_to_quantize=names,
                per_channel=True,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
            )
        elif args.method == "qdq":
            if fixed:
                static_fixed(args.root / (model_name + ".onnx"), output)
            else:
                quantize_static(
                    source,
                    output,
                    Samples(),
                    quant_format=QuantFormat.QDQ,
                    nodes_to_quantize=names,
                    per_channel=True,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul"],
                )
        else:
            cast_weights(source, output, {node.input[1] for node in matches})
        record["graphs"]["fixed" if fixed else "release"] = inspect_graph(output, fixed)
        record["transformed_projections"] = len(matches)
    manifest[model_name] = record
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        model_name, record["learned_parameters"], record["graphs"]["release"]["bytes"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("source")
    parser.add_argument("method", choices=("int8", "qdq", "fp16"))
    quantize(parser.parse_args())
