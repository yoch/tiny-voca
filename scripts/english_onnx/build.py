"""Export standalone English graphs from a pinned public checkpoint."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx
import torch
from torch import nn

from tiny_voca.export import Exportable, export
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize

CHECKPOINT_SHA = "ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70"
BASE_SHA = "fe6d918791a15b82e37465981d7f27a070e46b46635e5a3482fc8dff87a1ae46"
DOCS = Path(__file__).resolve().parents[2] / "docs" / "english-onnx"
EXACT = {
    "base": {},
    "mask": {"omit_mask": True},
    "packed": {"packed": True},
    "scan": {"scan_owners": True},
    "exact": {"omit_mask": True, "packed": True, "scan_owners": True},
    "matmulola": {
        "omit_mask": True,
        "packed": True,
        "scan_owners": True,
        "frame_ola": True,
    },
}
ACOUSTIC = "acoustic_stack"
PROSODY = "prosody_stack"
ENCODER = "encoder"
COMPRESSION = {
    "prune624": ("prune", {ACOUSTIC: 624}),
    "prune576": ("prune", {ACOUSTIC: 576}),
    "prune512": ("prune", {ACOUSTIC: 512}),
    "prune384": ("prune", {ACOUSTIC: 384}),
    "prune160": ("prune", {ACOUSTIC: 160}),
    "weighted512": ("budget", {ACOUSTIC: 512}),
    "solo384first": ("prune", {ACOUSTIC: [384, 640, 640, 640, 640]}),
    "solo384last": ("prune", {ACOUSTIC: [640, 640, 640, 640, 384]}),
    "svd144": ("svd", {ACOUSTIC: 144}),
    "svd128": ("svd", {ACOUSTIC: 128}),
    "svd96": ("svd", {ACOUSTIC: 96}),
    "svd128out": ("svd_out", {ACOUSTIC: 128}),
    "qkv128": ("svd_qkv", {ACOUSTIC: 128}),
    "attention80": ("svd_attention_out", {ACOUSTIC: 80}),
    "prosody576": ("prune", {PROSODY: 576}),
    "encoder576": ("prune", {ENCODER: 576}),
    "combined512": ("prune", {ACOUSTIC: 512, PROSODY: 512, ENCODER: 512}),
    "reconstruct576": ("reconstruct", {ACOUSTIC: 576}),
    "reconstruct512": ("reconstruct", {ACOUSTIC: 512}),
    "reconstruct448": ("reconstruct", {ACOUSTIC: 448}),
    "reconstruct384": ("reconstruct", {ACOUSTIC: 384}),
    "reconstruct160": ("reconstruct", {ACOUSTIC: 160}),
}
STACKS = (ACOUSTIC, PROSODY, ENCODER)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inputs(line, voice, speed=1.0, pitch=0.0, pitch_range=1.0):
    phones = phonemize(line)
    return (
        torch.tensor([phones.ids], dtype=torch.long),
        torch.tensor([phones.stress], dtype=torch.long),
        torch.tensor([voice], dtype=torch.long),
        *(torch.tensor([x], dtype=torch.float32) for x in (speed, pitch, pitch_range)),
    )


def linear(in_features, out_features, weight, bias=None):
    layer = nn.Linear(in_features, out_features, bias=bias is not None)
    layer.weight.copy_(weight)
    if bias is not None:
        layer.bias.copy_(bias)
    return layer


def factor(old, rank):
    out_features, in_features = old.weight.shape
    if (
        not 0 < rank < min(out_features, in_features)
        or rank * (out_features + in_features) >= old.weight.numel()
    ):
        raise ValueError("rank must reduce the physical coefficient count")
    u, s, vh = torch.linalg.svd(old.weight.double(), full_matrices=False)
    scale = s[:rank].sqrt()
    return nn.Sequential(
        linear(in_features, rank, scale[:, None] * vh[:rank]),
        linear(rank, out_features, u[:, :rank] * scale, old.bias),
    )


def prune(block, score, width, statistics=None):
    old_expand, old_project = block.expand, block.project
    if width == old_expand.out_features:
        return
    indices = score.topk(width).indices.sort().values
    projection = old_project.weight[:, indices]
    bias = old_project.bias
    if statistics is not None:
        mean, covariance = statistics
        # Restore the dropped FFN channels' contribution in a single
        # regularized linear projection on the retained activation channels.
        keep = torch.ones(len(score), dtype=torch.bool)
        keep[indices] = False
        auto = covariance[indices][:, indices]
        cross = covariance[indices][:, keep]
        omitted = old_project.weight[:, keep].double().T
        ridge = 0.005 * auto.diag().mean()
        auto.diagonal().add_(ridge)
        correction = torch.linalg.solve(auto, cross @ omitted)
        projection = projection.double() + correction.T
        bias = bias.double() + mean[keep] @ omitted - mean[indices] @ correction
    block.expand = linear(
        old_expand.in_features,
        width,
        old_expand.weight[indices],
        old_expand.bias[indices],
    )
    block.project = linear(width, old_project.out_features, projection, bias)


def budget(scores, width):
    """Allocate an identical group budget in multiples of 32, by proxy energy."""
    minimum = max(32, width - 128)
    widths = [minimum] * len(scores)
    ordered = [s.sort(descending=True).values for s in scores]
    for _ in range(len(scores) * (width - minimum) // 32):
        pool = [i for i, w in enumerate(widths) if w + 32 <= len(ordered[i])]
        owner = max(
            pool, key=lambda i: float(ordered[i][widths[i] : widths[i] + 32].sum())
        )
        widths[owner] += 32
    return widths


def spectrum(weight):
    singular = torch.linalg.svdvals(weight.double())
    energy = singular.square()
    cumulative = energy.cumsum(0) / energy.sum()
    p = singular / singular.sum()
    normalized_rows = torch.nn.functional.normalize(weight.double(), dim=1)
    cosine = (normalized_rows @ normalized_rows.T).abs()
    cosine.fill_diagonal_(0)
    return {
        "shape": list(weight.shape),
        "stable_rank": float(energy.sum() / energy[0]),
        "entropy_rank": float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum())),
        "rank_99pct_energy": int(torch.searchsorted(cumulative, 0.99)) + 1,
        "retained_energy": {
            str(r): float(cumulative[r - 1]) for r in (64, 96, 128, 144)
        },
        "median_max_row_cosine": float(cosine.max(dim=1).values.median()),
        "singular_values": singular.tolist(),
    }


def collect(model, texts, covariance=False):
    total, counts, hooks = {}, Counter(), []
    sums, products = {}, {}

    def hook(name):
        def accumulate(layer, values):
            activations = values[0].reshape(-1, layer.in_features).double()
            total[name] += activations.square().sum(0)
            counts[name] += activations.shape[0]
            if name in products:
                sums[name] += activations.sum(0)
                products[name] += activations.T @ activations

        return accumulate

    for stack in STACKS:
        for i, block in enumerate(getattr(model, stack).blocks):
            key = (stack, i)
            total[key] = torch.zeros(block.project.in_features, dtype=torch.float64)
            if covariance and stack == ACOUSTIC:
                sums[key] = torch.zeros_like(total[key])
                products[key] = torch.zeros(
                    len(total[key]), len(total[key]), dtype=torch.float64
                )
            hooks.append(block.project.register_forward_pre_hook(hook(key)))
    network = Exportable(model).eval()
    noise = torch.zeros(1, model.config.max_frames * model.config.hop)
    try:
        for text in texts:
            for voice in (0, 1):
                network(*inputs(text, voice), noise)
    finally:
        for item in hooks:
            item.remove()
    means = {key: value / counts[key] for key, value in sums.items()}
    centered = {
        key: (means[key], value / counts[key] - torch.outer(means[key], means[key]))
        for key, value in products.items()
    }
    return {key: (value / counts[key]).sqrt() for key, value in total.items()}, centered


def channel_scores(model, activations):
    return {
        (stack, i): activations[stack, i] * block.project.weight.double().norm(dim=0)
        for stack in STACKS
        for i, block in enumerate(getattr(model, stack).blocks)
    }


def analyses(model, activations, scores):
    report = {}
    for stack in STACKS:
        report[stack] = []
        for i, block in enumerate(getattr(model, stack).blocks):
            s = scores[stack, i].sort(descending=True).values
            item = {
                name: spectrum(layer.weight)
                for name, layer in (
                    ("expand", block.expand),
                    ("project", block.project),
                    ("qkv", block.attention.qkv),
                    ("out", block.attention.out),
                )
            }
            item["activation_rms"] = activations[stack, i].tolist()
            item["retained_proxy_energy"] = {
                str(width): float(s[:width].square().sum() / s.square().sum())
                for width in (160, 384, 512, 576, 624)
            }
            report[stack].append(item)
    return report


def configure(model, variant, scores, statistics):
    method, config = COMPRESSION[variant]
    applied = {}
    for stack, spec in config.items():
        blocks = getattr(model, stack).blocks
        widths = spec if isinstance(spec, list) else [spec] * len(blocks)
        if method == "budget":
            widths = budget([scores[stack, i] for i in range(len(blocks))], spec)
        applied[stack] = widths
        for i, (block, width) in enumerate(zip(blocks, widths, strict=True)):
            if method == "svd_qkv":
                block.attention.qkv = factor(block.attention.qkv, width)
            elif method == "svd_attention_out":
                block.attention.out = factor(block.attention.out, width)
            elif method.startswith("svd"):
                block.project = factor(block.project, width)
                if method == "svd":
                    block.expand = factor(block.expand, width)
            else:
                prune(
                    block,
                    scores[stack, i],
                    width,
                    statistics[stack, i] if method == "reconstruct" else None,
                )
    return applied


def rotary_constants(path):
    if sha(path) != BASE_SHA:
        raise ValueError("expected the released English ONNX graph")
    original = onnx.load(path)
    nodes = {node.name: node for node in original.graph.node}
    return {
        f"/{name}/Constant_4": nodes[f"/{name}/Constant_4"].attribute[0].t
        for name in STACKS
    }


def pin_rotary(path, reference):
    """Copy frequency constants from the release to avoid CPU pow() drift."""
    graph = onnx.load(path)
    nodes = {node.name: node for node in graph.graph.node}
    for name, tensor in reference.items():
        target = nodes[name]
        value = target.attribute[0].t
        if (
            target.op_type != "Constant"
            or value.data_type != tensor.data_type
            or value.dims != tensor.dims
        ):
            raise ValueError(f"unexpected rotary frequency node: {name}")
        if not torch.allclose(
            torch.from_numpy(onnx.numpy_helper.to_array(value).copy()),
            torch.from_numpy(onnx.numpy_helper.to_array(tensor).copy()),
            rtol=1e-6,
            atol=0,
        ):
            raise ValueError(f"unexpected rotary frequency values: {name}")
        value.CopyFrom(tensor)
    onnx.save(graph, path)


def learned_macs(model):
    counts = {}
    for name in STACKS:
        counts[name] = sum(
            layer.weight.numel()
            for layer in getattr(model, name).modules()
            if isinstance(layer, (nn.Linear, nn.Conv1d))
        )
    return counts


def inspect_graph(path, expected_noise):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if (
        any(node.domain not in ("", "ai.onnx") for node in model.graph.node)
        or model.functions
    ):
        raise ValueError(f"nonstandard ONNX graph: {path}")
    if {o.name for o in model.graph.output} != {"audio", "durations"}:
        raise ValueError("unexpected outputs")
    if ("noise" in {i.name for i in model.graph.input}) != expected_noise:
        raise ValueError("unexpected noise interface")
    if any(
        i.data_location == onnx.TensorProto.EXTERNAL for i in model.graph.initializer
    ):
        raise ValueError("external weights")
    ops = Counter(n.op_type for n in model.graph.node)
    const = sum(
        onnx.numpy_helper.to_array(attr.t).nbytes
        for node in model.graph.node
        if node.op_type == "Constant"
        for attr in node.attribute
        if attr.HasField("t")
    )
    return {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "operators": dict(ops),
        "constant_tensor_bytes": const,
    }


@torch.no_grad()
def build(args):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if (
        sha(args.checkpoint) != CHECKPOINT_SHA
        or match_frontend(args.checkpoint) != "own"
    ):
        raise ValueError("expected the public English checkpoint and matching dizione")
    reference = rotary_constants(args.public)
    original = load_model(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    scores, statistics = {}, {}
    if set(args.variants) & COMPRESSION.keys():
        texts = (DOCS / "calibration.txt").read_text().splitlines()
        covariance = any(
            COMPRESSION[v][0] == "reconstruct"
            for v in args.variants
            if v in COMPRESSION
        )
        activations, statistics = collect(original, texts, covariance)
        scores = channel_scores(original, activations)
        (args.output / "weights.json").write_text(
            json.dumps(analyses(original, activations, scores)) + "\n"
        )
    meta = args.output / "models.json"
    manifest = json.loads(meta.read_text()) if meta.exists() else {}
    for variant in args.variants:
        model = load_model(args.checkpoint)
        config = (
            configure(model, variant, scores, statistics)
            if variant in COMPRESSION
            else {}
        )
        model.eval()
        options = (
            EXACT[args.structural_base] if variant in COMPRESSION else EXACT[variant]
        )
        record = {
            "learned_parameters": sum(p.numel() for p in model.parameters()),
            "linear_and_depthwise_macs_per_position": learned_macs(model),
            "compression": config,
            "export_options": options,
            "graphs": {},
        }
        for fixed in (False, True):
            filename = variant + ("-fixed" if fixed else "") + ".onnx"
            path = args.output / filename
            export(model, path, external_noise=fixed, **options)
            pin_rotary(path, reference)
            record["graphs"]["fixed" if fixed else "release"] = inspect_graph(
                path, fixed
            )
        manifest[variant] = record
        meta.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            variant,
            record["learned_parameters"],
            record["graphs"]["release"]["bytes"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--public",
        type=Path,
        required=True,
        help="Verified public ONNX graph for the rotary constants",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structural-base", choices=EXACT, default="exact")
    parser.add_argument("variants", choices=(*EXACT, *COMPRESSION), nargs="+")
    build(parser.parse_args())
