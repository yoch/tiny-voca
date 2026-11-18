"""Export the inference model from a full training checkpoint (last.pt or step-*.pt).

The export holds the speech model only: no optimizer state, no auxiliary mel
head and no discriminators.
"""

import argparse
from pathlib import Path
import torch

from tiny_voca.config import CHECKPOINT_FORMAT, Config
from tiny_voca.model import TinyVoca
from tiny_voca.text import frontend_fingerprint
from tiny_voca.train import export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != CHECKPOINT_FORMAT or state["frontend"] != frontend_fingerprint():
        raise ValueError("checkpoint format or front end does not match this code")
    model = TinyVoca(Config.from_dict(state["config"]))
    model.load_state_dict({k.removeprefix("model."): v for k, v in state["system"].items() if k.startswith("model.")}, strict=True)
    export_model(model, args.output, state["step"], state["annotation"], frontend_rules=state.get("frontend_rules"))
    print(f"exported step {state['step']} with {sum(p.numel() for p in model.parameters()):,} parameters to {args.output}")


if __name__ == "__main__":
    main()
