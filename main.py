#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass, fields
import os
import torch
import torch.nn as nn


@dataclass
class TRMConfig:
    input_dim: int = 81 * 10
    hidden_dim: int = 512
    output_dim: int = 81 * 9
    L_layers: int = 3
    H_cycles: int = 4
    L_cycles: int = 8
    dropout: float = 0.1


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TinyRecursiveModel(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.latent_layers = nn.ModuleList(
            [ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.L_layers)]
        )
        self.output_layers = nn.ModuleList([ResidualBlock(config.hidden_dim, config.dropout) for _ in range(2)])
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.latent_gate = nn.Parameter(torch.ones(1))
        self.output_gate = nn.Parameter(torch.ones(1))

    def latent_recursion(self, x, y, z):
        combined = x + y + z
        for layer in self.latent_layers:
            combined = combined + self.latent_gate * layer(combined)
        return combined

    def output_refinement(self, y, z):
        combined = y + z
        for layer in self.output_layers:
            combined = combined + self.output_gate * layer(combined)
        return combined

    def forward(self, x):
        x_embedded = self.input_proj(x)
        y = torch.zeros_like(x_embedded)
        z = torch.zeros_like(x_embedded)
        for _h in range(self.config.H_cycles):
            for _l in range(self.config.L_cycles):
                z = self.latent_recursion(x_embedded, y, z)
            y = self.output_refinement(y, z)
        return self.output_proj(y)


def load_checkpoint(path, unsafe=False):
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        msg = str(e)
        try:
            import re

            missing = re.findall(r"__main__\.([A-Za-z_]\w*)", msg)
            if missing and hasattr(torch.serialization, "safe_globals"):
                objs = [globals().get(name) for name in set(missing) if globals().get(name) is not None]
                if objs:
                    try:
                        with torch.serialization.safe_globals(objs):
                            obj = torch.load(path, map_location="cpu")
                            return obj
                    except Exception:
                        obj = None
        except Exception:
            obj = None

        if obj is None:
            if ("weights_only" in msg or "WeightsUnpickler" in msg) and unsafe:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            else:
                raise
    return obj


def build_model_from_checkpoint(obj):
    if isinstance(obj, nn.Module):
        return obj

    state_dict = None
    cfg = None
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            state_dict = obj["model_state_dict"]
        elif "state_dict" in obj:
            state_dict = obj["state_dict"]
        else:
            if any(k.startswith("input_proj") or k.startswith("output_proj") for k in obj.keys()):
                state_dict = obj

        if "config" in obj:
            raw = obj["config"]
            if isinstance(raw, dict):
                try:
                    cfg = TRMConfig(**raw)
                except Exception:
                    cfg = TRMConfig()
            else:
                try:
                    names = [f.name for f in fields(TRMConfig)]
                    kw = {n: getattr(raw, n) for n in names if hasattr(raw, n)}
                    cfg = TRMConfig(**kw)
                except Exception:
                    cfg = TRMConfig()

    if state_dict is None:
        return None

    if cfg is None:
        for k, v in state_dict.items():
            if k.endswith("input_proj.weight"):
                hidden_dim, input_dim = v.shape
                cfg = TRMConfig(input_dim=int(input_dim), hidden_dim=int(hidden_dim), output_dim=81 * 9)
                break
        if cfg is None:
            cfg = TRMConfig()

    model = TinyRecursiveModel(cfg)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print("Warning: failed to load state dict strictly:", e, file=sys.stderr)
    model.eval()
    return model


def encode_puzzle_to_input(puzzle_digits):
    import numpy as np

    arr = np.zeros((81, 10), dtype=np.float32)
    for i, d in enumerate(puzzle_digits):
        if 0 <= d <= 9:
            arr[i, d] = 1.0
    flat = arr.flatten()
    return torch.from_numpy(flat).unsqueeze(0)


def render_solution(digits):
    one_line = "".join(str(d) if d != 0 else "." for d in digits)
    grid_lines = []
    for r in range(9):
        row = digits[r * 9:(r + 1) * 9]
        grid_lines.append("".join(str(d) if d != 0 else "." for d in row))
    return one_line + "\n\n" + "\n".join(grid_lines)


def main():
    p = argparse.ArgumentParser(description="Run production TRM model on a puzzle file")
    p.add_argument("-i", "--input", required=True, help="puzzle text file (81 digits, 0 for empty)")
    p.add_argument("-o", "--output", required=True, help="output text file for solved puzzle")
    p.add_argument("-m", "--model", default="checkpoints/trm_sudoku_production.pt", help="production model path")
    p.add_argument("--unsafe-load", action="store_true", help="allow unsafe torch.load fallback")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print("Input file not found:", args.input, file=sys.stderr)
        sys.exit(2)

    s = open(args.input, "r", encoding="utf-8").read()
    digits = [int(ch) for ch in s if ch.isdigit()]
    if len(digits) < 81:
        digits = digits + [0] * (81 - len(digits))
    digits = digits[:81]

    if not os.path.exists(args.model):
        print("Model file not found:", args.model, file=sys.stderr)
        sys.exit(3)

    obj = load_checkpoint(args.model, unsafe=args.unsafe_load)
    model = build_model_from_checkpoint(obj)
    if model is None:
        print("Could not construct model from checkpoint.", file=sys.stderr)
        sys.exit(4)

    inp = encode_puzzle_to_input(digits)
    with torch.no_grad():
        out = model(inp)

    t = out.squeeze(0)
    if t.numel() == 81 * 9:
        t = t.view(81, 9)
        preds = torch.argmax(t, dim=-1).cpu().tolist()
        digits_out = [p + 1 for p in preds]
    else:
        flat = t.view(-1)
        if flat.numel() >= 81:
            vals = flat[:81].cpu().tolist()
            digits_out = [int(round(v)) if float(v).is_integer() or abs(v) > 0.5 else 0 for v in vals]
            digits_out = [min(max(d, 0), 9) for d in digits_out]
        else:
            print("Unexpected model output shape", t.shape, file=sys.stderr)
            sys.exit(5)

    out_text = render_solution(digits_out)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)

    print("Wrote solved puzzle to", args.output)


if __name__ == "__main__":
    main()
