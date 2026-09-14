"""Alternate original and optimized FlyDSL graph timings on identical inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_conv import SHAPES, benchmark, load_flydsl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("before_after.json"))
    args = parser.parse_args()
    options = argparse.Namespace(device=0, seed=0, pool_slots=1243,
                                 state_layout="channel-first", history="fresh",
                                 rtol=0.016, atol=0.016, warmup=10, repeats=100, trials=5)
    paths = {"original": Path(__file__).with_name("causal_conv1d_baseline.py"),
             "optimized": ROOT / "causal_conv1d_flydsl.py"}
    # Reuse the comparison's validated capture/reset/event methodology. Rename
    # its fixed backend labels throughout the saved report to avoid ambiguity.
    functions = {"sglang": load_flydsl(paths["original"]),
                 "flydsl": load_flydsl(paths["optimized"])}

    def rename(value: object) -> object:
        labels = {"sglang": "original", "flydsl": "optimized"}
        if isinstance(value, dict):
            return {labels.get(key, key): rename(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [rename(item) for item in value]
        if isinstance(value, str):
            return labels.get(value, value)
        return value

    report = dict(method="Five alternating trials; GPU events around graph replay; reset excluded",
                  speedup_definition="original / optimized", options=vars(options),
                  gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  sources={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                           for name, path in paths.items()}, results=[])
    with torch.inference_mode():
        for name, lengths in SHAPES.items():
            row = benchmark(lengths, functions, options)
            report["results"].append(dict(name=name, **rename(row)))
            print(f"{name}: original={row['median_ms']['sglang']*1000:.3f} us "
                  f"optimized={row['median_ms']['flydsl']*1000:.3f} us "
                  f"speedup={row['speedup']:.3f}x", flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
