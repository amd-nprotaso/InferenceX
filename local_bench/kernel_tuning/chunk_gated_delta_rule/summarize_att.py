"""Normalize an attributed FlyDSL ATT instruction table by wave-iterations."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stats", type=Path)
    p.add_argument("--code-object", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [r for r in csv.DictReader(a.stats.open()) if int(r["CodeObj"]) == a.code_object]
    mfma_hits = sum(int(r["Hitcount"]) for r in rows if r["Instruction"].startswith("v_mfma"))
    if not mfma_hits:
        raise RuntimeError("No MFMA: reject this decode and retry capture")
    wave_iterations = mfma_hits / 8  # Four w@h MFMAs plus two per K half update.
    buckets = defaultdict(lambda: dict(latency=0, hits=0))
    for r in rows:
        op = r["Instruction"]
        if op.startswith("s_waitcnt"):
            key = "vmcnt wait" if "vmcnt" in op else "lgkmcnt wait" if "lgkmcnt" in op else "other wait"
        elif op.startswith("s_barrier"):
            key = "barrier"
        elif op.startswith("v_mfma"):
            key = "MFMA"
        elif op.startswith("ds_"):
            key = "LDS"
        elif op.startswith(("buffer_load", "global_load", "flat_load")):
            key = "global load"
        elif op.startswith(("buffer_store", "global_store", "flat_store")):
            key = "global store"
        elif op.startswith("v_"):
            key = "VALU"
        else:
            key = "other"
        buckets[key]["latency"] += int(r["Latency"])
        buckets[key]["hits"] += int(r["Hitcount"])
    total = sum(v["latency"] for v in buckets.values())
    result = dict(
        code_object=a.code_object,
        wave_iterations=wave_iterations,
        methodology="Sum of attributed instruction latency / wave-iterations; includes prologue and epilogue; not median wall-clock loop duration.",
        buckets={
            k: dict(
                cycles=v["latency"] / wave_iterations,
                issues=v["hits"] / wave_iterations,
                percent=100 * v["latency"] / total,
            )
            for k, v in buckets.items()
        },
    )
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
