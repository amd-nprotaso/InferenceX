"""Run paired device-dispatch experiments; keep every trace and exact command."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--launches", type=int, default=60)
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent / "results_flydsl")
    a = p.parse_args()
    root = a.outdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [("triton", 3), ("permuted", 3), ("triton", 4), ("permuted", 4), ("flydsl", 3)]
    for trial in range(a.trials):
        for backend, stages in configs[:: (-1 if trial % 2 else 1)]:
            out = root / f"{backend}_s{stages}_trial{trial}"
            cmd = [
                "rocprofv3",
                "--kernel-trace",
                "--output-format",
                "csv",
                "-d",
                str(out),
                "-o",
                "gdn",
                "--",
                sys.executable,
                str(Path(__file__).with_name("profile_flydsl.py").resolve()),
                "--backend",
                backend,
                "--stages",
                str(stages),
                "--launches",
                str(a.launches),
            ]
            env = dict(os.environ, HIP_VISIBLE_DEVICES="0")
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            out.mkdir(parents=True, exist_ok=True)
            (out / "command.json").write_text(json.dumps(cmd, indent=2))
            (out / "worker.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"{backend} failed; see {out}/worker.log")
            trace = next(out.glob("*kernel_trace.csv"))
            target = "kernel_0" if backend == "flydsl" else "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
            dispatches = [r for r in csv.DictReader(trace.open()) if target in r["Kernel_Name"]]
            timed = dispatches[-a.launches :]
            assert len(timed) == a.launches
            d = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6 for r in timed]
            row = dict(
                trial=trial,
                backend=backend,
                stages=stages,
                n=len(d),
                median_ms=statistics.median(d),
                std_ms=statistics.stdev(d),
                vgpr=int(timed[0]["VGPR_Count"]),
                scratch=int(timed[0]["Scratch_Size"]),
                trace=str(trace.relative_to(root)),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            (root / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
