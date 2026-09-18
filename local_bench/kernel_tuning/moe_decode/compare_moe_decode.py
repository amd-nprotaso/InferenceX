#!/usr/bin/env python3
"""A/B the FlyDSL fused MoE front-end against what SGLang runs today.

Shapes are Qwen3.5-397B-A17B MXFP4 at **TP=4**, per rank:

    model_dim (hidden) 4096
    num_experts        512
    topk               10
    inter_dim          256      (moe_intermediate_size 1024 / TP4)
    activation         SiLU, MXFP4 weights, bf16 activations quantised inline

Token counts are decode-side: with EAGLE MTP the verify step is
``bs x speculative-num-draft-tokens`` rows of gating input, so CONC=4 with
``--speculative-num-draft-tokens 4`` is 16 tokens.  The default sweep covers
CONC 1/2/4/8 plus one prefill-ish point.

Arms
----
``sglang``          what the server runs today: aiter ``topk_softmax`` router,
                    opus 2-kernel sort, tuned CSV as shipped (``reduce``
                    stage-2 epilogue at token=16, so ``moe_reduction`` runs).
``sglang+atomic``   change #1 only: same router/sort, tuned CSV row swapped to
                    the ``atomic`` stage-2 epilogue, which deletes
                    ``moe_reduction_kernel_0``.
``flydsl``          change #2 only: FlyDSL fused router+sort in one launch.
``flydsl+atomic``   changes #1 and #2 together.

Each arm runs in its own subprocess, because the arms differ by environment
(``AITER_CONFIG_FMOE``, ``SGLANG_FLYDSL_MOE_DECODE``) which aiter reads at
import time, and because a fresh process keeps JIT state from leaking between
arms.  Timing is torch-profiler device time, so kernels are attributed
individually and the launch count per step is exact.

Usage
-----
    python3 compare_moe_decode.py                        # default sweep
    python3 compare_moe_decode.py --tokens 16            # one shape
    python3 compare_moe_decode.py --arms sglang,flydsl+atomic
    python3 compare_moe_decode.py --json out.json
"""

import argparse
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Qwen3.5-397B-A17B MXFP4, TP=4, per rank.
MODEL_DIM = 4096
INTER_DIM = 256
NUM_EXPERTS = 512
TOPK = 10
NUM_LAYERS = 60

ATOMIC_CSV = os.path.join(HERE, "qwen3_5_397b_fp4_tuned_fmoe.atomic.csv")

ARMS = {
    "sglang": dict(flydsl=False, atomic=False),
    "sglang+atomic": dict(flydsl=False, atomic=True),
    "flydsl": dict(flydsl=True, atomic=False),
    "flydsl+atomic": dict(flydsl=True, atomic=True),
}

# Classification of kernel names into the pipeline stages the report groups by.
# Order matters: first match wins.
_STAGES = [
    ("router", ("topkGatingSoftmax", "topk_softmax")),
    ("route+sort", ("moe_sorting_oneshot_fused_kernel",)),
    ("sort", ("moe_sorting", "moe_sorting_entry", "mxfp4_moe_sort")),
    ("quant+scatter", ("fused_mx_quant_moe_sort", "fused_dynamic_mxfp4")),
    ("stage1", ("mfma_moe1", "moe1_", "flydsl_moe1")),
    ("stage2", ("gemm2_a4w4", "mfma_moe2", "moe2_", "flydsl_moe2")),
    ("reduce", ("moe_reduction", "moe_reduce")),
]


def classify(name):
    for stage, needles in _STAGES:
        if any(n in name for n in needles):
            return stage
    return "other"


# ---------------------------------------------------------------------------
# Worker: measure one (arm, shape) in this process
# ---------------------------------------------------------------------------
def worker(tokens, iters):
    import torch

    import aiter
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    sys.path.insert(0, HERE)
    if os.environ.get("SGLANG_FLYDSL_MOE_DECODE") == "1":
        import moe_decode_patch

        moe_decode_patch.install()

    import aiter.fused_moe as fm

    qtype = aiter.QuantType.per_1x32
    dev = "cuda"
    torch.manual_seed(0)

    x = torch.randn((tokens, MODEL_DIM), dtype=dtypes.bf16, device=dev)
    logits = torch.randn((tokens, NUM_EXPERTS), dtype=dtypes.bf16, device=dev)

    torch_quant = aiter.get_torch_quant(qtype)
    w1 = torch.randn((NUM_EXPERTS, INTER_DIM * 2, MODEL_DIM), dtype=dtypes.bf16, device=dev)
    w2 = torch.randn((NUM_EXPERTS, MODEL_DIM, INTER_DIM), dtype=dtypes.bf16, device=dev)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    w1_s = shuffle_weight(w1_qt.view(NUM_EXPERTS, INTER_DIM * 2, MODEL_DIM // 2), layout=(16, 16))
    w2_s = shuffle_weight(w2_qt.view(NUM_EXPERTS, MODEL_DIM, INTER_DIM // 2), layout=(16, 16))
    del w1_qt, w2_qt
    w1_scale = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale = fp4_utils.e8m0_shuffle(w2_scale)
    torch.cuda.empty_cache()

    kw = dict(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=qtype,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=False,
    )

    def step():
        # Mirrors SGLang: select_experts (aiter router) then fused_moe. With
        # the patch installed the router call stashes the logits instead of
        # launching, and the sort consumes them.
        tw, ti = fm.fused_topk(x, logits, TOPK, True)
        return fm.fused_moe(x, w1_s, w2_s, tw, ti, **kw)

    for _ in range(10):
        step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()

    per = collections.defaultdict(float)
    cnt = collections.defaultdict(float)
    for ev in prof.events():
        if str(ev.device_type) != "DeviceType.CUDA":
            continue
        d = getattr(ev, "self_device_time_total", 0) or 0
        if d:
            per[ev.key] += d / iters
            cnt[ev.key] += 1.0 / iters

    kernels = []
    for name, us in sorted(per.items(), key=lambda kv: -kv[1]):
        if us < 0.05:
            continue
        kernels.append(dict(name=name, us=us, launches=cnt[name], stage=classify(name)))

    return dict(tokens=tokens, kernels=kernels)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fmoe_chain(replacement):
    """AITER_CONFIG_FMOE value that swaps the qwen3.5 CSV for `replacement`."""
    sys.path.insert(0, HERE)
    from fmoe_chain import chain

    return chain(replacement)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def merge_repeats(runs):
    """Per-kernel median across repeated measurements of the same (arm, shape).

    stage1 alone varies by ~2 us run to run, which is larger than the effect
    being measured, so a single profile is not enough to rank arms.
    """
    by_name = collections.defaultdict(list)
    launches = {}
    stages = {}
    for r in runs:
        for k in r["kernels"]:
            by_name[k["name"]].append(k["us"])
            launches[k["name"]] = k["launches"]
            stages[k["name"]] = k["stage"]
    kernels = [
        dict(name=n, us=median(v), launches=launches[n], stage=stages[n], samples=len(v), spread=max(v) - min(v))
        for n, v in by_name.items()
    ]
    kernels.sort(key=lambda k: -k["us"])
    return dict(tokens=runs[0]["tokens"], kernels=kernels)


def run_arm(arm, tokens, iters, device):
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = str(device)
    cfg = ARMS[arm]
    if cfg["flydsl"]:
        env["SGLANG_FLYDSL_MOE_DECODE"] = "1"
        # Compile the fused kernel for exactly this shape. In the server this
        # would be the CUDA-graph capacity, not the live token count.
        env["SGLANG_FLYDSL_MOE_DECODE_MAX_TOKENS"] = str(max(8, tokens))
    else:
        env.pop("SGLANG_FLYDSL_MOE_DECODE", None)
    if cfg["atomic"]:
        if not os.path.exists(ATOMIC_CSV):
            raise SystemExit(f"missing {ATOMIC_CSV}; regenerate it (see README)")
        env["AITER_CONFIG_FMOE"] = fmoe_chain(ATOMIC_CSV)
    else:
        env.pop("AITER_CONFIG_FMOE", None)

    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--tokens", str(tokens), "--iters", str(iters)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    marker = "@@RESULT@@"
    for line in out.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    sys.stderr.write(out.stdout[-4000:])
    sys.stderr.write(out.stderr[-4000:])
    raise SystemExit(f"arm {arm!r} at tokens={tokens} produced no result")


def report(results, arms, token_list, per_layer_us=None):
    base = arms[0]
    for tokens in token_list:
        print(f"\n{'=' * 100}")
        print(
            f"Qwen3.5-397B-A17B MXFP4 TP=4  |  tokens={tokens}  rows={tokens * TOPK}  "
            f"model_dim={MODEL_DIM} inter_dim={INTER_DIM} E={NUM_EXPERTS} topk={TOPK}"
        )
        print("=" * 100)

        # Per-kernel detail for every arm.
        for arm in arms:
            r = results[(arm, tokens)]
            total = sum(k["us"] for k in r["kernels"])
            launches = sum(k["launches"] for k in r["kernels"])
            print(f"\n  -- {arm}  ({total:.2f} us, {launches:.1f} launches)")
            for k in r["kernels"]:
                spread = f" +-{k['spread'] / 2:.2f}" if k.get("samples", 1) > 1 else ""
                print(f"     {k['us']:8.2f}{spread} us  x{k['launches']:4.1f}  [{k['stage']:<13}] {k['name'][:64]}")

        # Stage roll-up across arms.
        stages = []
        for arm in arms:
            for k in results[(arm, tokens)]["kernels"]:
                if k["stage"] not in stages:
                    stages.append(k["stage"])
        order = ["router", "route+sort", "sort", "quant+scatter", "stage1", "stage2", "reduce", "other"]
        stages.sort(key=lambda s: order.index(s) if s in order else len(order))

        w = max(14, max(len(a) for a in arms) + 2)
        print("\n  stage roll-up (us per layer per rank)")
        print("     " + "stage".ljust(15) + "".join(a.rjust(w) for a in arms))
        for st in stages:
            row = "     " + st.ljust(15)
            for arm in arms:
                v = sum(k["us"] for k in results[(arm, tokens)]["kernels"] if k["stage"] == st)
                row += (f"{v:.2f}" if v else "-").rjust(w)
            print(row)
        # The two GEMMs do identical work in every arm and carry all the
        # run-to-run noise; the front-end subtotal is what these changes move.
        row = "     " + "FRONT-END".ljust(15)
        for arm in arms:
            v = sum(k["us"] for k in results[(arm, tokens)]["kernels"] if k["stage"] not in ("stage1", "stage2"))
            row += f"{v:.2f}".rjust(w)
        print(row)
        fe_base = sum(
            k["us"] for k in results[(base, tokens)]["kernels"] if k["stage"] not in ("stage1", "stage2")
        )
        row = "     " + "  vs base".ljust(15)
        for arm in arms:
            v = sum(k["us"] for k in results[(arm, tokens)]["kernels"] if k["stage"] not in ("stage1", "stage2"))
            row += (f"{v - fe_base:+.2f}" if arm != base else "--").rjust(w)
        print(row)
        row = "     " + "TOTAL".ljust(15)
        for arm in arms:
            row += f"{sum(k['us'] for k in results[(arm, tokens)]['kernels']):.2f}".rjust(w)
        print(row)
        row = "     " + "launches".ljust(15)
        for arm in arms:
            row += f"{sum(k['launches'] for k in results[(arm, tokens)]['kernels']):.1f}".rjust(w)
        print(row)
        base_total = sum(k["us"] for k in results[(base, tokens)]["kernels"])
        row = "     " + f"vs {base}".ljust(15)
        for arm in arms:
            t = sum(k["us"] for k in results[(arm, tokens)]["kernels"])
            row += (f"{t - base_total:+.2f}" if arm != base else "--").rjust(w)
        print(row)

    # Whole-model projection at the production decode shape.
    print(f"\n{'=' * 100}")
    print(f"projection over {NUM_LAYERS} layers (one rank, one decode/verify step)")
    print("=" * 100)
    w = max(14, max(len(a) for a in arms) + 2)
    print("  " + "tokens".ljust(10) + "".join(a.rjust(w) for a in arms))
    for tokens in token_list:
        row = "  " + str(tokens).ljust(10)
        for arm in arms:
            t = sum(k["us"] for k in results[(arm, tokens)]["kernels"])
            row += f"{t * NUM_LAYERS / 1000:.3f} ms".rjust(w)
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--tokens",
        default="4,8,16,32",
        help="comma-separated token counts (decode: bs x draft-tokens; default 4,8,16,32 = CONC 1/2/4/8 at MTP-4)",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=3, help="profile runs per (arm, shape); per-kernel median is reported")
    ap.add_argument("--arms", default="sglang,sglang+atomic,flydsl,flydsl+atomic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write raw results here")
    a = ap.parse_args()

    if a.worker:
        tokens = int(a.tokens)
        print("@@RESULT@@" + json.dumps(worker(tokens, a.iters)))
        return

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    token_list = [int(t) for t in a.tokens.split(",")]

    # Interleave repeats across arms rather than finishing one arm at a time.
    # Arms run back to back for minutes and the GPU clocks drift down as it
    # heats: running all repeats of arm A before arm B charges that drift
    # entirely to B. Measured that way stage1 -- byte-for-byte the same kernel
    # on the same inputs in every arm -- climbed monotonically by 3 us from the
    # first arm to the last, which is larger than the effect under test.
    raw = collections.defaultdict(list)
    for rep in range(a.repeat):
        for tokens in token_list:
            for arm in arms:
                print(f"... rep {rep + 1}/{a.repeat} {arm} tokens={tokens}", file=sys.stderr, flush=True)
                raw[(arm, tokens)].append(run_arm(arm, tokens, a.iters, a.device))
    results = {k: merge_repeats(v) for k, v in raw.items()}

    report(results, arms, token_list)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({f"{arm}|{tok}": v for (arm, tok), v in results.items()}, f, indent=2)
        print(f"\nraw results -> {a.json}")


if __name__ == "__main__":
    main()
