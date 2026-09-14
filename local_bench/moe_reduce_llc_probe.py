#!/usr/bin/env python3
"""Does the MoE topk-reduce go faster when its working set fits the 256 MB LLC?

`moe_reduction_kernel_0` reads [T, topk, model_dim] and writes [T, model_dim].
At the prefill shape that read set is 2.68 GB -> HBM. If the same kernel hits a
much higher effective bandwidth once the working set drops under the 256 MB L3,
then splitting stage2+reduce into output-column panels (so each panel's partial
stays resident) is worth building. If the curve is flat, it is not.

Total bytes are held constant across rows by scaling the repeat count, so the
only variable is residency.
"""

import torch

from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

N = 4096
TOPK = 10
REF_BYTES = 32768 * (TOPK + 1) * N * 2  # bytes moved by one full 32k-token reduce


def bench(T, iters):
    x = torch.randn((T * TOPK, N), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((T, N), dtype=torch.bfloat16, device="cuda")
    w = torch.rand((T, TOPK), dtype=torch.float32, device="cuda")
    for _ in range(3):
        _run_moe_reduction(x, y, T, TOPK, N, topk_weights=w)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        _run_moe_reduction(x, y, T, TOPK, N, topk_weights=w)
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1000 / iters
    rs = T * TOPK * N * 2
    tb = (rs + T * N * 2) / (us * 1e-6) / 1e12
    del x, y, w
    torch.cuda.empty_cache()
    return us, rs, tb


print(f"{'tokens':>8} {'read set':>10} {'us/call':>9} {'TB/s':>7} "
      f"{'us @32k-equiv':>14}")
for T, iters in [(256, 400), (512, 300), (1024, 200), (2048, 120),
                 (4096, 60), (8192, 40), (16384, 25), (32768, 20)]:
    us, rs, tb = bench(T, iters)
    # cost of doing the full 32768-token reduce in chunks of this size
    equiv = us * (32768 / T)
    print(f"{T:>8} {rs/1e6:>8.0f} MB {us:>9.1f} {tb:>7.2f} {equiv:>14.1f}")
