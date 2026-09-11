"""Correctness and GPU graph timing against PyTorch and installed SGLang."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from causal_conv1d_flydsl import causal_conv1d_fn


def check(lengths: list[int], channels: int = 65, width: int = 4,
          dtype: torch.dtype = torch.bfloat16, layout: str = "channel_last",
          bias_on: bool = True, activation: str | None = "silu", graph: bool = False,
          compare: bool = True, block: int = 128, tokens: int = 16,
          prefetch: int = 16, channels_per_thread: int | None = None,
          bias_dtype: torch.dtype | None = None) -> None:
    total, batch = sum(lengths), len(lengths)
    x = torch.randn(total, channels, device="cuda", dtype=dtype).T
    if layout == "channel_first":
        x = x.contiguous()
    elif layout == "strided":
        x = torch.randn(total*2, channels, device="cuda", dtype=dtype)[::2, :].T
    elif layout == "offset":
        x = torch.randn(total, channels+1, device="cuda", dtype=dtype)[:, 1:].T
    w = torch.randn(channels, width, device="cuda", dtype=dtype)*0.25
    bias = torch.randn(channels, device="cuda", dtype=bias_dtype or dtype) if bias_on else None
    state = torch.randn(batch+3, width+2, channels, device="cuda", dtype=dtype).transpose(1, 2)
    slots = list(reversed(range(batch)))
    if batch > 2:
        slots[1] = -1
    initial = [i % 2 == 0 for i in range(batch)]
    starts = [0]
    for n in lengths:
        starts.append(starts[-1]+n)
    q = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ids = torch.tensor(slots, device="cuda", dtype=torch.int64)
    init = torch.tensor(initial, device="cuda", dtype=torch.bool)
    original = state.clone()
    expected, expected_state = torch.zeros_like(x), state.clone()
    mask = torch.zeros(total, device="cuda", dtype=torch.bool)
    for i, n in enumerate(lengths):
        slot, lo = slots[i], starts[i]
        if slot == -1 or n == 0:
            continue
        mask[lo:lo+n] = True
        history = original[slot, :, :width-1].float()
        if not initial[i]:
            history = torch.zeros_like(history)
        joined = torch.cat((history, x[:, lo:lo+n].float()), dim=1)
        acc = torch.zeros_like(x[:, lo:lo+n], dtype=torch.float32)
        if bias is not None:
            acc += bias[:, None].float()
        for j in range(width):
            acc += joined[:, j:j+n]*w[:, j:j+1].float()
        if activation:
            acc = torch.nn.functional.silu(acc)
        expected[:, lo:lo+n] = acc
        expected_state[slot, :, :width-1] = joined[:, -(width-1):]
    kwargs = dict(cache_indices=ids, has_initial_state=init, activation=activation)
    tuning = dict(prefetch=prefetch, channels_per_thread=channels_per_thread)
    got = causal_conv1d_fn(x, w, bias, state, q, lengths, validate_data=True,
                          block=block, tokens=tokens, **tuning, **kwargs)
    tol = 0.016 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(got[:, mask], expected[:, mask], atol=tol, rtol=tol)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    if compare:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn as baseline
        ts = original.clone()
        ref = baseline(x, w, bias, ts, q, lengths, **kwargs)
        torch.testing.assert_close(got[:, mask], ref[:, mask], atol=tol, rtol=tol)
        torch.testing.assert_close(state, ts, atol=0, rtol=0)
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            causal_conv1d_fn(x, w, bias, state, q, lengths, block=block, tokens=tokens, **tuning, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        state.copy_(original)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = causal_conv1d_fn(x, w, bias, state, q, lengths, block=block, tokens=tokens, **tuning, **kwargs)
        state.copy_(original)
        g.replay()
        torch.testing.assert_close(yg[:, mask], expected[:, mask], atol=tol, rtol=tol)
        torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def bench(channels: int, length: int, batch: int, block: int, tokens: int) -> dict:
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn as baseline
    import triton.testing
    x = torch.randn(length*batch, channels, device="cuda", dtype=torch.bfloat16).T
    w = torch.randn(channels, 4, device="cuda", dtype=x.dtype)
    s = torch.zeros(batch, channels, 3, device="cuda", dtype=x.dtype)
    q = torch.arange(batch+1, device="cuda", dtype=torch.int32)*length
    lengths = [length]*batch
    def fly() -> torch.Tensor:
        return causal_conv1d_fn(x, w, None, s, q, lengths, block=block, tokens=tokens)
    def triton_call() -> torch.Tensor:
        return baseline(x, w, None, s, q, lengths)
    fly()
    triton_call()
    torch.cuda.synchronize()
    fly_ms = triton.testing.do_bench_cudagraph(fly, rep=100)
    triton_ms = triton.testing.do_bench_cudagraph(triton_call, rep=100)
    return dict(channels=channels, length=length, batch=batch, block=block, tokens=tokens,
                flydsl_ms=fly_ms, triton_ms=triton_ms, speedup=triton_ms/fly_ms)


def check_optional_state() -> None:
    """No cache, zero lengths, repeated dispatch, and invalid metadata."""
    x = torch.randn(19, 65, device="cuda", dtype=torch.bfloat16).T
    w = torch.randn(65, 4, device="cuda", dtype=x.dtype)
    q = torch.tensor([0, 19], device="cuda", dtype=torch.int32)
    expected = torch.nn.functional.conv1d(x.float()[None], w.float()[:, None], padding=3, groups=65)[0, :, :19].to(x.dtype)
    for _ in range(2):
        got = causal_conv1d_fn(x, w, None, None, q, [19], activation=None)
        torch.testing.assert_close(got, expected, atol=0.03125, rtol=0.016)
    state = torch.randn(1, 65, 3, device="cuda", dtype=x.dtype)
    original = state.clone()
    zero = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    got = causal_conv1d_fn(x[:, :0], w, None, state, zero, [0])
    assert got.shape == (65, 0)
    torch.testing.assert_close(state, original, atol=0, rtol=0)
    try:
        causal_conv1d_fn(x, w, None, state, q, [18], validate_data=True)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched sequence offsets accepted")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench-only", action="store_true")
    p.add_argument("--channels", type=int, default=3072)
    p.add_argument("--length", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.manual_seed(1234)
    if args.bench_only:
        result = bench(args.channels, args.length, args.batch, args.block, args.tokens)
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2)+"\n")
        return
    if args.smoke:
        check([17], compare=False)
        print("smoke passed", flush=True)
        return
    count = 0
    for width in (2, 3, 4, 5):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for layout in ("channel_last", "channel_first", "strided"):
                check([1, 7, 2, 0, 3, 17, 33], width=width, dtype=dtype, layout=layout,
                      compare=width < 5, block=args.block, tokens=args.tokens)
                count += 1
                print(f"passed {count}: {width=} {dtype=} {layout=}", flush=True)
    # Paired lanes with an incomplete final wave, strided input/state, padded
    # cache slots, and a prefetch window larger than a non-power-of-two tile.
    for width in (2, 3, 4, 5):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            check([1, 7, 2, 0, 3, 17, 33], channels=130, width=width,
                  dtype=dtype, layout="strided", tokens=7, prefetch=16,
                  channels_per_thread=2, compare=width < 5, graph=True)
            count += 1
            print(f"passed paired {count}: {width=} {dtype=}", flush=True)
    check([1, 15, 16, 17], channels=130, channels_per_thread=2,
          bias_dtype=torch.float32, prefetch=2, layout="offset", graph=True)
    count += 1
    for activation in (None, "swish"):
        check([65, 3], channels=256, bias_on=False, activation=activation, graph=True,
              block=args.block, tokens=args.tokens)
        count += 1
    check([args.length], channels=args.channels, bias_on=False, block=args.block, tokens=args.tokens)
    check_optional_state()
    result = dict(device=torch.cuda.get_device_name(), correctness_cases=count+1,
                  optional_state_checks="passed", torch_version=torch.__version__,
                  hip_version=torch.version.hip, seed=1234,
                  kernel_sha256=hashlib.sha256(Path(__file__).with_name("causal_conv1d_flydsl.py").read_bytes()).hexdigest())
    if args.bench:
        result["benchmark"] = bench(args.channels, args.length, args.batch, args.block, args.tokens)
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
