"""Opt-in FlyDSL GDN *decode* recurrence override, activated via PYTHONPATH.

An import hook rebinds ``fused_sigmoid_gating_delta_rule_update`` inside
``sglang.srt.layers.attention.linear.kernels.gdn_triton`` after that module
loads. It must be patched there, not in the defining module: gdn_triton does a
module-level ``from ... import fused_sigmoid_gating_delta_rule_update``, so it
holds its own reference and a patch to the source module would not be seen.

Only processes that start with this directory on PYTHONPATH *and*
SGLANG_FLYDSL_GDN_DECODE=1 are patched; an already-running server is not.

The FlyDSL kernel covers both ``gdn_triton.decode()`` and
``gdn_triton.target_verify()`` (the EAGLE path, which disables the state
write-back and snapshots each draft step into intermediate_states_buffer).
It requires a uniform sequence length, l2norm in kernel, no ReplaySSM ring, no
EAGLE tree mask and no KDA. Everything else -- notably the ``extend`` call in
the same module -- keeps Triton. Rather than raising (which would kill
the server on the first unsupported call), unsupported calls fall back and are
*counted*, so a run can be audited afterwards: "installed" is not "used", and a
low flydsl% means the arm measured mostly baseline.

Env:
  SGLANG_FLYDSL_GDN_DECODE=1          enable
  SGLANG_FLYDSL_GDN_DECODE_STATS_DIR  dump per-pid usage counters at exit
  SGLANG_FLYDSL_GDN_DECODE_LOG_EVERY  log a summary every N calls (0 = off)
"""

import atexit
import collections
import importlib.abc
import importlib.machinery
import json
import os
import sys

_TARGET = "sglang.srt.layers.attention.linear.kernels.gdn_triton"
_LOG_EVERY = int(os.environ.get("SGLANG_FLYDSL_GDN_DECODE_LOG_EVERY", "0"))
_STATS_DIR = os.environ.get("SGLANG_FLYDSL_GDN_DECODE_STATS_DIR", "")


class _Stats:
    def __init__(self):
        self.calls = 0
        self.flydsl = 0
        self.fallback = 0
        self.reasons = collections.Counter()

    def as_dict(self):
        return {
            "pid": os.getpid(),
            "calls": self.calls,
            "flydsl": self.flydsl,
            "fallback": self.fallback,
            "flydsl_pct": round(100.0 * self.flydsl / self.calls, 2) if self.calls else 0.0,
            "fallback_reasons": dict(self.reasons),
        }

    def summary(self):
        return "[FlyDSL-decode] pid=%d calls=%d flydsl=%d (%.1f%%) fallback=%s" % (
            os.getpid(), self.calls, self.flydsl,
            100.0 * self.flydsl / self.calls if self.calls else 0.0,
            dict(self.reasons) or "{}",
        )

    def dump(self):
        if not _STATS_DIR:
            return
        try:
            os.makedirs(_STATS_DIR, exist_ok=True)
            with open(os.path.join(_STATS_DIR, "flydsl_gdn_decode_%d.json" % os.getpid()), "w") as fh:
                json.dump(self.as_dict(), fh, indent=2, sort_keys=True)
        except Exception as exc:  # diagnostics only
            print("[FlyDSL-decode] stats dump failed: %r" % (exc,), file=sys.stderr, flush=True)

    def record(self, took, reason=None):
        self.calls += 1
        if took:
            self.flydsl += 1
        else:
            self.fallback += 1
            self.reasons[reason or "unsupported"] += 1
        if _LOG_EVERY and self.calls % _LOG_EVERY == 0:
            print(self.summary(), file=sys.stderr, flush=True)
            self.dump()


_STATS = _Stats()

# Keyword arguments that put the call outside this kernel's scope. Checked
# before dispatch so the common path never pays for an exception.
_BLOCKING = (
    "retrieve_parent_token",   # EAGLE tree mask; only linear chains are ported
    "replayssm_rawv",
    "replayssm_rawk",
    "replayssm_g",
    "replayssm_beta",
)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        from gdn_decode_flydsl import fused_sigmoid_gating_delta_rule_update as flydsl_fn

        original = module.fused_sigmoid_gating_delta_rule_update

        def dispatch(*args, **kwargs):
            if args:
                _STATS.record(False, "positional-args")
                return original(*args, **kwargs)
            reason = None
            if kwargs.get("is_kda") or kwargs.get("lower_bound") is not None:
                reason = "kda"
            elif kwargs.get("cache_ring"):
                reason = "replayssm-ring"
            elif any(kwargs.get(name) is not None for name in _BLOCKING):
                reason = "verify-buffers"
            elif not kwargs.get("use_qk_l2norm_in_kernel"):
                reason = "no-l2norm"
            elif kwargs.get("cu_seqlens") is None:
                reason = "no-cu-seqlens"
            if reason is not None:
                _STATS.record(False, reason)
                return original(**kwargs)
            try:
                out = flydsl_fn(**kwargs)
            except ValueError as exc:
                _STATS.record(False, str(exc)[:48])
                return original(**kwargs)
            _STATS.record(True)
            return out

        module.fused_sigmoid_gating_delta_rule_update = dispatch
        print("[FlyDSL] GDN decode recurrence hook installed. "
              "Installed != used; check the counters.", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("SGLANG_FLYDSL_GDN_DECODE") == "1":
    sys.meta_path.insert(0, _Finder())
    atexit.register(_STATS.dump)
    atexit.register(lambda: print(_STATS.summary(), file=sys.stderr, flush=True))
