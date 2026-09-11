"""Opt-in import hook for the specialized FlyDSL GDN chunk-state kernel.

The dispatch predicate below is a cheap pre-filter; `chunk_delta_h_flydsl` does
the authoritative geometry validation and raises ValueError when it cannot take
a call. Both outcomes are counted.

Counting is not optional instrumentation. Installing this hook prints a banner
whether or not the kernel is ever reached, and the fallback path is silent, so a
predicate that rejects 100% of production calls looks exactly like a working
kernel. That is not hypothetical: before the varlen rewrite this hook took the
FlyDSL path on 0 of 3200 calls in a real benchmark and nothing said so.

Env:
  SGLANG_FLYDSL_GDN_CHUNK_H=1   install the hook
  SGLANG_FLYDSL_GDN_LOG_EVERY=N log a summary every N calls (0 = off, default)
  SGLANG_FLYDSL_GDN_STATS_DIR=D dump per-pid JSON to D on exit
"""

import atexit
import collections
import importlib.abc
import importlib.machinery
import json
import os
import sys

_LOG_EVERY = int(os.environ.get("SGLANG_FLYDSL_GDN_LOG_EVERY", "0"))
_STATS_DIR = os.environ.get("SGLANG_FLYDSL_GDN_STATS_DIR", "")
# If most calls are still falling back after this many, say so once, loudly.
_WARN_AFTER = int(os.environ.get("SGLANG_FLYDSL_GDN_WARN_AFTER", "500"))


class _Stats:
    def __init__(self):
        self.calls = 0
        self.flydsl = 0
        self.fallback = 0
        self.reasons = collections.Counter()
        self.warned = False

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
        return "[FlyDSL] pid=%d calls=%d flydsl=%d (%.1f%%) fallback=%s" % (
            os.getpid(),
            self.calls,
            self.flydsl,
            100.0 * self.flydsl / self.calls if self.calls else 0.0,
            dict(self.reasons) or "{}",
        )

    def dump(self):
        if not _STATS_DIR:
            return
        try:
            os.makedirs(_STATS_DIR, exist_ok=True)
            with open(os.path.join(_STATS_DIR, "flydsl_gdn_%d.json" % os.getpid()), "w") as fh:
                json.dump(self.as_dict(), fh, indent=2, sort_keys=True)
        except Exception as exc:  # diagnostics only
            print("[FlyDSL] stats dump failed: %r" % (exc,), file=sys.stderr, flush=True)

    def record(self, took_flydsl, reason=None):
        self.calls += 1
        if took_flydsl:
            self.flydsl += 1
        else:
            self.fallback += 1
            self.reasons[reason or "predicate"] += 1
        if _LOG_EVERY and self.calls % _LOG_EVERY == 0:
            print(self.summary(), file=sys.stderr, flush=True)
            self.dump()
        if not self.warned and self.calls >= _WARN_AFTER and self.flydsl * 2 < self.calls:
            self.warned = True
            print(
                "[FlyDSL] WARNING: %.1f%% of the first %d chunk_h calls fell back to "
                "Triton -- the specialization is mostly unused. %s"
                % (100.0 * self.fallback / self.calls, self.calls, self.summary()),
                file=sys.stderr,
                flush=True,
            )


_STATS = _Stats()


class _Loader(importlib.abc.Loader):
    def __init__(self, original: importlib.abc.Loader) -> None:
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module) -> None:
        self.original.exec_module(module)
        from chunk_delta_h_flydsl import chunk_gated_delta_rule_fwd_h

        original = module.chunk_gated_delta_rule_fwd_h

        def dispatch(
            k,
            w,
            u,
            g=None,
            gk=None,
            initial_state=None,
            initial_state_indices=None,
            save_new_value=True,
            cu_seqlens=None,
            chunk_indices=None,
            use_exp2=False,
        ):
            args = (
                k,
                w,
                u,
                g,
                gk,
                initial_state,
                initial_state_indices,
                save_new_value,
                cu_seqlens,
                chunk_indices,
                use_exp2,
            )
            # Ragged token counts and multi-sequence batches are supported; the
            # kernel masks the tail chunk with per-sequence buffer bounds.
            supported = (
                k.shape[0] == 1
                and k.shape[-1] == 128
                and k.shape[1] > 0
                and u.shape[-1] % 16 == 0
                and u.shape[-2] % k.shape[2] == 0
            )
            if not supported:
                _STATS.record(False, "geometry")
                return original(*args)
            try:
                out = chunk_gated_delta_rule_fwd_h(*args)
            except ValueError as exc:
                # The kernel's own geometry validation declined this call. Any
                # other exception is a real fault and is left to propagate.
                _STATS.record(False, str(exc)[:80])
                return original(*args)
            _STATS.record(True)
            return out

        module.chunk_gated_delta_rule_fwd_h = dispatch
        atexit.register(_STATS.dump)
        print(
            "[FlyDSL] GDN chunk_h hook installed (varlen: any token count, any "
            "number of packed sequences). Installed != used; check the counters.",
            file=sys.stderr,
            flush=True,
        )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.kernels.ops.attention.fla.chunk_delta_h":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("SGLANG_FLYDSL_GDN_CHUNK_H") == "1":
    sys.meta_path.insert(0, _Finder())
