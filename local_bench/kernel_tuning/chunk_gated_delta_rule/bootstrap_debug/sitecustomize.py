"""Instrumented twin of ../bootstrap/sitecustomize.py.

Same dispatch predicate, byte for byte, plus a per-call counter that records
which guard rejected the call. Stats are dumped to
$SGLANG_FLYDSL_GDN_STATS_DIR/flydsl_gdn_<pid>.json on interpreter exit and,
every SGLANG_FLYDSL_GDN_LOG_EVERY calls, to stderr.
"""

import atexit
import collections
import importlib.abc
import importlib.machinery
import json
import os
import sys

_STATS_DIR = os.environ.get("SGLANG_FLYDSL_GDN_STATS_DIR", "/tmp/flydsl_gdn_stats")
_LOG_EVERY = int(os.environ.get("SGLANG_FLYDSL_GDN_LOG_EVERY", "200"))


class _Stats:
    def __init__(self) -> None:
        self.calls = 0
        self.flydsl = 0
        self.fallback = 0
        self.reasons = collections.Counter()
        self.seqlens_hist = collections.Counter()
        self.nseq_hist = collections.Counter()
        self.t_mod64_hist = collections.Counter()
        self.flydsl_errors = collections.Counter()

    def as_dict(self):
        return {
            "pid": os.getpid(),
            "calls": self.calls,
            "flydsl": self.flydsl,
            "fallback": self.fallback,
            "flydsl_pct": (100.0 * self.flydsl / self.calls) if self.calls else 0.0,
            "reject_reasons": dict(self.reasons),
            "n_sequences_in_batch": dict(self.nseq_hist),
            "T_mod_64": dict(self.t_mod64_hist),
            "T_values_top": dict(self.seqlens_hist.most_common(40)),
            "flydsl_errors": dict(self.flydsl_errors),
        }

    def dump(self):
        try:
            os.makedirs(_STATS_DIR, exist_ok=True)
            path = os.path.join(_STATS_DIR, "flydsl_gdn_%d.json" % os.getpid())
            with open(path, "w") as handle:
                json.dump(self.as_dict(), handle, indent=2, sort_keys=True)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print("[FlyDSL-dbg] stats dump failed: %r" % (exc,), file=sys.stderr, flush=True)


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
            # ---- identical predicate to bootstrap/sitecustomize.py ----
            supported = (
                k.shape[0] == 1
                and k.shape[-1] == 128
                and k.shape[1] > 0
                and k.shape[1] % 64 == 0
                and u.shape[-1] % 16 == 0
                and (cu_seqlens is None or cu_seqlens.numel() == 2)
            )
            # ---- instrumentation only below ----
            _STATS.calls += 1
            tokens = int(k.shape[1])
            nseq = (cu_seqlens.numel() - 1) if cu_seqlens is not None else 1
            _STATS.nseq_hist[nseq] += 1
            _STATS.t_mod64_hist[tokens % 64] += 1
            _STATS.seqlens_hist[tokens] += 1
            if not supported:
                if k.shape[0] != 1:
                    _STATS.reasons["B != 1"] += 1
                elif k.shape[-1] != 128:
                    _STATS.reasons["K != 128"] += 1
                elif k.shape[1] <= 0:
                    _STATS.reasons["T == 0"] += 1
                elif k.shape[1] % 64 != 0:
                    _STATS.reasons["T mod 64 != 0"] += 1
                elif u.shape[-1] % 16 != 0:
                    _STATS.reasons["V mod 16 != 0"] += 1
                else:
                    _STATS.reasons["cu_seqlens.numel() != 2 (multi-seq batch)"] += 1
                _STATS.fallback += 1
            else:
                _STATS.flydsl += 1

            if _LOG_EVERY and _STATS.calls % _LOG_EVERY == 0:
                print(
                    "[FlyDSL-dbg] pid=%d calls=%d flydsl=%d (%.2f%%) reasons=%s"
                    % (
                        os.getpid(),
                        _STATS.calls,
                        _STATS.flydsl,
                        100.0 * _STATS.flydsl / _STATS.calls,
                        dict(_STATS.reasons),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                _STATS.dump()

            fn = chunk_gated_delta_rule_fwd_h if supported else original
            try:
                return fn(
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
            except Exception as exc:
                if supported:
                    # The predicate said yes but the FlyDSL host function still
                    # refused -- record it, it would otherwise look like a crash.
                    _STATS.flydsl_errors[type(exc).__name__ + ": " + str(exc)[:200]] += 1
                    _STATS.dump()
                raise

        module.chunk_gated_delta_rule_fwd_h = dispatch
        atexit.register(_STATS.dump)
        print(
            "[FlyDSL-dbg] hook INSTALLED on %s in pid %d (stats -> %s)"
            % (module.__name__, os.getpid(), _STATS_DIR),
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
