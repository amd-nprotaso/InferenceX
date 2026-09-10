"""Opt-in import hook for the specialized FlyDSL GDN chunk-state kernel."""

import importlib.abc
import importlib.machinery
import os
import sys


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
            supported = (
                k.shape[0] == 1
                and k.shape[-1] == 128
                and k.shape[1] > 0
                and k.shape[1] % 64 == 0
                and u.shape[-1] % 16 == 0
                and (cu_seqlens is None or cu_seqlens.numel() == 2)
            )
            fn = chunk_gated_delta_rule_fwd_h if supported else original
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

        module.chunk_gated_delta_rule_fwd_h = dispatch
        print("[FlyDSL] experimental GDN chunk_h enabled for single-sequence K128 prefill", file=sys.stderr, flush=True)


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
