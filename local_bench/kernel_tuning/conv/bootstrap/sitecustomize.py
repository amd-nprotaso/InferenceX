"""Opt-in worker-local GDN patch; activated only through this PYTHONPATH entry."""
import importlib.abc
import importlib.machinery
import os
import sys


class _GDNLoader(importlib.abc.Loader):
    def __init__(self, original: importlib.abc.Loader) -> None:
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module) -> None:
        self.original.exec_module(module)
        from causal_conv1d_flydsl import causal_conv1d_fn
        module.causal_conv1d_fn = causal_conv1d_fn
        print("[FlyDSL] GDN causal prefill convolution enabled", file=sys.stderr, flush=True)


class _GDNFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.layers.attention.linear.gdn_backend":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _GDNLoader(spec.loader)
        return spec


if os.environ.get("SGLANG_FLYDSL_CAUSAL_CONV") == "1":
    sys.meta_path.insert(0, _GDNFinder())
