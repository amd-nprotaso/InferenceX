"""Opt-in worker-local decode-MoE patch; activated only through this PYTHONPATH entry.

Hooks ``aiter.fused_moe`` rather than an SGLang module. That is the seam the
patch actually rewrites (``fused_topk`` and ``moe_sorting``), and patching it at
its own import time means SGLang's module-level
``from aiter.fused_moe import fused_topk as aiter_fused_topk``
(python/sglang/srt/layers/moe/topk.py) binds the patched function without
needing a second hook -- while ``install()`` still rebinds that attribute
directly in case this hook lands after SGLang's import.

Gated on SGLANG_FLYDSL_MOE_DECODE=1. Install failures are logged and swallowed:
a tokenizer worker with no GPU visible must not take the server down just
because the kernel module could not resolve an arch.

The directory holding ``moe_decode_patch.py`` must be on PYTHONPATH -- the hook
does a bare ``import moe_decode_patch``, same convention as conv/bootstrap.
"""

import importlib.abc
import importlib.machinery
import os
import sys


class _MoeDecodeLoader(importlib.abc.Loader):
    def __init__(self, original: importlib.abc.Loader) -> None:
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module) -> None:
        self.original.exec_module(module)
        try:
            import moe_decode_patch

            moe_decode_patch.install()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[FlyDSL] decode-MoE patch NOT installed in pid {os.getpid()}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )


class _MoeDecodeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "aiter.fused_moe":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _MoeDecodeLoader(spec.loader)
        return spec


if os.environ.get("SGLANG_FLYDSL_MOE_DECODE") == "1":
    sys.meta_path.insert(0, _MoeDecodeFinder())
