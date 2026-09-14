"""Install the FlyDSL import hooks, optionally with a JIT-compile counter.

conv/bootstrap and chunk_gated_delta_rule/bootstrap are each named
`sitecustomize.py`. Python imports that name exactly once, from whichever
sys.path entry comes first, so putting both directories on PYTHONPATH silently
activates one kernel and drops the other -- with both banners absent for the
loser and no other symptom.

This module is the single `sitecustomize` on the path; it loads each hook file
under its own module name so both meta-path finders get installed. The kernel
source directories still have to be on PYTHONPATH, because each hook does a
bare `from <kernel>_flydsl import ...` when its target module is first imported.

Each hook stays gated by its own env flag (SGLANG_FLYDSL_CAUSAL_CONV /
SGLANG_FLYDSL_GDN_CHUNK_H / SGLANG_FLYDSL_MOE_DECODE), so this file activates
nothing on its own. That makes one launcher able to serve every arm of a
kernel A/B, and it is why adding a kernel means adding an entry to _HOOKS
rather than a third file named `sitecustomize.py`.

SGLANG_FLYDSL_COUNT_COMPILES=1 wraps flydsl.compiler.compile and logs one line
per JIT compile. Counting lines in the server log across a benchmark window is
how you tell "the kernel is slow" from "the kernel is still compiling".
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HOOKS = (
    ("_flydsl_hook_conv", os.path.join(_ROOT, "conv", "bootstrap", "sitecustomize.py")),
    ("_flydsl_hook_gdn", os.path.join(_ROOT, "chunk_gated_delta_rule", "bootstrap", "sitecustomize.py")),
    ("_flydsl_hook_moe_decode", os.path.join(_ROOT, "moe_decode", "bootstrap", "sitecustomize.py")),
)


class _CompileCounterLoader(importlib.abc.Loader):
    """Wrap flydsl.compiler.compile once the module finishes loading.

    Both kernels call `flyc.compile(...)` as an attribute lookup on the module
    object, so replacing the attribute here covers every caller regardless of
    import order.
    """

    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        inner = module.compile
        state = {"n": 0}

        def counting(func, *args):
            state["n"] += 1
            name = getattr(func, "__name__", "?")
            print(
                "[FlyDSL-compile] pid=%d n=%d fn=%s" % (os.getpid(), state["n"], name),
                file=sys.stderr,
                flush=True,
            )
            return inner(func, *args)

        module.compile = counting
        print("[FlyDSL] JIT compile counter installed in pid %d" % os.getpid(), file=sys.stderr, flush=True)


class _CompileCounterFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "flydsl.compiler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _CompileCounterLoader(spec.loader)
        return spec


if os.environ.get("SGLANG_FLYDSL_COUNT_COMPILES") == "1":
    sys.meta_path.insert(0, _CompileCounterFinder())

for _name, _path in _HOOKS:
    if not os.path.exists(_path):
        print(f"[FlyDSL] combined bootstrap: missing {_path}", file=sys.stderr, flush=True)
        continue
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
