#!/usr/bin/env python3
"""Print an AITER_CONFIG_FMOE chain that swaps one model CSV for a local variant.

aiter merges every ``model_configs/*tuned_fmoe*.csv`` into one table and raises
(after rewriting the source files in place!) if two rows share a shape key. So an
override cannot simply be prepended — the replaced file has to be dropped from
the chain. Usage:

    export AITER_CONFIG_FMOE=$(python3 fmoe_chain.py qwen3_5_397b_fp4_tuned_fmoe.atomic.csv)
"""

import glob
import os
import sys

AITER_CFG = "/sgl-workspace/aiter/aiter/configs"
REPLACED = "qwen3_5_397b_fp4_tuned_fmoe.csv"


def chain(replacement):
    """Reproduce aiter's default merge list, with REPLACED swapped out.

    Must match ``AITER_CONFIG.get_config_file`` in aiter/jit/core.py: the glob
    is ``*tuned_fmoe*`` but files whose name contains "untuned" are excluded.
    A plain ``*tuned_fmoe*.csv`` glob also matches ``*untuned_fmoe*.csv`` and
    would feed aiter ~30 untuned files it never normally merges.
    """
    files = [os.path.join(AITER_CFG, "tuned_fmoe.csv")]
    files += sorted(
        p
        for p in glob.glob(os.path.join(AITER_CFG, "model_configs", "*tuned_fmoe*.csv"))
        if os.path.basename(p) != REPLACED and "untuned" not in os.path.basename(p)
    )
    return ":".join([os.path.abspath(replacement)] + files)


if __name__ == "__main__":
    print(chain(sys.argv[1]))
