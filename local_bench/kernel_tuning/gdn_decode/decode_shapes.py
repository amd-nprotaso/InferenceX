"""Production decode/verify shapes for the Qwen3.5-397B-A17B GDN recurrence.

The prefill side already has ``chunk_gated_delta_rule/shapes.py``; this is its
decode twin. Geometry is read off the same model at TP=4:

    Hg (k/q heads) = 4      H / HV (v heads) = 16
    K = 128                 V = 128

Unlike prefill, the decode kernel's launch geometry depends on the *batch*, not
the sequence length -- ``grid = (NK, NV, N * HV)`` where ``N`` is the number of
sequences and ``NV = cdiv(V, BV)``. That is the whole reason this file exists:
at the serving config we actually run (concurrency 4, EAGLE with 4 draft
tokens) the grid is 1 x 4 x 64 = 256 single-wave workgroups on a 256-CU MI355X,
i.e. ~3% of the machine's wave slots.

``T`` is tokens-per-forward = batch x draft_token_num. The kernel is called with
B=1 and a cu_seqlens of length N+1, matching ``forward_decode`` in
sglang/srt/layers/attention/linear/gdn_backend.py.
"""

from dataclasses import dataclass

CU_COUNT = 256          # MI355X gfx950
SIMD_PER_CU = 4
WAVES_PER_SIMD = 8      # wave slots, from rocprofv3 agent_info


@dataclass(frozen=True)
class DecodeShape:
    name: str
    batch: int           # sequences in flight == N
    draft_tokens: int    # EAGLE --speculative-num-draft-tokens
    Hg: int = 4          # q/k heads per rank
    HV: int = 16         # v heads per rank
    K: int = 128
    V: int = 128
    pool_slots: int = 1243
    note: str = ""

    @property
    def T(self) -> int:
        """Tokens per forward pass (B=1, packed across sequences)."""
        return self.batch * self.draft_tokens

    def grid(self, BV: int) -> tuple[int, int, int]:
        """Mirror of the wrapper's ``grid = (NK, NV, N * HV)``."""
        NK = 1  # the wrapper asserts cdiv(K, BK) == 1
        NV = (self.V + BV - 1) // BV
        return (NK, NV, self.batch * self.HV)

    def workgroups(self, BV: int) -> int:
        g = self.grid(BV)
        return g[0] * g[1] * g[2]

    def wave_slot_pct(self, BV: int, num_warps: int) -> float:
        """Percent of the GPU's wave slots this launch can occupy."""
        waves = self.workgroups(BV) * num_warps
        return 100.0 * waves / (CU_COUNT * SIMD_PER_CU * WAVES_PER_SIMD)

    def state_bytes(self, dtype_size: int = 2) -> int:
        """SSM state touched per layer per call: batch x HV x V x K."""
        return self.batch * self.HV * self.V * self.K * dtype_size


SHAPES = {
    s.name: s
    for s in (
        DecodeShape(
            name="qwen35-397b-tp4-c4",
            batch=4,
            draft_tokens=4,
            note="Production: concurrency 4, EAGLE num-draft-tokens 4. "
                 "grid=(1,4,64)=256 WGs @ BV=32.",
        ),
        DecodeShape(
            name="qwen35-397b-tp4-c4-nospec",
            batch=4,
            draft_tokens=1,
            note="Same batch without speculative decoding (T=4).",
        ),
        DecodeShape(
            name="qwen35-397b-tp4-c8",
            batch=8,
            draft_tokens=4,
            note="Concurrency 8; grid z doubles to 128.",
        ),
        DecodeShape(
            name="qwen35-397b-tp4-c32",
            batch=32,
            draft_tokens=4,
            note="Concurrency 32 (the server default CONC); grid z = 512.",
        ),
    )
}
