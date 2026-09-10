import os, subprocess, sys, json
CASES = [
    ({}, "defaults"),
    ({"SGLANG_GDN_CHUNK_H_BV": "32"}, "BV=32 override -> wpe MUST drop"),
    ({"SGLANG_GDN_CHUNK_H_NUM_WARPS": "2"}, "warps=2 override -> wpe MUST drop"),
    ({"SGLANG_GDN_CHUNK_H_BV": "16", "SGLANG_GDN_CHUNK_H_NUM_WARPS": "4"}, "explicit tuned pair"),
    ({"SGLANG_GDN_CHUNK_H_WAVES_PER_EU": "0"}, "wpe=0 escape hatch"),
]
probe = (
    "from sglang.kernels.ops.attention.fla import chunk_delta_h as m;"
    "c=m.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.configs[0];"
    "import json;print('RESULT'+json.dumps({'kwargs':c.kwargs,"
    "'num_warps':c.num_warps,'num_stages':c.num_stages}))"
)
print(f"{'case':<38}{'BV':>4}{'warps':>7}{'stages':>8}{'waves_per_eu':>14}")
print("-" * 71)
bad = 0
for env, label in CASES:
    e = dict(os.environ); e.update(env)
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", probe],
                         capture_output=True, text=True, env=e)
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print(f"{label:<38}  FAILED: {(out.stderr or '').strip().splitlines()[-1][:80]}")
        bad += 1; continue
    d = json.loads(line[0][6:])
    wpe = d["kwargs"].get("waves_per_eu", "-")
    print(f"{label:<38}{d['kwargs']['BV']:>4}{d['num_warps']:>7}{d['num_stages']:>8}{str(wpe):>14}")
    # the whole point of the gate
    if d["kwargs"]["BV"] != 16 or d["num_warps"] != 4:
        if "waves_per_eu" in d["kwargs"]:
            print("   ^^ GATE FAILURE: waves_per_eu leaked onto a non-tuned config"); bad += 1
sys.exit(1 if bad else 0)
