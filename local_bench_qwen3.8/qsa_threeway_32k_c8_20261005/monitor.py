from pathlib import Path
import json
r=Path(__file__).resolve().parent
print((r/'driver.log').read_text()[-800:])
for arm in ['ck','triton','flydsl']:
 d=r/arm
 if not d.exists(): continue
 print('ARM',arm)
 p=d/'client-exact-result.json'
 if p.exists():
  x=json.loads(p.read_text()); print({k:x[k] for k in ['batch_wall_s','output_throughput_tok_s','tpot_ms','successful_requests']})
 else:
  p=d/'client.log'
  if p.exists(): print(p.read_text()[-300:])
  p=d/'server.log'
  if p.exists(): print('\n'.join(p.read_text(errors='replace').splitlines()[-2:])[:700])
 p=d/'server.log'
 if p.exists(): print('\n'.join(l for l in p.read_text(errors='replace').splitlines() if 'QSA_THREEWAY_EXECUTING' in l))
