from pathlib import Path
import json
r=Path(__file__).resolve().parent
print((r/'driver.log').read_text()[-1500:])
for a in ['off_1','on_1','on_2','off_2']:
 d=r/a
 if not d.exists(): continue
 print('TRIAL',a)
 p=d/'client-exact-result.json'
 if p.exists():
  x=json.loads(p.read_text()); print({k:v for k,v in x.items() if k!='requests'})
 else:
  p=d/'client.log'
  if p.exists(): print(p.read_text()[-600:])
 p=d/'server.log'
 if p.exists():
  lines=p.read_text(errors='replace').splitlines()
  print('\n'.join(l for l in lines if 'QSA_AB_' in l))
  print('\n'.join(lines[-2:])[:1200])
