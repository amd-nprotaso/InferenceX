"""Read-only progress and result inspection for this experiment."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for result in sorted(root.glob('round_*/*/trial_*/*.jsonl')):
    for line in result.read_text().splitlines():
        row = json.loads(line)
        print(result.parent.relative_to(root), 'completed=', row['completed'],
              'tok/s=', round(row['output_throughput'], 3),
              'accept=', round(row['accept_length'], 5))
logs = list(root.glob('round_*/*/server.log'))
if logs:
    latest = max(logs, key=lambda p: p.stat().st_mtime)
    text = latest.read_text(errors='replace')
    print('LATEST', latest.parent.relative_to(root), 'compile_calls=', text.count('[FlyDSL-compile]'))
    print(text[-800:])
    drivers = list(latest.parent.glob('*-driver.log'))
    if drivers:
        driver = max(drivers, key=lambda p: p.stat().st_mtime)
        print('CLIENT', driver.name, driver.read_text(errors='replace')[-500:])
    for f in sorted((latest.parent / 'gdn_stats').glob('*.json')):
        row = json.loads(f.read_text())
        if row['calls']:
            print('GDN', row['pid'], row['calls'], 'calls;', row['fallback'], 'fallbacks')
print('Trace files:', len(list(root.glob('round_*/*/profile/**/*.gz'))))
