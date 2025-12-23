import json
from pathlib import Path

DATA_PATH = Path("data/globalwoz/F2E_vi.json")
data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

def count_domains(goal):
    return sum(1 for d,g in goal.items() if g)

best = []
for did, dlg in data.items():
    k = count_domains(dlg["goal"])
    if k >= 2:
        best.append((k, did))
        if len(best) >= 5:
            break

print("Top candidates (>=2 domains):")
for k, did in best:
    print(k, did)
