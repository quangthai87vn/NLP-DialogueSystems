import json
from pathlib import Path

path = Path("data/globalwoz/F2E_vi.json")
data = json.loads(path.read_text(encoding="utf-8"))

print("Top-level type:", type(data))
keys = list(data.keys())
print("Num dialogs:", len(keys))
print("First dialog id:", keys[0])

ex = data[keys[0]]
print("Fields in one dialog:", list(ex.keys()))

# in thử vài dòng cho bạn nhìn
print("\n--- preview ---")
print(json.dumps(ex, ensure_ascii=False, indent=2)[:2000])
