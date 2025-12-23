import json
from pathlib import Path

DATA_PATH = Path("data/globalwoz/F2E_vi.json")
DIALOG_INDEX = 0   # đổi số này để lấy dialogue khác

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
dialog_ids = list(data.keys())
dialog_id = dialog_ids[DIALOG_INDEX]
dialog = data[dialog_id]

goal = dialog["goal"]
log = dialog["log"]

print(f"Dialog ID: {dialog_id}")
print("=== GOAL (tóm tắt domain có yêu cầu) ===")
domains = [d for d, g in goal.items() if g]  # domain nào có info
print("Domains:", domains)

print("\n=== TURNS ===")
for i, turn in enumerate(log):
    speaker = "USER" if i % 2 == 0 else "SYSTEM"
    text = turn.get("text", "")
    print(f"[{i:02d}] {speaker}: {text}")
