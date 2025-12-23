import json
from pathlib import Path

DATA_PATH = Path("data/globalwoz/F2E_vi.json")
OUT_MD = Path("demo_globalwoz_vi_slide.md")
OUT_JSON = Path("demo_globalwoz_vi_one_dialogue.json")

DIALOG_INDEX = 0  # đổi số để lấy dialogue khác

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
dialog_ids = list(data.keys())
dialog_id = dialog_ids[DIALOG_INDEX]
dialog = data[dialog_id]

# save raw one-dialogue (evidence)
OUT_JSON.write_text(json.dumps({dialog_id: dialog}, ensure_ascii=False, indent=2), encoding="utf-8")

goal = dialog["goal"]
log = dialog["log"]

lines = []
lines.append("# Demo: GlobalWoZ Vietnamese (F2E)\n")
lines.append(f"- Source: `{DATA_PATH}`")
lines.append(f"- Dialogue: `{dialog_id}`\n")

# Goal
lines.append("## Goal (belief target)\n```json")
lines.append(json.dumps(goal, ensure_ascii=False, indent=2))
lines.append("```\n")

# Turns
lines.append("## Turns (User/System)\n")
for i, turn in enumerate(log):
    speaker = "User" if i % 2 == 0 else "System"
    text = turn.get("text", "")
    lines.append(f"**Turn {i} – {speaker}:** {text}")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("Saved:", OUT_MD, "and", OUT_JSON)
print("Open demo_globalwoz_vi_slide.md để copy vào slide notes.")
