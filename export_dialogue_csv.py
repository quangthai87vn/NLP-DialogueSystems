import json
import csv
from pathlib import Path

IN_PATH = Path("data/globalwoz/F2E_vi.json")
OUT_CSV = Path("demo_globalwoz_vi_turns.csv")

data = json.loads(IN_PATH.read_text(encoding="utf-8"))
dialog_id = list(data.keys())[0]
dialog = data[dialog_id]
turns = dialog.get("log") or dialog.get("turns") or dialog.get("dialogue") or dialog.get("conversation") or []

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dialog_id", "turn", "speaker", "text"])
    for i, t in enumerate(turns):
        speaker = t.get("speaker") or t.get("role") or ("user" if i % 2 == 0 else "system")
        text = t.get("text") or t.get("utterance") or t.get("content") or ""
        w.writerow([dialog_id, i, speaker, text])

print("Saved:", OUT_CSV)
