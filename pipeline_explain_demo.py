import json
from pathlib import Path

DATA_PATH = Path("data/globalwoz/F2E_vi.json")
DIALOG_INDEX = 0

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
dialog_id = list(data.keys())[DIALOG_INDEX]
dialog = data[dialog_id]
goal = dialog["goal"]
log = dialog["log"]

def summarize_goal(goal: dict):
    out = []
    for domain, g in goal.items():
        if not g:
            continue
        info = g.get("info", {})
        book = g.get("book", {})
        reqt = g.get("reqt", [])
        if info:
            out.append(f"- {domain}.info: {info}")
        if book:
            out.append(f"- {domain}.book: {book}")
        if reqt:
            out.append(f"- {domain}.reqt: {reqt}")
    return "\n".join(out) if out else "(empty)"

print("=== DEMO PIPELINE (Explainable) ===")
print("Dialog:", dialog_id)
print("\n[Target Belief State = GOAL]")
print(summarize_goal(goal))

print("\n[Dialogue turns]")
for i, t in enumerate(log[:12]):  # lấy 12 turn đầu cho demo
    speaker = "USER" if i % 2 == 0 else "SYSTEM"
    print(f"{speaker}: {t.get('text','')}")
