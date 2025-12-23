'''
Trong demo này, lời của User được lấy trực tiếp từ dữ liệu GlobalWoZ (các lượt chẵn trong log), 
còn lời của System là phản hồi do baseline rule-based tự sinh ra (Policy: hỏi slot còn thiếu theo goal + NLG: 
câu trả lời theo template). Nhờ vậy ta thấy rõ pipeline TOD hoạt động: 
User utterance → NLU trích slot → DST cập nhật state → Policy chọn hành động → System response.
'''


import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

DATA_PATH = Path("data/globalwoz/F2E_vi.json")

# -----------------------------
# 1) NLU: rule-based slot extraction (Vietnamese-ish)
# -----------------------------
DAY_MAP = {
    "thứ hai": "monday", "thu hai": "monday",
    "thứ ba": "tuesday", "thu ba": "tuesday",
    "thứ tư": "wednesday", "thu tu": "wednesday",
    "thứ năm": "thursday", "thu nam": "thursday",
    "thứ sáu": "friday", "thu sau": "friday",
    "thứ bảy": "saturday", "thu bay": "saturday",
    "chủ nhật": "sunday", "chu nhat": "sunday",
}

AREA_MAP = {
    "trung tâm": "centre", "center": "centre", "centre": "centre",
    "phía bắc": "north", "mien bac": "north", "north": "north",
    "phía nam": "south", "mien nam": "south", "south": "south",
    "phía đông": "east", "mien dong": "east", "east": "east",
    "phía tây": "west", "mien tay": "west", "west": "west",
}

PRICE_MAP = {
    "rẻ": "cheap", "re": "cheap", "cheap": "cheap",
    "vừa": "moderate", "trung bình": "moderate", "moderate": "moderate",
    "đắt": "expensive", "dat": "expensive", "expensive": "expensive",
}

TYPE_MAP = {
    "guesthouse": "guesthouse", "nhà khách": "guesthouse", "nha khach": "guesthouse",
    "khách sạn": "hotel", "khach san": "hotel", "hotel": "hotel",
    "hostel": "hostel", "nhà trọ": "hostel", "nha tro": "hostel",
}

FOOD_MAP = {
    "ý": "italian", "italian": "italian",
    "trung": "chinese", "chinese": "chinese",
    "ấn": "indian", "indian": "indian",
    "thái": "thai", "thai": "thai",
    "nhật": "japanese", "japanese": "japanese",
}

def normalize(text: str) -> str:
    return text.strip().lower()

def extract_slots(user_utt: str) -> Dict[str, Any]:
    """Return extracted slot candidates from a user utterance."""
    t = normalize(user_utt)
    slots: Dict[str, Any] = {}

    # internet / wifi
    if "wifi" in t or "wi-fi" in t or "internet" in t:
        # naive polarity
        if "không" in t and ("wifi" in t or "internet" in t):
            slots["internet"] = "no"
        else:
            slots["internet"] = "yes"

    # parking
    if "đậu xe" in t or "dau xe" in t or "parking" in t:
        if "không" in t and ("đậu xe" in t or "dau xe" in t or "parking" in t):
            slots["parking"] = "no"
        else:
            slots["parking"] = "yes"

    # area
    for k, v in AREA_MAP.items():
        if k in t:
            slots["area"] = v
            break

    # price
    for k, v in PRICE_MAP.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            slots["pricerange"] = v
            break

    # type
    for k, v in TYPE_MAP.items():
        if k in t:
            slots["type"] = v
            break

    # food
    for k, v in FOOD_MAP.items():
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            slots["food"] = v
            break

    # people (book)
    m = re.search(r"(\d+)\s*(người|nguoi|person|people)", t)
    if m:
        slots["people"] = int(m.group(1))

    # stay nights
    m = re.search(r"(\d+)\s*(đêm|dem|night|nights)", t)
    if m:
        slots["stay"] = int(m.group(1))

    # day
    for k, v in DAY_MAP.items():
        if k in t:
            slots["day"] = v
            break
    # also english day direct
    for v in set(DAY_MAP.values()):
        if v in t:
            slots["day"] = v
            break

    # time HH:MM or HhMM
    m = re.search(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            slots["time"] = f"{hh:02d}:{mm:02d}"

    return slots


# -----------------------------
# 2) DST: dialogue state
# -----------------------------
def init_state() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {}  # domain -> {"info":{}, "book":{}}

def ensure_domain(state, domain: str):
    if domain not in state:
        state[domain] = {"info": {}, "book": {}}

def update_state(state, domain: str, extracted: Dict[str, Any]):
    ensure_domain(state, domain)

    # heuristic: map extracted slots into info vs book
    for k, v in extracted.items():
        if k in ["people", "stay", "day", "time"]:
            state[domain]["book"][k] = v
        else:
            state[domain]["info"][k] = v


# -----------------------------
# 3) Goal parsing: what slots are required?
# -----------------------------
def goal_domains(goal: Dict[str, Any]) -> List[str]:
    return [d for d, g in goal.items() if g]

def required_slots_for_domain(goal_domain: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (required_info_slots, required_book_slots)."""
    info = list((goal_domain.get("info") or {}).keys())
    book = list((goal_domain.get("book") or {}).keys())
    # ignore internal fields
    info = [s for s in info if s != "invalid"]
    book = [s for s in book if s != "invalid"]
    return info, book

def missing_slots(state, domain: str, goal_domain: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    ensure_domain(state, domain)
    req_info, req_book = required_slots_for_domain(goal_domain)
    miss_info = [s for s in req_info if s not in state[domain]["info"]]
    miss_book = [s for s in req_book if s not in state[domain]["book"]]
    return miss_info, miss_book


# -----------------------------
# 4) Policy + NLG templates
# -----------------------------
ASK_TEMPLATES = {
    "area": "Bạn muốn ở khu vực nào (trung tâm/bắc/nam/đông/tây) vậy?",
    "pricerange": "Bạn muốn mức giá rẻ / trung bình / đắt?",
    "type": "Bạn muốn loại hình gì (khách sạn/guesthouse/hostel)?",
    "internet": "Bạn có cần Wi-Fi không?",
    "parking": "Bạn có cần chỗ đậu xe không?",
    "food": "Bạn muốn món gì (Ý/Trung/Thái/…)?",
    "people": "Bạn đặt cho mấy người?",
    "stay": "Bạn ở mấy đêm?",
    "day": "Bạn muốn ngày nào (thứ…/chủ nhật)?",
    "time": "Bạn muốn giờ mấy (vd 18:45)?",
}

def choose_next_question(miss_info: List[str], miss_book: List[str]) -> Optional[str]:
    # prioritize book slots after basic info (tuỳ bạn, mình để info trước cho dễ)
    priority = ["area", "pricerange", "type", "internet", "parking", "food", "people", "stay", "day", "time"]
    candidates = miss_info + miss_book
    for p in priority:
        if p in candidates:
            return p
    return candidates[0] if candidates else None

def system_response(domain: str, state, goal_domain: Dict[str, Any]) -> Tuple[str, str]:
    """Return (sys_act, sys_utt) based on missing slots."""
    miss_info, miss_book = missing_slots(state, domain, goal_domain)
    nxt = choose_next_question(miss_info, miss_book)
    if nxt:
        act = f"REQUEST({domain}.{nxt})"
        utt = ASK_TEMPLATES.get(nxt, f"Bạn cho mình biết {nxt} được không?")
        return act, utt

    # if nothing missing: confirm completion for that domain
    act = f"CONFIRM({domain})"
    # show a compact summary
    info = state[domain]["info"]
    book = state[domain]["book"]
    utt = f"Ok, mình đã ghi nhận cho domain **{domain}**. Info={info}. Book={book}. Nếu muốn, mình có thể tiếp tục domain khác."
    return act, utt


# -----------------------------
# 5) Demo runner on one dialogue (using user turns from log)
# -----------------------------
def run_demo_one(dialog: Dict[str, Any], max_user_turns: int = 12) -> Dict[str, Any]:
    goal = dialog["goal"]
    log = dialog["log"]

    domains = goal_domains(goal)
    if not domains:
        return {"success": True, "state": {}, "trace": []}

    # pick current domain: first domain in goal (simple)
    current_idx = 0
    current_domain = domains[current_idx]

    state = init_state()
    trace = []

    # feed USER turns from dataset (even turns 0,2,4,...)
    user_turns = [(i, log[i]["text"]) for i in range(0, len(log), 2)]
    user_turns = user_turns[:max_user_turns]

    for turn_i, user_utt in user_turns:
        # NLU
        extracted = extract_slots(user_utt)

        # DST update
        update_state(state, current_domain, extracted)

        # policy+NLG
        sys_act, sys_utt = system_response(current_domain, state, goal[current_domain])

        # if domain completed, move to next domain (if any)
        miss_info, miss_book = missing_slots(state, current_domain, goal[current_domain])
        domain_done = (len(miss_info) == 0 and len(miss_book) == 0)
        
                # Lưu domain hiện tại của step này (trước khi switch)
        step_domain = current_domain

        # if domain completed, move to next domain (if any)
        miss_info, miss_book = missing_slots(state, step_domain, goal[step_domain])
        domain_done = (len(miss_info) == 0 and len(miss_book) == 0)

        if domain_done and current_idx + 1 < len(domains):
            current_idx += 1
            current_domain = domains[current_idx]
            ensure_domain(state, current_domain)  # <<< FIX: tạo state cho domain mới

        trace.append({
            "user_turn_index": turn_i,
            "domain": step_domain,  # domain của step này
            "user_utt": user_utt,
            "nlu_extracted": extracted,
            "dst_state_domain": state[step_domain],  # snapshot đúng domain vừa update
            "sys_act": sys_act,
            "sys_utt": sys_utt
        })


    # success definition (simple): state covers all goal required slots for all domains
    success = True
    for d in domains:
        miss_info, miss_book = missing_slots(state, d, goal[d])
        if miss_info or miss_book:
            success = False
            break

    return {"success": success, "domains": domains, "state": state, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog_id", type=str, default=None, help="e.g., MUL0003.json")
    ap.add_argument("--max_user_turns", type=int, default=12)
    args = ap.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if args.dialog_id is None:
        dialog_id = list(data.keys())[0]
    else:
        dialog_id = args.dialog_id
        if dialog_id not in data:
            raise KeyError(f"dialog_id not found: {dialog_id}")

    dialog = data[dialog_id]
    out = run_demo_one(dialog, max_user_turns=args.max_user_turns)

    print("=== RULE-BASED TOD (Level 2) ===")
    print("Dialog:", dialog_id)
    print("Domains:", out.get("domains"))
    print("SUCCESS:", out["success"])
    print("\n--- TRACE (first steps) ---")
    for i, step in enumerate(out["trace"]):
        print(f"\nStep {i}")
        print("User:", step["user_utt"])
        print("NLU extracted:", step["nlu_extracted"])
        print("DST (current domain state):", step["dst_state_domain"])
        print("Policy act:", step["sys_act"])
        print("System:", step["sys_utt"])

if __name__ == "__main__":
    main()
