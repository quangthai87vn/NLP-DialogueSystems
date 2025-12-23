#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based TOD demo (Movie booking + Navigation) with Evaluation outputs.

Outputs:
  1) turn-level CSV (utterances + state + acts)
  2) dialogue-level metrics CSV (task success, missing slots, turns, repair/confirm/oos counts)

Run:
  python run_rule_tod_demo_eval.py --turns rule_tod_turns.csv --metrics rule_tod_metrics.csv
"""

from __future__ import annotations
import re, json, csv, argparse
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

# -------------------------
# Config
# -------------------------

REQUIRED_SLOTS = {
    "movie": {
        "info": ["movie", "theater"],
        "book": ["date", "time", "tickets"],
    },
    "navigation": {
        "info": ["origin", "destination", "mode"],
        "book": [],
    }
}

POLICY_PRIORITY = {
    "movie": ["movie", "theater", "date", "time", "tickets", "seat_type", "format", "language"],
    "navigation": ["destination", "origin", "mode", "avoid", "time"],
}

ASK_TEMPLATES = {
    # movie
    "movie": "Bạn muốn xem phim nào?",
    "theater": "Bạn muốn xem ở rạp nào (CGV/Lotte/Galaxy/BHD...) hoặc khu vực nào?",
    "date": "Bạn xem ngày nào? (hôm nay/mai/25-12...)",
    "time": "Bạn muốn suất mấy giờ? (vd 19:45)",
    "tickets": "Bạn đặt mấy vé?",
    "seat_type": "Bạn chọn ghế thường hay VIP/couple?",
    "format": "Bạn muốn 2D/3D/IMAX?",
    "language": "Bạn muốn phụ đề hay lồng tiếng?",
    # navigation
    "origin": "Bạn đang ở đâu (điểm xuất phát)?",
    "destination": "Bạn muốn đi tới đâu?",
    "mode": "Bạn đi xe máy/ô tô/đi bộ/xe buýt?",
    "avoid": "Bạn muốn tránh gì không? (kẹt xe/cao tốc/phà)",
}

# Simple NLG for confirm
def nlg_confirm(domain: str, info: Dict[str, Any], book: Dict[str, Any]) -> str:
    if domain == "movie":
        return (f"Chốt đặt vé: phim **{info.get('movie','?')}**, rạp **{info.get('theater','?')}**, "
                f"ngày **{book.get('date','?')}**, giờ **{book.get('time','?')}**, "
                f"{book.get('tickets','?')} vé. Đúng không?")
    if domain == "navigation":
        return (f"Chốt: từ **{info.get('origin','?')}** đến **{info.get('destination','?')}**, "
                f"phương tiện **{info.get('mode','?')}**. Đúng không?")
    return f"Chốt domain {domain}. Đúng không?"

# -------------------------
# State
# -------------------------

@dataclass
class DomainState:
    info: Dict[str, Any] = field(default_factory=dict)
    book: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DialogState:
    domains: Dict[str, DomainState] = field(default_factory=dict)

    def ensure(self, domain: str):
        if domain not in self.domains:
            self.domains[domain] = DomainState()

# -------------------------
# NLU (rules)
# -------------------------

def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def _extract_time(t: str) -> Optional[str]:
    # HH:MM
    m = re.search(r"\b([01]?\d|2[0-3])[:h]([0-5]\d)\b", t)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return f"{hh:02d}:{mm:02d}"
    # "7h", "19h"
    m = re.search(r"\b([01]?\d|2[0-3])\s*h\b", t)
    if m:
        hh = int(m.group(1))
        return f"{hh:02d}:00"
    return None

def extract_slots(user_text: str) -> Dict[str, Any]:
    t = normalize(user_text)
    slots: Dict[str, Any] = {}

    # shared: time
    tm = _extract_time(t)
    if tm:
        slots["time"] = tm

    # ---- movie booking ----
    m = re.search(r"\b(\d{1,2})\s*(vé|ve)\b", t)
    if m:
        slots["tickets"] = int(m.group(1))

    if "hôm nay" in t:
        slots["date"] = "today"
    elif re.search(r"\bmai\b", t):
        slots["date"] = "tomorrow"
    elif "mốt" in t:
        slots["date"] = "day_after_tomorrow"
    elif "cuối tuần" in t or "cuoi tuan" in t:
        slots["date"] = "weekend"
    else:
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", t)
        if m:
            slots["date"] = f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    m = re.search(r"\bphim\s+([a-z0-9\s\-]+)", t)
    if m:
        name = m.group(1).strip()
        name = re.split(r"\b(ở|o|luc|lúc|ngay|ngày|tai|tại|rạp|rap)\b", name)[0].strip()
        if len(name) >= 2:
            slots["movie"] = name

    if "cgv" in t:
        slots["theater"] = "CGV"
    elif "lotte" in t:
        slots["theater"] = "Lotte"
    elif "galaxy" in t:
        slots["theater"] = "Galaxy"
    elif "bhd" in t:
        slots["theater"] = "BHD"

    if "vip" in t:
        slots["seat_type"] = "vip"
    if "couple" in t:
        slots["seat_type"] = "couple"
    if "imax" in t:
        slots["format"] = "imax"
    elif "3d" in t:
        slots["format"] = "3d"
    elif "2d" in t:
        slots["format"] = "2d"
    if "phụ đề" in t or "phu de" in t:
        slots["language"] = "sub"
    elif "lồng tiếng" in t or "long tieng" in t:
        slots["language"] = "dub"

    # ---- navigation ----
    if "xe máy" in t or "xe may" in t:
        slots["mode"] = "motorbike"
    elif "ô tô" in t or "oto" in t or "o to" in t:
        slots["mode"] = "car"
    elif "đi bộ" in t or "di bo" in t:
        slots["mode"] = "walk"
    elif "xe buýt" in t or "xe buyt" in t or "bus" in t:
        slots["mode"] = "bus"

    if "tránh kẹt" in t or "tranh ket" in t:
        slots["avoid"] = "traffic"
    elif "tránh cao tốc" in t or "tranh cao toc" in t:
        slots["avoid"] = "highway"
    elif "tránh phà" in t or "tranh pha" in t:
        slots["avoid"] = "ferry"

    m = re.search(r"\b(từ|tu)\s+(.+?)\s+\b(đến|den|tới|toi)\s+(.+)$", t)
    if m:
        slots["origin"] = m.group(2).strip()
        slots["destination"] = m.group(4).strip()

    return slots

# -------------------------
# DST + Policy (rules)
# -------------------------

def update_state(ds: DialogState, domain: str, extracted: Dict[str, Any]):
    ds.ensure(domain)
    st = ds.domains[domain]

    book_slots = set(REQUIRED_SLOTS.get(domain, {}).get("book", [])) | {"tickets", "date", "time", "seat_type", "format", "language"}
    for k, v in extracted.items():
        if k in book_slots:
            st.book[k] = v
        else:
            st.info[k] = v

def missing_slots(ds: DialogState, domain: str) -> Tuple[List[str], List[str]]:
    ds.ensure(domain)
    st = ds.domains[domain]
    req = REQUIRED_SLOTS.get(domain, {"info": [], "book": []})
    miss_info = [s for s in req["info"] if s not in st.info]
    miss_book = [s for s in req["book"] if s not in st.book]
    return miss_info, miss_book

def choose_next_question(domain: str, miss_info: List[str], miss_book: List[str]) -> Optional[str]:
    candidates = miss_info + miss_book
    if not candidates:
        return None
    for p in POLICY_PRIORITY.get(domain, []):
        if p in candidates:
            return p
    return candidates[0]

def system_response(domain: str, ds: DialogState) -> Tuple[str, str, List[str], List[str]]:
    miss_info, miss_book = missing_slots(ds, domain)
    nxt = choose_next_question(domain, miss_info, miss_book)
    if nxt:
        act = f"REQUEST({domain}.{nxt})"
        utt = ASK_TEMPLATES.get(nxt, f"Bạn cho mình biết {nxt} được không?")
        return act, utt, miss_info, miss_book

    st = ds.domains[domain]
    act = f"CONFIRM({domain})"
    utt = nlg_confirm(domain, st.info, st.book)
    return act, utt, miss_info, miss_book

# -------------------------
# Evaluation helpers
# -------------------------

OK_WORDS = {"ok", "oke", "okay", "đúng", "dung", "yes", "chuẩn", "chuan", "ừ", "u", "đồng ý", "dong y"}
SMALLTALK_WORDS = {"cảm ơn", "cam on", "thanks", "thank", "hi", "hello", "chào", "chao"}

def is_user_ok(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in OK_WORDS)

def is_smalltalk(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in SMALLTALK_WORDS)

def detect_repair(prev_state: DomainState, new_state: DomainState, user_text: str) -> int:
    """
    Heuristic repair detection:
      - user utterance contains "đổi"/"thôi"/"không phải"
      OR
      - any required slot changes value compared to previous state
    """
    t = normalize(user_text)
    hint = ("đổi" in t) or ("doi" in t) or ("à" in t and "đổi" in t) or ("không phải" in t) or ("khong phai" in t)

    changed = 0
    # compare all keys across info+book
    keys = set(prev_state.info.keys()) | set(prev_state.book.keys()) | set(new_state.info.keys()) | set(new_state.book.keys())
    for k in keys:
        pv = prev_state.info.get(k, prev_state.book.get(k, None))
        nv = new_state.info.get(k, new_state.book.get(k, None))
        if pv is not None and nv is not None and pv != nv:
            changed += 1
    if hint and (changed == 0):
        # still count as repair attempt
        return 1
    return 1 if changed > 0 else 0

def final_missing(ds: DialogState, domain: str) -> List[str]:
    mi, mb = missing_slots(ds, domain)
    return mi + mb

# -------------------------
# Demo conversations (you can replace with dataset loader later)
# -------------------------

DEMO_DIALOGS = [
    {
        "conv_id": "movie_01",
        "domain": "movie",
        "user_turns": [
            "Mình muốn đặt vé phim Conan tối mai.",
            "CGV Gò Vấp, 2 vé.",
            "19:45",
            "OK",
        ]
    },
    {
        "conv_id": "movie_02_repair",
        "domain": "movie",
        "user_turns": [
            "Đặt giúp mình vé phim avatar 2D hôm nay.",
            "cgv, 1 vé",
            "7h",
            "à đổi 19h",
            "OK"
        ]
    },
    {
        "conv_id": "nav_01",
        "domain": "navigation",
        "user_turns": [
            "Chỉ đường tới UBND Q1.",
            "Từ Bến Thành tới UBND Quận 1, đi bộ.",
            "tránh kẹt xe",
            "OK"
        ]
    },
    {
        "conv_id": "nav_02",
        "domain": "navigation",
        "user_turns": [
            "Từ Central Station đến City Hall, đi bus.",
            "OK"
        ]
    }
]

# -------------------------
# Runner -> CSV
# -------------------------

def run_and_write_csv(turns_out: str, metrics_out: str):
    turn_fields = [
        "conv_id","domain","turn","speaker","utterance",
        "extracted_slots","state_info","state_book",
        "missing_info","missing_book","system_act"
    ]
    metric_fields = [
        "conv_id","domain","task_success","missing_slots_final","num_turns",
        "repair_count","confirm_count","oos_count"
    ]

    with open(turns_out, "w", newline="", encoding="utf-8") as ft, \
         open(metrics_out, "w", newline="", encoding="utf-8") as fm:

        tw = csv.DictWriter(ft, fieldnames=turn_fields)
        mw = csv.DictWriter(fm, fieldnames=metric_fields)
        tw.writeheader()
        mw.writeheader()

        for d in DEMO_DIALOGS:
            conv_id = d["conv_id"]
            domain = d["domain"]
            ds = DialogState()
            ds.ensure(domain)

            repair_count = 0
            confirm_count = 0
            oos_count = 0

            # Initial system prompt
            act, utt, miss_info, miss_book = system_response(domain, ds)
            if act.startswith("CONFIRM"):
                confirm_count += 1

            tw.writerow({
                "conv_id": conv_id, "domain": domain, "turn": 0, "speaker": "system",
                "utterance": utt,
                "extracted_slots": "",
                "state_info": json.dumps(ds.domains[domain].info, ensure_ascii=False),
                "state_book": json.dumps(ds.domains[domain].book, ensure_ascii=False),
                "missing_info": json.dumps(miss_info, ensure_ascii=False),
                "missing_book": json.dumps(miss_book, ensure_ascii=False),
                "system_act": act
            })

            turn_idx = 1
            for user_utt in d["user_turns"]:
                # snapshot before update for repair detection
                prev = DomainState(info=dict(ds.domains[domain].info), book=dict(ds.domains[domain].book))

                slots = extract_slots(user_utt)
                update_state(ds, domain, slots)

                st = ds.domains[domain]
                miss_info, miss_book = missing_slots(ds, domain)

                # OOS heuristic
                if (not slots) and (not is_user_ok(user_utt)) and (not is_smalltalk(user_utt)):
                    oos_count += 1

                # repair heuristic (only meaningful if state already had something)
                if prev.info or prev.book:
                    repair_count += detect_repair(prev, st, user_utt)

                # log user
                tw.writerow({
                    "conv_id": conv_id, "domain": domain, "turn": turn_idx, "speaker": "user",
                    "utterance": user_utt,
                    "extracted_slots": json.dumps(slots, ensure_ascii=False),
                    "state_info": json.dumps(st.info, ensure_ascii=False),
                    "state_book": json.dumps(st.book, ensure_ascii=False),
                    "missing_info": json.dumps(miss_info, ensure_ascii=False),
                    "missing_book": json.dumps(miss_book, ensure_ascii=False),
                    "system_act": ""
                })
                turn_idx += 1

                # system reacts
                act, sys_utt, miss_info2, miss_book2 = system_response(domain, ds)
                if act.startswith("CONFIRM"):
                    confirm_count += 1

                tw.writerow({
                    "conv_id": conv_id, "domain": domain, "turn": turn_idx, "speaker": "system",
                    "utterance": sys_utt,
                    "extracted_slots": "",
                    "state_info": json.dumps(st.info, ensure_ascii=False),
                    "state_book": json.dumps(st.book, ensure_ascii=False),
                    "missing_info": json.dumps(miss_info2, ensure_ascii=False),
                    "missing_book": json.dumps(miss_book2, ensure_ascii=False),
                    "system_act": act
                })
                turn_idx += 1

            miss_final = final_missing(ds, domain)
            task_success = (len(miss_final) == 0)
            mw.writerow({
                "conv_id": conv_id,
                "domain": domain,
                "task_success": int(task_success),
                "missing_slots_final": json.dumps(miss_final, ensure_ascii=False),
                "num_turns": turn_idx,  # includes system/user rows
                "repair_count": repair_count,
                "confirm_count": confirm_count,
                "oos_count": oos_count
            })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="rule_tod_turns.csv", help="Turn-level CSV output")
    ap.add_argument("--metrics", default="rule_tod_metrics.csv", help="Dialogue-level metrics CSV output")
    args = ap.parse_args()
    run_and_write_csv(args.turns, args.metrics)
    print(f"Wrote turns:   {args.turns}")
    print(f"Wrote metrics: {args.metrics}")

if __name__ == "__main__":
    main()
