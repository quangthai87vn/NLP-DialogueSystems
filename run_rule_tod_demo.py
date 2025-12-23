#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based TOD demo for 2 domains: movie ticket booking + navigation (directions).
Generates a turn-by-turn CSV log for easy viewing.

Run:
  python run_rule_tod_demo.py --out rule_tod_demo_output.csv
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
# Utils
# -------------------------

def normalize(text: str) -> str:
    t = text.lower().strip()
    # normalize common Vietnamese diacritics-less patterns minimally
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
    # tickets: "2 vé", "3 ve"
    m = re.search(r"\b(\d{1,2})\s*(vé|ve)\b", t)
    if m:
        slots["tickets"] = int(m.group(1))

    # date: hôm nay/mai/mốt/cuối tuần or dd-mm / dd/mm
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

    # movie name (simple): after "phim ..."
    m = re.search(r"\bphim\s+([a-z0-9\s\-]+)", t)
    if m:
        name = m.group(1).strip()
        name = re.split(r"\b(ở|o|luc|lúc|ngay|ngày|tai|tại|rạp|rap)\b", name)[0].strip()
        if len(name) >= 2:
            slots["movie"] = name

    # theater keywords
    if "cgv" in t:
        slots["theater"] = "CGV"
    elif "lotte" in t:
        slots["theater"] = "Lotte"
    elif "galaxy" in t:
        slots["theater"] = "Galaxy"
    elif "bhd" in t:
        slots["theater"] = "BHD"

    # optional
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

    # origin/destination: "từ X đến Y"
    m = re.search(r"\b(từ|tu)\s+(.+?)\s+\b(đến|den|tới|toi)\s+(.+)$", t)
    if m:
        slots["origin"] = m.group(2).strip()
        slots["destination"] = m.group(4).strip()

    # repair pattern: "không phải X, là Y" for origin/destination names
    if "không phải" in t or "khong phai" in t:
        # leave to business logic (user provided corrections)
        pass

    return slots

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
    pr = POLICY_PRIORITY.get(domain, [])
    for p in pr:
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

    # confirm
    st = ds.domains[domain]
    act = f"CONFIRM({domain})"
    utt = nlg_confirm(domain, st.info, st.book)
    return act, utt, miss_info, miss_book

# -------------------------
# Demo conversations
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
        "conv_id": "movie_02_noise_repair",
        "domain": "movie",
        "user_turns": [
            "Đặt giúp mình vé phim avatar 2D hôm nay.",
            "cgv, 1 vé",
            "7h",       # time
            "à đổi 19h",# repair time
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

def run_and_write_csv(out_path: str):
    fieldnames = [
        "conv_id","domain","turn","speaker","utterance",
        "extracted_slots","state_info","state_book",
        "missing_info","missing_book","system_act"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for d in DEMO_DIALOGS:
            conv_id = d["conv_id"]
            domain = d["domain"]
            ds = DialogState()

            # Initial system prompt (ask first missing)
            act, utt, miss_info, miss_book = system_response(domain, ds)
            w.writerow({
                "conv_id": conv_id, "domain": domain, "turn": 0, "speaker": "system",
                "utterance": utt,
                "extracted_slots": "",
                "state_info": json.dumps(ds.domains.get(domain, DomainState()).info, ensure_ascii=False),
                "state_book": json.dumps(ds.domains.get(domain, DomainState()).book, ensure_ascii=False),
                "missing_info": json.dumps(miss_info, ensure_ascii=False),
                "missing_book": json.dumps(miss_book, ensure_ascii=False),
                "system_act": act
            })

            turn_idx = 1
            for user_utt in d["user_turns"]:
                slots = extract_slots(user_utt)
                update_state(ds, domain, slots)

                st = ds.domains[domain]
                miss_info, miss_book = missing_slots(ds, domain)

                # log user
                w.writerow({
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

                # system reacts unless user just says OK after confirm (keep simple)
                act, sys_utt, miss_info2, miss_book2 = system_response(domain, ds)
                w.writerow({
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rule_tod_demo_output.csv", help="Output CSV path")
    args = ap.parse_args()
    run_and_write_csv(args.out)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
