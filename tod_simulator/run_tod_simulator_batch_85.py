#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research-style batch simulation for Task-Oriented Dialogue (TOD)
Domains: movie ticket booking + navigation (directions)

This version is tuned so that average task success is ~85% (configurable).

Outputs:
  1) turn-level CSV
  2) dialogue-level metrics CSV (task success, turns, slot/state accuracy)
  3) summary CSV (aggregated by domain + overall)

Run:
  python run_tod_simulator_batch_85.py --n 1000 --turns tod_sim_turns_1000.csv --metrics tod_sim_metrics_1000.csv --summary tod_sim_summary.csv
"""

from __future__ import annotations
import re, json, csv, argparse, random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

def set_seed(seed: int):
    random.seed(seed)

# -------------------------
# KB (tiny)
# -------------------------
MOVIES = ["conan", "avatar", "dune", "wonka", "spider-man", "inside out", "oppenheimer", "barbie"]
THEATERS = ["CGV", "Lotte", "Galaxy", "BHD"]
DATES = ["today", "tomorrow", "weekend"]
SHOWTIMES = ["18:30", "19:45", "21:10", "22:00"]
MODES = ["walk", "motorbike", "car", "bus"]
PLACES = ["Bến Thành", "UBND Quận 1", "City Hall", "Central Station", "Central Stadium",
          "Chợ Bà Chiểu", "Landmark 81", "Sân bay Tân Sơn Nhất"]

CONFUSABLE = {
    "Central Station": "Central Stadium",
    "Central Stadium": "Central Station",
    "CGV": "Galaxy",
    "Galaxy": "CGV",
    "Lotte": "BHD",
    "BHD": "Lotte",
    "walk": "bus",
    "bus": "walk",
    "motorbike": "car",
    "car": "motorbike",
    "18:30": "19:45",
    "19:45": "18:30",
    "21:10": "22:00",
    "22:00": "21:10",
}

# -------------------------
# System config (rule-based)
# -------------------------
REQUIRED_SLOTS = {
    "movie": {"info": ["movie", "theater"], "book": ["date", "time", "tickets"]},
    "navigation": {"info": ["origin", "destination", "mode"], "book": []},
}
POLICY_PRIORITY = {
    "movie": ["movie", "theater", "date", "time", "tickets"],
    "navigation": ["destination", "origin", "mode", "avoid", "time"],
}
ASK_TEMPLATES = {
    "movie": "Bạn muốn xem phim nào?",
    "theater": "Bạn muốn xem ở rạp nào (CGV/Lotte/Galaxy/BHD...) hoặc khu vực nào?",
    "date": "Bạn xem ngày nào? (hôm nay/mai/cuối tuần)",
    "time": "Bạn muốn suất mấy giờ? (vd 19:45)",
    "tickets": "Bạn đặt mấy vé?",
    "origin": "Bạn đang ở đâu (điểm xuất phát)?",
    "destination": "Bạn muốn đi tới đâu?",
    "mode": "Bạn đi xe máy/ô tô/đi bộ/xe buýt?",
    "avoid": "Bạn muốn tránh gì không? (kẹt xe/cao tốc/phà)",
}
def nlg_confirm(domain: str, info: Dict[str, Any], book: Dict[str, Any]) -> str:
    if domain == "movie":
        return (f"Chốt đặt vé: phim {info.get('movie','?')}, rạp {info.get('theater','?')}, "
                f"ngày {book.get('date','?')}, giờ {book.get('time','?')}, {book.get('tickets','?')} vé. OK không?")
    if domain == "navigation":
        return (f"Chốt: từ {info.get('origin','?')} đến {info.get('destination','?')}, "
                f"phương tiện {info.get('mode','?')}. OK không?")
    return "OK không?"

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
# NLU rules
# -------------------------
def norm(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def extract_time(t: str) -> Optional[str]:
    m = re.search(r"\b([01]?\d|2[0-3])[:h]([0-5]\d)\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    m = re.search(r"\b([01]?\d|2[0-3])\s*h\b", t)
    if m:
        return f"{int(m.group(1)):02d}:00"
    return None

def snap_place(text_lower: str) -> Optional[str]:
    for p in PLACES:
        if p.lower() in text_lower:
            return p
    return None

def extract_slots(user_text: str, p_drop: float = 0.0) -> Dict[str, Any]:
    """Rule NLU + optional random slot drop to simulate NLU errors."""
    tl = norm(user_text)
    slots: Dict[str, Any] = {}

    # date
    if "hôm nay" in tl:
        slots["date"] = "today"
    elif re.search(r"\bmai\b", tl):
        slots["date"] = "tomorrow"
    elif "cuối tuần" in tl or "cuoi tuan" in tl:
        slots["date"] = "weekend"

    # time
    tm = extract_time(tl)
    if tm:
        slots["time"] = tm
    else:
        for s in SHOWTIMES:
            if s in tl:
                slots["time"] = s

    # tickets
    m = re.search(r"\b(\d{1,2})\s*(vé|ve)\b", tl)
    if m:
        slots["tickets"] = int(m.group(1))

    # theater
    for th in THEATERS:
        if th.lower() in tl:
            slots["theater"] = th

    # movie (phim <name>)
    m = re.search(r"\bphim\s+([a-z0-9\- ]+)", tl)
    if m:
        name = m.group(1).strip()
        name = re.split(r"\b(ở|o|lúc|luc|ngày|ngay|rạp|rap)\b", name)[0].strip()
        if name:
            slots["movie"] = name

    # mode
    if "xe buýt" in tl or "xe buyt" in tl or "bus" in tl:
        slots["mode"] = "bus"
    elif "đi bộ" in tl or "di bo" in tl or "walk" in tl:
        slots["mode"] = "walk"
    elif "xe máy" in tl or "xe may" in tl:
        slots["mode"] = "motorbike"
    elif "ô tô" in tl or "oto" in tl or "car" in tl:
        slots["mode"] = "car"

    # avoid
    if "tránh kẹt" in tl or "tranh ket" in tl:
        slots["avoid"] = "traffic"
    elif "tránh cao tốc" in tl or "tranh cao toc" in tl:
        slots["avoid"] = "highway"
    elif "tránh phà" in tl or "tranh pha" in tl:
        slots["avoid"] = "ferry"

    # origin/destination parsing
    m = re.search(r"\b(từ|tu)\s+(.+?)\s+\b(đến|den|tới|toi)\s+(.+)$", user_text, flags=re.IGNORECASE)
    if m:
        slots["origin"] = m.group(2).strip()
        slots["destination"] = m.group(4).strip()
    else:
        m = re.search(r"\b(từ|tu)\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            slots["origin"] = m.group(2).strip()
            p = snap_place(tl)
            if p:
                slots["origin"] = p
        m = re.search(r"\b(đến|den|tới|toi|đi tới|di toi)\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            slots["destination"] = m.group(2).strip()
            p = snap_place(tl)
            if p:
                slots["destination"] = p

    # NLU error simulation: randomly drop some extracted slots
    if p_drop > 0 and slots:
        for k in list(slots.keys()):
            if random.random() < p_drop:
                del slots[k]

    return slots

# -------------------------
# DST + Policy
# -------------------------
def update_state(ds: DialogState, domain: str, extracted: Dict[str, Any]):
    ds.ensure(domain)
    st = ds.domains[domain]
    book_slots = set(REQUIRED_SLOTS.get(domain, {}).get("book", [])) | {"tickets", "date", "time"}
    for k, v in extracted.items():
        if k in book_slots:
            st.book[k] = v
        else:
            st.info[k] = v

def missing_slots(ds: DialogState, domain: str) -> Tuple[List[str], List[str]]:
    ds.ensure(domain)
    st = ds.domains[domain]
    req = REQUIRED_SLOTS[domain]
    mi = [s for s in req["info"] if s not in st.info]
    mb = [s for s in req["book"] if s not in st.book]
    return mi, mb

def choose_next(domain: str, mi: List[str], mb: List[str]) -> Optional[str]:
    cand = mi + mb
    if not cand:
        return None
    for p in POLICY_PRIORITY[domain]:
        if p in cand:
            return p
    return cand[0]

def system_step(domain: str, ds: DialogState) -> Tuple[str, str, List[str], List[str]]:
    mi, mb = missing_slots(ds, domain)
    nxt = choose_next(domain, mi, mb)
    if nxt:
        act = f"REQUEST({domain}.{nxt})"
        return act, ASK_TEMPLATES.get(nxt, f"Bạn cho mình biết {nxt}?"), mi, mb
    st = ds.domains[domain]
    act = f"CONFIRM({domain})"
    return act, nlg_confirm(domain, st.info, st.book), mi, mb

# -------------------------
# User simulator
# -------------------------
def sample_goal(domain: str) -> Dict[str, Any]:
    if domain == "movie":
        return {
            "movie": random.choice(MOVIES),
            "theater": random.choice(THEATERS),
            "date": random.choice(DATES),
            "time": random.choice(SHOWTIMES),
            "tickets": random.randint(1, 4),
        }
    origin = random.choice(PLACES)
    dest = random.choice([p for p in PLACES if p != origin])
    goal = {"origin": origin, "destination": dest, "mode": random.choice(MODES)}
    if random.random() < 0.25:
        goal["avoid"] = random.choice(["traffic","highway","ferry"])
    return goal

def value_to_utter(domain: str, slot: str, value: Any) -> str:
    if domain == "movie":
        if slot == "movie": return f"Mình muốn đặt vé phim {value}."
        if slot == "theater": return f"{value}."
        if slot == "date":
            return {"today":"Hôm nay.","tomorrow":"Mai.","weekend":"Cuối tuần."}.get(value, str(value))
        if slot == "time": return f"{value}"
        if slot == "tickets": return f"{value} vé"
    if domain == "navigation":
        if slot == "destination": return f"Đi tới {value}."
        if slot == "origin": return f"Từ {value}."
        if slot == "mode":
            return {"walk":"Đi bộ.","bus":"Đi bus.","motorbike":"Đi xe máy.","car":"Đi ô tô."}.get(value, str(value))
        if slot == "avoid":
            return {"traffic":"Tránh kẹt xe.","highway":"Tránh cao tốc.","ferry":"Tránh phà."}.get(value, str(value))
    return str(value)

def maybe_corrupt(goal: Dict[str, Any], slot: str, force_noise: bool) -> Tuple[Any, bool]:
    if not force_noise:
        return goal.get(slot), False
    v = goal.get(slot)
    if v is None:
        return None, False
    if isinstance(v, int):
        v2 = max(1, min(6, v + random.choice([-1, 1])))
        return v2, True
    vs = str(v)
    if vs in CONFUSABLE:
        return CONFUSABLE[vs], True
    return v, False

def user_reply(domain: str,
               goal: Dict[str, Any],
               sys_act: str,
               noisy_slot_once: Optional[str],
               p_multi: float = 0.35) -> Tuple[str, Dict[str, Any], Optional[str]]:
    if sys_act.startswith("REQUEST("):
        slot = sys_act[len("REQUEST("):-1].split(".", 1)[1]

        provided = [slot]
        if random.random() < p_multi:
            req_all = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
            others = [s for s in req_all if s != slot]
            random.shuffle(others)
            provided += others[: random.randint(0, 2)]

        parts = []
        out_slots = {}

        for s in provided:
            force_noise = (noisy_slot_once == s)
            val, is_noisy = maybe_corrupt(goal, s, force_noise)
            if val is None:
                continue
            parts.append(value_to_utter(domain, s, val))
            out_slots[s] = val
            if is_noisy:
                noisy_slot_once = None

        return " ".join(parts) if parts else "Mình chưa rõ.", {"act":"inform","slots":out_slots}, noisy_slot_once

    return "OK", {"act":"confirm","slots":{}}, noisy_slot_once

def build_repair_utter(domain: str, goal: Dict[str, Any], wrong_slots: List[str]) -> Tuple[str, Dict[str, Any]]:
    parts = ["Không đúng, mình sửa lại:"]
    out = {}
    for s in wrong_slots:
        v = goal.get(s)
        if v is None:
            continue
        parts.append(value_to_utter(domain, s, v))
        out[s] = v
    return " ".join(parts), {"act":"deny+inform","slots":out}

# -------------------------
# Metrics
# -------------------------
def compare_state_to_goal(domain: str, st: DomainState, goal: Dict[str, Any]) -> Dict[str, Any]:
    req = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
    pred = {}
    pred.update(st.info); pred.update(st.book)

    tp=fp=fn=0
    correct=0
    for s in req:
        if s in pred:
            if pred[s] == goal.get(s):
                tp += 1; correct += 1
            else:
                fp += 1; fn += 1
        else:
            fn += 1

    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    state_acc = correct/len(req) if req else 1.0
    joint = 1 if correct == len(req) else 0
    missing = [s for s in req if (s not in pred) or (pred[s] != goal.get(s))]
    return {"slot_precision":prec,"slot_recall":rec,"slot_f1":f1,"state_slot_accuracy":state_acc,"joint_goal_accuracy":joint,"missing_or_wrong_slots":missing}

# -------------------------
# Batch runner
# -------------------------
def run_batch(n: int,
              turns_out: str,
              metrics_out: str,
              summary_out: str,
              seed: int,
              p_force_noise_dialog: float,
              p_user_repairs: float,
              p_nlu_drop: float,
              max_sys_steps: int):
    set_seed(seed)

    turn_fields = [
        "conv_id","domain","row_id","speaker","utterance",
        "sys_act","usr_act","extracted_slots",
        "state_info","state_book","missing_info","missing_book"
    ]
    metric_fields = [
        "conv_id","domain","task_success","num_rows","repairs","confirms","oos",
        "slot_precision","slot_recall","slot_f1","state_slot_accuracy","joint_goal_accuracy",
        "missing_or_wrong_slots","goal"
    ]

    metrics_rows = []

    with open(turns_out, "w", newline="", encoding="utf-8") as ft, \
         open(metrics_out, "w", newline="", encoding="utf-8") as fm:

        tw = csv.DictWriter(ft, fieldnames=turn_fields); tw.writeheader()
        mw = csv.DictWriter(fm, fieldnames=metric_fields); mw.writeheader()

        for i in range(n):
            domain = "movie" if random.random() < 0.5 else "navigation"
            goal = sample_goal(domain)

            # Force exactly one slot to be noisy in some dialogues
            noisy_slot_once = None
            if random.random() < p_force_noise_dialog:
                req_all = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
                noisy_slot_once = random.choice(req_all)

            conv_id = f"{domain}_{i:04d}"
            ds = DialogState(); ds.ensure(domain)
            row_id = 0
            repairs = 0
            confirms = 0
            oos = 0

            for _ in range(max_sys_steps):
                sys_act, sys_text, mi, mb = system_step(domain, ds)
                if sys_act.startswith("CONFIRM"):
                    confirms += 1

                tw.writerow({
                    "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"system",
                    "utterance":sys_text,"sys_act":sys_act,"usr_act":"",
                    "extracted_slots":"","state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                    "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                    "missing_info":json.dumps(mi,ensure_ascii=False),
                    "missing_book":json.dumps(mb,ensure_ascii=False),
                })
                row_id += 1

                if sys_act.startswith("CONFIRM"):
                    st = ds.domains[domain]
                    cmp = compare_state_to_goal(domain, st, goal)
                    wrong = cmp["missing_or_wrong_slots"]
                    if wrong and (random.random() < p_user_repairs):
                        random.shuffle(wrong)
                        pick = wrong[: random.randint(1, min(2, len(wrong)))]
                        user_text, usr_act = build_repair_utter(domain, goal, pick)
                        repairs += 1
                    else:
                        # either nothing wrong OR user refuses to repair -> ends
                        user_text, usr_act, noisy_slot_once = user_reply(domain, goal, sys_act, noisy_slot_once)
                        tw.writerow({
                            "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"user",
                            "utterance":user_text,"sys_act":"","usr_act":json.dumps(usr_act,ensure_ascii=False),
                            "extracted_slots":json.dumps(extract_slots(user_text, p_drop=p_nlu_drop),ensure_ascii=False),
                            "state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                            "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                            "missing_info":json.dumps(mi,ensure_ascii=False),
                            "missing_book":json.dumps(mb,ensure_ascii=False),
                        })
                        row_id += 1
                        break
                else:
                    user_text, usr_act, noisy_slot_once = user_reply(domain, goal, sys_act, noisy_slot_once)

                extracted = extract_slots(user_text, p_drop=p_nlu_drop)
                if not extracted and "ok" not in norm(user_text):
                    oos += 1
                update_state(ds, domain, extracted)

                mi2, mb2 = missing_slots(ds, domain)
                tw.writerow({
                    "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"user",
                    "utterance":user_text,"sys_act":"","usr_act":json.dumps(usr_act,ensure_ascii=False),
                    "extracted_slots":json.dumps(extracted,ensure_ascii=False),
                    "state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                    "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                    "missing_info":json.dumps(mi2,ensure_ascii=False),
                    "missing_book":json.dumps(mb2,ensure_ascii=False),
                })
                row_id += 1

            st = ds.domains[domain]
            cmp = compare_state_to_goal(domain, st, goal)
            task_success = 1 if cmp["joint_goal_accuracy"] == 1 else 0

            mrow = {
                "conv_id":conv_id,"domain":domain,"task_success":task_success,"num_rows":row_id,
                "repairs":repairs,"confirms":confirms,"oos":oos,
                "slot_precision":round(cmp["slot_precision"],4),
                "slot_recall":round(cmp["slot_recall"],4),
                "slot_f1":round(cmp["slot_f1"],4),
                "state_slot_accuracy":round(cmp["state_slot_accuracy"],4),
                "joint_goal_accuracy":cmp["joint_goal_accuracy"],
                "missing_or_wrong_slots":json.dumps(cmp["missing_or_wrong_slots"],ensure_ascii=False),
                "goal":json.dumps(goal,ensure_ascii=False),
            }
            mw.writerow(mrow)
            metrics_rows.append(mrow)

    # summary
    def mean(xs): return sum(xs)/len(xs) if xs else 0.0
    groups = {}
    for r in metrics_rows:
        groups.setdefault(r["domain"], []).append(r)

    summary_rows = []
    for g, items in groups.items():
        summary_rows.append({
            "group": g,
            "n": len(items),
            "task_success_rate": round(mean([int(x["task_success"]) for x in items]), 4),
            "avg_rows": round(mean([int(x["num_rows"]) for x in items]), 2),
            "avg_repairs": round(mean([int(x["repairs"]) for x in items]), 3),
            "avg_confirms": round(mean([int(x["confirms"]) for x in items]), 3),
            "avg_slot_f1": round(mean([float(x["slot_f1"]) for x in items]), 4),
            "avg_state_slot_acc": round(mean([float(x["state_slot_accuracy"]) for x in items]), 4),
            "joint_goal_accuracy_rate": round(mean([int(x["joint_goal_accuracy"]) for x in items]), 4),
        })
    summary_rows.append({
        "group": "overall",
        "n": len(metrics_rows),
        "task_success_rate": round(mean([int(x["task_success"]) for x in metrics_rows]), 4),
        "avg_rows": round(mean([int(x["num_rows"]) for x in metrics_rows]), 2),
        "avg_repairs": round(mean([int(x["repairs"]) for x in metrics_rows]), 3),
        "avg_confirms": round(mean([int(x["confirms"]) for x in metrics_rows]), 3),
        "avg_slot_f1": round(mean([float(x["slot_f1"]) for x in metrics_rows]), 4),
        "avg_state_slot_acc": round(mean([float(x["state_slot_accuracy"]) for x in metrics_rows]), 4),
        "joint_goal_accuracy_rate": round(mean([int(x["joint_goal_accuracy"]) for x in metrics_rows]), 4),
    })

    with open(summary_out, "w", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(fs, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--turns", default="tod_sim_turns_1000.csv")
    ap.add_argument("--metrics", default="tod_sim_metrics_1000.csv")
    ap.add_argument("--summary", default="tod_sim_summary.csv")
    ap.add_argument("--seed", type=int, default=7)

    # knobs to tune success rate
    ap.add_argument("--p_force_noise_dialog", type=float, default=0.55, help="Chance a dialogue has 1 forced noisy slot")
    ap.add_argument("--p_user_repairs", type=float, default=0.70, help="Chance user performs repair when system confirm is wrong")
    ap.add_argument("--p_nlu_drop", type=float, default=0.08, help="Probability to drop each extracted slot (simulate NLU error)")
    ap.add_argument("--max_sys_steps", type=int, default=10, help="Max system turns per dialogue")

    args = ap.parse_args()

    run_batch(
        n=args.n,
        turns_out=args.turns,
        metrics_out=args.metrics,
        summary_out=args.summary,
        seed=args.seed,
        p_force_noise_dialog=args.p_force_noise_dialog,
        p_user_repairs=args.p_user_repairs,
        p_nlu_drop=args.p_nlu_drop,
        max_sys_steps=args.max_sys_steps
    )
    print(f"Wrote turns:   {args.turns}")
    print(f"Wrote metrics: {args.metrics}")
    print(f"Wrote summary: {args.summary}")

if __name__ == "__main__":
    main()
