#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HARD TOD DEMO + TRAINING (Imitation Learning) - v4 (hard defaults ~85% success)

Hardness knobs (defaults tuned for failures):
- Higher NLU drop/confuse
- More OOS/smalltalk/refuse
- Lower repair probability
- Smaller max_steps

Run:
  python tod_hard_sim_and_train_v4.py --n 1000 --seed 7
"""

from __future__ import annotations
import re, json, csv, argparse, random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

SKLEARN_OK = True
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report
except Exception:
    SKLEARN_OK = False

def set_seed(seed: int):
    random.seed(seed)

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

REQUIRED_SLOTS = {
    "movie": {"info": ["movie", "theater"], "book": ["date", "time", "tickets"]},
    "navigation": {"info": ["origin", "destination", "mode"], "book": []},
}
PRIORITY = {
    "movie": ["movie", "theater", "date", "time", "tickets"],
    "navigation": ["destination", "origin", "mode", "avoid"],
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
        return (f"Mình chốt: phim {info.get('movie','?')}, rạp {info.get('theater','?')}, "
                f"ngày {book.get('date','?')}, giờ {book.get('time','?')}, {book.get('tickets','?')} vé. Đúng không?")
    return (f"Mình chốt: từ {info.get('origin','?')} đến {info.get('destination','?')}, "
            f"đi {info.get('mode','?')}. Đúng không?")

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

# NLU
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

def extract_slots(user_text: str) -> Dict[str, Any]:
    tl = norm(user_text)
    slots: Dict[str, Any] = {}

    if "hôm nay" in tl:
        slots["date"] = "today"
    elif re.search(r"\bmai\b", tl):
        slots["date"] = "tomorrow"
    elif "cuối tuần" in tl or "cuoi tuan" in tl:
        slots["date"] = "weekend"

    tm = extract_time(tl)
    if tm:
        slots["time"] = tm
    else:
        for s in SHOWTIMES:
            if s in tl:
                slots["time"] = s

    m = re.search(r"\b(\d{1,2})\s*(vé|ve)\b", tl)
    if m:
        slots["tickets"] = int(m.group(1))

    for th in THEATERS:
        if th.lower() in tl:
            slots["theater"] = th

    m = re.search(r"\bphim\s+([a-z0-9\- ]+)", tl)
    if m:
        name = m.group(1).strip()
        name = re.split(r"\b(ở|o|lúc|luc|ngày|ngay|rạp|rap)\b", name)[0].strip()
        if name:
            slots["movie"] = name

    if "xe buýt" in tl or "xe buyt" in tl or "bus" in tl:
        slots["mode"] = "bus"
    elif "đi bộ" in tl or "di bo" in tl or "walk" in tl:
        slots["mode"] = "walk"
    elif "xe máy" in tl or "xe may" in tl:
        slots["mode"] = "motorbike"
    elif "ô tô" in tl or "oto" in tl or "car" in tl:
        slots["mode"] = "car"

    if "tránh kẹt" in tl or "tranh ket" in tl:
        slots["avoid"] = "traffic"
    elif "tránh cao tốc" in tl or "tranh cao toc" in tl:
        slots["avoid"] = "highway"
    elif "tránh phà" in tl or "tranh pha" in tl:
        slots["avoid"] = "ferry"

    m = re.search(r"\b(từ|tu)\s+(.+?)\s+\b(đến|den|tới|toi)\s+(.+)$", user_text, flags=re.IGNORECASE)
    if m:
        slots["origin"] = m.group(2).strip()
        slots["destination"] = m.group(4).strip()
    else:
        m = re.search(r"\b(đến|den|tới|toi|đi tới|di toi)\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            slots["destination"] = m.group(2).strip()
        m = re.search(r"\b(từ|tu)\s+(.+)$", user_text, flags=re.IGNORECASE)
        if m:
            slots["origin"] = m.group(2).strip()

    if "origin" in slots:
        p = snap_place(tl)
        if p:
            slots["origin"] = p
    if "destination" in slots:
        p = snap_place(tl)
        if p:
            slots["destination"] = p

    return slots

def nlu_apply_noise(slots: Dict[str, Any], p_drop: float, p_confuse: float) -> Dict[str, Any]:
    out = dict(slots)
    for k in list(out.keys()):
        if random.random() < p_drop:
            out.pop(k, None)
    for k, v in list(out.items()):
        if random.random() < p_confuse:
            vs = str(v)
            if vs in CONFUSABLE:
                out[k] = CONFUSABLE[vs]
    return out

# DST + Oracle
def update_state(ds: DialogState, domain: str, extracted: Dict[str, Any]):
    ds.ensure(domain)
    st = ds.domains[domain]
    book_slots = set(REQUIRED_SLOTS[domain]["book"]) | {"tickets", "date", "time"}
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

def oracle_next(domain: str, ds: DialogState) -> str:
    mi, mb = missing_slots(ds, domain)
    cand = mi + mb
    if domain == "navigation" and ("avoid" not in ds.domains[domain].info) and random.random() < 0.30:
        core_missing = [s for s in REQUIRED_SLOTS[domain]["info"] if s not in ds.domains[domain].info]
        if not core_missing:
            return "avoid"
    if not cand:
        return "CONFIRM"
    for p in PRIORITY[domain]:
        if p in cand:
            return p
    return cand[0]

def system_step(domain: str, ds: DialogState, action_label: str) -> Tuple[str, str, List[str], List[str]]:
    mi, mb = missing_slots(ds, domain)
    if action_label == "CONFIRM":
        st = ds.domains[domain]
        return f"CONFIRM({domain})", nlg_confirm(domain, st.info, st.book), mi, mb
    return f"REQUEST({domain}.{action_label})", ASK_TEMPLATES.get(action_label, f"Bạn cho mình biết {action_label}?"), mi, mb

# User
SMALLTALK = ["Cảm ơn nha!", "Ok ok.", "Hi bot!", "Cho mình hỏi chút.", "Ủa alo?"]
OOS = ["Bạn biết thời tiết hôm nay không?", "Tư vấn mua laptop giúp mình?", "Kể chuyện vui đi.", "Bạn là ai vậy?"]

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
    if random.random() < 0.30:
        goal["avoid"] = random.choice(["traffic", "highway", "ferry"])
    return goal

def value_to_utter(domain: str, slot: str, value: Any) -> str:
    if domain == "movie":
        if slot == "movie": return f"Mình muốn đặt vé phim {value}."
        if slot == "theater": return f"{value}."
        if slot == "date": return {"today":"Hôm nay.","tomorrow":"Mai.","weekend":"Cuối tuần."}.get(value, str(value))
        if slot == "time": return f"{value}"
        if slot == "tickets": return f"{value} vé"
    if slot == "destination": return f"Đi tới {value}."
    if slot == "origin": return f"Từ {value}."
    if slot == "mode":
        return {"walk":"Đi bộ.","bus":"Đi bus.","motorbike":"Đi xe máy.","car":"Đi ô tô."}.get(value, str(value))
    if slot == "avoid":
        return {"traffic":"Tránh kẹt xe.","highway":"Tránh cao tốc.","ferry":"Tránh phà."}.get(value, str(value))
    return str(value)

def build_repair_utter(domain: str, goal: Dict[str, Any], wrong_slots: List[str]) -> Tuple[str, Dict[str, Any]]:
    parts = ["Không đúng, mình sửa lại:"]
    out = {}
    for s in wrong_slots:
        if s in goal:
            parts.append(value_to_utter(domain, s, goal[s]))
            out[s] = goal[s]
    return " ".join(parts), {"act":"deny+inform","slots":out}

def user_reply(domain: str, goal: Dict[str, Any], sys_act: str, asked_slot: Optional[str],
               p_multi: float, p_refuse: float, p_smalltalk: float, p_oos: float) -> Tuple[str, Dict[str, Any]]:
    r = random.random()
    if r < p_oos:
        return random.choice(OOS), {"act":"oos","slots":{}}
    if r < p_oos + p_smalltalk:
        return random.choice(SMALLTALK), {"act":"smalltalk","slots":{}}
    if asked_slot and random.random() < p_refuse:
        return "Mình không tiện nói, hỏi cái khác được không?", {"act":"refuse","slots":{}}

    if sys_act.startswith("REQUEST(") and asked_slot:
        provided = [asked_slot]
        if random.random() < p_multi:
            req_all = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
            others = [s for s in req_all if s != asked_slot]
            random.shuffle(others)
            provided += others[: random.randint(0, 2)]

        parts = []
        out = {}
        for s in provided:
            if s in goal:
                parts.append(value_to_utter(domain, s, goal[s]))
                out[s] = goal[s]
        return " ".join(parts) if parts else "Mình chưa rõ.", {"act":"inform","slots":out}

    return "OK", {"act":"confirm","slots":{}}

# Eval
def compare_state_to_goal(domain: str, st: DomainState, goal: Dict[str, Any]) -> Dict[str, Any]:
    req = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
    pred = {}
    pred.update(st.info); pred.update(st.book)
    correct=0
    for s in req:
        if s in pred and pred[s] == goal.get(s):
            correct += 1
    joint = 1 if correct == len(req) else 0
    state_acc = correct/len(req) if req else 1.0
    wrong = [s for s in req if (s not in pred) or (pred[s] != goal.get(s))]
    tp = correct
    fp = sum(1 for s in req if s in pred and pred[s] != goal.get(s))
    fn = len(wrong)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {"slot_f1": f1, "state_acc": state_acc, "joint": joint, "wrong": wrong}

# Features
def make_feature_row(domain: str, ds: DialogState, turn: int, last_sys_type: str) -> Dict[str, Any]:
    ds.ensure(domain)
    st = ds.domains[domain]
    req = REQUIRED_SLOTS[domain]["info"] + REQUIRED_SLOTS[domain]["book"]
    row = {"domain": domain, "turn": turn, "last_sys_type": last_sys_type}
    for s in req:
        row[f"has_{s}"] = int((s in st.info) or (s in st.book))
    row["has_avoid"] = int("avoid" in st.info)
    return row

# Simulation
def run_sim(n: int, seed: int, turns_out: str, metrics_out: str, summary_out: str, train_out: str,
            policy_mode: str, pipeline=None, feature_cols: Optional[List[str]]=None,
            p_drop: float=0.22, p_confuse: float=0.12, max_steps: int=7,
            p_refuse: float=0.14, p_smalltalk: float=0.08, p_oos: float=0.08, p_multi: float=0.55,
            p_repair: float=0.55):
    set_seed(seed)
    turn_fields = ["conv_id","domain","row_id","speaker","utterance","sys_act","usr_act","asked_slot",
                   "extracted_slots","state_info","state_book","missing_info","missing_book"]
    metric_fields = ["conv_id","domain","task_success","num_rows","repairs","confirms","oos","refusals",
                     "slot_f1","state_slot_accuracy","joint_goal_accuracy","wrong_slots","goal"]

    train_rows = []

    with open(turns_out, "w", newline="", encoding="utf-8") as ft, \
         open(metrics_out, "w", newline="", encoding="utf-8") as fm:
        tw = csv.DictWriter(ft, fieldnames=turn_fields); tw.writeheader()
        mw = csv.DictWriter(fm, fieldnames=metric_fields); mw.writeheader()

        metrics_rows = []

        for i in range(n):
            domain = "movie" if random.random() < 0.5 else "navigation"
            goal = sample_goal(domain)
            ds = DialogState(); ds.ensure(domain)

            conv_id = f"{policy_mode}_{domain}_{i:05d}"
            row_id = 0
            repairs=confirms=oos=refusals=0
            last_sys_type = "NONE"

            for step in range(max_steps):
                if policy_mode == "oracle":
                    label = oracle_next(domain, ds)
                else:
                    feat = make_feature_row(domain, ds, step, last_sys_type)
                    xdf = pd.DataFrame([feat])
                    if feature_cols is not None:
                        xdf = xdf.reindex(columns=feature_cols)
                    label = pipeline.predict(xdf)[0]
                    if label != "CONFIRM":
                        mi, mb = missing_slots(ds, domain)
                        if label not in (mi+mb) and not (domain=="navigation" and label=="avoid"):
                            label = oracle_next(domain, ds)

                if policy_mode == "oracle":
                    feat = make_feature_row(domain, ds, step, last_sys_type)
                    feat["label"] = label
                    train_rows.append(feat)

                sys_act, sys_text, mi, mb = system_step(domain, ds, label)
                asked_slot = ""
                if sys_act.startswith("REQUEST("):
                    asked_slot = sys_act[len("REQUEST("):-1].split(".",1)[1]
                    last_sys_type = "REQUEST"
                else:
                    confirms += 1
                    last_sys_type = "CONFIRM"

                tw.writerow({
                    "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"system",
                    "utterance":sys_text,"sys_act":sys_act,"usr_act":"","asked_slot":asked_slot,
                    "extracted_slots":"",
                    "state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                    "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                    "missing_info":json.dumps(mi,ensure_ascii=False),
                    "missing_book":json.dumps(mb,ensure_ascii=False),
                })
                row_id += 1

                if sys_act.startswith("CONFIRM"):
                    st = ds.domains[domain]
                    cmp = compare_state_to_goal(domain, st, goal)
                    if cmp["wrong"]:
                        if random.random() < p_repair:
                            wrong = list(cmp["wrong"]); random.shuffle(wrong)
                            pick = wrong[: random.randint(1, min(2, len(wrong)))]
                            user_text, usr_act = build_repair_utter(domain, goal, pick)
                            repairs += 1
                        else:
                            user_text, usr_act = "OK", {"act":"confirm","slots":{}}
                    else:
                        user_text, usr_act = "OK", {"act":"confirm","slots":{}}
                        extracted = nlu_apply_noise(extract_slots(user_text), p_drop, p_confuse)
                        tw.writerow({
                            "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"user",
                            "utterance":user_text,"sys_act":"","usr_act":json.dumps(usr_act,ensure_ascii=False),
                            "asked_slot":"",
                            "extracted_slots":json.dumps(extracted,ensure_ascii=False),
                            "state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                            "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                            "missing_info":json.dumps(mi,ensure_ascii=False),
                            "missing_book":json.dumps(mb,ensure_ascii=False),
                        })
                        row_id += 1
                        break
                else:
                    user_text, usr_act = user_reply(domain, goal, sys_act, asked_slot,
                                                   p_multi=p_multi, p_refuse=p_refuse,
                                                   p_smalltalk=p_smalltalk, p_oos=p_oos)

                extracted = nlu_apply_noise(extract_slots(user_text), p_drop, p_confuse)
                if usr_act.get("act") == "oos": oos += 1
                if usr_act.get("act") == "refuse": refusals += 1
                update_state(ds, domain, extracted)

                mi2, mb2 = missing_slots(ds, domain)
                tw.writerow({
                    "conv_id":conv_id,"domain":domain,"row_id":row_id,"speaker":"user",
                    "utterance":user_text,"sys_act":"","usr_act":json.dumps(usr_act,ensure_ascii=False),
                    "asked_slot":"",
                    "extracted_slots":json.dumps(extracted,ensure_ascii=False),
                    "state_info":json.dumps(ds.domains[domain].info,ensure_ascii=False),
                    "state_book":json.dumps(ds.domains[domain].book,ensure_ascii=False),
                    "missing_info":json.dumps(mi2,ensure_ascii=False),
                    "missing_book":json.dumps(mb2,ensure_ascii=False),
                })
                row_id += 1

            st = ds.domains[domain]
            cmp = compare_state_to_goal(domain, st, goal)
            task_success = 1 if cmp["joint"] == 1 else 0
            mrow = {
                "conv_id":conv_id,"domain":domain,"task_success":task_success,"num_rows":row_id,
                "repairs":repairs,"confirms":confirms,"oos":oos,"refusals":refusals,
                "slot_f1":round(cmp["slot_f1"],4),
                "state_slot_accuracy":round(cmp["state_acc"],4),
                "joint_goal_accuracy":cmp["joint"],
                "wrong_slots":json.dumps(cmp["wrong"],ensure_ascii=False),
                "goal":json.dumps(goal,ensure_ascii=False),
            }
            mw.writerow(mrow)
            metrics_rows.append(mrow)

    if policy_mode == "oracle":
        cols = sorted({k for r in train_rows for k in r.keys()})
        cols = [c for c in cols if c != "label"] + ["label"]
        with open(train_out, "w", newline="", encoding="utf-8") as ftr:
            w = csv.DictWriter(ftr, fieldnames=cols)
            w.writeheader()
            for r in train_rows:
                w.writerow(r)

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
            "avg_oos": round(mean([int(x["oos"]) for x in items]), 3),
            "avg_refusals": round(mean([int(x["refusals"]) for x in items]), 3),
            "avg_slot_f1": round(mean([float(x["slot_f1"]) for x in items]), 4),
            "joint_goal_acc_rate": round(mean([int(x["joint_goal_accuracy"]) for x in items]), 4),
        })
    summary_rows.append({
        "group":"overall",
        "n": len(metrics_rows),
        "task_success_rate": round(mean([int(x["task_success"]) for x in metrics_rows]), 4),
        "avg_rows": round(mean([int(x["num_rows"]) for x in metrics_rows]), 2),
        "avg_repairs": round(mean([int(x["repairs"]) for x in metrics_rows]), 3),
        "avg_oos": round(mean([int(x["oos"]) for x in metrics_rows]), 3),
        "avg_refusals": round(mean([int(x["refusals"]) for x in metrics_rows]), 3),
        "avg_slot_f1": round(mean([float(x["slot_f1"]) for x in metrics_rows]), 4),
        "joint_goal_acc_rate": round(mean([int(x["joint_goal_accuracy"]) for x in metrics_rows]), 4),
    })

    with open(summary_out, "w", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(fs, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

# Training
def train_policy(train_csv: str, report_json: str):
    if not SKLEARN_OK:
        rep = {"sklearn_available": False}
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        return None, None, rep

    df = pd.read_csv(train_csv)
    cat_cols = ["domain", "last_sys_type"]
    for c in cat_cols:
        df[c] = df[c].fillna("NONE").astype(str)

    y = df["label"].fillna("CONFIRM").astype(str)
    X = df.drop(columns=["label"])

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = X[c].fillna(0)

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))]), num_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=400, solver="lbfgs", multi_class="auto")
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    rep = {
        "sklearn_available": True,
        "n_samples": int(len(df)),
        "n_labels": int(y.nunique()),
        "val_accuracy": acc,
        "labels": sorted(y.unique().tolist()),
        "feature_columns": list(X.columns),
        "classification_report": classification_report(y_val, y_pred, output_dict=True, zero_division=0),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return pipe, list(X.columns), rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--turns", default="tod_hard_turns_v4.csv")
    ap.add_argument("--metrics", default="tod_hard_metrics_v4.csv")
    ap.add_argument("--summary", default="tod_hard_summary_v4.csv")

    ap.add_argument("--train_csv", default="tod_policy_train_v4.csv")
    ap.add_argument("--policy_report", default="tod_policy_report_v4.json")

    ap.add_argument("--learned_turns", default="tod_hard_turns_learned_v4.csv")
    ap.add_argument("--learned_metrics", default="tod_hard_metrics_learned_v4.csv")
    ap.add_argument("--learned_summary", default="tod_hard_summary_learned_v4.csv")

    # hardness knobs
    ap.add_argument("--p_nlu_drop", type=float, default=0.22)
    ap.add_argument("--p_nlu_confuse", type=float, default=0.12)
    ap.add_argument("--max_steps", type=int, default=7)
    ap.add_argument("--p_refuse", type=float, default=0.14)
    ap.add_argument("--p_smalltalk", type=float, default=0.08)
    ap.add_argument("--p_oos", type=float, default=0.08)
    ap.add_argument("--p_multi", type=float, default=0.55)
    ap.add_argument("--p_repair", type=float, default=0.55)

    args = ap.parse_args()

    run_sim(args.n, args.seed, args.turns, args.metrics, args.summary, args.train_csv,
            policy_mode="oracle",
            p_drop=args.p_nlu_drop, p_confuse=args.p_nlu_confuse, max_steps=args.max_steps,
            p_refuse=args.p_refuse, p_smalltalk=args.p_smalltalk, p_oos=args.p_oos, p_multi=args.p_multi,
            p_repair=args.p_repair)

    pipe, feature_cols, rep = train_policy(args.train_csv, args.policy_report)

    if pipe is not None:
        run_sim(args.n, args.seed + 1, args.learned_turns, args.learned_metrics, args.learned_summary, args.train_csv,
                policy_mode="learned", pipeline=pipe, feature_cols=feature_cols,
                p_drop=args.p_nlu_drop, p_confuse=args.p_nlu_confuse, max_steps=args.max_steps,
                p_refuse=args.p_refuse, p_smalltalk=args.p_smalltalk, p_oos=args.p_oos, p_multi=args.p_multi,
                p_repair=args.p_repair)

    print("DONE")

if __name__ == "__main__":
    main()
