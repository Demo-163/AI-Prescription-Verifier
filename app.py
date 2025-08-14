# app.py
# ---------------------------
# AI Prescription Verifier (Offline Demo) – Streamlit single file
# Clean & aligned UI with better visuals.
#
# Run:
#   pip install streamlit pandas numpy
#   streamlit run app.py
# ---------------------------

import re
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Prescription Verifier",
    page_icon="💊",
    layout="wide",
)

# --- CSS for Better UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
    }
    .stTextArea textarea {
        background-color: #c3e0e5;
        border-radius: 10px;
        color: #000000;
    }
    .stButton button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Knowledge Base
# ---------------------------

DRUGS: Dict[str, dict] = {
    "paracetamol": {
        "display": "Paracetamol (Acetaminophen)", "ingredient": "paracetamol",
        "adult_range_mg": (500, 1000), "child_range_mg": (120, 250),
        "contra_child_under": None, "alternatives": ["Tylenol", "Crocin"],
        "forms": ["tablet", "syrup", "capsule"], "notes": "Max single dose 1g in adults (demo)."
                    },
    "ibuprofen": {
        "display": "Ibuprofen", "ingredient": "ibuprofen",
        "adult_range_mg": (200, 400), "child_range_mg": (100, 200),
        "contra_child_under": 6, "alternatives": ["Brufen", "Advil"],
        "forms": ["tablet", "suspension", "capsule"], "notes": "Take with food (demo)."
                  },
    "amoxicillin": {
        "display": "Amoxicillin", "ingredient": "amoxicillin",
        "adult_range_mg": (250, 500), "child_range_mg": (125, 250),
        "contra_child_under": None, "alternatives": ["Amoxil"],
        "forms": ["tablet", "capsule", "suspension"], "notes": "Demo antibiotic ranges."
                    },
    "cetirizine": {
        "display": "Cetirizine", "ingredient": "cetirizine",
        "adult_range_mg": (10, 10), "child_range_mg": (5, 5),
        "contra_child_under": 2, "alternatives": ["Zyrtec"],
        "forms": ["tablet",], "notes": "May cause drowsiness (demo)."
                   },
    "ammonium_chloride": {
        "display": "Ammonium Chloride",
        "ingredient": "ammonium chloride",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Mycodryl Syrup"],
        "forms": ["syrup"],
        "notes": "Auto-generated from dataset, contains Ammonium Chloride (138mg/5ml) ."
    },
    "vildagliptin": {
        "display": "Vildagliptin",
        "ingredient": "vildagliptin",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Vildapin 50mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Vildagliptin (50mg)."
    },
    "propranolol": {
        "display": "Propranolol",
        "ingredient": "propranolol",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Propol 20mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Propranolol (20mg)."
    },
    "cefixime": {
        "display": "Cefixime",
        "ingredient": "cefixime",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["B Cef DS Syrup"],
        "forms": ["syrup"],
        "notes": "Auto-generated from dataset, contains Cefixime (50mg/5ml)."
    },
    "cyclophosphamide": {
        "display": "Cyclophosphamide",
        "ingredient": "cyclophosphamide",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Chophos 200mg Injection"],
        "forms": ["injection"],
        "notes": "Auto-generated from dataset, contains Cyclophosphamide (200mg)."
    },
    "ceftriaxone": {
        "display": "Ceftriaxone",
        "ingredient": "ceftriaxone",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Tazo C 250mg/31.25mg Injection"],
        "forms": ["injection"],
        "notes": "Auto-generated from dataset, contains Ceftriaxone (250mg) ."
    },
    "nimesulide": {
        "display": "Nimesulide",
        "ingredient": "nimesulide",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Nisu 100mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Nimesulide (100mg)."
    },
    "amoxycillin": {
        "display": "Amoxycillin",
        "ingredient": "amoxycillin",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Metclav 1.2gm Injection"],
        "forms": ["injection"],
        "notes": "Auto-generated from dataset, contains Amoxycillin  (1000mg) ."
    },
    "fenofibrate": {
        "display": "Fenofibrate",
        "ingredient": "fenofibrate",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Rosuless-F Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Fenofibrate (160mg) ."
    },
    "ibuprofen": {
        "display": "Ibuprofen",
        "ingredient": "ibuprofen",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Brucet 400 mg/333 mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Ibuprofen (400mg) ."
    },
    "alprazolam": {
        "display": "Alprazolam",
        "ingredient": "alprazolam",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["ALPRAQUIL 0.5 MG TABLET"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Alprazolam (0.5mg)."
    },
    "diclofenac": {
        "display": "Diclofenac",
        "ingredient": "diclofenac",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Decsir-P 50mg/500mg/10mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Diclofenac (50mg) ."
    },
    "sitagliptin": {
        "display": "Sitagliptin",
        "ingredient": "sitagliptin",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Sit DC-M 1000 Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Sitagliptin  (50mg) ."
    },
    "clobazam": {
        "display": "Clobazam",
        "ingredient": "clobazam",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Closum 10mg Tablet MD"],
        "forms": ["md"],
        "notes": "Auto-generated from dataset, contains Clobazam (10mg)."
    },
    "etodolac": {
        "display": "Etodolac",
        "ingredient": "etodolac",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Etudolo MR 400mg/4mg Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Etodolac (400mg) ."
    },
    "caffeine": {
        "display": "Caffeine",
        "ingredient": "caffeine",
        "adult_range_mg": (100, 500),
        "child_range_mg": (50, 250),
        "contra_child_under": None,
        "alternatives": ["Forecold Tablet"],
        "forms": ["tablets"],
        "notes": "Auto-generated from dataset, contains Caffeine (30mg) ."
    }

}


INTERACTIONS: Dict[Tuple[str, str], dict] = {
    tuple(sorted(["ibuprofen", "aspirin"])): {
        "severity": "moderate", "summary": "Ibuprofen may reduce aspirin’s effect (demo).",
        "advice": "Separate dosing and consult provider."},
    tuple(sorted(["paracetamol", "ibuprofen"])): {
        "severity": "minor", "summary": "Often co-used; watch combined dosing schedules (demo).",
        "advice": "Stagger if needed."}
            
}

# ---------------------------
# Parsing & Logic
# ---------------------------
UNIT_PAT = r"(mg|milligram|ml|milliliter|mcg|µg)"
DOSE_PAT = rf"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>{UNIT_PAT})\b"
FORM_WORDS = ["tablet", "cap", "capsule", "syrup", "suspension", "tab"]

KNOWN_NAMES = sorted(set(list(DRUGS.keys()) + sum([d["alternatives"] for d in DRUGS.values()], [])))
NAME_PAT = r"|".join(sorted(map(re.escape, KNOWN_NAMES), key=len, reverse=True))

def normalize_drug_name(raw: str) -> Optional[str]:
    s = raw.lower().strip()
    for key, meta in DRUGS.items():
        if s == key or s == meta["ingredient"] or s in [a.lower() for a in meta["alternatives"]]:
            return key
    return None

def parse_prescription(text: str) -> List[dict]:
    results = []
    if not text: return results
    chunks = re.split(r"[;\n,.]+", text, flags=re.I)
    for chunk in chunks:
        if not chunk.strip(): continue
        name_match = re.search(rf"\b({NAME_PAT})\b", chunk, flags=re.I)
        if not name_match: continue
        key = normalize_drug_name(name_match.group(1))
        if not key: continue
        dose_mg, unit = None, None
        dose_match = re.search(DOSE_PAT, chunk, flags=re.I)
        if dose_match:
            unit = dose_match.group("unit").lower()
            val = float(dose_match.group("dose"))
            if unit in ("mg", "milligram"): dose_mg = val
        form = next((fw for fw in FORM_WORDS if re.search(rf"\b{fw}\b", chunk, re.I)), "")
        results.append({"key": key, "name": DRUGS[key]["display"], "dose_mg": dose_mg, "form": form})
    return results

def dosage_check(row: dict, age: int) -> dict:
    meta = DRUGS[row["key"]]
    ag = "adult" if age >= 12 else "child"
    lo, hi = meta[f"{ag}_range_mg"]
    if row["dose_mg"] is None: return {"status": "unknown", "explanation": "Dose missing."}
    if row["dose_mg"] < lo: return {"status": "low", "explanation": f"Below typical {ag} range {lo}-{hi} mg"}
    if row["dose_mg"] > hi: return {"status": "high", "explanation": f"Above typical {ag} range {lo}-{hi} mg"}
    return {"status": "ok", "explanation": f"Within typical {ag} range {lo}-{hi} mg"}

def find_interactions(keys: List[str]) -> List[dict]:
    findings = []
    for a, b in combinations(sorted(set(keys)), 2):
        pair = tuple(sorted([a, b]))
        if pair in INTERACTIONS:
            info = INTERACTIONS[pair]
            findings.append({"drug_a": DRUGS[a]["display"], "drug_b": DRUGS[b]["display"], **info})
    return findings

# ---------------------------
# UI
# ---------------------------
st.markdown("<h1 style='text-align: center;'>💊 AI Prescription Verifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Educational use only.</p>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("🧍 Patient Info")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    st.subheader("📋 Example Prescription")
    example = "Paracetamol 500 mg tablet, Ibuprofen 200 mg tablet, Cetirizine 10 mg"
    if st.button("Use Example"): st.session_state.prescription_text = example

with col1:
    st.subheader("📝 Enter Prescription")
    text = st.text_area("Type or paste your prescription here:", key="prescription_text", height=150)

st.write("")
if st.button("🔍 Analyze Prescription"):
    meds = parse_prescription(text)
    if not meds:
        st.error("No medicines detected. Please check spelling or try example.")
    else:
        rows = []
        for m in meds:
            check = dosage_check(m, age)
            rows.append({"Medicine": m["name"], "Dose (mg)": m["dose_mg"], "Form": m["form"],
                         "Status": check["status"], "Details": check["explanation"]})
        st.success("✅ Analysis complete!")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        interactions = find_interactions([m["key"] for m in meds])
        if interactions:
            st.warning("⚠️ Potential Interactions Found")
            st.dataframe(pd.DataFrame(interactions), use_container_width=True)
        else:
            st.info("No interactions detected in demo database.")

st.markdown("---")
st.caption("⚠️ This tool is for demonstration purposes only and not a substitute for professional medical advice.")
