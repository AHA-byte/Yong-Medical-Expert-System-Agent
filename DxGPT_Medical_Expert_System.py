import os
import re
import json
import uuid
import requests
from typing import Dict, List, Any, Tuple

import streamlit as st

# --- Config ---
from dotenv import load_dotenv
load_dotenv()
#using DxGpt
DXGPT_SUBSCRIPTION_KEY = os.getenv("DXGPT_SUBSCRIPTION_KEY")
DXGPT_BASE_URL = (os.getenv("DXGPT_BASE_URL") or "https://dxgpt-apim.azure-api.net/api").rstrip("/")

SYSTEM_PROMPT = """You are a cautious clinical triage assistant.
You must NOT provide a medical diagnosis.
Given 4–10 comma-separated symptoms, return the TOP 3 likely DIFFERENTIAL diagnoses as JSON.
Rules:
- Be concise, evidence-based, adult general population by default.
- Include a calibrated probability percentage for EACH diagnosis that sums to 100.
- Add a one-sentence rationale per item.
- Add a short red_flags note advising urgent care if any are present.
- Never claim certainty. Never give treatment plans or drug dosing.
Output JSON ONLY with this schema:
{
  "diagnoses": [
    {"name": string, "probability_percent": number, "rationale": string},
    {"name": string, "probability_percent": number, "rationale": string},
    {"name": string, "probability_percent": number, "rationale": string}
  ],
  "red_flags": string,
  "disclaimer": "This is not medical advice. Seek professional evaluation."
}
"""

# --- Helpers for normalization / cleaning ---

_PUNCT_RE = re.compile(r"[,\.\(\)\[\]\{\};:!?\-_/]+")

def _norm(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (for robust comparisons)."""
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _normalize_symptom_list(symptoms_text: str) -> List[str]:
    """
    Prefer semicolons or newlines as separators; keep commas inside a symptom.
    If the user only uses commas, fall back to comma-split but keep long adjectival phrases intact.
    """
    text = (symptoms_text or "").strip()
    if not text:
        return []

    if ("\n" in text) or (";" in text):
        parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
    else:
        # Fall back to comma-split; if it looks like one descriptive phrase, keep it as one.
        comma_parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(comma_parts) >= 3 and sum(len(p) <= 5 for p in comma_parts) >= 2:
            parts = [" ".join(comma_parts)]
        else:
            parts = comma_parts

    # de-duplicate (case/punct insensitive) while preserving original phrasing
    seen, out = set(), []
    for p in parts:
        k = _norm(p)
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out

def _clean_overlap(matches: List[str], mismatches: List[str], user_symptoms: List[str]) -> Tuple[List[str], List[str]]:
    """
    - Deduplicate matches/mismatches (case/punct insensitive).
    - Remove any mismatch that equals/contains/is contained by any match.
    - If API provided no mismatches, compute them from user symptoms, then apply the same filter.
    """
    # Dedup matches
    m_seen, m_dedup = set(), []
    for m in (matches or []):
        n = _norm(m)
        if n and n not in m_seen:
            m_seen.add(n)
            m_dedup.append(m)

    # Base mismatches: API list or user symptoms not listed as matches
    if mismatches:
        raw_mm = mismatches
    else:
        m_norms = set(_norm(x) for x in m_dedup)
        raw_mm = [s for s in (user_symptoms or []) if _norm(s) not in m_norms]

    # Remove any mismatch overlapping with a match (identical, substring, superstring)
    m_norm_list = [_norm(x) for x in m_dedup]
    mm_seen, mm_clean = set(), []
    for nt in raw_mm:
        ntn = _norm(nt)
        if not ntn or ntn in mm_seen:
            continue
        overlaps = any(ntn in mn or mn in ntn for mn in m_norm_list)
        if not overlaps:
            mm_seen.add(ntn)
            mm_clean.append(nt)

    return m_dedup, mm_clean

# --- Logic ---

def _heuristic_probabilities(items: List[Dict], original_symptoms: List[str]) -> List[float]:
    scores = []
    for it in items:
        matches = len((it.get("symptoms_in_common") or it.get("matching_symptoms") or []) or [])
        mismatches = len((it.get("symptoms_not_in_common") or it.get("non_matching_symptoms") or []) or [])
        score = max(0, matches - mismatches)
        scores.append(score)
    if not any(scores):
        n = max(1, len(items))
        return [round(100.0 / n, 1)] * n
    total = float(sum(scores))
    probs = [round((s / total) * 100.0, 1) for s in scores]
    drift = round(100.0 - sum(probs), 1)
    if probs:
        probs[0] = round(probs[0] + drift, 1)
    return probs

def call_dxgpt_diagnose(description: str) -> Any:
    url = f"{DXGPT_BASE_URL}/diagnose"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": DXGPT_SUBSCRIPTION_KEY
    }
    payload = {
        "description": description,
        "myuuid": str(uuid.uuid4()),
        "timezone": "Asia/Karachi",
        "lang": "en",
        "model": "gpt4o",
        "response_mode": "direct"
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"DXGPT error {resp.status_code}: {resp.text}") from e
    return resp.json()

def _extract_items(dxgpt: Any) -> List[Dict]:
    if isinstance(dxgpt, list):
        return dxgpt
    if isinstance(dxgpt, dict):
        data = dxgpt.get("data")
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data, list):
            return data
        if isinstance(dxgpt.get("diagnoses"), list):
            return dxgpt["diagnoses"]
        if any(k in dxgpt for k in ("diagnosis", "name", "disease")):
            return [dxgpt]
    return []

def _build_context_section(
    age_group: str,
    sex: str,
    duration: str,
    onset: str,
    fever: str,
    comorbid: str,
    meds: str,
    allergies: str,
    smoking: str,
    travel: str,
    pregnancy: str,
    noticed: str = ""
) -> str:
    fields = []
    if age_group:  fields.append(f"Age group: {age_group}")
    if sex:        fields.append(f"Sex: {sex}")
    if duration:   fields.append(f"Duration: {duration}")
    if onset:      fields.append(f"Onset: {onset}")
    if fever:      fields.append(f"Fever: {fever}")
    if comorbid:   fields.append(f"Previous illnesses/comorbidities: {comorbid}")
    if meds:       fields.append(f"Current medications: {meds}")
    if allergies:  fields.append(f"Allergies: {allergies}")
    if smoking:    fields.append(f"Smoking status: {smoking}")
    if travel:     fields.append(f"Recent travel or exposure: {travel}")
    if pregnancy:  fields.append(f"Pregnancy status: {pregnancy}")
    if noticed and noticed.strip():
        fields.append(f"Symptoms first noticed: {noticed.strip()}")
    return " ".join(fields)

def get_top3_differentials_with_mismatch(symptoms_csv: str, context: str, optional_note: str = "") -> dict:
    user_symptoms = _normalize_symptom_list(symptoms_csv)
    base = f"Adult patient with the following symptoms: {', '.join(user_symptoms) if user_symptoms else symptoms_csv.strip()}."
    desc_parts = [base]
    if context.strip():
        desc_parts.append(context.strip())
    if optional_note.strip():
        desc_parts.append(f"Additional patient note: {optional_note.strip()}")
    description = " ".join(desc_parts)

    dxgpt = call_dxgpt_diagnose(description)
    items = _extract_items(dxgpt)

    if not items and isinstance(dxgpt, dict) and dxgpt.get("result") != "success":
        msg = dxgpt.get("message") or dxgpt.get("error") or "Unknown error"
        raise RuntimeError(f"DXGPT returned non-success: {msg}")

    top = items[:3] if len(items) >= 3 else items
    if not top:
        return {
            "diagnoses": [],
            "red_flags": "None reported by the model.",
            "disclaimer": "This is not medical advice. Seek professional evaluation."
        }

    probs = _heuristic_probabilities(top, user_symptoms)
    diagnoses_fmt = []
    for it, p in zip(top, probs):
        if p > 0:
            name = it.get("diagnosis") or it.get("name") or it.get("disease") or "Unspecified"
            api_matches = (it.get("symptoms_in_common") or it.get("matching_symptoms") or []) or []
            api_mismatches = (it.get("symptoms_not_in_common") or it.get("non_matching_symptoms") or []) or []

            # NEW: strong overlap cleaning
            matches, not_typical = _clean_overlap(api_matches, api_mismatches, user_symptoms)

            rationale_bits = []
            if matches:
                rationale_bits.append(f"matches: {', '.join(matches[:3])}")
            if not_typical:
                rationale_bits.append(f"not typical: {', '.join(not_typical[:2])}")
            rationale = "; ".join(rationale_bits) or "Pattern partially overlaps the provided symptoms."

            diagnoses_fmt.append({
                "name": name,
                "probability_percent": float(p),
                "rationale": rationale,
                "matching_symptoms": matches,
                "not_typical_symptoms": not_typical
            })

    diagnoses_fmt = diagnoses_fmt[:3]

    red_flags = "Seek urgent care for severe chest pain, confusion, unilateral weakness, severe shortness of breath, or rapid deterioration."

    return {
        "diagnoses": diagnoses_fmt,
        "red_flags": red_flags,
        "disclaimer": "This is not medical advice. Seek professional evaluation."
    }

#streamlit

st.set_page_config(page_title="Yong Differential Helper (Minimal)", page_icon="🩺", layout="centered")
st.title("🩺 Yong Differential Helper (Minimal)")
st.caption("For educational triage support only. Not medical advice.")

symptoms_csv = st.text_area(
    "Enter 4–10 symptoms (use semicolons or new lines to separate; commas can be part of a symptom).",
    placeholder="e.g., dark velvety skin in body folds; weight gain; skin tags"
)

with st.expander("Optional clinical context (click to expand)"):
    col1, col2 = st.columns(2)
    with col1:
        age_group = st.selectbox("Age group", ["", "Infant", "Child", "Adolescent", "Adult", "Older adult"], index=4)
        sex = st.selectbox("Sex", ["", "Female", "Male", "Intersex", "Prefer not to say"], index=0)
        duration = st.text_input("Symptom duration (e.g., 3 days, 2 weeks)")
        onset = st.selectbox("Onset", ["", "Sudden", "Gradual"])
        fever = st.selectbox("Fever", ["", "Yes", "No", "Unknown"])
        pregnancy = st.selectbox("Pregnancy status", ["", "Pregnant", "Not pregnant", "Unknown"])
    with col2:
        comorbid = st.text_input("Previous illnesses or comorbidities")
        meds = st.text_input("Current medications")
        allergies = st.text_input("Allergies")
        smoking = st.selectbox("Smoking status", ["", "Never", "Former", "Current"])
        travel = st.text_input("Recent travel or exposure")
        noticed = st.text_input("When did you notice symptoms")

optional_note = st.text_input(
    "Optional Note",
    placeholder="e.g., Already saw a doctor yesterday, suspected common cold.",
    key="optional_note",
)

left, right = st.columns([1, 1])
with left:
    run_btn = st.button("Give me differential")
with right:
    st.write("")

if run_btn:
    if not symptoms_csv.strip():
        st.error("Please enter at least a few symptoms.")
    else:
        try:
            context = _build_context_section(
                age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed
            )
            with st.spinner("Querying DxGPT..."):
                result = get_top3_differentials_with_mismatch(symptoms_csv, context, optional_note)

            diags = result.get("diagnoses", [])
            if not diags:
                st.warning("No results returned.")
            else:
                st.subheader("Top differentials")
                rows = [
                    {
                        "Diagnosis": d["name"],
                        "Probability %": d.get("probability_percent", 0.0),
                        "Rationale": d.get("rationale", "")
                    }
                    for d in diags
                ]
                st.dataframe(rows, use_container_width=True)

                for i, d in enumerate(diags, start=1):
                    with st.expander(f"{i}. {d['name']}"):
                        st.markdown(f"**Estimated probability:** {d.get('probability_percent', 0.0)}%")
                        matches = d.get("matching_symptoms") or []
                        not_typical = d.get("not_typical_symptoms") or []

                        st.markdown("**Matching symptoms:** " + (", ".join(matches) if matches else "none reported"))
                        st.markdown("**Your entered symptoms not typical for this illness:** " + (", ".join(not_typical) if not_typical else "none reported"))
                        st.markdown("**Rationale:** " + d.get("rationale", ""))

                st.info("Red flags: " + result.get("red_flags", ""))
                st.caption(result.get("disclaimer", ""))

        except Exception as ex:
            st.error(str(ex))
            st.stop()

st.caption("This tool is not a substitute for a real Doctor.")
