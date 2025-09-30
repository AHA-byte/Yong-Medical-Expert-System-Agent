import os
import json
import uuid
import requests
from typing import Dict, List, Any, Tuple

import streamlit as st

#Config 
from dotenv import load_dotenv
load_dotenv()
DXGPT_SUBSCRIPTION_KEY = os.getenv("DXGPT_SUBSCRIPTION_KEY")
DXGPT_BASE_URL = os.getenv("DXGPT_BASE_URL")

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
#Logic
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

def _normalize_symptom_list(symptoms_csv: str) -> List[str]:
    return [s.strip() for s in symptoms_csv.split(",") if s.strip()]

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
    symptoms_list = _normalize_symptom_list(symptoms_csv)
    base = f"Adult patient with the following symptoms: {', '.join(symptoms_list)}."
    # keep context separate from the optional note
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

    # Only keep diagnoses with probability > 0, up to 3
    top = items[:3] if len(items) >= 3 else items
    if not top:
        return {
            "diagnoses": [],
            "red_flags": "None reported by the model.",
            "disclaimer": "This is not medical advice. Seek professional evaluation."
        }

    probs = _heuristic_probabilities(top, symptoms_list)
    diagnoses_fmt = []
    for it, p in zip(top, probs):
        if p > 0:
            name = it.get("diagnosis") or it.get("name") or it.get("disease") or "Unspecified"
            matches = (it.get("symptoms_in_common") or it.get("matching_symptoms") or []) or []
            mismatches = (it.get("symptoms_not_in_common") or it.get("non_matching_symptoms") or []) or []

            user_set = set([s.lower() for s in symptoms_list])
            match_set = set([m.lower() for m in matches])
            not_typical = mismatches if mismatches else sorted([s for s in symptoms_list if s.lower() not in match_set])

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
    # Only keep up to 3 diagnoses with >0 probability
    diagnoses_fmt = diagnoses_fmt[:3]

    red_flags = "Seek urgent care for severe chest pain, confusion, unilateral weakness, severe shortness of breath, or rapid deterioration."

    return {
        "diagnoses": diagnoses_fmt,
        "red_flags": red_flags,
        "disclaimer": "This is not medical advice. Seek professional evaluation."
    }

#prepares the prompt, calls the API, processes the response, and formats the output for the UI
# Streamlit Part


st.set_page_config(page_title="DxGPT Triage (Differentials)", page_icon="🩺", layout="centered")


st.title("🩺 Yong Differential Helper")

st.caption("For educational triage support only. Not a medical tool. Consult Doctor for critical conditions.")

symptoms_csv = st.text_area(
    "Enter 4 to 10 symptoms (comma separated), Please give Optional Clinical Context below to improve results.",
    placeholder="e.g., persistent cough, chest pain, shortness of breath, fatigue"
)

with st.expander("Optional clinical context(CLick to Expand)"):
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

# NEW: small optional message box, not part of the clinical context

optional_note = st.text_input(
    "Optional Note",
    placeholder="e.g., Already saw a doctor yesterday, suspected common cold.",
    key="optional_note",
    help="Anything you want the model to know that isn't clinical context (opinions received, prior visits, preferences, etc.)."
)


left, right = st.columns([1,1])
with left:
    run_btn = st.button("Give me diagnosis right away!")
with right:
    st.write("")

if run_btn:
    if not symptoms_csv.strip():
        st.error("Please enter at least a few comma separated symptoms.")
    else:
        try:
            context = _build_context_section(
                age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed
            )

            with st.spinner("Querying DxGPT..."):
                #st.image("https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif", caption="AI is thinking...", width='stretch')
                result = get_top3_differentials_with_mismatch(symptoms_csv, context, optional_note)


            diags = result.get("diagnoses", [])
            if not diags:
                st.warning("No results returned.")
            else:
                # Summary table
                st.subheader("Top differentials")
                rows = [
                    {
                        "Diagnosis": d["name"],
                        "Probability %": d.get("probability_percent", 0.0),
                        "Rationale": d.get("rationale", "")
                    }
                    for d in diags
                ]
                st.dataframe(rows, width='stretch')

                # Details per diagnosis
                for i, d in enumerate(diags, start=1):
                    with st.expander(f"{i}. {d['name']}"):
                        st.markdown(f"**Estimated probability:** {d.get('probability_percent', 0.0)}%")
                        matches = d.get("matching_symptoms") or []
                        not_typical = d.get("not_typical_symptoms") or []

                        if matches:
                            st.markdown("**Matching symptoms:** " + ", ".join(matches))
                        else:
                            st.markdown("**Matching symptoms:** none reported")

                        if not_typical:
                            st.markdown("**Your entered symptoms not typical for this illness:** " + ", ".join(not_typical))
                        else:
                            st.markdown("**Your entered symptoms not typical for this illness:** none reported")

                        st.markdown("**Rationale:** " + d.get("rationale", ""))

                # Red flags and disclaimer
                st.info("Red flags: " + result.get("red_flags", ""))
                st.caption(result.get("disclaimer", ""))

        except Exception as ex:
            st.error(str(ex))
            st.stop()

# Footer note
st.caption("This tool is not a substitute for a real Doctor.")
#st.image("https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif", caption="Stay healthy!", width='stretch')
#st.markdown("the above image wont be in final product, Compliments from Ludiac")