# DxGPT2.py — Streamlit UI + robust DXGPT workflow with overlap cleaning
# ------------------------------------------------------------------------------
# Features:
# - Model selector (gpt4o / o3), direct mode
# - Stable myuuid per session
# - diseases_list exclusions to widen differential
# - Emergency & follow-up questions (+ optional patient/update)
# - Probability heuristic: recall/precision with mismatch penalty
# - Handles result=queued/processing, retries 429 with backoff
# - Shows detected language and anonymized text
# - NEW: robust overlap cleaner so "matching" won't reappear in "not typical"
# - NEW: better symptom splitting (prefer semicolons/newlines over commas)
# ------------------------------------------------------------------------------

import os
import re
import json
import time
import uuid
import requests
from typing import Dict, List, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

# ------------------------------- Config -------------------------------

load_dotenv()
DXGPT_SUBSCRIPTION_KEY = os.getenv("DXGPT_SUBSCRIPTION_KEY", "").strip()
DXGPT_BASE_URL = (os.getenv("DXGPT_BASE_URL") or "https://dxgpt-apim.azure-api.net/api").rstrip("/")

if not DXGPT_SUBSCRIPTION_KEY:
    st.stop()
    raise SystemExit("DXGPT_SUBSCRIPTION_KEY is required (env or .env).")

if "myuuid" not in st.session_state:
    st.session_state["myuuid"] = str(uuid.uuid4())

# ----------------------------- Utilities ------------------------------

# Normalize phrases for comparison: lowercase, collapse spaces, strip select punctuation
_PUNCT_RE = re.compile(r"[,\.\(\)\[\]\{\};:!?\-_/]+")

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)        # remove common punctuation
    s = re.sub(r"\s+", " ", s)       # collapse whitespace
    return s.strip()

def _normalize_symptom_list(symptoms_text: str) -> List[str]:
    """
    Prefer semicolons or newlines as separators; keep commas inside a symptom.
    If only commas are used, fall back to comma-split, but try to keep long phrases intact.
    """
    text = (symptoms_text or "").strip()
    if not text:
        return []

    if ("\n" in text) or (";" in text):
        parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
    else:
        # Fall back to commas, but if it looks like one long adjectival phrase, keep it whole.
        comma_parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(comma_parts) >= 3 and sum(len(p) <= 5 for p in comma_parts) >= 2:
            parts = [" ".join(comma_parts)]     # treat as one symptom
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
    - Deduplicate matches and mismatches (case/punct insensitive).
    - Remove any mismatch that is identical to, a substring of, or a superstring of any match.
    - If API gave no mismatches, compute from user symptoms, then apply the same overlap removal.
    """
    # Dedup matches
    m_seen, m_dedup = set(), []
    for m in (matches or []):
        n = _norm(m)
        if n and n not in m_seen:
            m_seen.add(n)
            m_dedup.append(m)

    # Base mismatches set: API list or user symptoms not present in matches
    if mismatches:
        raw_mm = mismatches
    else:
        # Compute "not typical" from user's list
        m_norm_set = set(_norm(m) for m in m_dedup)
        raw_mm = [s for s in (user_symptoms or []) if _norm(s) not in m_norm_set]

    # Remove overlap with matches
    m_norms = [_norm(m) for m in m_dedup]
    mm_seen, mm_clean = set(), []
    for nt in raw_mm:
        ntn = _norm(nt)
        if not ntn or ntn in mm_seen:
            continue
        overlaps = any(ntn in mn or mn in ntn for mn in m_norms)
        if not overlaps:
            mm_seen.add(ntn)
            mm_clean.append(nt)

    return m_dedup, mm_clean

# --------------------------- Scoring heuristic ------------------------

def _heuristic_probabilities(items: List[Dict], original_symptoms: List[str]) -> List[float]:
    """Score by explanatory power: recall + precision, penalize mismatches."""
    u = [_norm(s) for s in original_symptoms if s.strip()]
    u_set = set(u)
    scores = []
    for it in items:
        m = [_norm(x) for x in (it.get("symptoms_in_common") or [])]
        nm = [_norm(x) for x in (it.get("symptoms_not_in_common") or [])]
        inter = len(u_set & set(m))
        recall = inter / max(1, len(u_set))
        precision = inter / max(1, len(set(m))) if m else 0.0
        penalty = 0.1 * len([x for x in nm if x])  # penalize true mismatches
        score = max(0.0, 0.7*recall + 0.3*precision - penalty)
        scores.append(score)

    if not any(scores):
        return [round(100.0 / max(1, len(items)), 1)] * len(items)

    total = sum(scores) or 1.0
    probs = [round((s / total) * 100.0, 1) for s in scores]
    drift = round(100.0 - sum(probs), 1)
    if probs:
        i = probs.index(max(probs))
        probs[i] = round(probs[i] + drift, 1)
    return probs

# ------------------------------- API Calls ----------------------------

def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": DXGPT_SUBSCRIPTION_KEY
    }

def call_dxgpt_diagnose(description: str, *, model: str, diseases_list: str = "", direct: bool = True) -> Any:
    url = f"{DXGPT_BASE_URL}/diagnose"
    payload = {
        "description": description,
        "myuuid": st.session_state["myuuid"],
        "timezone": "Asia/Karachi",
        "lang": "en",
        "model": model,
        "response_mode": "direct" if direct else None
    }
    if diseases_list.strip():
        payload["diseases_list"] = diseases_list.strip()
    payload = {k: v for k, v in payload.items() if v is not None}

    last_resp = None
    for attempt in range(3):
        resp = requests.post(url, headers=_headers(), json=payload, timeout=90)
        last_resp = resp
        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()

        status = str(data.get("result", "")).lower()
        if status in {"queued", "processing"}:
            q = data.get("queueInfo", {}) or {}
            msg = data.get("message", "Request is being processed.")
            raise RuntimeError(f"DXGPT is processing (pos={q.get('position')}, ~{q.get('estimatedWaitTime')}m). {msg}")
        return data

    raise RuntimeError(f"DXGPT error {last_resp.status_code}: {last_resp.text}")

def call_questions_emergency(description: str) -> List[str]:
    url = f"{DXGPT_BASE_URL}/questions/emergency"
    payload = {"description": description, "myuuid": st.session_state["myuuid"], "lang": "en", "timezone": "Asia/Karachi"}
    r = requests.post(url, headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return (data.get("data", {}) or {}).get("questions", []) or []

def call_questions_followup(description: str, diseases_csv: str) -> List[str]:
    url = f"{DXGPT_BASE_URL}/questions/followup"
    payload = {"description": description, "diseases": diseases_csv, "myuuid": st.session_state["myuuid"], "lang": "en", "timezone": "Asia/Karachi"}
    r = requests.post(url, headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return (data.get("data", {}) or {}).get("questions", []) or []

def call_patient_update(description: str, qa_pairs: List[Tuple[str, str]]) -> str:
    url = f"{DXGPT_BASE_URL}/patient/update"
    payload = {
        "description": description,
        "answers": [{"question": q, "answer": a} for (q, a) in qa_pairs],
        "myuuid": st.session_state["myuuid"],
        "lang": "en",
        "timezone": "Asia/Karachi"
    }
    r = requests.post(url, headers=_headers(), json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return (data.get("data", {}) or {}).get("updatedDescription", "") or ""

# --------------------------- App-Level Logic --------------------------

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

def _get_meta(dxgpt: dict) -> Tuple[str, dict]:
    if not isinstance(dxgpt, dict):
        return None, {}
    return dxgpt.get("detectedLang"), dxgpt.get("anonymization") or {}

def make_description(symptoms_text: str, context: str, optional_note: str = "") -> str:
    # Keep the raw text phrasing; context and note appended for better 200–2000 char target
    base = f"Patient with the following symptoms: {symptoms_text.strip()}."
    parts = [base]
    if context.strip():
        parts.append(context.strip())
    if optional_note.strip():
        parts.append(f"Additional note: {optional_note.strip()}")
    return " ".join(parts)

def get_top3_differentials_with_mismatch(symptoms_text: str, context: str, optional_note: str = "",
                                         *, model: str, diseases_list: str = "", direct: bool = True) -> Tuple[dict, dict]:
    user_symptoms = _normalize_symptom_list(symptoms_text)
    description = make_description(symptoms_text, context, optional_note)
    dxgpt = call_dxgpt_diagnose(description, model=model, diseases_list=diseases_list, direct=direct)

    items = _extract_items(dxgpt)
    if not items and isinstance(dxgpt, dict) and dxgpt.get("result") != "success":
        msg = dxgpt.get("message") or dxgpt.get("error") or "Unknown error"
        raise RuntimeError(f"DXGPT returned non-success: {msg}")

    top = items[:3] if len(items) >= 3 else items
    if not top:
        return ({
            "diagnoses": [],
            "red_flags": "None reported by the model.",
            "disclaimer": "This is not medical advice. Seek professional evaluation."
        }, dxgpt)

    probs = _heuristic_probabilities(top, user_symptoms)

    diagnoses_fmt = []
    for it, p in zip(top, probs):
        name = it.get("diagnosis") or it.get("name") or it.get("disease") or "Unspecified"
        matches = (it.get("symptoms_in_common") or it.get("matching_symptoms") or []) or []
        mismatches = (it.get("symptoms_not_in_common") or it.get("non_matching_symptoms") or []) or []
        # Strong overlap cleaning (handles identical, substring, superstring cases)
        matches, not_typical = _clean_overlap(matches, mismatches, user_symptoms)

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

    red_flags = "Seek urgent care for severe chest pain, confusion, unilateral weakness, severe shortness of breath, or rapid deterioration."

    return ({
        "diagnoses": diagnoses_fmt[:3],
        "red_flags": red_flags,
        "disclaimer": "This is not medical advice. Seek professional evaluation."
    }, dxgpt)

# --------------------------------- UI ---------------------------------

st.set_page_config(page_title="Yong Differential Helper", page_icon="🩺", layout="centered")
st.title("🩺 Yong Differential Helper")
st.caption("Educational triage support only — not medical advice.")

with st.sidebar:
    st.subheader("Engine & Options")
    model = st.selectbox("DXGPT model", ["gpt4o", "o3"], index=0, help="gpt4o: fast & good. o3: deeper reasoning.")
    direct_mode = st.checkbox("Direct mode (no WebSocket)", value=True)
    prior_exclusions = st.text_input("Exclude diagnoses (comma-separated)", help="Send as diseases_list to explore alternatives.")
    st.caption(f"Session ID: {st.session_state['myuuid']}")

symptoms_text = st.text_area(
    "Enter 4–10 symptoms (use semicolons or new lines to separate distinct symptoms).",
    placeholder="e.g., dark velvety skin in body folds; weight gain; skin tags"
)

with st.expander("Optional clinical context (improves results)"):
    col1, col2 = st.columns(2)
    with col1:
        age_group = st.selectbox("Age group", ["", "Infant", "Child", "Adolescent", "Adult", "Older adult"], index=4)
        sex = st.selectbox("Sex", ["", "Female", "Male", "Intersex", "Prefer not to say"], index=0)
        duration = st.text_input("Symptom duration (e.g., 3 days, 2 weeks)")
        onset = st.selectbox("Onset", ["", "Sudden", "Gradual"])
        fever = st.selectbox("Fever", ["", "Yes", "No", "Unknown"])
        pregnancy = st.selectbox("Pregnancy status", ["", "Pregnant", "Not pregnant", "Unknown"])
    with col2:
        comorbid = st.text_input("Previous illnesses / comorbidities")
        meds = st.text_input("Current medications")
        allergies = st.text_input("Allergies")
        smoking = st.selectbox("Smoking status", ["", "Never", "Former", "Current"])
        travel = st.text_input("Recent travel or exposure")
        noticed = st.text_input("When did you first notice symptoms?")

optional_note = st.text_input(
    "Optional note (non-clinical)",
    placeholder="e.g., Already saw a doctor yesterday, suspected common cold."
)

left, right = st.columns([1, 1])
with left:
    run_btn = st.button("Get top 3 differentials")
with right:
    st.write("")

# ------------------------- Emergency questions path -------------------

def _build_context_section(age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed) -> str:
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
    if noticed:    fields.append(f"Symptoms first noticed: {noticed.strip()}")
    return " ".join(fields)

if run_btn:
    if not symptoms_text.strip():
        st.error("Please enter at least a few symptoms.")
        st.stop()

    user_symptoms = _normalize_symptom_list(symptoms_text)
    if len(user_symptoms) < 4:
        st.warning("Fewer than 4 distinct symptoms detected. Fetching initial questions first...")
        description_seed = make_description(symptoms_text, _build_context_section(
            age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed
        ), optional_note)
        with st.spinner("Getting initial questions..."):
            try:
                qlist = call_questions_emergency(description_seed)
            except Exception as e:
                st.error(f"Error getting initial questions: {e}")
                st.stop()

        if not qlist:
            st.info("No initial questions returned. Please add more detail and try again.")
            st.stop()

        st.subheader("Initial questions to complete the clinical picture")
        answers = []
        for q in qlist:
            a = st.text_input(q, key=f"q_{hash(q)}")
            answers.append((q, a))

        if st.button("Incorporate answers and re-run"):
            try:
                updated_desc = call_patient_update(description_seed, answers) or description_seed
                with st.spinner("Diagnosing with updated description..."):
                    result, raw = get_top3_differentials_with_mismatch(
                        symptoms_text, "", "", model=model, diseases_list=prior_exclusions, direct=direct_mode
                    )
            except Exception as e:
                st.error(f"Error during update/diagnose: {e}")
                st.stop()
        else:
            st.stop()
    else:
        context = _build_context_section(
            age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed
        )
        try:
            with st.spinner("Querying DxGPT..."):
                result, raw = get_top3_differentials_with_mismatch(
                    symptoms_text, context, optional_note, model=model, diseases_list=prior_exclusions, direct=direct_mode
                )
        except Exception as ex:
            st.error(str(ex))
            st.stop()

    # ------------------------- Render results -------------------------
    diags = result.get("diagnoses", [])
    if not diags:
        st.warning("No results returned.")
        st.stop()

    st.subheader("Top differentials")
    rows = [{"Diagnosis": d["name"], "Probability %": d.get("probability_percent", 0.0), "Rationale": d.get("rationale", "")} for d in diags]
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

    detected_lang, anonym = _get_meta(raw)
    if detected_lang:
        st.caption(f"Detected language: {detected_lang}")
    if anonym.get("hasPersonalInfo"):
        with st.expander("View anonymized text (PII masked)"):
            st.text(anonym.get("anonymizedText", ""))

    st.divider()
    st.subheader("Refine")
    bad_pick = st.multiselect("Does anything here feel wrong? Exclude to see alternatives:", [d["name"] for d in diags])
    if st.button("Try alternatives (exclude selected)"):
        excl = ", ".join(bad_pick)
        try:
            with st.spinner("Re-running with exclusions..."):
                result2, raw2 = get_top3_differentials_with_mismatch(
                    symptoms_text,
                    _build_context_section(
                        age_group, sex, duration, onset, fever, comorbid, meds, allergies, smoking, travel, pregnancy, noticed
                    ),
                    optional_note,
                    model=model,
                    diseases_list=excl,
                    direct=direct_mode
                )
            st.success("Updated alternatives below:")
            diags2 = result2.get("diagnoses", [])
            if diags2:
                st.dataframe(
                    [{"Diagnosis": d["name"], "Probability %": d.get("probability_percent", 0.0), "Rationale": d.get("rationale", "")} for d in diags2],
                    use_container_width=True
                )
            else:
                st.info("No alternatives returned.")
        except Exception as e:
            st.error(f"Re-run error: {e}")

# Footer
st.caption("This tool is not a substitute for a doctor. If you have severe or worsening symptoms, seek urgent care.")
