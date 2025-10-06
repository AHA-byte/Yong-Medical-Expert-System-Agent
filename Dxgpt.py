import os
import json
import uuid
import requests
from typing import Dict, List, Any

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

def _heuristic_probabilities(items: List[Dict], original_symptoms: List[str]) -> List[float]:
    # score = matches - mismatches (min 0), then normalize to 100
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
        "model": "o3",
        "response_mode": "direct"
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Show body to help debugging 4xx/5xx
        raise RuntimeError(f"DXGPT error {resp.status_code}: {resp.text}") from e
    # Return parsed JSON (may be dict OR list depending on backend path)
    return resp.json()

def _extract_items(dxgpt: Any) -> List[Dict]:
    """
    Accepts either:
      - dict with shape {"result":"success","data":{"data":[...]}}
      - dict with shape {"data":[...]} or {"diagnoses":[...]}
      - plain list [...]
    Returns a list of diagnosis-like dicts.
    """
    if isinstance(dxgpt, list):
        return dxgpt

    if isinstance(dxgpt, dict):
        # Most common documented success shape
        data = dxgpt.get("data")
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]

        # Sometimes APIs shortcut to a list in "data"
        if isinstance(data, list):
            return data

        # Alternate key
        if isinstance(dxgpt.get("diagnoses"), list):
            return dxgpt["diagnoses"]

        # Fallback: if dict itself looks like an item, wrap it
        if any(k in dxgpt for k in ("diagnosis", "name", "disease")):
            return [dxgpt]

    # Last resort: empty list
    return []

def get_top3_differentials(symptoms_csv: str) -> dict:
    symptoms_list = [s.strip() for s in symptoms_csv.split(",") if s.strip()]
    description = f"Adult patient with the following symptoms: {', '.join(symptoms_list)}."

    dxgpt = call_dxgpt_diagnose(description)

    items = _extract_items(dxgpt)
    if not items and isinstance(dxgpt, dict) and dxgpt.get("result") != "success":
        # surface server message if present
        msg = dxgpt.get("message") or dxgpt.get("error") or "Unknown error"
        raise RuntimeError(f"DXGPT returned non-success: {msg}")

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
        name = it.get("diagnosis") or it.get("name") or it.get("disease") or "Unspecified"
        matches = (it.get("symptoms_in_common") or it.get("matching_symptoms") or []) or []
        mismatches = (it.get("symptoms_not_in_common") or it.get("non_matching_symptoms") or []) or []
        rationale_bits = []
        if matches:
            rationale_bits.append(f"matches: {', '.join(matches[:3])}")
        if mismatches:
            rationale_bits.append(f"not typical: {', '.join(mismatches[:2])}")
        rationale = "; ".join(rationale_bits) or "Pattern partially overlaps the provided symptoms."
        diagnoses_fmt.append({
            "name": name,
            "probability_percent": float(p),
            "rationale": rationale
        })

    red_flags = "Seek urgent care for severe chest pain, confusion, unilateral weakness, severe dyspnea, or rapid deterioration."

    return {
        "diagnoses": diagnoses_fmt,
        "red_flags": red_flags,
        "disclaimer": "This is not medical advice. Seek professional evaluation."
    }

def pretty_print_results(result: dict):
    print("\nTop 3 differentials:")
    total = 0.0
    for i, d in enumerate(result.get("diagnoses", []), start=1):
        pct = float(d.get("probability_percent", 0))
        total += pct
        print(f"{i}. {d.get('name','?')} — {pct:.1f}%")
        print(f"   Rationale: {d.get('rationale','')}")
    print(f"\nSum of probabilities: {total:.1f}%")
    print(f"\nRed flags: {result.get('red_flags','')}")
    print(f"\nDisclaimer: {result.get('disclaimer','')}")

if __name__ == "__main__":
    symptoms = "Persistent cough, Coughing up blood, Shortness of breath, Chest pain, Unexplained weight loss, Fatigue"
    result = get_top3_differentials(symptoms)
    pretty_print_results(result)