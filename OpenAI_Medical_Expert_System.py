import os
import json
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# 1) Put your key in an env var once:
#    Windows (PowerShell):  setx OPENAI_API_KEY "sk-proj-..."; restart terminal
#    macOS/Linux (bash):    echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc && source ~/.bashrc
#
# If you insist on passing directly (not recommended), do: client = OpenAI(api_key="sk-proj-...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")
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

def get_top3_differentials(symptoms_csv: str) -> dict:
    user_prompt = (
        f"Symptoms: {symptoms_csv.strip()}\n"
        "Return only the JSON object described in the schema. No prose outside JSON."
    )

    # NOTE: Do NOT pass temperature/top_p/etc. for gpt-5 here (causes 400).
    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
    )

    # Prefer the convenience property if available:
    text = getattr(resp, "output_text", None)
    if not text:
        # Fallback parsing for older client versions / alt formats
        parts = []
        for out in getattr(resp, "output", []) or []:
            for blk in getattr(out, "content", []) or []:
                if getattr(blk, "type", "") == "output_text":
                    parts.append(blk.text)
        text = "".join(parts)

    data = json.loads(text)
    return data

def pretty_print_results(result: dict):
    print("\nTop 3 differentials:")
    total = 0.0
    for i, d in enumerate(result["diagnoses"], start=1):
        pct = float(d["probability_percent"])
        total += pct
        print(f"{i}. {d['name']} — {pct:.1f}%")
        print(f"   Rationale: {d['rationale']}")
    print(f"\nSum of probabilities: {total:.1f}%")
    print(f"\nRed flags: {result['red_flags']}")
    print(f"\nDisclaimer: {result['disclaimer']}")

if __name__ == "__main__":
    symptoms = "Persistent cough, Coughing up blood, Shortness of breath, Chest pain, Unexplained weight loss, Fatigue"
    result = get_top3_differentials(symptoms)
    pretty_print_results(result)