"""
utils/groq_client.py — Groq API Integration (LLaMA 3 70B)
Fake News Detection System | Ayush Deval | 2026-27

WHY GROQ:
    Groq runs LLaMA 3 70B at the fastest speed available anywhere.
    It understands Indian politics, culture, and current affairs natively.
    It explains WHY news is fake — not just a label.
    Free tier: 14,400 requests/day — more than enough.

WHY llama3-70b-8192:
    70 billion parameters = deepest contextual understanding.
    Best accuracy among all free Groq models.
    Knows Indian PMs, parties, events, geography out of the box.
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_MODEL  = "llama3-70b-8192"
MAX_TOKENS  = 300   # enough for a clear explanation, not too long


# ──────────────────────────────────────────────
# SYSTEM PROMPT
# This tells LLaMA exactly what role to play
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert fake news detection AI specialized in both Indian and international news.

Your job is to analyze a news statement or article and determine if it is REAL or FAKE.

Rules:
- You have deep knowledge of Indian politics, government, history, science, and current affairs
- You know all Indian Prime Ministers, Presidents, Chief Ministers, and major political figures
- You know international leaders, scientific facts, and world events
- Be direct and confident in your verdict
- Keep your explanation under 3 sentences — clear and simple
- Always end with either [VERDICT: FAKE] or [VERDICT: REAL]

Response format (always follow this exactly):
Explanation: <your explanation in 2-3 sentences>
Confidence: <High / Medium / Low>
[VERDICT: FAKE] or [VERDICT: REAL]"""


# ──────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────

def analyze_with_groq(text: str) -> dict:
    """
    Send news text to Groq LLaMA 3 70B for analysis.

    Args:
        text: Raw news text from user

    Returns:
        {
            "verdict":      "Fake" | "Real",
            "explanation":  "This is false because...",
            "confidence":   "High" | "Medium" | "Low",
            "groq_used":    True
        }
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping Groq analysis.")
        return _groq_unavailable()

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.1,        # low temperature = more factual, less creative
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Analyze this news: {text}"}
            ]
        )

        raw = response.choices[0].message.content.strip()
        logger.info("Groq raw response: %s", raw)
        return _parse_groq_response(raw)

    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return _groq_unavailable(str(exc))


# ──────────────────────────────────────────────
# RESPONSE PARSER
# ──────────────────────────────────────────────

def _parse_groq_response(raw: str) -> dict:
    """
    Parse Groq's text response into structured data.
    Handles slight variations in formatting gracefully.
    """
    lines      = raw.strip().splitlines()
    verdict    = "Unknown"
    explanation = raw   # fallback = full response
    confidence  = "Medium"

    # Extract verdict
    if "[VERDICT: FAKE]" in raw.upper():
        verdict = "Fake"
    elif "[VERDICT: REAL]" in raw.upper():
        verdict = "Real"

    # Extract explanation line
    for line in lines:
        if line.lower().startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
            break

    # Extract confidence
    for line in lines:
        if line.lower().startswith("confidence:"):
            conf_text = line.split(":", 1)[1].strip().lower()
            if "high" in conf_text:
                confidence = "High"
            elif "low" in conf_text:
                confidence = "Low"
            else:
                confidence = "Medium"
            break

    return {
        "verdict":     verdict,
        "explanation": explanation,
        "confidence":  confidence,
        "groq_used":   True
    }


def _groq_unavailable(error: str = "") -> dict:
    """Fallback when Groq is unavailable."""
    return {
        "verdict":     "Unavailable",
        "explanation": "Groq AI explanation unavailable. Using ML model result only.",
        "confidence":  "N/A",
        "groq_used":   False,
        "error":       error
    }


# ──────────────────────────────────────────────
# COMBINE ML + GROQ VERDICT
# ──────────────────────────────────────────────

def combined_verdict(ml_prediction: int, ml_confidence: float, groq_result: dict) -> dict:
    """
    Combines your trained ML model result with Groq's analysis.

    Logic:
        Both agree FAKE  → FAKE, very high confidence
        Both agree REAL  → REAL, very high confidence
        They disagree    → trust Groq (70B model) but flag disagreement
        Groq unavailable → use ML result only

    Args:
        ml_prediction:  0 = Real, 1 = Fake  (from your trained model)
        ml_confidence:  0.0 to 1.0
        groq_result:    dict from analyze_with_groq()

    Returns:
        Combined verdict dict for API response
    """
    ml_label = "Fake" if ml_prediction == 1 else "Real"

    # Groq not available — use ML only
    if not groq_result.get("groq_used"):
        return {
            "final_verdict":    ml_label,
            "final_confidence": round(ml_confidence * 100),
            "ml_verdict":       ml_label,
            "ml_confidence":    round(ml_confidence * 100),
            "groq_verdict":     "Unavailable",
            "groq_explanation": groq_result.get("explanation", ""),
            "agreement":        "N/A",
            "verdict_source":   "ML Model only"
        }

    groq_label = groq_result.get("verdict", "Unknown")
    groq_conf  = groq_result.get("confidence", "Medium")

    # Both agree
    if ml_label == groq_label:
        # Boost confidence when both models agree
        boosted = min(ml_confidence + 0.05, 0.99)
        return {
            "final_verdict":    ml_label,
            "final_confidence": round(boosted * 100),
            "ml_verdict":       ml_label,
            "ml_confidence":    round(ml_confidence * 100),
            "groq_verdict":     groq_label,
            "groq_explanation": groq_result.get("explanation", ""),
            "groq_confidence":  groq_conf,
            "agreement":        "✅ Both models agree",
            "verdict_source":   "ML + Groq (agreed)"
        }

    # They disagree — trust Groq (larger, more knowledgeable model)
    return {
        "final_verdict":    groq_label,
        "final_confidence": round(ml_confidence * 100),
        "ml_verdict":       ml_label,
        "ml_confidence":    round(ml_confidence * 100),
        "groq_verdict":     groq_label,
        "groq_explanation": groq_result.get("explanation", ""),
        "groq_confidence":  groq_conf,
        "agreement":        "⚠️ Models disagree — Groq verdict used",
        "verdict_source":   "Groq (overrode ML)"
    }
