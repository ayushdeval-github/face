"""
app.py — Main Flask Application Entry Point
Fake News Detection System | Ayush Deval | Branch: ayush-deval | 2026-27

Architecture:
    User input → ML Model (LR/LSTM/BERT) → Groq LLaMA 3 70B → Combined verdict
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time, logging, os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from utils.model_loader import load_all_models
models_cache = load_all_models()
logger.info("All ML models loaded into memory.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": list(models_cache.keys()),
        "groq_enabled":  bool(os.environ.get("GROQ_API_KEY"))
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Request JSON:
        { "text": "news content here", "model": "lr" | "lstm" | "bert" }

    Response JSON:
        {
            "final_verdict":    "Fake" | "Real",
            "final_confidence": 94,
            "ml_verdict":       "Fake",
            "ml_confidence":    91,
            "groq_verdict":     "Fake",
            "groq_explanation": "This is false because...",
            "groq_confidence":  "High",
            "agreement":        "Both models agree",
            "model_used":       "BERT",
            "time_ms":          1420.5
        }
    """
    start_time = time.time()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    text         = data.get("text", "").strip()
    model_choice = data.get("model", "lr").lower()

    if not text or len(text) < 10:
        return jsonify({"error": "Text must be at least 10 characters."}), 400
    if model_choice not in ("lr", "lstm", "bert"):
        return jsonify({"error": "Invalid model. Choose: lr, lstm, or bert."}), 400

    # Step 1: ML Model
    try:
        from utils.predict import run_prediction
        ml_prediction, ml_confidence = run_prediction(text, model_choice, models_cache)
    except Exception as exc:
        logger.error("ML inference error: %s", exc, exc_info=True)
        return jsonify({"error": "ML model inference failed.", "detail": str(exc)}), 500

    # Step 2: Groq LLaMA 3 70B
    try:
        from utils.groq_client import analyze_with_groq
        groq_result = analyze_with_groq(text)
    except Exception as exc:
        logger.error("Groq error: %s", exc)
        groq_result = {"verdict": "Unavailable", "explanation": str(exc),
                       "confidence": "N/A", "groq_used": False}

    # Step 3: Combine
    from utils.groq_client import combined_verdict
    result             = combined_verdict(ml_prediction, ml_confidence, groq_result)
    result["model_used"] = model_choice.upper()
    result["time_ms"]    = round((time.time() - start_time) * 1000, 2)

    return jsonify(result)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)