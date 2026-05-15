"""
Flask web application for real-time face recognition.
Supports webcam capture, image upload, and dual-model prediction.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.utils.predictor import ClassicalMLPredictor, TeachableMachinePredictor

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Lazy-loaded predictors
_ml_predictor = None
_tm_predictor = None


def get_ml_predictor():
    global _ml_predictor
    if _ml_predictor is None:
        _ml_predictor = ClassicalMLPredictor()
    return _ml_predictor


def get_tm_predictor():
    global _tm_predictor
    if _tm_predictor is None:
        _tm_predictor = TeachableMachinePredictor()
    return _tm_predictor


def decode_image_from_request() -> np.ndarray | None:
    """Decode image from file upload or base64 JSON (webcam frame)."""
    if "image" in request.files:
        file = request.files["image"]
        if file.filename:
            data = np.frombuffer(file.read(), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    data = request.get_json(silent=True) or {}
    b64 = data.get("image_base64") or data.get("image")
    if b64:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


def encode_image_bgr(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image)
    return base64.b64encode(buf).decode("utf-8")


def format_response(result: dict) -> dict:
    """Build JSON response with optional annotated image."""
    out = {
        "success": result["success"],
        "label": result.get("label"),
        "confidence": result.get("confidence", 0),
        "is_me": result.get("is_me", False),
        "model": result.get("model"),
        "error": result.get("error"),
    }
    if result.get("annotated_image") is not None:
        out["image_base64"] = encode_image_bgr(result["annotated_image"])
    return out


@app.route("/")
def index():
    tm_available = get_tm_predictor().is_available()
    ml_ready = config.BEST_MODEL_PATH.exists()
    best_model_name = "kNN"
    if config.METADATA_PATH.exists():
        import json
        with open(config.METADATA_PATH) as f:
            best_model_name = json.load(f).get("best_model", best_model_name)
    return render_template(
        "index.html",
        tm_available=tm_available,
        ml_ready=ml_ready,
        best_model_name=best_model_name,
    )


@app.route("/api/status")
def status():
    return jsonify(
        {
            "ml_model_ready": config.BEST_MODEL_PATH.exists(),
            "tm_model_ready": get_tm_predictor().is_available(),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict using classical ML (default) or teachable machine via ?model=tm|ml"""
    image = decode_image_from_request()
    if image is None:
        return jsonify({"success": False, "error": "No image provided"}), 400

    model_type = request.args.get("model", "ml").lower()
    try:
        if model_type == "tm":
            result = get_tm_predictor().predict(image)
        else:
            result = get_ml_predictor().predict(image)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    return jsonify(format_response(result))


@app.route("/api/predict/both", methods=["POST"])
def predict_both():
    """Run both models and return comparison."""
    image = decode_image_from_request()
    if image is None:
        return jsonify({"success": False, "error": "No image provided"}), 400

    responses = {}
    try:
        responses["ml"] = format_response(get_ml_predictor().predict(image.copy()))
    except Exception as e:
        responses["ml"] = {"success": False, "error": str(e)}

    responses["tm"] = format_response(get_tm_predictor().predict(image))
    return jsonify(responses)


if __name__ == "__main__":
    print(f"Starting server at http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
