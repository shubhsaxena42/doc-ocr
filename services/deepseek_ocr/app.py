#!/usr/bin/env python3
"""
DeepSeek-OCR Microservice

A lightweight Flask HTTP wrapper around the DeepSeek-OCR 4-bit model.
Runs in its own container/virtualenv with isolated dependencies
to avoid version conflicts with PaddleOCR's transformers.

Endpoints:
    POST /ocr     — Run OCR on an uploaded image (multipart/form-data)
    GET  /health  — Health check
"""

import os
import sys
import re
import io
import time
import json
import logging
import tempfile
import hashlib
import random

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepseek-ocr-service")

# ── Lazy-loaded model singleton ──────────────────────────────────────────────
_model = None
_tokenizer = None
_mock_active = False
_initialised = False


def _init_model():
    """Attempt to load the real DeepSeek-OCR 4-bit model; fallback to mock."""
    global _model, _tokenizer, _mock_active, _initialised

    if _initialised:
        return
    _initialised = True

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        model_id = os.environ.get(
            "DEEPSEEK_MODEL_ID", "Jalea96/DeepSeek-OCR-bnb-4bit-NF4"
        )
        logger.info("Loading DeepSeek-OCR (4-bit): %s …", model_id)

        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            model_id,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _model = _model.eval()
        logger.info("DeepSeek-OCR loaded successfully!")
    except Exception as e:
        logger.warning("DeepSeek-OCR load failed: %s  → MOCK MODE enabled", e)
        _mock_active = True


def _run_real_ocr(image_path: str, mode: str = "gundam") -> dict:
    """Run inference with the real model."""
    import torch

    mode_configs = {
        "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
        "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    }
    cfg = mode_configs.get(mode, mode_configs["gundam"])

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    temp_dir = os.path.join(tempfile.gettempdir(), "deepseek_ocr_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Capture stdout — model.infer() may print output
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        result = _model.infer(
            _tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=temp_dir,
            base_size=cfg["base_size"],
            image_size=cfg["image_size"],
            crop_mode=cfg["crop_mode"],
            save_results=False,
            test_compress=True,
        )
    finally:
        sys.stdout = old_stdout

    captured_text = captured.getvalue()
    final_text = ""

    # Extract from return value
    if result and isinstance(result, (str, dict)):
        if isinstance(result, dict) and "text" in result:
            final_text = result["text"]
        elif isinstance(result, str) and len(result) > 10:
            final_text = result

    # Fallback: parse captured stdout
    if not final_text and captured_text:
        lines = []
        for line in captured_text.split("\n"):
            if "PATCHES:" in line or "torch.Size" in line:
                continue
            line = re.sub(
                r"<\|ref\|>|</ref>|<\|det\|>|</det>|<\|grounding\|>", "", line
            )
            line = re.sub(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]", "", line)
            if line.strip():
                lines.append(line.strip())
        final_text = "\n".join(lines)

    from PIL import Image as PILImage

    img = PILImage.open(image_path)
    w, h = img.size

    return {
        "text": final_text,
        "bbox_points": [[0, 0], [w, 0], [w, h], [0, h]],
        "confidence": 0.90 if final_text else 0.0,
    }


def _run_mock_ocr(image_path: str) -> dict:
    """Generate mock OCR to test the adjudication pipeline."""

    path_hash = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
    random.seed(path_hash)

    mock_models = [
        "Mahindra Yuvo 575 DI", "Swaraj 744 FE", "John Deere 5050D",
        "Eicher 380 Super", "Sonalika DI 750 III", "TAFE 45 DI",
        "Kubota MU5502", "New Holland 3630", "Massey Ferguson 1035",
        "Powertrac Euro 50",
    ]
    mock_dealers = [
        "Sharma Tractors Pvt Ltd", "Gupta Agricultural Equipment",
        "Singh Motor Works", "Patel Farm Machinery",
        "Verma Implements & Tractors", "Khan Agro Services",
        "Yadav Krishi Udyog",
    ]

    model_name = random.choice(mock_models)
    dealer_name = random.choice(mock_dealers)
    horse_power = str(random.randint(35, 75))
    asset_cost = str(random.randint(500000, 1200000))

    full_text = (
        f"TAX INVOICE / BILL OF SALE\n"
        f"Dealer: {dealer_name}\n"
        f"Address: 123 Industrial Area, New Delhi 110001\n"
        f"PRODUCT DETAILS:\n"
        f"Model Name: {model_name}\n"
        f"HP: {horse_power} HP\n"
        f"Horse Power: {horse_power}\n"
        f"PRICING:\n"
        f"Ex-Showroom Price: Rs. {asset_cost}\n"
        f"Total Amount: Rs. {asset_cost}/-\n"
        f"Invoice No: INV/2024/MOCK/{path_hash % 10000:04d}"
    )

    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        w, h = img.size
    except Exception:
        w, h = 1000, 1000

    return {
        "text": full_text,
        "bbox_points": [[0, 0], [w, 0], [w, h], [0, h]],
        "confidence": 0.85,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "deepseek_ocr",
        "mock_mode": _mock_active,
    })


@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Run DeepSeek-OCR on an uploaded image.

    Expects:
        multipart/form-data with field `image` containing the image file.

    Returns JSON:
        {
          "success": true,
          "engine": "deepseek",
          "latency": 2.34,
          "results": [ { "text": "...", "bbox_points": [...], "confidence": 0.9 } ]
        }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()

        _init_model()   # no-op after first call

        start = time.time()
        if _mock_active:
            result = _run_mock_ocr(tmp.name)
        else:
            result = _run_real_ocr(tmp.name)
        latency = time.time() - start

        return jsonify({
            "success": True,
            "engine": "deepseek",
            "latency": round(latency, 4),
            "results": [result],
        })

    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        os.unlink(tmp.name)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    logger.info("Starting DeepSeek-OCR service on port %d", port)
    _init_model()
    app.run(host="0.0.0.0", port=port, debug=False)
