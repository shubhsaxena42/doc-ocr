#!/usr/bin/env python3
"""
PaddleOCR Microservice

A lightweight Flask HTTP wrapper around PaddleOCR.
Runs in its own container/virtualenv with isolated dependencies
to avoid version conflicts with DeepSeek-OCR's transformers.

Endpoints:
    POST /ocr     — Run OCR on an uploaded image (multipart/form-data)
    GET  /health  — Health check
"""

import os
import sys
import time
import json
import logging
import tempfile

# PaddlePaddle environment flags — must be set before any paddle import
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_executor"] = "0"
os.environ["FLAGS_use_mkl"] = "0"
os.environ["MKLDNN_DISABLE"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"

from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddle-ocr-service")

# ── Lazy-loaded singleton ────────────────────────────────────────────────────
_ocr_engine = None


def _get_engine():
    """Lazy-load PaddleOCR once and reuse across requests."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    from paddleocr import PaddleOCR

    has_gpu = False
    try:
        import paddle
        has_gpu = paddle.device.is_compiled_with_cuda()
    except Exception:
        pass

    try:
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            use_textline_orientation=True,
            lang="en",
            use_gpu=has_gpu,
            show_log=False,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR engine ready  (GPU=%s)", has_gpu)
    except Exception:
        logger.warning("GPU init failed, falling back to CPU")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            use_textline_orientation=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR engine ready  (CPU fallback)")

    return _ocr_engine


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Readiness probe."""
    return jsonify({"status": "ok", "engine": "paddle_ocr"})


@app.route("/ocr", methods=["POST"])
def ocr():
    """
    Run PaddleOCR on an uploaded image.

    Expects:
        multipart/form-data with field `image` containing the image file.

    Returns JSON:
        {
          "success": true,
          "engine": "paddle",
          "latency": 1.23,
          "results": [
            {
              "text": "...",
              "bbox_points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
              "confidence": 0.97
            },
            ...
          ]
        }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()

        start = time.time()
        engine = _get_engine()
        ocr_results = engine.ocr(tmp.name, cls=True)
        latency = time.time() - start

        results = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                if line and len(line) >= 2:
                    bbox_points = line[0]  # [[x1,y1], ...]
                    text = line[1][0]
                    confidence = float(line[1][1])
                    # Convert numpy floats → native for JSON
                    bbox_points = [[float(c) for c in pt] for pt in bbox_points]
                    results.append({
                        "text": text,
                        "bbox_points": bbox_points,
                        "confidence": confidence,
                    })

        return jsonify({
            "success": True,
            "engine": "paddle",
            "latency": round(latency, 4),
            "results": results,
        })

    except Exception as e:
        logger.exception("OCR failed")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        os.unlink(tmp.name)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info("Starting PaddleOCR service on port %d", port)
    # Warm-up: load the engine once before serving
    _get_engine()
    app.run(host="0.0.0.0", port=port, debug=False)
