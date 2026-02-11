"""
OCR Engines Module — Microservice Client

Calls PaddleOCR and DeepSeek-OCR through isolated HTTP microservices.
Each service runs in its own Docker container with its own transformers
version, eliminating dependency conflicts completely.

The main pipeline imports this module exactly as before — the public API
(parallel_ocr, run_paddle_ocr, run_deepseek_ocr, OCREngineOutput, etc.)
is unchanged.

Service URLs are read from config.py (or environment variables).
"""

import os
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service URLs — read from environment so Docker Compose can inject them.
# Defaults point to localhost for local-dev (running services manually).
# ---------------------------------------------------------------------------
PADDLE_OCR_URL = os.environ.get("PADDLE_OCR_URL", "http://localhost:5001")
DEEPSEEK_OCR_URL = os.environ.get("DEEPSEEK_OCR_URL", "http://localhost:5002")

# HTTP timeout for OCR calls (seconds)
OCR_TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "120"))


# ===========================================================================
# Data classes (unchanged public API)
# ===========================================================================

@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_points(cls, points: List[List[float]]) -> 'BoundingBox':
        """Create from list of corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x = int(min(xs))
        y = int(min(ys))
        width = int(max(xs) - x)
        height = int(max(ys) - y)
        return cls(x=x, y=y, width=width, height=height)

    def to_list(self) -> List[int]:
        """Convert to [x, y, width, height] format."""
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> List[int]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.x, self.y, self.x + self.width, self.y + self.height]


@dataclass
class OCRResult:
    """Result from a single OCR detection."""
    text: str
    bbox: BoundingBox
    confidence: float
    language: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class OCREngineOutput:
    """Complete output from an OCR engine."""
    engine: str
    results: List[OCRResult]
    latency: float
    full_text: str = ""
    avg_confidence: float = 0.0

    def __post_init__(self):
        if self.results:
            self.full_text = " ".join(r.text for r in self.results)
            self.avg_confidence = float(np.mean([r.confidence for r in self.results]))

    def to_dict(self) -> Dict:
        return {
            "engine": self.engine,
            "results": [r.to_dict() for r in self.results],
            "latency": self.latency,
            "full_text": self.full_text,
            "avg_confidence": self.avg_confidence,
        }


# ===========================================================================
# HTTP Client helpers
# ===========================================================================

def _call_ocr_service(
    service_url: str,
    image_path: str,
    engine_name: str,
) -> OCREngineOutput:
    """
    POST the image file to an OCR microservice and parse the response
    into an OCREngineOutput.

    Args:
        service_url: Base URL of the service (e.g. http://paddle-ocr:5001)
        image_path:  Local path to the image file
        engine_name: "paddle" or "deepseek"

    Returns:
        OCREngineOutput
    """
    start = time.time()

    try:
        url = f"{service_url.rstrip('/')}/ocr"
        filename = Path(image_path).name

        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                files={"image": (filename, f)},
                timeout=OCR_TIMEOUT,
            )

        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", False):
            logger.warning(
                "%s service returned error: %s",
                engine_name,
                data.get("error", "unknown"),
            )
            return OCREngineOutput(
                engine=engine_name, results=[], latency=time.time() - start
            )

        # Parse results → OCRResult list
        results: List[OCRResult] = []
        for item in data.get("results", []):
            bbox_points = item.get("bbox_points", [[0, 0], [1, 0], [1, 1], [0, 1]])
            results.append(
                OCRResult(
                    text=item.get("text", ""),
                    bbox=BoundingBox.from_points(bbox_points),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )

        latency = time.time() - start
        return OCREngineOutput(
            engine=engine_name, results=results, latency=latency
        )

    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot connect to %s service at %s — is it running?",
            engine_name,
            service_url,
        )
    except requests.exceptions.Timeout:
        logger.error("%s service timed out after %ds", engine_name, OCR_TIMEOUT)
    except Exception as e:
        logger.exception("Error calling %s service: %s", engine_name, e)

    return OCREngineOutput(
        engine=engine_name, results=[], latency=time.time() - start
    )


# ===========================================================================
# Public API  (drop-in replacement — same signatures as before)
# ===========================================================================

def run_paddle_ocr(image_path: str) -> OCREngineOutput:
    """
    Run PaddleOCR on an image by calling the PaddleOCR microservice.

    Args:
        image_path: Path to the image file

    Returns:
        OCREngineOutput with all detected text regions
    """
    return _call_ocr_service(PADDLE_OCR_URL, image_path, "paddle")


def run_deepseek_ocr(image_path: str) -> OCREngineOutput:
    """
    Run DeepSeek-OCR on an image by calling the DeepSeek-OCR microservice.

    Args:
        image_path: Path to the image file

    Returns:
        OCREngineOutput with all detected text regions
    """
    return _call_ocr_service(DEEPSEEK_OCR_URL, image_path, "deepseek")


def parallel_ocr(
    image_path: str, use_deepseek: bool = True
) -> Dict[str, OCREngineOutput]:
    """
    Call both OCR services concurrently using ThreadPoolExecutor.

    Args:
        image_path:   Path to the image file
        use_deepseek: Whether to include DeepSeek-OCR

    Returns:
        Dict mapping engine name → OCREngineOutput
    """
    outputs: Dict[str, OCREngineOutput] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_paddle_ocr, image_path): "paddle",
        }
        if use_deepseek:
            futures[executor.submit(run_deepseek_ocr, image_path)] = "deepseek"

        for future in as_completed(futures):
            engine = futures[future]
            try:
                outputs[engine] = future.result()
            except Exception as e:
                logger.exception("Error running %s OCR", engine)
                outputs[engine] = OCREngineOutput(
                    engine=engine, results=[], latency=0.0
                )

    return outputs


# ===========================================================================
# Utility functions (unchanged)
# ===========================================================================

def merge_ocr_results(
    paddle_output: OCREngineOutput,
    deepseek_output: OCREngineOutput,
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """Merge results from both OCR engines, handling overlapping detections."""
    merged = []
    used_deepseek = set()

    for paddle_result in paddle_output.results:
        paddle_bbox = paddle_result.bbox.to_xyxy()
        best_match = None
        best_iou = 0

        for i, deepseek_result in enumerate(deepseek_output.results):
            if i in used_deepseek:
                continue
            deepseek_bbox = deepseek_result.bbox.to_xyxy()
            iou = _calculate_iou(paddle_bbox, deepseek_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = (i, deepseek_result)

        merged_item = {
            "paddle": paddle_result.to_dict(),
            "deepseek": None,
            "iou": 0.0,
            "agreement": False,
        }

        if best_match:
            used_deepseek.add(best_match[0])
            merged_item["deepseek"] = best_match[1].to_dict()
            merged_item["iou"] = best_iou
            merged_item["agreement"] = _texts_agree(
                paddle_result.text, best_match[1].text
            )

        merged.append(merged_item)

    for i, deepseek_result in enumerate(deepseek_output.results):
        if i not in used_deepseek:
            merged.append({
                "paddle": None,
                "deepseek": deepseek_result.to_dict(),
                "iou": 0.0,
                "agreement": False,
            })

    return merged


def _calculate_iou(box1: List[int], box2: List[int]) -> float:
    """IoU for two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def _texts_agree(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts agree using simple character overlap."""
    if not text1 or not text2:
        return False
    t1, t2 = text1.lower().strip(), text2.lower().strip()
    if t1 == t2:
        return True
    s1, s2 = set(t1), set(t2)
    union = len(s1 | s2)
    return (len(s1 & s2) / union) >= threshold if union else False


def get_full_text(ocr_outputs: Dict[str, OCREngineOutput]) -> Dict[str, str]:
    """Extract full text from each OCR engine output."""
    return {engine: output.full_text for engine, output in ocr_outputs.items()}


def compute_confidence_score(
    ocr_confidence: float,
    parser_certainty: float,
    detection_iou: float = 1.0,
) -> float:
    """Compute per-field confidence: OCR confidence × parser certainty × detection IoU."""
    return ocr_confidence * parser_certainty * detection_iou
