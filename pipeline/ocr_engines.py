"""
OCR Engines Module

Parallel execution of PaddleOCR and DeepSeek-OCR for ensemble extraction.
Computes per-field confidence scores: OCR confidence × parser certainty.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import cv2

# Lazy imports for OCR engines
_paddle_ocr = None
_deepseek_model = None


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
            "metadata": self.metadata
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
            self.avg_confidence = np.mean([r.confidence for r in self.results])
    
    def to_dict(self) -> Dict:
        return {
            "engine": self.engine,
            "results": [r.to_dict() for r in self.results],
            "latency": self.latency,
            "full_text": self.full_text,
            "avg_confidence": self.avg_confidence
        }


def _get_paddle_ocr():
    """Lazy load PaddleOCR."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Will detect multiple languages
                use_gpu=True,
                show_log=False,
                enable_mkldnn=False
            )
        except ImportError:
            print("Warning: PaddleOCR not installed. Install with: pip install paddleocr")
            _paddle_ocr = None
    return _paddle_ocr


def _get_deepseek_model():
    """Lazy load DeepSeek-OCR model (quantized for efficiency)."""
    global _deepseek_model
    if _deepseek_model is None:
        try:
            # DeepSeek-OCR implementation
            # For production, this would load the actual DeepSeek model
            # Here we provide a placeholder that can be swapped with actual implementation
            _deepseek_model = DeepSeekOCRWrapper()
        except Exception as e:
            print(f"Warning: DeepSeek-OCR initialization failed: {e}")
            _deepseek_model = None
    return _deepseek_model


class DeepSeekOCRWrapper:
    """
    Wrapper for DeepSeek-OCR model.
    
    In production, this would load the actual DeepSeek model.
    For now, it provides a compatible interface that can run
    alongside PaddleOCR for ensemble extraction.
    """
    
    def __init__(self):
        """Initialize DeepSeek-OCR wrapper."""
        self.model = None
        self.processor = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization of model components."""
        if self._initialized:
            return
            
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            # Note: Replace with actual DeepSeek-OCR model when available
            # This is a placeholder using a compatible VLM architecture
            model_name = "microsoft/trocr-base-handwritten"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._initialized = True
        except Exception as e:
            print(f"DeepSeek-OCR initialization skipped: {e}")
            self._initialized = True  # Prevent repeated attempts
    
    def ocr(self, image_path: str) -> List[Dict]:
        """
        Run OCR on image.
        
        Returns list of detections with text, bbox, and confidence.
        """
        self._lazy_init()
        
        if self.model is None:
            # Fallback: return empty results if model not available
            return []
        
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # For full document OCR, we'd use a text detection model first
            # then run recognition on each detected region
            # Here we provide a simplified single-pass approach
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            generated_ids = self.model.generate(pixel_values, max_length=256)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Return as single detection covering the whole image
            w, h = image.size
            return [{
                "text": text,
                "bbox": [[0, 0], [w, 0], [w, h], [0, h]],
                "confidence": 0.85  # Estimated confidence
            }]
            
        except Exception as e:
            print(f"DeepSeek-OCR error: {e}")
            return []


def run_paddle_ocr(image_path: str) -> OCREngineOutput:
    """
    Run PaddleOCR on an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        OCREngineOutput with all detected text regions
    """
    start_time = time.time()
    results = []
    
    ocr = _get_paddle_ocr()
    if ocr is None:
        return OCREngineOutput(
            engine="paddle",
            results=[],
            latency=time.time() - start_time
        )
    
    try:
        # Run PaddleOCR
        ocr_results = ocr.ocr(image_path, cls=True)
        
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                if line and len(line) >= 2:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    results.append(OCRResult(
                        text=text,
                        bbox=BoundingBox.from_points(bbox_points),
                        confidence=confidence
                    ))
    except Exception as e:
        print(f"PaddleOCR error: {e}")
    
    latency = time.time() - start_time
    return OCREngineOutput(
        engine="paddle",
        results=results,
        latency=latency
    )


def run_deepseek_ocr(image_path: str) -> OCREngineOutput:
    """
    Run DeepSeek-OCR on an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        OCREngineOutput with all detected text regions
    """
    start_time = time.time()
    results = []
    
    model = _get_deepseek_model()
    if model is None:
        return OCREngineOutput(
            engine="deepseek",
            results=[],
            latency=time.time() - start_time
        )
    
    try:
        ocr_results = model.ocr(image_path)
        
        for detection in ocr_results:
            results.append(OCRResult(
                text=detection.get("text", ""),
                bbox=BoundingBox.from_points(detection.get("bbox", [[0,0],[1,0],[1,1],[0,1]])),
                confidence=detection.get("confidence", 0.0)
            ))
    except Exception as e:
        print(f"DeepSeek-OCR error: {e}")
    
    latency = time.time() - start_time
    return OCREngineOutput(
        engine="deepseek",
        results=results,
        latency=latency
    )


def parallel_ocr(image_path: str, use_deepseek: bool = True) -> Dict[str, OCREngineOutput]:
    """
    Run both OCR engines in parallel using ThreadPoolExecutor.
    
    Args:
        image_path: Path to the image file
        use_deepseek: Whether to include DeepSeek-OCR (can be disabled for speed)
        
    Returns:
        Dictionary with engine names as keys and OCREngineOutput as values
    """
    outputs = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_paddle_ocr, image_path): "paddle"
        }
        
        if use_deepseek:
            futures[executor.submit(run_deepseek_ocr, image_path)] = "deepseek"
        
        for future in as_completed(futures):
            engine = futures[future]
            try:
                output = future.result()
                outputs[engine] = output
            except Exception as e:
                print(f"Error running {engine} OCR: {e}")
                outputs[engine] = OCREngineOutput(
                    engine=engine,
                    results=[],
                    latency=0.0
                )
    
    return outputs


def merge_ocr_results(
    paddle_output: OCREngineOutput,
    deepseek_output: OCREngineOutput,
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Merge results from both OCR engines, handling overlapping detections.
    
    Args:
        paddle_output: Results from PaddleOCR
        deepseek_output: Results from DeepSeek-OCR
        iou_threshold: IoU threshold for considering boxes as overlapping
        
    Returns:
        List of merged results with both engine outputs where applicable
    """
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
            "agreement": False
        }
        
        if best_match:
            used_deepseek.add(best_match[0])
            merged_item["deepseek"] = best_match[1].to_dict()
            merged_item["iou"] = best_iou
            # Check if texts match (using simple comparison)
            merged_item["agreement"] = _texts_agree(
                paddle_result.text, 
                best_match[1].text
            )
        
        merged.append(merged_item)
    
    # Add unmatched DeepSeek results
    for i, deepseek_result in enumerate(deepseek_output.results):
        if i not in used_deepseek:
            merged.append({
                "paddle": None,
                "deepseek": deepseek_result.to_dict(),
                "iou": 0.0,
                "agreement": False
            })
    
    return merged


def _calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union for two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def _texts_agree(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts agree using simple character overlap."""
    if not text1 or not text2:
        return False
    
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return True
    
    # Simple character overlap ratio
    set1 = set(text1)
    set2 = set(text2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return False
    
    return (intersection / union) >= threshold


def get_full_text(ocr_outputs: Dict[str, OCREngineOutput]) -> Dict[str, str]:
    """
    Extract full text from each OCR engine output.
    
    Args:
        ocr_outputs: Dictionary of engine outputs from parallel_ocr
        
    Returns:
        Dictionary mapping engine names to full extracted text
    """
    return {
        engine: output.full_text 
        for engine, output in ocr_outputs.items()
    }


def compute_confidence_score(
    ocr_confidence: float,
    parser_certainty: float,
    detection_iou: float = 1.0
) -> float:
    """
    Compute per-field confidence score.
    
    Formula: OCR confidence × detection IoU × parser certainty
    
    Args:
        ocr_confidence: Confidence from OCR engine (0-1)
        parser_certainty: Certainty from field parser (0-1)
        detection_iou: IoU with detection bbox if applicable (0-1)
        
    Returns:
        Combined confidence score (0-1)
    """
    return ocr_confidence * parser_certainty * detection_iou
