"""
Detectors Module

YOLO-based signature and stamp detection with verification layer.
Includes stroke density, connected components, HSV color masking,
and circularity checks for detection validation.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from pathlib import Path

# For downloading trained models from HuggingFace Hub
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Will use fallback YOLO model.")


@dataclass
class Detection:
    """Single detection result."""
    class_name: str  # "signature" or "stamp"
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    verified: bool = False
    verification_reason: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "verified": self.verified,
            "verification_reason": self.verification_reason,
            "metadata": self.metadata
        }
    
    @property
    def present(self) -> bool:
        """Whether this detection is valid and verified."""
        return self.verified and self.confidence > 0


@dataclass
class DetectorOutput:
    """Output from a detector."""
    detector_type: str
    detections: List[Detection]
    latency: float
    image_size: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict:
        return {
            "detector_type": self.detector_type,
            "detections": [d.to_dict() for d in self.detections],
            "latency": self.latency,
            "image_size": list(self.image_size)
        }
    
    @property
    def best_detection(self) -> Optional[Detection]:
        """Get highest confidence verified detection."""
        verified = [d for d in self.detections if d.verified]
        if not verified:
            return None
        return max(verified, key=lambda x: x.confidence)


class YOLODetector:
    """
    YOLO-based object detector.
    
    Loads ONNX quantized model for efficient inference.
    Supports both signature and stamp detection.
    """
    
    def __init__(
        self,
        model_path: str,
        detector_type: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_gpu: bool = True
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to ONNX model file
            detector_type: "signature" or "stamp"
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_gpu = use_gpu
        self.model = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return
            
        try:
            # Try loading with ultralytics YOLO
            from ultralytics import YOLO
            
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained YOLOv8 as fallback
                # In production, train custom model on signature/stamp data
                self.model = YOLO("yolov8n.pt")
                print(f"Warning: Using pretrained YOLOv8n. Train custom model for {self.detector_type}")
            
            self._initialized = True
        except Exception as e:
            print(f"YOLO initialization error: {e}")
            self._initialized = True
    
    def detect(self, image_path: str) -> DetectorOutput:
        """
        Run detection on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            DetectorOutput with all detections
        """
        start_time = time.time()
        detections = []
        image_size = (0, 0)
        
        self._lazy_init()
        
        try:
            # Load image to get size
            image = cv2.imread(image_path)
            if image is not None:
                image_size = (image.shape[1], image.shape[0])  # width, height
            
            if self.model is not None:
                # Run YOLO inference
                results = self.model(
                    image_path,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            
                            detection = Detection(
                                class_name=self.detector_type,
                                bbox=[int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                confidence=conf
                            )
                            detections.append(detection)
            
            # If no model, use fallback heuristic detection
            if not detections and image is not None:
                detections = self._heuristic_detect(image)
                
        except Exception as e:
            print(f"Detection error: {e}")
        
        latency = time.time() - start_time
        return DetectorOutput(
            detector_type=self.detector_type,
            detections=detections,
            latency=latency,
            image_size=image_size
        )
    
    def _heuristic_detect(self, image: np.ndarray) -> List[Detection]:
        """
        Fallback heuristic detection when YOLO model not available.
        Uses contour analysis to find potential signatures/stamps.
        """
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.detector_type == "signature":
            # Look for handwriting-like regions
            detections = self._detect_signature_regions(image, gray)
        elif self.detector_type == "stamp":
            # Look for colored circular/rectangular regions
            detections = self._detect_stamp_regions(image)
        
        return detections
    
    def _detect_signature_regions(
        self, 
        image: np.ndarray, 
        gray: np.ndarray
    ) -> List[Detection]:
        """Heuristic signature detection using edge analysis."""
        detections = []
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Group nearby contours
        h, w = gray.shape
        min_area = (w * h) * 0.001  # At least 0.1% of image
        max_area = (w * h) * 0.15   # At most 15% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = cw / ch if ch > 0 else 0
                
                # Signatures tend to be wider than tall
                if 1.5 < aspect_ratio < 8:
                    detection = Detection(
                        class_name="signature",
                        bbox=[x, y, cw, ch],
                        confidence=0.5  # Low confidence for heuristic
                    )
                    detections.append(detection)
        
        return detections[:5]  # Limit to top 5
    
    def _detect_stamp_regions(self, image: np.ndarray) -> List[Detection]:
        """Heuristic stamp detection using color analysis."""
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Red color ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue color range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        mask = mask_red1 | mask_red2 | mask_blue
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = (w * h) * 0.002  # At least 0.2% of image
        max_area = (w * h) * 0.1    # At most 10% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                detection = Detection(
                    class_name="stamp",
                    bbox=[x, y, cw, ch],
                    confidence=0.5
                )
                detections.append(detection)
        
        return detections[:3]  # Limit to top 3


class SignatureDetector(YOLODetector):
    """
    Specialized detector for signatures.
    
    Uses the trained YOLOv8s signature detector from HuggingFace Hub:
    tech4humans/yolov8s-signature-detector
    """
    
    # HuggingFace Hub model configuration (from working notebook)
    HF_REPO_ID = "tech4humans/yolov8s-signature-detector"
    HF_MODEL_FILENAME = "yolov8s.pt"
    
    def __init__(
        self,
        model_path: str = None,  # If None, will download from HuggingFace
        confidence_threshold: float = 0.5,
        use_hf_model: bool = True  # Whether to use HuggingFace model
    ):
        """
        Initialize signature detector.
        
        Args:
            model_path: Path to local model (optional, uses HF model by default)
            confidence_threshold: Minimum confidence for detections
            use_hf_model: If True, download model from HuggingFace Hub
        """
        self.use_hf_model = use_hf_model
        
        # Determine model path
        if model_path is None and use_hf_model and HF_HUB_AVAILABLE:
            # Will be downloaded in _lazy_init
            model_path = "models/yolo_signature.pt"  # Placeholder, actual download happens later
        elif model_path is None:
            model_path = "models/yolo_signature.onnx"
        
        super().__init__(
            model_path=model_path,
            detector_type="signature",
            confidence_threshold=confidence_threshold
        )
    
    def _lazy_init(self):
        """Lazy initialization with HuggingFace Hub model download."""
        if self._initialized:
            return
        
        try:
            from ultralytics import YOLO
            
            # Try to download from HuggingFace Hub (matching notebook approach)
            if self.use_hf_model and HF_HUB_AVAILABLE:
                try:
                    print(f"🔄 Loading signature detector from HuggingFace Hub...")
                    model_path = hf_hub_download(
                        repo_id=self.HF_REPO_ID,
                        filename=self.HF_MODEL_FILENAME
                    )
                    self.model = YOLO(model_path)
                    print(f"✓ Signature detector loaded from {self.HF_REPO_ID}")
                    self._initialized = True
                    return
                except Exception as e:
                    print(f"Warning: Failed to download from HuggingFace Hub: {e}")
                    print("Falling back to local model or pretrained YOLOv8...")
            
            # Try local model path
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"✓ Signature detector loaded from local: {self.model_path}")
            else:
                # Fallback to pretrained YOLOv8
                self.model = YOLO("yolov8n.pt")
                print(f"Warning: Using pretrained YOLOv8n. For better results, install huggingface_hub.")
            
            self._initialized = True
        except Exception as e:
            print(f"YOLO initialization error: {e}")
            self._initialized = True


class StampDetector(YOLODetector):
    """Specialized detector for stamps."""
    
    def __init__(
        self,
        model_path: str = "models/yolo_stamp.onnx",
        confidence_threshold: float = 0.5
    ):
        super().__init__(
            model_path=model_path,
            detector_type="stamp",
            confidence_threshold=confidence_threshold
        )


class VerificationLayer:
    """
    Verification layer for validating YOLO detections.
    
    Implements additional checks:
    - Signatures: stroke density, connected components
    - Stamps: HSV color masking, circularity
    - Both: reject if too much printed text (OCR word count > 10)
    """
    
    def __init__(
        self,
        # Signature verification params
        stroke_density_min: float = 0.05,
        stroke_density_max: float = 0.30,
        min_connected_components: int = 5,
        # Stamp verification params
        circularity_threshold: float = 0.5,
        # Text rejection params
        max_ocr_words: int = 10
    ):
        self.stroke_density_min = stroke_density_min
        self.stroke_density_max = stroke_density_max
        self.min_connected_components = min_connected_components
        self.circularity_threshold = circularity_threshold
        self.max_ocr_words = max_ocr_words
    
    def verify_detection(
        self,
        detection: Detection,
        image: np.ndarray,
        ocr_results: List[Dict] = None
    ) -> Detection:
        """
        Verify a single detection.
        
        Args:
            detection: Detection to verify
            image: Full image as numpy array
            ocr_results: OCR results for text density check
            
        Returns:
            Detection with updated verified flag and reason
        """
        # Extract region of interest
        x, y, w, h = detection.bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            detection.verified = False
            detection.verification_reason = "invalid_bbox"
            return detection
        
        roi = image[y:y+h, x:x+w]
        
        # Check OCR word count in bbox
        if ocr_results:
            word_count = self._count_words_in_bbox(ocr_results, detection.bbox)
            if word_count > self.max_ocr_words:
                detection.verified = False
                detection.verification_reason = f"text_density_too_high: {word_count} words"
                detection.metadata["word_count"] = word_count
                return detection
        
        if detection.class_name == "signature":
            return self._verify_signature(detection, roi)
        elif detection.class_name == "stamp":
            return self._verify_stamp(detection, roi)
        
        return detection
    
    def _verify_signature(
        self, 
        detection: Detection, 
        roi: np.ndarray
    ) -> Detection:
        """Verify signature detection."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Calculate stroke density
        stroke_density = self._compute_stroke_density(gray)
        detection.metadata["stroke_density"] = stroke_density
        
        if not (self.stroke_density_min <= stroke_density <= self.stroke_density_max):
            detection.verified = False
            detection.verification_reason = f"stroke_density_out_of_range: {stroke_density:.3f}"
            return detection
        
        # Count connected components
        num_components = self._count_connected_components(gray)
        detection.metadata["connected_components"] = num_components
        
        if num_components < self.min_connected_components:
            detection.verified = False
            detection.verification_reason = f"insufficient_components: {num_components}"
            return detection
        
        # All checks passed
        detection.verified = True
        detection.verification_reason = "passed_all_checks"
        return detection
    
    def _verify_stamp(
        self, 
        detection: Detection, 
        roi: np.ndarray
    ) -> Detection:
        """Verify stamp detection."""
        # Check for colored regions (red/blue stamps)
        color_score = self._compute_color_score(roi)
        detection.metadata["color_score"] = color_score
        
        if color_score < 0.1:  # At least 10% colored pixels
            detection.verified = False
            detection.verification_reason = f"insufficient_color: {color_score:.3f}"
            return detection
        
        # Check circularity
        circularity = self._compute_circularity(roi)
        detection.metadata["circularity"] = circularity
        
        if circularity < self.circularity_threshold:
            # Allow rectangular stamps too
            rectangularity = self._compute_rectangularity(roi)
            detection.metadata["rectangularity"] = rectangularity
            
            if rectangularity < 0.7:
                detection.verified = False
                detection.verification_reason = f"shape_irregular: circ={circularity:.3f}, rect={rectangularity:.3f}"
                return detection
        
        # All checks passed
        detection.verified = True
        detection.verification_reason = "passed_all_checks"
        return detection
    
    def _compute_stroke_density(self, gray: np.ndarray) -> float:
        """Compute stroke density (ratio of edge pixels to total pixels)."""
        edges = cv2.Canny(gray, 50, 150)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_pixels = np.sum(edges > 0)
        return edge_pixels / total_pixels if total_pixels > 0 else 0
    
    def _count_connected_components(self, gray: np.ndarray) -> int:
        """Count connected components (strokes) in the image."""
        # Threshold and find components
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(binary)
        return num_labels - 1  # Subtract background
    
    def _compute_color_score(self, roi: np.ndarray) -> float:
        """Compute ratio of colored (red/blue) pixels."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red masks
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        
        # Blue mask
        mask_blue = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
        
        # Black mask (low saturation stamps)
        mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 80]))
        
        combined_mask = mask_red1 | mask_red2 | mask_blue | mask_black
        
        total_pixels = roi.shape[0] * roi.shape[1]
        colored_pixels = np.sum(combined_mask > 0)
        
        return colored_pixels / total_pixels if total_pixels > 0 else 0
    
    def _compute_circularity(self, roi: np.ndarray) -> float:
        """Compute circularity score (1.0 = perfect circle)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity = 4π × area / perimeter²
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)
    
    def _compute_rectangularity(self, roi: np.ndarray) -> float:
        """Compute rectangularity score (1.0 = perfect rectangle)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest)
        rect_area = w * h
        
        if rect_area == 0:
            return 0.0
        
        return area / rect_area
    
    def _count_words_in_bbox(
        self, 
        ocr_results: List[Dict], 
        bbox: List[int]
    ) -> int:
        """Count OCR words that fall within the detection bbox."""
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        word_count = 0
        for result in ocr_results:
            if "bbox" in result:
                rx, ry, rw, rh = result["bbox"]
                rx2, ry2 = rx + rw, ry + rh
                
                # Check overlap
                if not (rx2 < x1 or rx > x2 or ry2 < y1 or ry > y2):
                    # Count words in this result
                    text = result.get("text", "")
                    word_count += len(text.split())
        
        return word_count


def run_signature_detection(
    image_path: str,
    verify: bool = True,
    ocr_results: List[Dict] = None
) -> DetectorOutput:
    """
    Run signature detection with optional verification.
    
    Args:
        image_path: Path to image file
        verify: Whether to run verification layer
        ocr_results: OCR results for text density check
        
    Returns:
        DetectorOutput with verified detections
    """
    detector = SignatureDetector()
    output = detector.detect(image_path)
    
    if verify and output.detections:
        image = cv2.imread(image_path)
        if image is not None:
            verifier = VerificationLayer()
            for detection in output.detections:
                verifier.verify_detection(detection, image, ocr_results)
    
    return output


def run_stamp_detection(
    image_path: str,
    verify: bool = True,
    ocr_results: List[Dict] = None
) -> DetectorOutput:
    """
    Run stamp detection with optional verification.
    
    Args:
        image_path: Path to image file
        verify: Whether to run verification layer
        ocr_results: OCR results for text density check
        
    Returns:
        DetectorOutput with verified detections
    """
    detector = StampDetector()
    output = detector.detect(image_path)
    
    if verify and output.detections:
        image = cv2.imread(image_path)
        if image is not None:
            verifier = VerificationLayer()
            for detection in output.detections:
                verifier.verify_detection(detection, image, ocr_results)
    
    return output


def run_all_detections(
    image_path: str,
    verify: bool = True,
    ocr_results: List[Dict] = None
) -> Dict[str, DetectorOutput]:
    """
    Run both signature and stamp detection.
    
    Args:
        image_path: Path to image file
        verify: Whether to run verification layer
        ocr_results: OCR results for text density check
        
    Returns:
        Dictionary with 'signature' and 'stamp' DetectorOutputs
    """
    return {
        "signature": run_signature_detection(image_path, verify, ocr_results),
        "stamp": run_stamp_detection(image_path, verify, ocr_results)
    }
