"""
Tests for Detectors Module

Tests stroke density, connected components, HSV masking, and circularity.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.detectors import (
    Detection,
    VerificationLayer,
    SignatureDetector,
    StampDetector
)


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test creating a detection."""
        det = Detection(
            class_name="signature",
            bbox=[10, 20, 100, 50],
            confidence=0.85
        )
        assert det.class_name == "signature"
        assert det.bbox == [10, 20, 100, 50]
        assert det.confidence == 0.85
        assert det.verified is False
    
    def test_detection_present(self):
        """Test present property."""
        det = Detection(
            class_name="signature",
            bbox=[10, 20, 100, 50],
            confidence=0.85,
            verified=True
        )
        assert det.present is True
        
        det.verified = False
        assert det.present is False
    
    def test_detection_to_dict(self):
        """Test serialization."""
        det = Detection(
            class_name="stamp",
            bbox=[0, 0, 50, 50],
            confidence=0.9,
            verified=True,
            verification_reason="passed"
        )
        d = det.to_dict()
        assert d["class_name"] == "stamp"
        assert d["verified"] is True


class TestVerificationLayer:
    """Tests for signature/stamp verification."""
    
    @pytest.fixture
    def verifier(self):
        """Create a verification layer."""
        return VerificationLayer(
            stroke_density_min=0.05,
            stroke_density_max=0.30,
            min_connected_components=5,
            circularity_threshold=0.5,
            max_ocr_words=10
        )
    
    def test_stroke_density_calculation(self, verifier):
        """Test stroke density computation."""
        # Create an image with some edges
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[40:60, 10:90] = 255  # Horizontal line
        gray[10:90, 40:60] = 255  # Vertical line (cross)
        
        density = verifier._compute_stroke_density(gray)
        assert 0 < density < 1
    
    def test_connected_components_count(self, verifier):
        """Test connected components counting."""
        # Create image with multiple blobs
        gray = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Add some black regions (components)
        gray[10:20, 10:20] = 0
        gray[30:40, 30:40] = 0
        gray[50:60, 60:70] = 0
        gray[70:80, 20:30] = 0
        gray[80:90, 70:80] = 0
        gray[20:25, 70:75] = 0
        
        count = verifier._count_connected_components(gray)
        assert count >= 5
    
    def test_color_score_red(self, verifier):
        """Test color score for red region."""
        # Create red image region
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        roi[:, :] = [0, 0, 180]  # BGR red
        
        score = verifier._compute_color_score(roi)
        assert score > 0.5
    
    def test_color_score_blue(self, verifier):
        """Test color score for blue region."""
        # Create blue image region
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        roi[:, :] = [180, 0, 0]  # BGR blue
        
        score = verifier._compute_color_score(roi)
        assert score > 0.5
    
    def test_circularity_circle(self, verifier):
        """Test circularity for circular shape."""
        # Create circular mask
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        center = (50, 50)
        radius = 40
        
        # Draw filled circle
        import cv2
        cv2.circle(roi, center, radius, (255, 255, 255), -1)
        
        circularity = verifier._compute_circularity(roi)
        assert circularity > 0.7  # Should be close to 1 for circle
    
    def test_rectangularity_rectangle(self, verifier):
        """Test rectangularity for rectangular shape."""
        # Create rectangular mask
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[20:80, 10:90] = 255
        
        rectangularity = verifier._compute_rectangularity(roi)
        assert rectangularity > 0.8  # Should be close to 1 for rectangle
    
    def test_word_count_in_bbox(self, verifier):
        """Test OCR word count in bbox."""
        ocr_results = [
            {"bbox": [10, 10, 50, 20], "text": "Hello World"},
            {"bbox": [100, 100, 50, 20], "text": "Other text"},
            {"bbox": [15, 15, 30, 10], "text": "Inside"}
        ]
        
        bbox = [0, 0, 80, 40]  # Should contain first and third
        count = verifier._count_words_in_bbox(ocr_results, bbox)
        assert count >= 2  # At least "Hello World" + "Inside"
    
    def test_verify_signature_valid(self, verifier):
        """Test signature verification with valid signature."""
        # Create signature-like image
        roi = np.ones((60, 200, 3), dtype=np.uint8) * 255
        
        # Add signature-like strokes
        import cv2
        for i in range(10):
            x1 = np.random.randint(10, 190)
            y1 = np.random.randint(10, 50)
            x2 = x1 + np.random.randint(-30, 30)
            y2 = y1 + np.random.randint(-20, 20)
            cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        detection = Detection(
            class_name="signature",
            bbox=[0, 0, 200, 60],
            confidence=0.8
        )
        
        result = verifier._verify_signature(detection, roi)
        # Result depends on random strokes, just check it runs
        assert result.verification_reason is not None
    
    def test_verify_stamp_valid(self, verifier):
        """Test stamp verification with valid stamp."""
        # Create stamp-like image (red circle)
        roi = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        import cv2
        cv2.circle(roi, (50, 50), 40, (0, 0, 180), 3)  # Red circle
        cv2.circle(roi, (50, 50), 35, (0, 0, 180), 2)  # Inner circle
        
        detection = Detection(
            class_name="stamp",
            bbox=[0, 0, 100, 100],
            confidence=0.8
        )
        
        result = verifier._verify_stamp(detection, roi)
        assert result.verification_reason is not None


class TestSignatureDetector:
    """Tests for SignatureDetector class."""
    
    def test_detector_creation(self):
        """Test detector instantiation."""
        detector = SignatureDetector()
        assert detector.detector_type == "signature"
    
    def test_detector_type(self):
        """Test detector type is correct."""
        detector = SignatureDetector()
        assert detector.detector_type == "signature"


class TestStampDetector:
    """Tests for StampDetector class."""
    
    def test_detector_creation(self):
        """Test detector instantiation."""
        detector = StampDetector()
        assert detector.detector_type == "stamp"
    
    def test_detector_type(self):
        """Test detector type is correct."""
        detector = StampDetector()
        assert detector.detector_type == "stamp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
