"""
Calibration Module

Isotonic regression calibration on golden validation set.
Maps raw confidence scores to actual accuracy per field.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


@dataclass
class CalibrationMetrics:
    """Metrics from calibration."""
    field_name: str
    n_samples: int
    raw_accuracy: float
    calibrated_accuracy: float
    calibration_error: float  # Expected Calibration Error
    
    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "n_samples": self.n_samples,
            "raw_accuracy": self.raw_accuracy,
            "calibrated_accuracy": self.calibrated_accuracy,
            "calibration_error": self.calibration_error
        }


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for confidence scores.
    
    Maps raw model confidence to actual accuracy using
    non-parametric isotonic regression.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.calibrators: Dict[str, Any] = {}
        self._fitted = False
    
    def fit(
        self,
        field_name: str,
        confidences: np.ndarray,
        correct: np.ndarray
    ):
        """
        Fit isotonic calibrator for a field.
        
        Args:
            field_name: Name of the field
            confidences: Array of raw confidence scores
            correct: Array of binary correctness labels
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            
            calibrator = IsotonicRegression(
                out_of_bounds='clip',
                y_min=0.0,
                y_max=1.0
            )
            calibrator.fit(confidences, correct)
            self.calibrators[field_name] = calibrator
            self._fitted = True
            
        except ImportError:
            # Fallback: store mean accuracy per confidence bin
            self._fit_binned(field_name, confidences, correct)
    
    def _fit_binned(
        self,
        field_name: str,
        confidences: np.ndarray,
        correct: np.ndarray
    ):
        """Fallback binned calibration."""
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        bin_means = {}
        for i in range(10):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_means[i] = np.mean(correct[mask])
            else:
                bin_means[i] = (bins[i] + bins[i+1]) / 2  # Use bin center
        
        self.calibrators[field_name] = ('binned', bins, bin_means)
        self._fitted = True
    
    def calibrate(
        self,
        field_name: str,
        confidence: float
    ) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            field_name: Name of the field
            confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        if field_name not in self.calibrators:
            return confidence  # No calibration available
        
        calibrator = self.calibrators[field_name]
        
        if isinstance(calibrator, tuple) and calibrator[0] == 'binned':
            # Binned calibration
            _, bins, bin_means = calibrator
            bin_idx = min(int(confidence * 10), 9)
            return bin_means.get(bin_idx, confidence)
        else:
            # Isotonic regression
            return float(calibrator.predict([confidence])[0])
    
    def calibrate_batch(
        self,
        field_name: str,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate multiple confidence scores.
        
        Args:
            field_name: Name of the field
            confidences: Array of raw confidence scores
            
        Returns:
            Array of calibrated confidence scores
        """
        if field_name not in self.calibrators:
            return confidences
        
        calibrator = self.calibrators[field_name]
        
        if isinstance(calibrator, tuple) and calibrator[0] == 'binned':
            return np.array([self.calibrate(field_name, c) for c in confidences])
        else:
            return calibrator.predict(confidences)
    
    def compute_ece(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Args:
            confidences: Array of confidence scores
            correct: Array of binary correctness labels
            n_bins: Number of bins
            
        Returns:
            ECE value
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)
        
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct[mask])
                bin_confidence = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / total_samples
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def save(self, filepath: str):
        """Save calibrators to file."""
        # For simplicity, save as JSON with binned representation
        data = {}
        for field_name, calibrator in self.calibrators.items():
            if isinstance(calibrator, tuple) and calibrator[0] == 'binned':
                _, bins, bin_means = calibrator
                data[field_name] = {
                    "type": "binned",
                    "bins": bins.tolist(),
                    "bin_means": bin_means
                }
            else:
                # Convert isotonic to binned for serialization
                test_points = np.linspace(0, 1, 11)
                calibrated = calibrator.predict(test_points)
                data[field_name] = {
                    "type": "isotonic_approx",
                    "points": test_points.tolist(),
                    "values": calibrated.tolist()
                }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load calibrators from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for field_name, calib_data in data.items():
            if calib_data["type"] == "binned":
                bins = np.array(calib_data["bins"])
                bin_means = calib_data["bin_means"]
                self.calibrators[field_name] = ('binned', bins, bin_means)
            else:
                # Approximate isotonic with interpolation
                points = np.array(calib_data["points"])
                values = np.array(calib_data["values"])
                self.calibrators[field_name] = ('interp', points, values)
        
        self._fitted = True


@dataclass 
class GoldenDocument:
    """A document in the golden set."""
    doc_id: str
    image_path: str
    ground_truth: Dict[str, Any]
    layout_cluster: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GoldenDocument':
        return cls(
            doc_id=data["doc_id"],
            image_path=data["image_path"],
            ground_truth=data["ground_truth"],
            layout_cluster=data.get("layout_cluster", 0)
        )


class GoldenSet:
    """
    Golden validation set for calibration.
    
    Minimum 50 manually verified documents required.
    """
    
    def __init__(self, filepath: str = None):
        """
        Initialize golden set.
        
        Args:
            filepath: Path to golden_set.json
        """
        self.documents: List[GoldenDocument] = []
        self.by_id: Dict[str, GoldenDocument] = {}
        
        if filepath and Path(filepath).exists():
            self.load(filepath)
    
    def load(self, filepath: str):
        """Load golden set from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        docs = data.get("documents", [])
        for doc_data in docs:
            doc = GoldenDocument.from_dict(doc_data)
            self.documents.append(doc)
            self.by_id[doc.doc_id] = doc
    
    def save(self, filepath: str):
        """Save golden set to JSON."""
        data = {
            "_schema_version": "1.0",
            "_description": "Golden set for IDAI pipeline calibration",
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "image_path": doc.image_path,
                    "ground_truth": doc.ground_truth,
                    "layout_cluster": doc.layout_cluster
                }
                for doc in self.documents
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_document(
        self,
        doc_id: str,
        image_path: str,
        ground_truth: Dict[str, Any],
        layout_cluster: int = 0
    ):
        """Add a document to the golden set."""
        doc = GoldenDocument(
            doc_id=doc_id,
            image_path=image_path,
            ground_truth=ground_truth,
            layout_cluster=layout_cluster
        )
        self.documents.append(doc)
        self.by_id[doc_id] = doc
    
    def get_ground_truth(self, doc_id: str) -> Optional[Dict]:
        """Get ground truth for a document."""
        doc = self.by_id.get(doc_id)
        return doc.ground_truth if doc else None
    
    @property
    def size(self) -> int:
        """Number of documents in golden set."""
        return len(self.documents)
    
    def is_sufficient(self, min_size: int = 50) -> bool:
        """Check if golden set has minimum required documents."""
        return self.size >= min_size


class Calibrator:
    """
    Main calibration interface.
    
    Uses golden set to train per-field calibrators and
    applies calibration to pipeline outputs.
    """
    
    def __init__(
        self,
        golden_set_path: str = None,
        calibrator_path: str = None
    ):
        """
        Initialize calibrator.
        
        Args:
            golden_set_path: Path to golden_set.json
            calibrator_path: Path to saved calibrator
        """
        self.golden_set = GoldenSet(golden_set_path)
        self.isotonic = IsotonicCalibrator()
        
        if calibrator_path and Path(calibrator_path).exists():
            self.isotonic.load(calibrator_path)
    
    def train(
        self,
        predictions: Dict[str, List[Tuple[Any, float]]],
        fields: List[str]
    ) -> Dict[str, CalibrationMetrics]:
        """
        Train calibrators on predictions vs golden set.
        
        Args:
            predictions: Dict mapping doc_id to {field: (pred_value, confidence)}
            fields: List of field names to calibrate
            
        Returns:
            Dict mapping field name to CalibrationMetrics
        """
        metrics = {}
        
        for field_name in fields:
            confidences = []
            correct = []
            
            for doc_id, field_preds in predictions.items():
                if field_name not in field_preds:
                    continue
                
                pred_value, confidence = field_preds[field_name]
                ground_truth = self.golden_set.get_ground_truth(doc_id)
                
                if ground_truth is None or field_name not in ground_truth:
                    continue
                
                gt_value = ground_truth[field_name]
                is_correct = self._check_correct(pred_value, gt_value, field_name)
                
                confidences.append(confidence)
                correct.append(1.0 if is_correct else 0.0)
            
            if len(confidences) >= 10:  # Minimum samples for calibration
                conf_array = np.array(confidences)
                correct_array = np.array(correct)
                
                # Fit calibrator
                self.isotonic.fit(field_name, conf_array, correct_array)
                
                # Compute metrics
                calibrated = self.isotonic.calibrate_batch(field_name, conf_array)
                
                metrics[field_name] = CalibrationMetrics(
                    field_name=field_name,
                    n_samples=len(confidences),
                    raw_accuracy=np.mean(correct_array),
                    calibrated_accuracy=np.mean(correct_array),  # Same, but confidence is now calibrated
                    calibration_error=self.isotonic.compute_ece(conf_array, correct_array)
                )
        
        return metrics
    
    def calibrate_confidence(
        self,
        field_name: str,
        confidence: float
    ) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            field_name: Name of the field
            confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        return self.isotonic.calibrate(field_name, confidence)
    
    def compute_document_confidence(
        self,
        field_confidences: Dict[str, float],
        method: str = "min"
    ) -> float:
        """
        Compute document-level confidence from field confidences.
        
        Uses MIN (strict) instead of geometric mean.
        
        Args:
            field_confidences: Dict mapping field name to calibrated confidence
            method: "min" (default, strict) or "geomean"
            
        Returns:
            Document-level confidence
        """
        if not field_confidences:
            return 0.0
        
        confidences = list(field_confidences.values())
        
        if method == "min":
            return min(confidences)
        elif method == "geomean":
            return float(np.exp(np.mean(np.log(np.array(confidences) + 1e-10))))
        else:
            return min(confidences)
    
    def _check_correct(
        self,
        predicted: Any,
        ground_truth: Any,
        field_name: str
    ) -> bool:
        """Check if prediction matches ground truth."""
        if predicted is None and ground_truth is None:
            return True
        if predicted is None or ground_truth is None:
            return False
        
        # Numeric fields
        if field_name in ["horse_power", "asset_cost"]:
            try:
                p = float(predicted)
                gt = float(ground_truth)
                # Allow 5% tolerance
                return abs(p - gt) / max(abs(gt), 1) < 0.05
            except (ValueError, TypeError):
                return str(predicted) == str(ground_truth)
        
        # Visual fields
        if field_name in ["signature", "stamp"]:
            p_present = predicted.get("present", False) if isinstance(predicted, dict) else bool(predicted)
            gt_present = ground_truth.get("present", False) if isinstance(ground_truth, dict) else bool(ground_truth)
            return p_present == gt_present
        
        # Text fields
        return str(predicted).lower().strip() == str(ground_truth).lower().strip()
    
    def save(self, filepath: str):
        """Save calibrator state."""
        self.isotonic.save(filepath)
    
    def load(self, filepath: str):
        """Load calibrator state."""
        self.isotonic.load(filepath)


def load_golden_set(filepath: str) -> GoldenSet:
    """Load golden set from file."""
    return GoldenSet(filepath)


def train_calibrator(
    golden_set_path: str,
    predictions: Dict[str, Dict[str, Tuple[Any, float]]],
    fields: List[str],
    output_path: str = None
) -> Calibrator:
    """
    Train and optionally save calibrator.
    
    Args:
        golden_set_path: Path to golden_set.json
        predictions: Predictions to calibrate
        fields: Fields to calibrate
        output_path: Optional path to save calibrator
        
    Returns:
        Trained Calibrator instance
    """
    calibrator = Calibrator(golden_set_path=golden_set_path)
    calibrator.train(predictions, fields)
    
    if output_path:
        calibrator.save(output_path)
    
    return calibrator
