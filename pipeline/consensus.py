"""
Consensus Engine Module

Conflict detection and weighted voting for ensemble OCR results.
Implements per-field conflict detection with different strategies
for numeric vs textual fields.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .field_parser import (
    compute_fuzzy_score,
    compute_token_overlap,
    compute_numeric_diff,
    ParseResult
)


@dataclass
class ConflictResult:
    """Result of conflict detection between two values."""
    field_name: str
    has_conflict: bool
    value1: Any
    value2: Any
    confidence1: float
    confidence2: float
    conflict_type: Optional[str] = None  # "numeric", "textual", "visual"
    conflict_metric: float = 0.0  # diff for numeric, 1-score for textual
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "has_conflict": self.has_conflict,
            "value1": self.value1,
            "value2": self.value2,
            "confidence1": self.confidence1,
            "confidence2": self.confidence2,
            "conflict_type": self.conflict_type,
            "conflict_metric": self.conflict_metric,
            "metadata": self.metadata
        }


@dataclass
class ConsensusResult:
    """Result of consensus voting."""
    field_name: str
    final_value: Any
    final_confidence: float
    source: str  # "engine1", "engine2", "weighted", "adjudicated"
    had_conflict: bool
    conflict_resolved: bool = False
    resolution_method: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "final_value": self.final_value,
            "final_confidence": self.final_confidence,
            "source": self.source,
            "had_conflict": self.had_conflict,
            "conflict_resolved": self.conflict_resolved,
            "resolution_method": self.resolution_method,
            "metadata": self.metadata
        }


class ConflictDetector:
    """
    Detects conflicts between OCR engine outputs.
    
    Uses different strategies for:
    - Numeric fields: >5% difference triggers conflict
    - Textual fields: fuzzy score <90% OR token overlap <70% triggers conflict
    """
    
    def __init__(
        self,
        numeric_diff_threshold: float = 0.05,
        fuzzy_match_threshold: float = 90.0,
        token_overlap_threshold: float = 0.70
    ):
        """
        Initialize conflict detector.
        
        Args:
            numeric_diff_threshold: Max allowed relative difference for numeric (0.05 = 5%)
            fuzzy_match_threshold: Min fuzzy score for textual agreement (0-100)
            token_overlap_threshold: Min token overlap for textual agreement (0-1)
        """
        self.numeric_diff_threshold = numeric_diff_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.token_overlap_threshold = token_overlap_threshold
        
        # Define field types
        self.numeric_fields = {"horse_power", "asset_cost"}
        self.textual_fields = {"dealer_name", "model_name"}
        self.visual_fields = {"signature", "stamp"}
    
    def detect_conflict(
        self,
        field_name: str,
        value1: Any,
        value2: Any,
        confidence1: float = 1.0,
        confidence2: float = 1.0
    ) -> ConflictResult:
        """
        Detect if there's a conflict between two values.
        
        Args:
            field_name: Name of the field being compared
            value1: Value from first OCR engine
            value2: Value from second OCR engine
            confidence1: Confidence of first value
            confidence2: Confidence of second value
            
        Returns:
            ConflictResult with conflict details
        """
        # Handle None values
        if value1 is None and value2 is None:
            return ConflictResult(
                field_name=field_name,
                has_conflict=False,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2
            )
        
        if value1 is None or value2 is None:
            return ConflictResult(
                field_name=field_name,
                has_conflict=True,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2,
                conflict_type="missing_value",
                conflict_metric=1.0
            )
        
        # Route to appropriate detector
        if field_name in self.numeric_fields:
            return self._detect_numeric_conflict(
                field_name, value1, value2, confidence1, confidence2
            )
        elif field_name in self.textual_fields:
            return self._detect_textual_conflict(
                field_name, value1, value2, confidence1, confidence2
            )
        elif field_name in self.visual_fields:
            return self._detect_visual_conflict(
                field_name, value1, value2, confidence1, confidence2
            )
        else:
            # Default to textual comparison
            return self._detect_textual_conflict(
                field_name, value1, value2, confidence1, confidence2
            )
    
    def _detect_numeric_conflict(
        self,
        field_name: str,
        value1: float,
        value2: float,
        confidence1: float,
        confidence2: float
    ) -> ConflictResult:
        """Detect conflict for numeric fields."""
        try:
            v1 = float(value1)
            v2 = float(value2)
            diff = compute_numeric_diff(v1, v2)
            
            return ConflictResult(
                field_name=field_name,
                has_conflict=diff > self.numeric_diff_threshold,
                value1=v1,
                value2=v2,
                confidence1=confidence1,
                confidence2=confidence2,
                conflict_type="numeric",
                conflict_metric=diff,
                metadata={"diff_percent": diff * 100}
            )
        except (ValueError, TypeError):
            # If conversion fails, treat as conflict
            return ConflictResult(
                field_name=field_name,
                has_conflict=True,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2,
                conflict_type="numeric_parse_error"
            )
    
    def _detect_textual_conflict(
        self,
        field_name: str,
        value1: str,
        value2: str,
        confidence1: float,
        confidence2: float
    ) -> ConflictResult:
        """Detect conflict for textual fields."""
        v1 = str(value1).strip()
        v2 = str(value2).strip()
        
        # Compute both metrics
        fuzzy_score = compute_fuzzy_score(v1, v2)
        token_overlap = compute_token_overlap(v1, v2)
        
        # Conflict if EITHER metric fails
        has_conflict = (
            fuzzy_score < self.fuzzy_match_threshold or
            token_overlap < self.token_overlap_threshold
        )
        
        return ConflictResult(
            field_name=field_name,
            has_conflict=has_conflict,
            value1=v1,
            value2=v2,
            confidence1=confidence1,
            confidence2=confidence2,
            conflict_type="textual",
            conflict_metric=1.0 - (fuzzy_score / 100.0),
            metadata={
                "fuzzy_score": fuzzy_score,
                "token_overlap": token_overlap
            }
        )
    
    def _detect_visual_conflict(
        self,
        field_name: str,
        value1: Dict,
        value2: Dict,
        confidence1: float,
        confidence2: float
    ) -> ConflictResult:
        """Detect conflict for visual fields (signature/stamp)."""
        # Visual fields are compared by presence and bbox overlap
        present1 = value1.get("present", False) if isinstance(value1, dict) else bool(value1)
        present2 = value2.get("present", False) if isinstance(value2, dict) else bool(value2)
        
        if present1 != present2:
            return ConflictResult(
                field_name=field_name,
                has_conflict=True,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2,
                conflict_type="visual_presence",
                conflict_metric=1.0
            )
        
        if not present1:  # Both not present
            return ConflictResult(
                field_name=field_name,
                has_conflict=False,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2
            )
        
        # Compare bboxes if both present
        bbox1 = value1.get("bbox", []) if isinstance(value1, dict) else []
        bbox2 = value2.get("bbox", []) if isinstance(value2, dict) else []
        
        if bbox1 and bbox2:
            iou = self._compute_bbox_iou(bbox1, bbox2)
            has_conflict = iou < 0.5  # Conflict if IoU < 0.5
            
            return ConflictResult(
                field_name=field_name,
                has_conflict=has_conflict,
                value1=value1,
                value2=value2,
                confidence1=confidence1,
                confidence2=confidence2,
                conflict_type="visual_bbox",
                conflict_metric=1.0 - iou,
                metadata={"iou": iou}
            )
        
        return ConflictResult(
            field_name=field_name,
            has_conflict=False,
            value1=value1,
            value2=value2,
            confidence1=confidence1,
            confidence2=confidence2
        )
    
    def _compute_bbox_iou(self, bbox1: List, bbox2: List) -> float:
        """Compute IoU between two bboxes in [x, y, w, h] format."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to xyxy
        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
        
        # Intersection
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area


class ConsensusEngine:
    """
    Consensus engine for combining OCR outputs.
    
    Implements:
    - Conflict detection using ConflictDetector
    - Weighted voting using learned weights
    - Fallback to weighted average when no adjudication needed
    """
    
    def __init__(
        self,
        conflict_detector: ConflictDetector = None,
        default_weights: Dict[str, float] = None
    ):
        """
        Initialize consensus engine.
        
        Args:
            conflict_detector: ConflictDetector instance
            default_weights: Default weights for each engine {"paddle": 0.5, "deepseek": 0.5}
        """
        self.conflict_detector = conflict_detector or ConflictDetector()
        self.weights = default_weights or {
            "paddle": 0.55,  # Slightly prefer PaddleOCR by default
            "deepseek": 0.45
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set engine weights (learned from golden set).
        
        Args:
            weights: Dictionary mapping engine names to weights
        """
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def compute_consensus(
        self,
        field_name: str,
        engine_results: Dict[str, Tuple[Any, float]]
    ) -> Tuple[ConsensusResult, Optional[ConflictResult]]:
        """
        Compute consensus for a field from multiple engine results.
        
        Args:
            field_name: Name of the field
            engine_results: Dict mapping engine name to (value, confidence) tuple
            
        Returns:
            Tuple of (ConsensusResult, ConflictResult or None if no conflict)
        """
        engines = list(engine_results.keys())
        
        if len(engines) == 0:
            return ConsensusResult(
                field_name=field_name,
                final_value=None,
                final_confidence=0.0,
                source="none",
                had_conflict=False
            ), None
        
        if len(engines) == 1:
            engine = engines[0]
            value, conf = engine_results[engine]
            return ConsensusResult(
                field_name=field_name,
                final_value=value,
                final_confidence=conf,
                source=engine,
                had_conflict=False
            ), None
        
        # Get two engine results
        e1, e2 = engines[0], engines[1]
        v1, c1 = engine_results[e1]
        v2, c2 = engine_results[e2]
        
        # Detect conflict
        conflict = self.conflict_detector.detect_conflict(
            field_name, v1, v2, c1, c2
        )
        
        if not conflict.has_conflict:
            # No conflict - use weighted average
            return self._weighted_consensus(
                field_name, e1, v1, c1, e2, v2, c2
            ), conflict
        else:
            # Conflict detected - mark for adjudication
            # Return the higher confidence value as preliminary
            if c1 >= c2:
                return ConsensusResult(
                    field_name=field_name,
                    final_value=v1,
                    final_confidence=c1 * 0.7,  # Reduce confidence due to conflict
                    source=e1,
                    had_conflict=True,
                    conflict_resolved=False,
                    resolution_method="needs_adjudication"
                ), conflict
            else:
                return ConsensusResult(
                    field_name=field_name,
                    final_value=v2,
                    final_confidence=c2 * 0.7,
                    source=e2,
                    had_conflict=True,
                    conflict_resolved=False,
                    resolution_method="needs_adjudication"
                ), conflict
    
    def _weighted_consensus(
        self,
        field_name: str,
        e1: str, v1: Any, c1: float,
        e2: str, v2: Any, c2: float
    ) -> ConsensusResult:
        """Compute weighted consensus for agreeing values."""
        w1 = self.weights.get(e1, 0.5)
        w2 = self.weights.get(e2, 0.5)
        
        # Weighted confidence
        final_conf = (c1 * w1 + c2 * w2) / (w1 + w2)
        
        # For numeric fields, compute weighted average
        if field_name in self.conflict_detector.numeric_fields:
            try:
                weighted_value = (float(v1) * w1 * c1 + float(v2) * w2 * c2) / (w1 * c1 + w2 * c2)
                return ConsensusResult(
                    field_name=field_name,
                    final_value=weighted_value,
                    final_confidence=final_conf,
                    source="weighted",
                    had_conflict=False,
                    metadata={"engine_values": {e1: v1, e2: v2}}
                )
            except (ValueError, TypeError):
                pass
        
        # For non-numeric, use higher confidence value
        if c1 * w1 >= c2 * w2:
            return ConsensusResult(
                field_name=field_name,
                final_value=v1,
                final_confidence=final_conf,
                source=e1,
                had_conflict=False
            )
        else:
            return ConsensusResult(
                field_name=field_name,
                final_value=v2,
                final_confidence=final_conf,
                source=e2,
                had_conflict=False
            )
    
    def resolve_conflict(
        self,
        consensus: ConsensusResult,
        conflict: ConflictResult,
        resolved_value: Any,
        resolved_confidence: float,
        resolution_method: str
    ) -> ConsensusResult:
        """
        Update consensus result after adjudication resolves a conflict.
        
        Args:
            consensus: Original consensus result
            conflict: Conflict that was resolved
            resolved_value: Value determined by adjudication
            resolved_confidence: Confidence in resolved value
            resolution_method: Method used to resolve (tier1, tier2_slm, tier3_vlm)
            
        Returns:
            Updated ConsensusResult
        """
        return ConsensusResult(
            field_name=consensus.field_name,
            final_value=resolved_value,
            final_confidence=resolved_confidence,
            source="adjudicated",
            had_conflict=True,
            conflict_resolved=True,
            resolution_method=resolution_method,
            metadata={
                "original_values": {
                    "value1": conflict.value1,
                    "value2": conflict.value2
                },
                "conflict_type": conflict.conflict_type
            }
        )


def detect_all_conflicts(
    fields: List[str],
    engine1_results: Dict[str, Tuple[Any, float]],
    engine2_results: Dict[str, Tuple[Any, float]],
    detector: ConflictDetector = None
) -> Dict[str, ConflictResult]:
    """
    Detect conflicts across all fields.
    
    Args:
        fields: List of field names to check
        engine1_results: Dict mapping field name to (value, confidence)
        engine2_results: Dict mapping field name to (value, confidence)
        detector: ConflictDetector instance
        
    Returns:
        Dict mapping field name to ConflictResult
    """
    detector = detector or ConflictDetector()
    conflicts = {}
    
    for field_name in fields:
        v1, c1 = engine1_results.get(field_name, (None, 0.0))
        v2, c2 = engine2_results.get(field_name, (None, 0.0))
        
        conflicts[field_name] = detector.detect_conflict(
            field_name, v1, v2, c1, c2
        )
    
    return conflicts


def compute_all_consensus(
    fields: List[str],
    engine_results: Dict[str, Dict[str, Tuple[Any, float]]],
    consensus_engine: ConsensusEngine = None
) -> Dict[str, Tuple[ConsensusResult, Optional[ConflictResult]]]:
    """
    Compute consensus across all fields from all engines.
    
    Args:
        fields: List of field names
        engine_results: Dict mapping engine name to {field: (value, conf)}
        consensus_engine: ConsensusEngine instance
        
    Returns:
        Dict mapping field name to (ConsensusResult, ConflictResult or None)
    """
    engine = consensus_engine or ConsensusEngine()
    results = {}
    
    for field_name in fields:
        # Collect results for this field from all engines
        field_results = {}
        for eng_name, eng_data in engine_results.items():
            if field_name in eng_data:
                field_results[eng_name] = eng_data[field_name]
        
        results[field_name] = engine.compute_consensus(field_name, field_results)
    
    return results
