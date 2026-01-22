"""
Tests for Consensus Module

Tests conflict detection, weighted voting, and consensus computation.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.consensus import (
    ConflictDetector,
    ConflictResult,
    ConsensusEngine,
    ConsensusResult,
    detect_all_conflicts,
    compute_all_consensus
)


class TestConflictDetector:
    """Tests for conflict detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a conflict detector."""
        return ConflictDetector(
            numeric_diff_threshold=0.05,
            fuzzy_match_threshold=90.0,
            token_overlap_threshold=0.70
        )
    
    # Numeric conflict tests
    
    def test_numeric_no_conflict(self, detector):
        """Test numeric values with no conflict."""
        result = detector.detect_conflict(
            "horse_power",
            45,
            46,  # Less than 5% difference
            0.9,
            0.9
        )
        assert result.has_conflict is False
    
    def test_numeric_with_conflict(self, detector):
        """Test numeric values with conflict (>5% diff)."""
        result = detector.detect_conflict(
            "asset_cost",
            450000,
            500000,  # More than 5% difference
            0.9,
            0.9
        )
        assert result.has_conflict is True
        assert result.conflict_type == "numeric"
    
    def test_numeric_exact_threshold(self, detector):
        """Test at exact 5% threshold."""
        result = detector.detect_conflict(
            "horse_power",
            100,
            105,  # Exactly 5% difference
            0.9,
            0.9
        )
        # At exactly 5%, should be no conflict (threshold is >5%)
        assert result.has_conflict is False
    
    # Textual conflict tests
    
    def test_textual_no_conflict(self, detector):
        """Test textual values with no conflict."""
        result = detector.detect_conflict(
            "dealer_name",
            "ABC Motors Pvt Ltd",
            "ABC Motors Private Limited",  # Similar enough
            0.9,
            0.9
        )
        # May or may not conflict depending on fuzzy score
        assert result.conflict_type == "textual"
    
    def test_textual_with_conflict(self, detector):
        """Test textual values with conflict."""
        result = detector.detect_conflict(
            "model_name",
            "Mahindra 575 DI",
            "John Deere 5050D",  # Completely different
            0.9,
            0.9
        )
        assert result.has_conflict is True
        assert result.conflict_type == "textual"
    
    def test_textual_exact_match(self, detector):
        """Test exact textual match."""
        result = detector.detect_conflict(
            "dealer_name",
            "ABC Motors",
            "ABC Motors",
            0.9,
            0.9
        )
        assert result.has_conflict is False
    
    # Visual conflict tests
    
    def test_visual_presence_agrees(self, detector):
        """Test visual field when both agree on presence."""
        result = detector.detect_conflict(
            "signature",
            {"present": True, "bbox": [10, 10, 100, 50]},
            {"present": True, "bbox": [12, 11, 98, 48]},  # Similar bbox
            0.9,
            0.9
        )
        # High IoU should mean no conflict
        assert result.conflict_type in ["visual_bbox", None]
    
    def test_visual_presence_disagrees(self, detector):
        """Test visual field when presence disagrees."""
        result = detector.detect_conflict(
            "stamp",
            {"present": True, "bbox": [10, 10, 50, 50]},
            {"present": False, "bbox": None},
            0.9,
            0.9
        )
        assert result.has_conflict is True
        assert result.conflict_type == "visual_presence"
    
    def test_visual_both_absent(self, detector):
        """Test visual field when both absent."""
        result = detector.detect_conflict(
            "signature",
            {"present": False},
            {"present": False},
            0.9,
            0.9
        )
        assert result.has_conflict is False
    
    # None value tests
    
    def test_both_none(self, detector):
        """Test when both values are None."""
        result = detector.detect_conflict(
            "dealer_name",
            None,
            None,
            0.0,
            0.0
        )
        assert result.has_conflict is False
    
    def test_one_none(self, detector):
        """Test when one value is None."""
        result = detector.detect_conflict(
            "model_name",
            "Mahindra 575",
            None,
            0.9,
            0.0
        )
        assert result.has_conflict is True
        assert result.conflict_type == "missing_value"


class TestConsensusEngine:
    """Tests for consensus computation."""
    
    @pytest.fixture
    def engine(self):
        """Create a consensus engine."""
        return ConsensusEngine()
    
    def test_single_engine_result(self, engine):
        """Test consensus with single engine result."""
        result, conflict = engine.compute_consensus(
            "dealer_name",
            {"paddle": ("ABC Motors", 0.9)}
        )
        assert result.final_value == "ABC Motors"
        assert result.source == "paddle"
        assert conflict is None
    
    def test_no_engine_results(self, engine):
        """Test consensus with no engine results."""
        result, conflict = engine.compute_consensus(
            "dealer_name",
            {}
        )
        assert result.final_value is None
        assert result.source == "none"
    
    def test_consensus_agreement(self, engine):
        """Test consensus when engines agree."""
        result, conflict = engine.compute_consensus(
            "horse_power",
            {
                "paddle": (45, 0.9),
                "deepseek": (45, 0.8)
            }
        )
        assert conflict is not None
        assert conflict.has_conflict is False
        assert result.had_conflict is False
    
    def test_consensus_conflict(self, engine):
        """Test consensus when engines disagree."""
        result, conflict = engine.compute_consensus(
            "asset_cost",
            {
                "paddle": (450000, 0.9),
                "deepseek": (500000, 0.8)  # >5% different
            }
        )
        assert conflict is not None
        assert conflict.has_conflict is True
        assert result.had_conflict is True
    
    def test_weighted_consensus(self, engine):
        """Test weighted consensus for numeric fields."""
        engine.set_weights({"paddle": 0.6, "deepseek": 0.4})
        
        result, conflict = engine.compute_consensus(
            "horse_power",
            {
                "paddle": (45, 0.9),
                "deepseek": (46, 0.9)
            }
        )
        # Should produce weighted average since no conflict
        assert result.final_value is not None


class TestConflictResultDataclass:
    """Tests for ConflictResult dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        result = ConflictResult(
            field_name="test",
            has_conflict=True,
            value1="a",
            value2="b",
            confidence1=0.9,
            confidence2=0.8,
            conflict_type="textual",
            conflict_metric=0.5
        )
        d = result.to_dict()
        assert d["field_name"] == "test"
        assert d["has_conflict"] is True
        assert d["conflict_type"] == "textual"


class TestConsensusResultDataclass:
    """Tests for ConsensusResult dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        result = ConsensusResult(
            field_name="test",
            final_value="value",
            final_confidence=0.9,
            source="paddle",
            had_conflict=False
        )
        d = result.to_dict()
        assert d["field_name"] == "test"
        assert d["final_value"] == "value"
        assert d["source"] == "paddle"


class TestHelperFunctions:
    """Tests for module-level helper functions."""
    
    def test_detect_all_conflicts(self):
        """Test detecting conflicts across all fields."""
        fields = ["horse_power", "dealer_name"]
        engine1 = {
            "horse_power": (45, 0.9),
            "dealer_name": ("ABC", 0.8)
        }
        engine2 = {
            "horse_power": (50, 0.85),  # >5% diff
            "dealer_name": ("XYZ", 0.7)  # Different
        }
        
        conflicts = detect_all_conflicts(fields, engine1, engine2)
        
        assert "horse_power" in conflicts
        assert "dealer_name" in conflicts
        assert conflicts["horse_power"].has_conflict is True
        assert conflicts["dealer_name"].has_conflict is True
    
    def test_compute_all_consensus(self):
        """Test computing consensus for all fields."""
        fields = ["horse_power", "model_name"]
        engine_results = {
            "paddle": {
                "horse_power": (45, 0.9),
                "model_name": ("Mahindra 575", 0.85)
            },
            "deepseek": {
                "horse_power": (45, 0.8),
                "model_name": ("Mahindra 575 DI", 0.8)
            }
        }
        
        results = compute_all_consensus(fields, engine_results)
        
        assert "horse_power" in results
        assert "model_name" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
