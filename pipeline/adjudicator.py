"""
Adjudicator Module

Three-tier adjudication ladder for conflict resolution:
- Tier 1: Deterministic rules (<1ms)
- Tier 2: SLM Judge (~$0.001)
- Tier 3: VLM Judge (~$0.02, <10% invocation rate)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .field_parser import (
    normalize_native_digits,
    strip_currency,
    compute_fuzzy_score,
    compute_token_overlap
)
from .slm_judge import SLMJudge, get_slm_judge, SLMJudgment
from .vlm_judge import VLMJudge, get_vlm_judge, VLMJudgment
from .consensus import ConflictResult, ConsensusResult


@dataclass
class AdjudicationResult:
    """Result from the adjudication ladder."""
    field_name: str
    resolved_value: Any
    confidence: float
    tier_used: int  # 1, 2, or 3
    resolution_method: str
    latency: float
    cost: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "field_name": self.field_name,
            "resolved_value": self.resolved_value,
            "confidence": self.confidence,
            "tier_used": self.tier_used,
            "resolution_method": self.resolution_method,
            "latency": self.latency,
            "cost": self.cost,
            "metadata": self.metadata
        }


class Tier1Resolver:
    """
    Tier 1: Deterministic rules for conflict resolution.
    
    Handles:
    - Devanagari/Gujarati digit normalization
    - Currency stripping
    - Range parsing
    - Token-set Jaccard similarity
    - Common OCR error corrections
    """
    
    # Common OCR character confusions
    OCR_CORRECTIONS = {
        'O': '0', 'o': '0', 'l': '1', 'I': '1',
        'S': '5', 'B': '8', 'Z': '2', 'G': '6',
        'q': '9', 'g': '9'
    }
    
    def __init__(
        self,
        fuzzy_threshold: float = 95.0,
        jaccard_threshold: float = 0.85
    ):
        """
        Initialize Tier 1 resolver.
        
        Args:
            fuzzy_threshold: Fuzzy score for considering values equivalent
            jaccard_threshold: Jaccard threshold for token set similarity
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.jaccard_threshold = jaccard_threshold
    
    def resolve(
        self,
        conflict: ConflictResult,
        context: str = ""
    ) -> Optional[AdjudicationResult]:
        """
        Attempt to resolve conflict using deterministic rules.
        
        Args:
            conflict: ConflictResult to resolve
            context: Additional context from document
            
        Returns:
            AdjudicationResult if resolved, None otherwise
        """
        start_time = time.time()
        
        value1 = conflict.value1
        value2 = conflict.value2
        field_name = conflict.field_name
        
        # Handle None values
        if value1 is None and value2 is not None:
            return AdjudicationResult(
                field_name=field_name,
                resolved_value=value2,
                confidence=conflict.confidence2 * 0.8,
                tier_used=1,
                resolution_method="prefer_non_null",
                latency=time.time() - start_time,
                cost=0.0
            )
        elif value2 is None and value1 is not None:
            return AdjudicationResult(
                field_name=field_name,
                resolved_value=value1,
                confidence=conflict.confidence1 * 0.8,
                tier_used=1,
                resolution_method="prefer_non_null",
                latency=time.time() - start_time,
                cost=0.0
            )
        elif value1 is None and value2 is None:
            return None
        
        # Route to appropriate resolver based on conflict type
        if conflict.conflict_type == "numeric":
            return self._resolve_numeric(conflict, context, start_time)
        elif conflict.conflict_type == "textual":
            return self._resolve_textual(conflict, context, start_time)
        else:
            return None
    
    def _resolve_numeric(
        self,
        conflict: ConflictResult,
        context: str,
        start_time: float
    ) -> Optional[AdjudicationResult]:
        """Resolve numeric conflicts."""
        # Normalize both values
        v1_str = str(conflict.value1)
        v2_str = str(conflict.value2)
        
        v1_normalized = self._normalize_numeric_string(v1_str)
        v2_normalized = self._normalize_numeric_string(v2_str)
        
        # Check if normalization resolves conflict
        if v1_normalized == v2_normalized:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=float(v1_normalized) if v1_normalized else conflict.value1,
                confidence=max(conflict.confidence1, conflict.confidence2),
                tier_used=1,
                resolution_method="numeric_normalization",
                latency=time.time() - start_time,
                cost=0.0
            )
        
        # Try OCR correction
        v1_corrected = self._apply_ocr_corrections(v1_str)
        v2_corrected = self._apply_ocr_corrections(v2_str)
        
        v1_corr_normalized = self._normalize_numeric_string(v1_corrected)
        v2_corr_normalized = self._normalize_numeric_string(v2_corrected)
        
        if v1_corr_normalized == v2_corr_normalized:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=float(v1_corr_normalized) if v1_corr_normalized else conflict.value1,
                confidence=max(conflict.confidence1, conflict.confidence2) * 0.9,
                tier_used=1,
                resolution_method="ocr_correction",
                latency=time.time() - start_time,
                cost=0.0
            )
        
        # Check if one value appears in context
        if context:
            if v1_normalized in context and v2_normalized not in context:
                return AdjudicationResult(
                    field_name=conflict.field_name,
                    resolved_value=conflict.value1,
                    confidence=conflict.confidence1 * 0.95,
                    tier_used=1,
                    resolution_method="context_match",
                    latency=time.time() - start_time,
                    cost=0.0
                )
            elif v2_normalized in context and v1_normalized not in context:
                return AdjudicationResult(
                    field_name=conflict.field_name,
                    resolved_value=conflict.value2,
                    confidence=conflict.confidence2 * 0.95,
                    tier_used=1,
                    resolution_method="context_match",
                    latency=time.time() - start_time,
                    cost=0.0
                )
        
        return None  # Cannot resolve at Tier 1
    
    def _resolve_textual(
        self,
        conflict: ConflictResult,
        context: str,
        start_time: float
    ) -> Optional[AdjudicationResult]:
        """Resolve textual conflicts."""
        v1 = str(conflict.value1).strip().lower()
        v2 = str(conflict.value2).strip().lower()
        
        # Normalize and compare
        v1_normalized = self._normalize_text(v1)
        v2_normalized = self._normalize_text(v2)
        
        # Check if normalization resolves conflict
        if v1_normalized == v2_normalized:
            # Prefer longer original (more complete)
            resolved = conflict.value1 if len(str(conflict.value1)) >= len(str(conflict.value2)) else conflict.value2
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=resolved,
                confidence=max(conflict.confidence1, conflict.confidence2),
                tier_used=1,
                resolution_method="text_normalization",
                latency=time.time() - start_time,
                cost=0.0
            )
        
        # Token-set Jaccard similarity
        tokens1 = set(v1_normalized.split())
        tokens2 = set(v2_normalized.split())
        
        if tokens1 and tokens2:
            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            
            if jaccard >= self.jaccard_threshold:
                # High overlap - prefer longer value
                resolved = conflict.value1 if len(v1) >= len(v2) else conflict.value2
                return AdjudicationResult(
                    field_name=conflict.field_name,
                    resolved_value=resolved,
                    confidence=max(conflict.confidence1, conflict.confidence2) * jaccard,
                    tier_used=1,
                    resolution_method="jaccard_similarity",
                    latency=time.time() - start_time,
                    cost=0.0,
                    metadata={"jaccard": jaccard}
                )
        
        # Check subset relationship
        if v1_normalized in v2_normalized:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=conflict.value2,
                confidence=conflict.confidence2 * 0.95,
                tier_used=1,
                resolution_method="subset_match",
                latency=time.time() - start_time,
                cost=0.0
            )
        elif v2_normalized in v1_normalized:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=conflict.value1,
                confidence=conflict.confidence1 * 0.95,
                tier_used=1,
                resolution_method="subset_match",
                latency=time.time() - start_time,
                cost=0.0
            )
        
        return None  # Cannot resolve at Tier 1
    
    def _normalize_numeric_string(self, s: str) -> str:
        """Normalize a numeric string."""
        # Convert native digits
        s = normalize_native_digits(s)
        # Strip currency
        s = strip_currency(s)
        # Remove commas and spaces
        s = re.sub(r'[,\s]', '', s)
        # Extract number
        match = re.search(r'(\d+\.?\d*)', s)
        return match.group(1) if match else s
    
    def _apply_ocr_corrections(self, s: str) -> str:
        """Apply common OCR error corrections."""
        result = s
        for wrong, correct in self.OCR_CORRECTIONS.items():
            result = result.replace(wrong, correct)
        return result
    
    def _normalize_text(self, s: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        s = s.lower()
        # Remove punctuation except spaces
        s = re.sub(r'[^\w\s]', '', s)
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        return s


class AdjudicationLadder:
    """
    Three-tier adjudication ladder.
    
    Escalation:
    - Tier 1: Deterministic rules (always try first)
    - Tier 2: SLM Judge (if Tier 1 fails)
    - Tier 3: VLM Judge (if Tier 2 returns UNCERTAIN, and mode allows)
    """
    
    # Cost constants
    COSTS = {
        1: 0.0,
        2: 0.001,
        3: 0.02
    }
    
    def __init__(
        self,
        tier1_resolver: Tier1Resolver = None,
        slm_judge: SLMJudge = None,
        vlm_judge: VLMJudge = None,
        use_tier3: bool = True
    ):
        """
        Initialize adjudication ladder.
        
        Args:
            tier1_resolver: Tier 1 deterministic resolver
            slm_judge: Tier 2 SLM judge
            vlm_judge: Tier 3 VLM judge
            use_tier3: Whether to use Tier 3 (disabled in cpu-lite mode)
        """
        self.tier1 = tier1_resolver or Tier1Resolver()
        self.slm_judge = slm_judge
        self.vlm_judge = vlm_judge
        self.use_tier3 = use_tier3
    
    def resolve(
        self,
        conflict: ConflictResult,
        context: str = "",
        image_path: str = None,
        bbox: List[int] = None
    ) -> AdjudicationResult:
        """
        Resolve a conflict through the adjudication ladder.
        
        Args:
            conflict: ConflictResult to resolve
            context: Document context for SLM
            image_path: Image path for VLM
            bbox: Bounding box for VLM crop
            
        Returns:
            AdjudicationResult from the tier that resolved it
        """
        total_start = time.time()
        
        # Tier 1: Deterministic rules
        result = self.tier1.resolve(conflict, context)
        if result is not None:
            return result
        
        # Tier 2: SLM Judge
        slm = self.slm_judge or get_slm_judge()
        slm_result = slm.judge(
            conflict.field_name,
            str(conflict.value1),
            str(conflict.value2),
            context
        )
        
        if slm_result.status == "RESOLVED":
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=slm_result.resolved_value,
                confidence=slm_result.confidence,
                tier_used=2,
                resolution_method="slm_judge",
                latency=slm_result.latency,
                cost=self.COSTS[2],
                metadata={"slm_reasoning": slm_result.reasoning}
            )
        
        # Tier 3: VLM Judge (if enabled and SLM uncertain)
        if self.use_tier3 and slm_result.status == "UNCERTAIN":
            if image_path and bbox:
                vlm = self.vlm_judge or get_vlm_judge()
                vlm_result = vlm.judge_from_image(
                    image_path,
                    bbox,
                    conflict.field_name,
                    str(conflict.value1),
                    str(conflict.value2)
                )
                
                if vlm_result.status == "RESOLVED":
                    return AdjudicationResult(
                        field_name=conflict.field_name,
                        resolved_value=vlm_result.resolved_value,
                        confidence=vlm_result.confidence,
                        tier_used=3,
                        resolution_method="vlm_judge",
                        latency=slm_result.latency + vlm_result.latency,
                        cost=self.COSTS[2] + self.COSTS[3],
                        metadata={"vlm_reasoning": vlm_result.reasoning}
                    )
        
        # Fallback: Use higher confidence value
        total_latency = time.time() - total_start
        
        if conflict.confidence1 >= conflict.confidence2:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=conflict.value1,
                confidence=conflict.confidence1 * 0.6,  # Low confidence
                tier_used=2 if not self.use_tier3 else 3,
                resolution_method="confidence_fallback",
                latency=total_latency,
                cost=self.COSTS[2] + (self.COSTS[3] if self.use_tier3 else 0),
                metadata={"fallback": True}
            )
        else:
            return AdjudicationResult(
                field_name=conflict.field_name,
                resolved_value=conflict.value2,
                confidence=conflict.confidence2 * 0.6,
                tier_used=2 if not self.use_tier3 else 3,
                resolution_method="confidence_fallback",
                latency=total_latency,
                cost=self.COSTS[2] + (self.COSTS[3] if self.use_tier3 else 0),
                metadata={"fallback": True}
            )


def resolve_conflict(
    conflict: ConflictResult,
    context: str = "",
    image_path: str = None,
    bbox: List[int] = None,
    use_tier3: bool = True
) -> AdjudicationResult:
    """
    Convenience function to resolve a conflict.
    
    Args:
        conflict: ConflictResult to resolve
        context: Document context
        image_path: Image path for VLM
        bbox: Bounding box for VLM
        use_tier3: Whether to use Tier 3
        
    Returns:
        AdjudicationResult
    """
    ladder = AdjudicationLadder(use_tier3=use_tier3)
    return ladder.resolve(conflict, context, image_path, bbox)
