"""
Main Pipeline Module

Full document processing orchestration combining:
- Parallel OCR (PaddleOCR + DeepSeek)
- YOLO signature/stamp detection with verification
- Field parsing and extraction
- Conflict detection and consensus
- Three-tier adjudication
- Confidence calibration
- Cost and latency tracking
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ocr_engines import parallel_ocr, get_full_text, OCREngineOutput
from .detectors import run_all_detections, DetectorOutput
from .field_parser import (
    parse_dealer_name, parse_model_name,
    parse_horse_power, parse_asset_cost,
    extract_field_from_text, ParseResult
)
from .consensus import (
    ConsensusEngine, ConflictDetector,
    ConsensusResult, ConflictResult,
    compute_all_consensus
)
from .adjudicator import AdjudicationLadder, resolve_conflict, AdjudicationResult
from .slm_judge import get_slm_judge
from .vlm_judge import get_vlm_judge
from .calibration import Calibrator, GoldenSet
from .cost_tracker import CostTracker, get_tracker
from .json_logger import get_json_logger, JSONResultLogger
from pipeline import run_logger


@dataclass
class FieldExtraction:
    """Extraction result for a single field."""
    field_name: str
    value: Any
    confidence: float
    calibrated_confidence: float = 0.0
    source: str = ""  # "paddle", "deepseek", "consensus", "adjudicated"
    bbox: Optional[List[int]] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Return full field details."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence": self.confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "source": self.source,
            "bbox": self.bbox,
            "metadata": self.metadata
        }
    
    def to_simple_value(self) -> Any:
        """
        Return simplified value for output.
        For visual fields (signature/stamp), returns {present, bbox}.
        For text/numeric fields, returns just the value.
        """
        if self.field_name in ["signature", "stamp"]:
            if isinstance(self.value, dict):
                return {
                    "present": self.value.get("present", False),
                    "bbox": self.value.get("bbox")
                }
            return {"present": False, "bbox": None}
        return self.value


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""
    doc_id: str
    image_path: str
    fields: Dict[str, FieldExtraction]
    document_confidence: float
    total_cost: float
    total_latency: float
    success: bool
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """
        Return simplified output matching the required format:
        {
            "doc_id": "invoice_001",
            "fields": {
                "dealer_name": "ABC Tractors Pvt Ltd",
                "model_name": "Mahindra 575 DI",
                "horse_power": 50,
                "asset_cost": 525000,
                "signature": {"present": true, "bbox": [100, 200, 300, 250]},
                "stamp": {"present": true, "bbox": [400, 500, 500, 550]}
            },
            "confidence": 0.96,
            "processing_time_sec": 3.8,
            "cost_estimate_usd": 0.002
        }
        """
        return {
            "doc_id": self.doc_id,
            "fields": {k: v.to_simple_value() for k, v in self.fields.items()},
            "confidence": round(self.document_confidence, 2),
            "processing_time_sec": round(self.total_latency, 1),
            "cost_estimate_usd": round(self.total_cost, 4)
        }
    
    def to_detailed_dict(self) -> Dict:
        """Return full detailed output for debugging/analysis."""
        return {
            "doc_id": self.doc_id,
            "image_path": self.image_path,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "document_confidence": self.document_confidence,
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
            "success": self.success,
            "errors": self.errors,
            "metadata": self.metadata
        }
    
    def get_field_value(self, field_name: str) -> Any:
        """Get value for a specific field."""
        if field_name in self.fields:
            return self.fields[field_name].value
        return None


class DocumentProcessor:
    """
    Main document processing pipeline.
    
    Orchestrates the full extraction flow:
    1. Parallel OCR with PaddleOCR + DeepSeek
    2. YOLO detection for signature/stamp
    3. Field parsing with conflict detection
    4. Adjudication ladder for conflicts
    5. Confidence calibration
    6. Cost tracking
    """
    
    # Fields to extract
    FIELDS = [
        "dealer_name",
        "model_name",
        "horse_power",
        "asset_cost",
        "signature",
        "stamp"
    ]
    
    def __init__(
        self,
        use_tier3: bool = True,
        calibrator: Calibrator = None,
        cost_tracker: CostTracker = None,
        dealer_master_list: List[str] = None
    ):
        """
        Initialize document processor.
        
        Args:
            use_tier3: Whether to use Tier 3 VLM (disabled in cpu-lite mode)
            calibrator: Calibrator for confidence scores
            cost_tracker: Cost tracker instance
            dealer_master_list: List of known dealer names for matching
        """
        self.use_tier3 = use_tier3
        self.calibrator = calibrator
        self.tracker = cost_tracker or get_tracker()
        self.dealer_master_list = dealer_master_list or []
        
        # Initialize components
        self.conflict_detector = ConflictDetector()
        self.consensus_engine = ConsensusEngine(conflict_detector=self.conflict_detector)
        self.adjudicator = AdjudicationLadder(use_tier3=use_tier3)
    
    def process_document(self, image_path: str, doc_id: str = None) -> ExtractionResult:
        """
        Process a single document image.
        
        Args:
            image_path: Path to document image
            doc_id: Optional document ID (defaults to filename)
            
        Returns:
            ExtractionResult with all extracted fields
        """
        start_time = time.time()
        doc_id = doc_id or Path(image_path).stem
        errors = []
        
        # Start cost tracking
        self.tracker.start_document(doc_id, image_path)
        
        # Log document start (run_logger should already be initialized)
        run_logger.start_new_run()  # Idempotent - only creates file on first call
        run_logger.log_document_start(doc_id, image_path)
        
        try:
            # Step 1: Parallel OCR
            ocr_outputs = self._run_ocr(image_path)
            
            # Step 2: YOLO Detection
            detector_outputs = self._run_detection(image_path, ocr_outputs)
            
            # Step 3: Field Extraction
            field_extractions = self._extract_fields(ocr_outputs, detector_outputs)
            
            # Step 4: Consensus & Adjudication
            resolved_fields = self._resolve_conflicts(
                field_extractions, 
                image_path,
                ocr_outputs
            )
            
            # Step 5: Calibration
            calibrated_fields = self._calibrate(resolved_fields)
            
            # Compute document confidence (min of all fields - strict)
            field_confidences = [f.calibrated_confidence for f in calibrated_fields.values()]
            doc_confidence = min(field_confidences) if field_confidences else 0.0
            
        except Exception as e:
            import traceback
            error_msg = f"Processing error: {str(e)}"
            errors.append(error_msg)
            run_logger.logger.error(f"PROCESSING_ERROR | DocID: {doc_id} | Error: {error_msg}")
            run_logger.logger.error(f"TRACEBACK | {traceback.format_exc()}")
            calibrated_fields = {}
            doc_confidence = 0.0
        
        # End tracking
        self.tracker.end_document()
        
        total_latency = time.time() - start_time
        total_cost = self.tracker.get_document_cost(doc_id)
        
        # Log to JSON logger
        json_logger = get_json_logger()
        json_logger.start_document(doc_id, image_path)
        
        for field_name, extraction in calibrated_fields.items():
            bbox = extraction.bbox if hasattr(extraction, 'bbox') else None
            json_logger.log_field(
                field_name=field_name,
                value=extraction.value,
                confidence=extraction.calibrated_confidence,
                source=extraction.source,
                bbox=bbox
            )
        
        json_logger.end_document(
            confidence=doc_confidence,
            processing_time=total_latency,
            cost_usd=total_cost
        )
        
        return ExtractionResult(
            doc_id=doc_id,
            image_path=image_path,
            fields=calibrated_fields,
            document_confidence=doc_confidence,
            total_cost=total_cost,
            total_latency=total_latency,
            success=len(errors) == 0,
            errors=errors,
            metadata={
                "use_tier3": self.use_tier3,
                "vlm_invocation_rate": self.tracker.vlm_invocation_rate
            }
        )
    
    def _run_ocr(self, image_path: str) -> Dict[str, OCREngineOutput]:
        """Run parallel OCR engines."""
        ocr_start = time.time()
        
        outputs = parallel_ocr(image_path, use_deepseek=True)
        
        # Log costs
        for engine, output in outputs.items():
            self.tracker.log_ocr(engine, output.latency, {
                "word_count": len(output.results),
                "avg_confidence": output.avg_confidence
            })
        
        return outputs
    
    def _run_detection(
        self,
        image_path: str,
        ocr_outputs: Dict[str, OCREngineOutput]
    ) -> Dict[str, DetectorOutput]:
        """Run YOLO detection with verification."""
        detect_start = time.time()
        
        # Convert OCR results to format expected by detectors
        ocr_results = []
        for output in ocr_outputs.values():
            for result in output.results:
                ocr_results.append(result.to_dict())
        
        outputs = run_all_detections(image_path, verify=True, ocr_results=ocr_results)
        
        # Log costs
        for det_type, output in outputs.items():
            self.tracker.log_yolo(det_type, output.latency, {
                "detections": len(output.detections),
                "verified": sum(1 for d in output.detections if d.verified)
            })
        
        return outputs
    
    def _extract_fields(
        self,
        ocr_outputs: Dict[str, OCREngineOutput],
        detector_outputs: Dict[str, DetectorOutput]
    ) -> Dict[str, Dict[str, Tuple[Any, float]]]:
        """
        Extract fields from OCR and detector outputs.
        
        Returns dict mapping engine -> {field: (value, confidence)}
        """
        extractions = {}
        
        for engine, ocr_output in ocr_outputs.items():
            full_text = ocr_output.full_text
            extractions[engine] = {}
            
            # Text fields
            for field_name in ["dealer_name", "model_name", "horse_power", "asset_cost"]:
                result = extract_field_from_text(
                    full_text, 
                    field_name,
                    self.dealer_master_list
                )
                extractions[engine][field_name] = (
                    result.parsed_value,
                    result.confidence
                )
                
                run_logger.log_ocr_extraction(
                    field_name, engine, result.parsed_value, result.confidence
                )
        
        # Visual fields from detectors
        for det_type in ["signature", "stamp"]:
            det_output = detector_outputs.get(det_type)
            if det_output:
                best = det_output.best_detection
                if best:
                    value = {
                        "present": True,
                        "bbox": best.bbox,
                        "verified": best.verified
                    }
                    conf = best.confidence
                else:
                    value = {"present": False, "bbox": None, "verified": False}
                    conf = 0.7  # Default confidence for absence
                
                # Add to all engines (same detection result)
                for engine in extractions:
                    extractions[engine][det_type] = (value, conf)
        
        return extractions
    
    def _resolve_conflicts(
        self,
        field_extractions: Dict[str, Dict[str, Tuple[Any, float]]],
        image_path: str,
        ocr_outputs: Dict[str, OCREngineOutput]
    ) -> Dict[str, FieldExtraction]:
        """
        Resolve conflicts using FIELD-LEVEL (atomic) escalation.
        
        KEY FIX: Only escalate the SPECIFIC conflicting field to SLM/VLM.
        Keep the consensus values for fields that passed Tier 1.
        This prevents VLM hallucinations from voiding correct OCR extractions.
        """
        resolved = {}
        full_context = " ".join(o.full_text for o in ocr_outputs.values())[:3000]
        
        # Master-list for fuzzy pre-filtering (from field_parser.py)
        from .field_parser import parse_dealer_name, parse_model_name
        
        for field_name in self.FIELDS:
            # Collect values from all engines
            engine_results = {}
            for engine, fields in field_extractions.items():
                if field_name in fields:
                    engine_results[engine] = fields[field_name]
            
            # Compute consensus
            consensus, conflict = self.consensus_engine.compute_consensus(
                field_name, engine_results
            )
            
            # Log consensus check
            all_values = [str(v[0]) for v in engine_results.values()]
            run_logger.log_consensus_check(
                field_name, 
                conflict.has_conflict if conflict else False, 
                all_values[0] if len(all_values) > 0 else "None", 
                all_values[1] if len(all_values) > 1 else "None"
            )
            
            # === TIER 1: No conflict or consensus reached ===
            if not conflict or not conflict.has_conflict:
                resolved[field_name] = FieldExtraction(
                    field_name=field_name,
                    value=consensus.final_value,
                    confidence=consensus.final_confidence,
                    source="tier1_consensus",
                    metadata={"method": "ocr_agreement"}
                )
                run_logger.log_adjudication_result(
                    field_name, "Tier 1", "consensus",
                    consensus.final_value, consensus.final_confidence
                )
                continue
            
            # === PRE-FILTER: Fuzzy match against master-list ===
            if field_name == "dealer_name":
                # Try to auto-correct using master-list
                fuzzy_result = self._fuzzy_match_master_list(
                    conflict.value1, conflict.value2, field_name
                )
                if fuzzy_result:
                    resolved[field_name] = FieldExtraction(
                        field_name=field_name,
                        value=fuzzy_result["value"],
                        confidence=fuzzy_result["confidence"],
                        source="tier1_fuzzy_match",
                        metadata={"matched_to": fuzzy_result.get("matched_to")}
                    )
                    run_logger.log_adjudication_result(
                        field_name, "Tier 1", "fuzzy_match",
                        fuzzy_result["value"], fuzzy_result["confidence"]
                    )
                    continue
            
            # === TIER 1.5: Rule-based resolution ===
            tier1_result = self.adjudicator.tier1.resolve(conflict, full_context[:500])
            if tier1_result and tier1_result.confidence >= 0.7:
                self.tracker.log_tier1(tier1_result.latency)
                resolved[field_name] = FieldExtraction(
                    field_name=field_name,
                    value=tier1_result.resolved_value,
                    confidence=tier1_result.confidence,
                    source=f"tier1_{tier1_result.resolution_method}",
                    metadata=tier1_result.metadata
                )
                run_logger.log_adjudication_start(field_name, "Tier 1 (Rules)")
                run_logger.log_adjudication_result(
                    field_name, "Tier 1", tier1_result.resolution_method,
                    tier1_result.resolved_value, tier1_result.confidence
                )
                continue
            
            # === TIER 2: SLM for text fields only ===
            if field_name not in ["signature", "stamp"]:
                run_logger.log_adjudication_start(field_name, "Tier 2 (SLM)")
                
                slm = self.adjudicator.slm_judge or get_slm_judge()
                
                # Build layout-aware hint
                layout_hint = self._get_layout_hint(field_name, engine_results, ocr_outputs)
                
                # Call SLM with verification prompt (choose between OCR values)
                slm_result = slm.judge(
                    field_name,
                    str(conflict.value1),
                    str(conflict.value2),
                    context=full_context[:1500],
                    layout_hint=layout_hint,
                    master_hint=""
                )
                self.tracker.log_slm(slm_result.latency)
                
                if slm_result.status == "RESOLVED" and slm_result.confidence >= 0.6:
                    resolved[field_name] = FieldExtraction(
                        field_name=field_name,
                        value=slm_result.resolved_value,
                        confidence=slm_result.confidence,
                        source="tier2_slm",
                        metadata={"reasoning": slm_result.reasoning}
                    )
                    run_logger.log_adjudication_result(
                        field_name, "Tier 2", "slm_verification",
                        slm_result.resolved_value, slm_result.confidence
                    )
                    continue
            
            # === TIER 3: VLM for visual fields OR as final tie-breaker ===
            if self.adjudicator.use_tier3:
                run_logger.log_adjudication_start(field_name, "Tier 3 (VLM)")
                
                vlm = self.adjudicator.vlm_judge or get_vlm_judge()
                
                # For visual fields, use tight crop
                if field_name in ["signature", "stamp"]:
                    # Get bbox from detection
                    bbox = None
                    for engine, (val, _) in engine_results.items():
                        if isinstance(val, dict) and val.get("bbox"):
                            bbox = val["bbox"]
                            break
                    
                    if bbox:
                        vlm_result = vlm.judge_crop_from_path(
                            image_path, field_name, bbox,
                            padding=0.2  # 20% padding
                        )
                    else:
                        vlm_result = vlm.adjudicate_conflict(
                            image_path, field_name,
                            str(conflict.value1), str(conflict.value2)
                        )
                else:
                    # For text fields, use verification mode
                    vlm_result = vlm.adjudicate_conflict(
                        image_path, field_name,
                        str(conflict.value1), str(conflict.value2)
                    )
                
                self.tracker.log_vlm(vlm_result.latency)
                
                if vlm_result.status == "RESOLVED":
                    resolved[field_name] = FieldExtraction(
                        field_name=field_name,
                        value=vlm_result.resolved_value,
                        confidence=vlm_result.confidence,
                        source="tier3_vlm",
                        metadata={"reasoning": vlm_result.reasoning}
                    )
                    run_logger.log_adjudication_result(
                        field_name, "Tier 3", "vlm_verification",
                        vlm_result.resolved_value, vlm_result.confidence
                    )
                    continue
            
            # === FALLBACK: Use best available value ===
            # Never return None - use quality-based selection
            fallback_value, fallback_conf = self._quality_fallback(
                conflict.value1, conflict.confidence1,
                conflict.value2, conflict.confidence2,
                field_name
            )
            resolved[field_name] = FieldExtraction(
                field_name=field_name,
                value=fallback_value,
                confidence=fallback_conf * 0.5,  # Penalize fallback
                source="tier_fallback",
                metadata={"note": "All tiers failed, using quality heuristic"}
            )
            run_logger.log_adjudication_result(
                field_name, "Fallback", "quality_heuristic",
                fallback_value, fallback_conf * 0.5
            )
        
        return resolved
    
    def _fuzzy_match_master_list(
        self,
        value1: str,
        value2: str,
        field_name: str
    ) -> Optional[Dict]:
        """
        Try to auto-correct OCR values using fuzzy matching against master-list.
        
        If one value has >95% fuzzy match to a master entry, lock it.
        This turns extraction into verification and prevents unnecessary escalation.
        """
        from rapidfuzz import fuzz, process
        
        # Master lists (would be loaded from config in production)
        DEALER_MASTER_LIST = [
            "ABC Motors", "XYZ Tractors", "Mahindra Authorized Dealer",
            "John Deere Agri Services", "Swaraj Motors Pvt Ltd",
            "Sonalika Tractors", "TAFE Dealers", "New Holland India"
            # Add more from your data
        ]
        
        if field_name != "dealer_name":
            return None
        
        master_list = DEALER_MASTER_LIST
        threshold = 90  # 90% similarity threshold
        
        for candidate in [value1, value2]:
            if not candidate:
                continue
            
            result = process.extractOne(
                str(candidate),
                master_list,
                scorer=fuzz.token_sort_ratio
            )
            
            if result and result[1] >= threshold:
                return {
                    "value": result[0],  # Use the master-list value
                    "confidence": result[1] / 100.0,
                    "matched_to": result[0],
                    "original": candidate
                }
        
        return None
    
    def _get_layout_hint(
        self,
        field_name: str,
        engine_results: Dict[str, Tuple[Any, float]],
        ocr_outputs: Dict[str, OCREngineOutput]
    ) -> str:
        """
        Generate a layout-aware hint for the SLM based on field position.
        
        Helps the model distinguish between similar fields (e.g., dealer vs customer).
        """
        hints = {
            "dealer_name": "Usually found at the top of the invoice, often near a logo or letterhead",
            "model_name": "Located in the item description or product details section",
            "horse_power": "Often appears near the model name, labeled as 'HP' or 'BHP'",
            "asset_cost": "Found in the totals section at the bottom, after 'Total' or 'Grand Total'",
            "signature": "Located at the bottom left or right, may have a date nearby",
            "stamp": "Usually a circular or rectangular seal, often blue or red in color"
        }
        return hints.get(field_name, "")
    
    def _quality_fallback(
        self,
        value1: Any,
        conf1: float,
        value2: Any,
        conf2: float,
        field_name: str
    ) -> Tuple[Any, float]:
        """
        Quality-based fallback when all tiers fail.
        
        Uses heuristics to pick the better value:
        - Prefers non-null values
        - Prefers shorter/cleaner values (less hallucination)
        - Penalizes values with excessive special characters
        """
        def quality_score(value, conf):
            if value is None:
                return 0
            
            v_str = str(value)
            score = conf
            
            # Penalize very long values (likely hallucinations)
            if len(v_str) > 100:
                score *= 0.5
            
            # Penalize excessive special characters
            special_ratio = sum(1 for c in v_str if not c.isalnum() and c != ' ') / max(len(v_str), 1)
            if special_ratio > 0.3:
                score *= 0.6
            
            return score
        
        s1 = quality_score(value1, conf1)
        s2 = quality_score(value2, conf2)
        
        if s1 >= s2:
            return value1, conf1
        return value2, conf2
    
    def _calibrate(
        self,
        fields: Dict[str, FieldExtraction]
    ) -> Dict[str, FieldExtraction]:
        """Apply confidence calibration to fields."""
        for field_name, extraction in fields.items():
            if self.calibrator:
                calibrated = self.calibrator.calibrate_confidence(
                    field_name, extraction.confidence
                )
                extraction.calibrated_confidence = calibrated
            else:
                extraction.calibrated_confidence = extraction.confidence
        
        return fields
    
    def process_batch(
        self,
        image_paths: List[str],
        max_workers: int = 4,
        show_progress: bool = True
    ) -> List[ExtractionResult]:
        """
        Process multiple documents.
        
        Args:
            image_paths: List of image paths
            max_workers: Number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            List of ExtractionResult
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Processing documents")
            except ImportError:
                iterator = image_paths
        else:
            iterator = image_paths
        
        # Process sequentially for now (models may not be thread-safe)
        for image_path in iterator:
            result = self.process_document(image_path)
            results.append(result)
        
        return results
    
    def process_directory(
        self,
        directory: str,
        extensions: List[str] = None
    ) -> List[ExtractionResult]:
        """
        Process all images in a directory.
        
        Args:
            directory: Path to directory
            extensions: File extensions to process (default: png, jpg, jpeg)
            
        Returns:
            List of ExtractionResult
        """
        extensions = extensions or ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(directory).glob(f'*{ext}'))
            image_paths.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        return self.process_batch([str(p) for p in image_paths])


def create_processor(
    mode: str = "full",
    golden_set_path: str = None,
    dealer_list_path: str = None
) -> DocumentProcessor:
    """
    Factory function to create a DocumentProcessor.
    
    Args:
        mode: "full" or "cpu-lite"
        golden_set_path: Optional path to golden set for calibration
        dealer_list_path: Optional path to dealer master list
        
    Returns:
        Configured DocumentProcessor
    """
    use_tier3 = mode != "cpu-lite"
    
    # Load calibrator if golden set available
    calibrator = None
    if golden_set_path and Path(golden_set_path).exists():
        calibrator = Calibrator(golden_set_path=golden_set_path)
    
    # Load dealer list
    dealer_list = []
    if dealer_list_path and Path(dealer_list_path).exists():
        with open(dealer_list_path, 'r') as f:
            dealer_list = json.load(f)
    
    return DocumentProcessor(
        use_tier3=use_tier3,
        calibrator=calibrator,
        dealer_master_list=dealer_list
    )


def process_single(
    image_path: str,
    mode: str = "full"
) -> Dict:
    """
    Quick function to process a single document.
    
    Args:
        image_path: Path to document image
        mode: "full" or "cpu-lite"
        
    Returns:
        Extraction result as dictionary
    """
    processor = create_processor(mode=mode)
    result = processor.process_document(image_path)
    return result.to_dict()
