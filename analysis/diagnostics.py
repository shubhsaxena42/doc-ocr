"""
Diagnostics Module

Error analysis and reporting for the IDAI pipeline.
Categorizes failures and provides per-language accuracy breakdown.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from pathlib import Path


@dataclass
class ErrorRecord:
    """Record of a single extraction error."""
    doc_id: str
    field_name: str
    error_category: str
    predicted_value: Any
    ground_truth: Any
    confidence: float
    metadata: Dict = field(default_factory=dict)


class ErrorCategorizer:
    """
    Categorizes extraction errors.
    
    Categories:
    - OCR_Mismatch: Both OCR engines disagree significantly
    - SLM_Uncertain: SLM could not resolve conflict
    - VLM_Error: VLM returned incorrect value
    - Handwriting_Noise: Signature/handwriting related errors
    - Stamp_Overlap: Stamp detection errors due to overlaps
    - Low_Confidence: Correct extraction but low confidence
    - Parse_Error: Field parsing failure
    - Missing_Value: Expected field not found
    """
    
    CATEGORIES = [
        "OCR_Mismatch",
        "SLM_Uncertain", 
        "VLM_Error",
        "Handwriting_Noise",
        "Stamp_Overlap",
        "Low_Confidence",
        "Parse_Error",
        "Missing_Value"
    ]
    
    def categorize(
        self,
        field_name: str,
        predicted: Any,
        ground_truth: Any,
        extraction_metadata: Dict
    ) -> str:
        """
        Categorize an extraction error.
        
        Args:
            field_name: Name of the field
            predicted: Predicted value
            ground_truth: Ground truth value
            extraction_metadata: Metadata from extraction result
            
        Returns:
            Error category string
        """
        source = extraction_metadata.get("source", "")
        
        # Check for missing value
        if predicted is None:
            return "Missing_Value"
        
        # Check visual field errors
        if field_name == "signature":
            if "handwriting" in str(extraction_metadata).lower():
                return "Handwriting_Noise"
            return "OCR_Mismatch"
        
        if field_name == "stamp":
            if "overlap" in str(extraction_metadata).lower():
                return "Stamp_Overlap"
            return "OCR_Mismatch"
        
        # Check adjudication-related errors
        if "tier2" in source and "uncertain" in source.lower():
            return "SLM_Uncertain"
        
        if "tier3" in source or "vlm" in source.lower():
            return "VLM_Error"
        
        # Check for OCR mismatch
        if extraction_metadata.get("conflict", False):
            return "OCR_Mismatch"
        
        # Check confidence
        confidence = extraction_metadata.get("confidence", 1.0)
        if confidence < 0.6:
            return "Low_Confidence"
        
        # Default to parse error
        return "Parse_Error"


class DiagnosticsReport:
    """
    Generates diagnostic reports for pipeline results.
    """
    
    LANGUAGES = ["Hindi", "Gujarati", "English", "Mixed"]
    
    def __init__(self):
        self.errors: List[ErrorRecord] = []
        self.categorizer = ErrorCategorizer()
    
    def add_error(
        self,
        doc_id: str,
        field_name: str,
        predicted: Any,
        ground_truth: Any,
        confidence: float,
        metadata: Dict = None
    ):
        """Add an error record."""
        category = self.categorizer.categorize(
            field_name, predicted, ground_truth, metadata or {}
        )
        
        self.errors.append(ErrorRecord(
            doc_id=doc_id,
            field_name=field_name,
            error_category=category,
            predicted_value=predicted,
            ground_truth=ground_truth,
            confidence=confidence,
            metadata=metadata or {}
        ))
    
    def analyze_results(
        self,
        results: List[Dict],
        golden_set: Dict[str, Dict]
    ) -> Dict:
        """
        Analyze extraction results against golden set.
        
        Args:
            results: List of extraction result dicts
            golden_set: Dict mapping doc_id to ground truth
            
        Returns:
            Analysis report dict
        """
        total_docs = len(results)
        correct_docs = 0
        field_correct = defaultdict(int)
        field_total = defaultdict(int)
        
        for result in results:
            doc_id = result.get("doc_id", "")
            gt = golden_set.get(doc_id, {})
            
            if not gt:
                continue
            
            all_correct = True
            
            for field_name, field_data in result.get("fields", {}).items():
                pred_value = field_data.get("value")
                gt_value = gt.get(field_name)
                confidence = field_data.get("confidence", 0)
                metadata = field_data.get("metadata", {})
                
                field_total[field_name] += 1
                
                if self._check_correct(pred_value, gt_value, field_name):
                    field_correct[field_name] += 1
                else:
                    all_correct = False
                    self.add_error(
                        doc_id, field_name, pred_value, gt_value,
                        confidence, metadata
                    )
            
            if all_correct:
                correct_docs += 1
        
        return {
            "document_level_accuracy": correct_docs / total_docs if total_docs > 0 else 0,
            "field_level_accuracy": {
                field: field_correct[field] / field_total[field]
                for field in field_total
            },
            "total_documents": total_docs,
            "correct_documents": correct_docs,
            "error_count": len(self.errors)
        }
    
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
    
    def get_error_breakdown(self) -> Dict[str, int]:
        """Get count of errors by category."""
        breakdown = defaultdict(int)
        for error in self.errors:
            breakdown[error.error_category] += 1
        return dict(breakdown)
    
    def get_field_error_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get error breakdown per field."""
        breakdown = defaultdict(lambda: defaultdict(int))
        for error in self.errors:
            breakdown[error.field_name][error.error_category] += 1
        return {k: dict(v) for k, v in breakdown.items()}
    
    def generate_report(
        self,
        results: List[Dict],
        golden_set: Dict[str, Dict],
        cost_summary: Dict = None
    ) -> Dict:
        """
        Generate full diagnostic report.
        
        Args:
            results: Extraction results
            golden_set: Ground truth data
            cost_summary: Cost tracking summary
            
        Returns:
            Complete diagnostic report
        """
        analysis = self.analyze_results(results, golden_set)
        
        report = {
            "summary": {
                "document_level_accuracy": analysis["document_level_accuracy"],
                "total_documents": analysis["total_documents"],
                "correct_documents": analysis["correct_documents"],
                "error_count": analysis["error_count"]
            },
            "field_accuracy": analysis["field_level_accuracy"],
            "error_breakdown": self.get_error_breakdown(),
            "field_error_breakdown": self.get_field_error_breakdown(),
            "errors": [
                {
                    "doc_id": e.doc_id,
                    "field": e.field_name,
                    "category": e.error_category,
                    "predicted": str(e.predicted_value)[:100],
                    "ground_truth": str(e.ground_truth)[:100],
                    "confidence": e.confidence
                }
                for e in self.errors[:100]  # Limit to first 100 errors
            ]
        }
        
        if cost_summary:
            report["cost_metrics"] = {
                "cost_per_document": cost_summary.get("average_cost_per_document", 0),
                "vlm_invocation_rate": cost_summary.get("vlm_invocation_rate", 0),
                "within_budget": cost_summary.get("within_cost_target", False)
            }
        
        return report
    
    def save_report(self, report: Dict, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def generate_diagnostics(
    results_path: str,
    golden_set_path: str,
    output_path: str = None
) -> Dict:
    """
    Generate diagnostics report from results file.
    
    Args:
        results_path: Path to results.json
        golden_set_path: Path to golden_set.json
        output_path: Optional path to save report
        
    Returns:
        Diagnostic report dict
    """
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    with open(golden_set_path, 'r') as f:
        golden_data = json.load(f)
    
    # Convert golden set to lookup dict
    golden_set = {}
    for doc in golden_data.get("documents", []):
        golden_set[doc["doc_id"]] = doc["ground_truth"]
    
    # Generate report
    diagnostics = DiagnosticsReport()
    report = diagnostics.generate_report(
        results_data.get("results", []),
        golden_set,
        results_data.get("summary", {})
    )
    
    if output_path:
        diagnostics.save_report(report, output_path)
    
    return report


def print_diagnostics_summary(report: Dict):
    """Print a human-readable summary of diagnostics."""
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    summary = report.get("summary", {})
    print(f"\nDocument-Level Accuracy: {summary.get('document_level_accuracy', 0)*100:.1f}%")
    print(f"Total Documents: {summary.get('total_documents', 0)}")
    print(f"Correct Documents: {summary.get('correct_documents', 0)}")
    print(f"Total Errors: {summary.get('error_count', 0)}")
    
    print("\nField-Level Accuracy:")
    for field, acc in report.get("field_accuracy", {}).items():
        print(f"  {field}: {acc*100:.1f}%")
    
    print("\nError Breakdown by Category:")
    for category, count in report.get("error_breakdown", {}).items():
        print(f"  {category}: {count}")
    
    if "cost_metrics" in report:
        print("\nCost Metrics:")
        metrics = report["cost_metrics"]
        print(f"  Cost/Document: ${metrics.get('cost_per_document', 0):.4f}")
        print(f"  VLM Invocation Rate: {metrics.get('vlm_invocation_rate', 0)*100:.1f}%")
        print(f"  Within Budget: {'Yes' if metrics.get('within_budget') else 'No'}")
    
    print("="*60)
