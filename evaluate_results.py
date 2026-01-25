"""
Evaluation Script for Document OCR Pipeline

Compares pipeline results (results_with_sources.json) against golden dataset.

Evaluation Criteria:
- Dealer Name: Fuzzy match (using RapidFuzz, threshold 80%)
- Model Name: Exact match (case-insensitive)
- Horse Power: Exact match (numeric)
- Asset Cost: Exact match (numeric)
- Signature: Binary + IOU > 0.5 with ground truth bounding box
- Stamp: Binary + IOU > 0.5 with ground truth bounding box

Usage:
    python evaluate_results.py --results results_with_sources.json --golden data/golden_set.json
"""

import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import rapidfuzz, fall back to basic matching if not available
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Using basic string matching.")
    print("Install with: pip install rapidfuzz")


@dataclass
class FieldResult:
    """Result for a single field evaluation."""
    field_name: str
    correct: bool
    predicted: Any
    ground_truth: Any
    score: float = 0.0  # For fuzzy match or IOU
    details: str = ""


@dataclass
class DocumentResult:
    """Result for a single document evaluation."""
    doc_id: str
    field_results: Dict[str, FieldResult]
    overall_correct: int = 0
    overall_total: int = 0


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IOU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IOU score between 0 and 1
    """
    if box1 is None or box2 is None:
        return 0.0
    
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    # Check for invalid boxes (all zeros or negative dimensions)
    if box1 == [0, 0, 0, 0] or box2 == [0, 0, 0, 0]:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def fuzzy_match(pred: str, gt: str, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Perform fuzzy string matching.
    
    Args:
        pred: Predicted string
        gt: Ground truth string
        threshold: Minimum score to consider a match (0-100)
        
    Returns:
        Tuple of (is_match, score)
    """
    if pred is None or gt is None:
        return False, 0.0
    
    pred_clean = str(pred).strip().lower()
    gt_clean = str(gt).strip().lower()
    
    if RAPIDFUZZ_AVAILABLE:
        # Use token_set_ratio for better matching with word reordering
        score = fuzz.token_set_ratio(pred_clean, gt_clean)
    else:
        # Basic matching: check if one contains the other
        if pred_clean == gt_clean:
            score = 100.0
        elif pred_clean in gt_clean or gt_clean in pred_clean:
            score = 80.0
        else:
            # Very basic Jaccard similarity on words
            pred_words = set(pred_clean.split())
            gt_words = set(gt_clean.split())
            if len(pred_words | gt_words) > 0:
                score = 100 * len(pred_words & gt_words) / len(pred_words | gt_words)
            else:
                score = 0.0
    
    return score >= threshold, score


def exact_match_text(pred: str, gt: str) -> Tuple[bool, float]:
    """
    Perform exact text matching (case-insensitive).
    
    Args:
        pred: Predicted string
        gt: Ground truth string
        
    Returns:
        Tuple of (is_match, score)
    """
    if pred is None or gt is None:
        return False, 0.0
    
    pred_clean = str(pred).strip().lower()
    gt_clean = str(gt).strip().lower()
    
    is_match = pred_clean == gt_clean
    return is_match, 100.0 if is_match else 0.0


def exact_match_numeric(pred: Any, gt: Any, tolerance: float = 0.0) -> Tuple[bool, float]:
    """
    Perform exact numeric matching with optional tolerance.
    
    Args:
        pred: Predicted value
        gt: Ground truth value
        tolerance: Relative tolerance (0.0 for exact, 0.05 for 5%)
        
    Returns:
        Tuple of (is_match, score)
    """
    if pred is None or gt is None:
        return False, 0.0
    
    try:
        pred_num = float(pred)
        gt_num = float(gt)
    except (ValueError, TypeError):
        return False, 0.0
    
    if gt_num == 0:
        is_match = pred_num == 0
    else:
        diff = abs(pred_num - gt_num) / abs(gt_num)
        is_match = diff <= tolerance
    
    return is_match, 100.0 if is_match else 0.0


def evaluate_visual_field(
    pred: Dict, 
    gt: Dict, 
    iou_threshold: float = 0.5
) -> Tuple[bool, float, str]:
    """
    Evaluate a visual field (signature/stamp).
    
    Evaluation:
    1. First check if presence (true/false) matches
    2. If both present, check IOU > threshold
    
    Args:
        pred: Predicted dict with 'present' and 'bbox'
        gt: Ground truth dict with 'present' and 'bbox'
        iou_threshold: Minimum IOU for bounding box match
        
    Returns:
        Tuple of (is_correct, iou_score, details)
    """
    # Handle None or missing values
    if pred is None:
        pred = {"present": False, "bbox": None}
    if gt is None:
        gt = {"present": False, "bbox": None}
    
    # Extract presence
    pred_present = pred.get("present", False) if isinstance(pred, dict) else False
    gt_present = gt.get("present", False) if isinstance(gt, dict) else False
    
    # Handle string values like "YES"/"NO"
    if isinstance(pred_present, str):
        pred_present = pred_present.upper() in ["YES", "TRUE", "1"]
    if isinstance(gt_present, str):
        gt_present = gt_present.upper() in ["YES", "TRUE", "1"]
    
    # If presence doesn't match, it's wrong
    if pred_present != gt_present:
        return False, 0.0, f"Presence mismatch: pred={pred_present}, gt={gt_present}"
    
    # If both absent, it's correct
    if not pred_present and not gt_present:
        return True, 1.0, "Both correctly identified as absent"
    
    # Both present - check IOU
    pred_bbox = pred.get("bbox") if isinstance(pred, dict) else None
    gt_bbox = gt.get("bbox") if isinstance(gt, dict) else None
    
    if pred_bbox is None or gt_bbox is None:
        # Presence correct but no bbox to compare
        return True, 0.5, "Presence correct, bbox not available for comparison"
    
    iou = compute_iou(pred_bbox, gt_bbox)
    is_correct = iou >= iou_threshold
    
    details = f"IOU={iou:.3f} (threshold={iou_threshold})"
    return is_correct, iou, details


def evaluate_document(
    pred_doc: Dict,
    gt_doc: Dict,
    fuzzy_threshold: float = 80.0,
    iou_threshold: float = 0.5
) -> DocumentResult:
    """
    Evaluate a single document's predictions against ground truth.
    
    Args:
        pred_doc: Predicted document with 'fields' dict
        gt_doc: Ground truth document with 'fields' dict
        fuzzy_threshold: Threshold for fuzzy text matching (0-100)
        iou_threshold: Threshold for bounding box IOU
        
    Returns:
        DocumentResult with all field evaluations
    """
    doc_id = gt_doc.get("doc_id", "unknown")
    result = DocumentResult(doc_id=doc_id, field_results={})
    
    pred_fields = pred_doc.get("fields", {})
    gt_fields = gt_doc.get("fields", {})
    
    # 1. Dealer Name (Fuzzy Match)
    pred_dealer = pred_fields.get("dealer_name")
    gt_dealer = gt_fields.get("dealer_name")
    is_correct, score = fuzzy_match(pred_dealer, gt_dealer, fuzzy_threshold)
    result.field_results["dealer_name"] = FieldResult(
        field_name="dealer_name",
        correct=is_correct,
        predicted=pred_dealer,
        ground_truth=gt_dealer,
        score=score,
        details=f"Fuzzy score: {score:.1f}% (threshold: {fuzzy_threshold}%)"
    )
    
    # 2. Model Name (Exact Match)
    pred_model = pred_fields.get("model_name")
    gt_model = gt_fields.get("model_name")
    is_correct, score = exact_match_text(pred_model, gt_model)
    result.field_results["model_name"] = FieldResult(
        field_name="model_name",
        correct=is_correct,
        predicted=pred_model,
        ground_truth=gt_model,
        score=score,
        details="Exact match (case-insensitive)"
    )
    
    # 3. Horse Power (Exact Match)
    pred_hp = pred_fields.get("horse_power")
    gt_hp = gt_fields.get("horse_power")
    is_correct, score = exact_match_numeric(pred_hp, gt_hp)
    result.field_results["horse_power"] = FieldResult(
        field_name="horse_power",
        correct=is_correct,
        predicted=pred_hp,
        ground_truth=gt_hp,
        score=score,
        details="Exact numeric match"
    )
    
    # 4. Asset Cost (Exact Match)
    pred_cost = pred_fields.get("asset_cost")
    gt_cost = gt_fields.get("asset_cost")
    is_correct, score = exact_match_numeric(pred_cost, gt_cost)
    result.field_results["asset_cost"] = FieldResult(
        field_name="asset_cost",
        correct=is_correct,
        predicted=pred_cost,
        ground_truth=gt_cost,
        score=score,
        details="Exact numeric match"
    )
    
    # 5. Signature (Binary + IOU)
    pred_sig = pred_fields.get("signature")
    gt_sig = gt_fields.get("signature")
    is_correct, iou, details = evaluate_visual_field(pred_sig, gt_sig, iou_threshold)
    result.field_results["signature"] = FieldResult(
        field_name="signature",
        correct=is_correct,
        predicted=pred_sig,
        ground_truth=gt_sig,
        score=iou * 100,
        details=details
    )
    
    # 6. Stamp (Binary + IOU)
    pred_stamp = pred_fields.get("stamp")
    gt_stamp = gt_fields.get("stamp")
    is_correct, iou, details = evaluate_visual_field(pred_stamp, gt_stamp, iou_threshold)
    result.field_results["stamp"] = FieldResult(
        field_name="stamp",
        correct=is_correct,
        predicted=pred_stamp,
        ground_truth=gt_stamp,
        score=iou * 100,
        details=details
    )
    
    # Calculate overall stats
    result.overall_total = len(result.field_results)
    result.overall_correct = sum(1 for fr in result.field_results.values() if fr.correct)
    
    return result


def run_evaluation(
    results_path: str,
    golden_path: str,
    fuzzy_threshold: float = 80.0,
    iou_threshold: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Run full evaluation of results against golden set.
    
    Args:
        results_path: Path to results_with_sources.json
        golden_path: Path to golden_set.json
        fuzzy_threshold: Threshold for fuzzy text matching
        iou_threshold: Threshold for bounding box IOU
        verbose: Print detailed results
        
    Returns:
        Summary dict with metrics
    """
    # Load files
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden = json.load(f)
    
    # Build lookup by doc_id
    results_by_id = {}
    for doc in results.get("results", []):
        doc_id = doc.get("doc_id")
        if doc_id:
            results_by_id[doc_id] = doc
    
    golden_by_id = {}
    for doc in golden.get("results", []):
        doc_id = doc.get("doc_id")
        if doc_id:
            golden_by_id[doc_id] = doc
    
    # Track metrics
    field_correct = {
        "dealer_name": 0,
        "model_name": 0,
        "horse_power": 0,
        "asset_cost": 0,
        "signature": 0,
        "stamp": 0
    }
    field_total = {k: 0 for k in field_correct}
    
    all_doc_results = []
    matched_docs = 0
    skipped_docs = []
    
    print("=" * 80)
    print("📊 DOCUMENT OCR PIPELINE EVALUATION")
    print("=" * 80)
    print(f"Results file: {results_path}")
    print(f"Golden file: {golden_path}")
    print(f"Fuzzy threshold: {fuzzy_threshold}%")
    print(f"IOU threshold: {iou_threshold}")
    print("=" * 80)
    
    # Evaluate each golden document
    for doc_id, gt_doc in golden_by_id.items():
        # Skip documents with errors in golden set
        if "error" in gt_doc:
            skipped_docs.append((doc_id, "Golden has error"))
            continue
        
        # Check if we have prediction
        if doc_id not in results_by_id:
            skipped_docs.append((doc_id, "Not in results"))
            continue
        
        pred_doc = results_by_id[doc_id]
        
        # Skip documents with errors in prediction
        if "error" in pred_doc:
            skipped_docs.append((doc_id, f"Prediction error: {pred_doc.get('error', 'unknown')[:50]}"))
            continue
        
        # Skip if no fields
        if "fields" not in pred_doc or "fields" not in gt_doc:
            skipped_docs.append((doc_id, "Missing fields"))
            continue
        
        # Evaluate
        doc_result = evaluate_document(pred_doc, gt_doc, fuzzy_threshold, iou_threshold)
        all_doc_results.append(doc_result)
        matched_docs += 1
        
        # Update totals
        for field_name, field_result in doc_result.field_results.items():
            field_total[field_name] += 1
            if field_result.correct:
                field_correct[field_name] += 1
        
        # Print document result
        if verbose:
            status = "✅" if doc_result.overall_correct == doc_result.overall_total else "⚠️"
            print(f"\n{status} {doc_id} ({doc_result.overall_correct}/{doc_result.overall_total} correct)")
            for field_name, fr in doc_result.field_results.items():
                icon = "✓" if fr.correct else "✗"
                print(f"   {icon} {field_name}: {fr.details}")
                if not fr.correct:
                    print(f"      Predicted: {fr.predicted}")
                    print(f"      Expected:  {fr.ground_truth}")
    
    # Calculate accuracy per field
    print("\n" + "=" * 80)
    print("📈 FIELD-LEVEL ACCURACY")
    print("=" * 80)
    
    field_accuracy = {}
    for field_name in field_correct:
        total = field_total[field_name]
        correct = field_correct[field_name]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        field_accuracy[field_name] = accuracy
        bar = "█" * int(accuracy / 2) + "░" * (50 - int(accuracy / 2))
        print(f"{field_name:20s} {bar} {accuracy:6.2f}% ({correct}/{total})")
    
    # Overall accuracy
    total_fields = sum(field_total.values())
    total_correct = sum(field_correct.values())
    overall_accuracy = (total_correct / total_fields * 100) if total_fields > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Documents evaluated: {matched_docs}")
    print(f"Documents skipped: {len(skipped_docs)}")
    print(f"Total fields evaluated: {total_fields}")
    print(f"Total fields correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    # Print skipped docs
    if skipped_docs and verbose:
        print(f"\n📝 Skipped documents ({len(skipped_docs)}):")
        for doc_id, reason in skipped_docs[:10]:  # Show first 10
            print(f"   - {doc_id}: {reason}")
        if len(skipped_docs) > 10:
            print(f"   ... and {len(skipped_docs) - 10} more")
    
    print("=" * 80)
    
    # Return summary
    return {
        "documents_evaluated": matched_docs,
        "documents_skipped": len(skipped_docs),
        "field_accuracy": field_accuracy,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_fields": total_fields
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR pipeline results against golden set")
    parser.add_argument("--results", "-r", default="results_with_sources.json",
                        help="Path to results JSON file")
    parser.add_argument("--golden", "-g", default="data/golden_set.json",
                        help="Path to golden set JSON file")
    parser.add_argument("--fuzzy-threshold", "-f", type=float, default=80.0,
                        help="Fuzzy match threshold for dealer name (0-100)")
    parser.add_argument("--iou-threshold", "-i", type=float, default=0.5,
                        help="IOU threshold for signature/stamp bounding boxes")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only show summary, not per-document details")
    
    args = parser.parse_args()
    
    summary = run_evaluation(
        results_path=args.results,
        golden_path=args.golden,
        fuzzy_threshold=args.fuzzy_threshold,
        iou_threshold=args.iou_threshold,
        verbose=not args.quiet
    )
    
    # Return exit code based on accuracy
    if summary["overall_accuracy"] >= 90:
        return 0
    elif summary["overall_accuracy"] >= 70:
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main())
