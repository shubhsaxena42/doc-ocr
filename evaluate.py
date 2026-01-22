#!/usr/bin/env python3
"""
Evaluation Script

Computes metrics against golden set:
- Document-Level Accuracy (DLA): % docs with all 6 fields correct
- Field-Level mAP: For signature/stamp IoU >0.5
- Cost per document (average)
- Latency per document (average)
- VLM Invocation Rate (must be <10%)

Usage:
    python evaluate.py --predictions results.json --golden data/golden_set.json
    python evaluate.py --predictions results.json --golden data/golden_set.json --output evaluation_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict


def compute_iou(bbox1: List, bbox2: List) -> float:
    """
    Compute IoU between two bboxes in [x, y, w, h] format.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU value (0-1)
    """
    if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
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


def check_field_correct(
    predicted: Any,
    ground_truth: Any,
    field_name: str,
    iou_threshold: float = 0.5
) -> Tuple[bool, Dict]:
    """
    Check if a field prediction is correct.
    
    Args:
        predicted: Predicted value
        ground_truth: Ground truth value
        field_name: Name of the field
        iou_threshold: IoU threshold for visual fields
        
    Returns:
        Tuple of (is_correct, metrics_dict)
    """
    metrics = {"field": field_name}
    
    # Handle None cases
    if predicted is None and ground_truth is None:
        return True, metrics
    if predicted is None or ground_truth is None:
        return False, metrics
    
    # Numeric fields
    if field_name in ["horse_power", "asset_cost"]:
        try:
            p = float(predicted)
            gt = float(ground_truth)
            rel_diff = abs(p - gt) / max(abs(gt), 1)
            metrics["relative_diff"] = rel_diff
            return rel_diff < 0.05, metrics
        except (ValueError, TypeError):
            return str(predicted) == str(ground_truth), metrics
    
    # Visual fields (signature, stamp)
    if field_name in ["signature", "stamp"]:
        # Check presence
        p_present = predicted.get("present", False) if isinstance(predicted, dict) else bool(predicted)
        gt_present = ground_truth.get("present", False) if isinstance(ground_truth, dict) else bool(ground_truth)
        
        if p_present != gt_present:
            metrics["presence_match"] = False
            return False, metrics
        
        metrics["presence_match"] = True
        
        if not p_present:  # Both not present
            return True, metrics
        
        # Check bbox IoU if both present
        p_bbox = predicted.get("bbox") if isinstance(predicted, dict) else None
        gt_bbox = ground_truth.get("bbox") if isinstance(ground_truth, dict) else None
        
        if p_bbox and gt_bbox:
            iou = compute_iou(p_bbox, gt_bbox)
            metrics["iou"] = iou
            return iou >= iou_threshold, metrics
        
        # Presence match is enough if no bbox
        return True, metrics
    
    # Text fields
    p_str = str(predicted).lower().strip()
    gt_str = str(ground_truth).lower().strip()
    return p_str == gt_str, metrics


def evaluate_results(
    predictions: Dict,
    golden_set: Dict,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Evaluate predictions against golden set.
    
    Args:
        predictions: Predictions dict with "results" list
        golden_set: Golden set dict with "documents" list
        iou_threshold: IoU threshold for visual fields
        
    Returns:
        Evaluation report dict
    """
    # Build ground truth lookup
    gt_lookup = {}
    for doc in golden_set.get("documents", []):
        if doc.get("ground_truth"):
            gt_lookup[doc["doc_id"]] = doc["ground_truth"]
    
    # Get predictions
    results = predictions.get("results", [])
    
    # Track metrics
    total_docs = 0
    correct_docs = 0
    
    field_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "ious": []})
    
    cost_total = 0.0
    latency_total = 0.0
    
    detailed_results = []
    
    for result in results:
        doc_id = result.get("doc_id", "")
        
        if doc_id not in gt_lookup:
            continue
        
        gt = gt_lookup[doc_id]
        total_docs += 1
        
        # Track document-level correctness
        doc_correct = True
        doc_details = {"doc_id": doc_id, "fields": {}}
        
        # Check each field
        fields = result.get("fields", {})
        for field_name in ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]:
            pred_field = fields.get(field_name, {})
            pred_value = pred_field.get("value")
            gt_value = gt.get(field_name)
            
            is_correct, metrics = check_field_correct(
                pred_value, gt_value, field_name, iou_threshold
            )
            
            field_metrics[field_name]["total"] += 1
            if is_correct:
                field_metrics[field_name]["correct"] += 1
            else:
                doc_correct = False
            
            # Track IoU for visual fields
            if "iou" in metrics:
                field_metrics[field_name]["ious"].append(metrics["iou"])
            
            doc_details["fields"][field_name] = {
                "correct": is_correct,
                "predicted": str(pred_value)[:100] if pred_value else None,
                "ground_truth": str(gt_value)[:100] if gt_value else None,
                "metrics": metrics
            }
        
        if doc_correct:
            correct_docs += 1
        
        # Track cost and latency
        cost_total += result.get("total_cost", 0)
        latency_total += result.get("total_latency", 0)
        
        doc_details["correct"] = doc_correct
        detailed_results.append(doc_details)
    
    # Compute aggregate metrics
    dla = correct_docs / total_docs if total_docs > 0 else 0
    avg_cost = cost_total / total_docs if total_docs > 0 else 0
    avg_latency = latency_total / total_docs if total_docs > 0 else 0
    
    # Compute field-level accuracy
    field_accuracy = {}
    for field, metrics in field_metrics.items():
        field_accuracy[field] = {
            "accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
            "correct": metrics["correct"],
            "total": metrics["total"]
        }
        
        # Add mAP for visual fields
        if metrics["ious"]:
            field_accuracy[field]["mean_iou"] = sum(metrics["ious"]) / len(metrics["ious"])
            field_accuracy[field]["iou_samples"] = len(metrics["ious"])
    
    # Get VLM rate from predictions summary
    vlm_rate = predictions.get("summary", {}).get("vlm_invocation_rate", 0)
    
    return {
        "summary": {
            "document_level_accuracy": dla,
            "total_documents": total_docs,
            "correct_documents": correct_docs,
            "average_cost_per_document": avg_cost,
            "average_latency_per_document": avg_latency,
            "vlm_invocation_rate": vlm_rate,
            "targets": {
                "dla_target": 0.95,
                "dla_met": dla >= 0.95,
                "cost_target": 0.01,
                "cost_met": avg_cost < 0.01,
                "latency_target": 30.0,
                "latency_met": avg_latency < 30.0,
                "vlm_rate_target": 0.10,
                "vlm_rate_met": vlm_rate < 0.10
            }
        },
        "field_accuracy": field_accuracy,
        "detailed_results": detailed_results
    }


def run_ablation_analysis(
    results_files: Dict[str, str],
    golden_set: Dict,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Run ablation analysis comparing different configurations.
    
    Args:
        results_files: Dict mapping config name to results file path
        golden_set: Golden set dict
        iou_threshold: IoU threshold
        
    Returns:
        Ablation analysis report
    """
    ablations = {}
    
    for config_name, results_path in results_files.items():
        with open(results_path, 'r') as f:
            predictions = json.load(f)
        
        evaluation = evaluate_results(predictions, golden_set, iou_threshold)
        
        ablations[config_name] = {
            "dla": evaluation["summary"]["document_level_accuracy"],
            "cost": evaluation["summary"]["average_cost_per_document"],
            "latency": evaluation["summary"]["average_latency_per_document"],
            "vlm_rate": evaluation["summary"]["vlm_invocation_rate"],
            "field_accuracy": {
                k: v["accuracy"] for k, v in evaluation["field_accuracy"].items()
            }
        }
    
    return ablations


def print_evaluation_report(report: Dict):
    """Print evaluation report to console."""
    summary = report["summary"]
    
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\n{'Metric':<35} {'Value':<15} {'Target':<10} {'Status'}")
    print("-"*70)
    
    dla = summary["document_level_accuracy"]
    print(f"{'Document-Level Accuracy (DLA)':<35} {dla*100:>6.2f}%       {'≥95%':<10} {'✓' if summary['targets']['dla_met'] else '✗'}")
    
    cost = summary["average_cost_per_document"]
    print(f"{'Average Cost/Document':<35} ${cost:>6.4f}       {'<$0.01':<10} {'✓' if summary['targets']['cost_met'] else '✗'}")
    
    latency = summary["average_latency_per_document"]
    print(f"{'Average Latency/Document':<35} {latency:>6.2f}s       {'<30s':<10} {'✓' if summary['targets']['latency_met'] else '✗'}")
    
    vlm = summary["vlm_invocation_rate"]
    print(f"{'VLM Invocation Rate':<35} {vlm*100:>6.2f}%       {'<10%':<10} {'✓' if summary['targets']['vlm_rate_met'] else '✗'}")
    
    print("\nField-Level Accuracy:")
    print("-"*50)
    for field, metrics in report["field_accuracy"].items():
        acc = metrics["accuracy"]
        extra = ""
        if "mean_iou" in metrics:
            extra = f" (mIoU: {metrics['mean_iou']:.3f})"
        print(f"  {field:<20} {acc*100:>6.2f}%{extra}")
    
    print("\n" + "="*60)
    
    # Overall verdict
    all_met = all([
        summary["targets"]["dla_met"],
        summary["targets"]["cost_met"],
        summary["targets"]["latency_met"],
        summary["targets"]["vlm_rate_met"]
    ])
    
    if all_met:
        print("✅ ALL TARGETS MET")
    else:
        failed = []
        if not summary["targets"]["dla_met"]:
            failed.append("DLA")
        if not summary["targets"]["cost_met"]:
            failed.append("Cost")
        if not summary["targets"]["latency_met"]:
            failed.append("Latency")
        if not summary["targets"]["vlm_rate_met"]:
            failed.append("VLM Rate")
        print(f"❌ TARGETS NOT MET: {', '.join(failed)}")
    
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate IDAI pipeline results against golden set"
    )
    
    parser.add_argument(
        '--predictions', '-p',
        required=True,
        help='Path to predictions/results JSON file'
    )
    
    parser.add_argument(
        '--golden', '-g',
        required=True,
        help='Path to golden_set.json'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to save evaluation report (optional)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for visual field matching (default: 0.5)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.predictions).exists():
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    if not Path(args.golden).exists():
        print(f"Error: Golden set file not found: {args.golden}")
        sys.exit(1)
    
    # Load data
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    with open(args.golden, 'r') as f:
        golden_set = json.load(f)
    
    # Evaluate
    report = evaluate_results(predictions, golden_set, args.iou_threshold)
    
    # Output
    if not args.quiet:
        print_evaluation_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        if not args.quiet:
            print(f"\nReport saved to: {args.output}")
    
    # Return exit code based on DLA target
    if report["summary"]["targets"]["dla_met"]:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
