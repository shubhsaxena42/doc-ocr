"""
JSON Result Logger

Outputs extraction results in clean JSON format with source attribution
(OCR, SLM, or VLM) for each field.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class JSONResultLogger:
    """
    Logger that outputs extraction results in JSON format.
    
    For each document, logs:
    - doc_id
    - fields with values
    - source for each field (ocr/slm/vlm)
    - confidence
    - processing time
    - cost estimate
    """
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize JSON logger.
        
        Args:
            output_dir: Directory to save JSON output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict] = []
        self.current_doc: Optional[Dict] = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def start_document(self, doc_id: str, image_path: str = ""):
        """Start logging a new document."""
        self.current_doc = {
            "doc_id": doc_id,
            "image_path": image_path,
            "fields": {},
            "field_sources": {},  # Track source for each field
            "confidence": 0.0,
            "processing_time_sec": 0.0,
            "cost_estimate_usd": 0.0
        }
    
    def log_field(
        self, 
        field_name: str, 
        value: Any, 
        confidence: float,
        source: str,  # "ocr", "slm", "vlm", "tier1", "consensus"
        bbox: Optional[List[int]] = None
    ):
        """
        Log a field extraction result.
        
        Args:
            field_name: Name of the field
            value: Extracted value
            confidence: Confidence score
            source: Source of the value (ocr/slm/vlm/tier1/consensus)
            bbox: Bounding box for visual fields [x1, y1, x2, y2]
        """
        if self.current_doc is None:
            return
        
        # Normalize source name
        source_map = {
            "tier1": "ocr",
            "tier1_rules": "ocr", 
            "tier1_dealer_match": "ocr",
            "tier1_hp_pattern": "ocr",
            "tier1_confidence": "ocr",
            "consensus": "ocr",
            "paddle": "ocr",
            "deepseek": "ocr",
            "tier2": "slm",
            "tier2_slm": "slm",
            "slm_judge": "slm",
            "slm_document_extraction": "slm",
            "tier2_slm_document_extraction": "slm",
            "tier3": "vlm",
            "tier3_vlm": "vlm",
            "vlm_judge": "vlm",
            "vlm_document_extraction": "vlm",
            "tier3_vlm_document_extraction": "vlm",
            "quality_fallback": "ocr",
            "confidence_fallback": "ocr"
        }
        
        normalized_source = source_map.get(source.lower(), source.lower())
        if "slm" in source.lower():
            normalized_source = "slm"
        elif "vlm" in source.lower():
            normalized_source = "vlm"
        
        # Format value based on field type
        if field_name in ["signature", "stamp"]:
            if isinstance(value, dict):
                field_value = value
            else:
                field_value = {
                    "present": bool(value) if value is not None else False
                }
                if bbox:
                    field_value["bbox"] = bbox
        else:
            field_value = value
        
        self.current_doc["fields"][field_name] = field_value
        self.current_doc["field_sources"][field_name] = normalized_source
    
    def end_document(
        self, 
        confidence: float, 
        processing_time: float, 
        cost_usd: float = 0.0
    ):
        """End logging for current document."""
        if self.current_doc is None:
            return
        
        self.current_doc["confidence"] = round(confidence, 4)
        self.current_doc["processing_time_sec"] = round(processing_time, 3)
        self.current_doc["cost_estimate_usd"] = round(cost_usd, 6)
        
        self.results.append(self.current_doc)
        self.current_doc = None
    
    def get_formatted_result(self, doc_result: Dict) -> Dict:
        """
        Format a document result for JSON output.
        
        Returns clean JSON structure with source attribution.
        """
        # Create the main output structure
        output = {
            "doc_id": doc_result["doc_id"],
            "fields": doc_result["fields"],
            "field_sources": doc_result["field_sources"],
            "confidence": doc_result["confidence"],
            "processing_time_sec": doc_result["processing_time_sec"],
            "cost_estimate_usd": doc_result["cost_estimate_usd"]
        }
        
        return output
    
    def save_results(self, filename: str = None) -> str:
        """
        Save all results to a JSON file.
        
        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"extraction_results_{self.run_id}.json"
        
        output_path = self.output_dir / filename
        
        # Format all results
        formatted_results = [
            self.get_formatted_result(r) for r in self.results
        ]
        
        # Compute summary statistics
        total_docs = len(formatted_results)
        source_counts = {"ocr": 0, "slm": 0, "vlm": 0}
        
        for result in formatted_results:
            for field, source in result.get("field_sources", {}).items():
                if source in source_counts:
                    source_counts[source] += 1
        
        output = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "total_documents": total_docs,
            "source_statistics": {
                "ocr_fields": source_counts["ocr"],
                "slm_fields": source_counts["slm"],
                "vlm_fields": source_counts["vlm"]
            },
            "results": formatted_results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        return str(output_path)
    
    def print_result(self, doc_result: Dict):
        """Print a single document result as formatted JSON."""
        formatted = self.get_formatted_result(doc_result)
        print(json.dumps(formatted, indent=2, ensure_ascii=False))
    
    def print_all_results(self):
        """Print all results as formatted JSON."""
        for result in self.results:
            self.print_result(result)
            print()  # Empty line between documents


# Global instance for easy access
_json_logger: Optional[JSONResultLogger] = None


def get_json_logger(output_dir: str = ".") -> JSONResultLogger:
    """Get or create the global JSON logger instance."""
    global _json_logger
    if _json_logger is None:
        _json_logger = JSONResultLogger(output_dir)
    return _json_logger


def reset_json_logger():
    """Reset the global JSON logger."""
    global _json_logger
    _json_logger = None
