"""
Cost Tracker Module

Tracks per-component costs for each document processed through the pipeline.
Ensures compliance with <$0.01/document target.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class ComponentInvocation:
    """Record of a single component invocation."""
    component: str
    cost: float
    latency: float  # in seconds
    timestamp: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class DocumentCostRecord:
    """Cost and latency record for a single document."""
    doc_id: str
    image_path: str
    invocations: List[ComponentInvocation] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost for this document."""
        return sum(inv.cost for inv in self.invocations)
    
    @property
    def total_latency(self) -> float:
        """Calculate total latency for this document."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(inv.latency for inv in self.invocations)
    
    @property
    def component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get cost and latency breakdown by component."""
        breakdown = {}
        for inv in self.invocations:
            if inv.component not in breakdown:
                breakdown[inv.component] = {"cost": 0.0, "latency": 0.0, "count": 0}
            breakdown[inv.component]["cost"] += inv.cost
            breakdown[inv.component]["latency"] += inv.latency
            breakdown[inv.component]["count"] += 1
        return breakdown
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "doc_id": self.doc_id,
            "image_path": self.image_path,
            "total_cost": self.total_cost,
            "total_latency": self.total_latency,
            "component_breakdown": self.component_breakdown,
            "invocations": [
                {
                    "component": inv.component,
                    "cost": inv.cost,
                    "latency": inv.latency,
                    "timestamp": inv.timestamp,
                    "metadata": inv.metadata
                }
                for inv in self.invocations
            ]
        }


class CostTracker:
    """
    Tracks costs and latency across the entire pipeline.
    
    Usage:
        tracker = CostTracker()
        tracker.start_document("doc_001", "/path/to/image.png")
        
        # Log component invocations
        tracker.log_ocr("paddle", latency=2.5)
        tracker.log_yolo("signature", latency=1.2)
        tracker.log_slm(latency=4.8)
        
        tracker.end_document()
        
        # Get results
        record = tracker.get_document_record("doc_001")
        print(f"Total cost: ${record.total_cost:.4f}")
    """
    
    # Cost constants (from config, but defined here for standalone use)
    COSTS = {
        "paddle_ocr": 0.0,
        "deepseek_ocr": 0.0,
        "yolo_signature": 0.0,
        "yolo_stamp": 0.0,
        "tier1_rules": 0.0,
        "tier2_slm": 0.001,
        "tier3_vlm": 0.02
    }
    
    def __init__(self):
        self.documents: Dict[str, DocumentCostRecord] = {}
        self.current_doc_id: Optional[str] = None
        self._vlm_invocations: int = 0
        self._total_documents: int = 0
    
    def start_document(self, doc_id: str, image_path: str):
        """Start tracking a new document."""
        self.current_doc_id = doc_id
        self.documents[doc_id] = DocumentCostRecord(
            doc_id=doc_id,
            image_path=image_path,
            start_time=time.time()
        )
        self._total_documents += 1
    
    def end_document(self):
        """End tracking for current document."""
        if self.current_doc_id and self.current_doc_id in self.documents:
            self.documents[self.current_doc_id].end_time = time.time()
        self.current_doc_id = None
    
    def _log_invocation(self, component: str, latency: float, metadata: Dict = None):
        """Internal method to log a component invocation."""
        if not self.current_doc_id:
            raise ValueError("No document currently being tracked. Call start_document first.")
        
        cost = self.COSTS.get(component, 0.0)
        invocation = ComponentInvocation(
            component=component,
            cost=cost,
            latency=latency,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.documents[self.current_doc_id].invocations.append(invocation)
        
        # Track VLM invocations for rate monitoring
        if component == "tier3_vlm":
            self._vlm_invocations += 1
    
    def log_ocr(self, engine: str, latency: float, metadata: Dict = None):
        """
        Log OCR engine invocation.
        
        Args:
            engine: "paddle" or "deepseek"
            latency: Time taken in seconds
            metadata: Additional info (e.g., word count, language detected)
        """
        component = f"{engine}_ocr"
        self._log_invocation(component, latency, metadata)
    
    def log_yolo(self, detection_type: str, latency: float, metadata: Dict = None):
        """
        Log YOLO detection invocation.
        
        Args:
            detection_type: "signature" or "stamp"
            latency: Time taken in seconds
            metadata: Additional info (e.g., detections found, confidence)
        """
        component = f"yolo_{detection_type}"
        self._log_invocation(component, latency, metadata)
    
    def log_tier1(self, latency: float, metadata: Dict = None):
        """Log Tier 1 deterministic rules invocation."""
        self._log_invocation("tier1_rules", latency, metadata)
    
    def log_slm(self, latency: float, metadata: Dict = None):
        """
        Log Tier 2 SLM judge invocation.
        Cost: $0.001 per call
        """
        self._log_invocation("tier2_slm", latency, metadata)
    
    def log_vlm(self, latency: float, metadata: Dict = None):
        """
        Log Tier 3 VLM judge invocation.
        Cost: $0.02 per call
        """
        self._log_invocation("tier3_vlm", latency, metadata)
    
    def get_document_record(self, doc_id: str) -> Optional[DocumentCostRecord]:
        """Get the cost record for a specific document."""
        return self.documents.get(doc_id)
    
    def get_document_cost(self, doc_id: str) -> float:
        """Get total cost for a specific document."""
        record = self.documents.get(doc_id)
        return record.total_cost if record else 0.0
    
    def get_document_latency(self, doc_id: str) -> float:
        """Get total latency for a specific document."""
        record = self.documents.get(doc_id)
        return record.total_latency if record else 0.0
    
    @property
    def vlm_invocation_rate(self) -> float:
        """Calculate VLM invocation rate across all documents."""
        if self._total_documents == 0:
            return 0.0
        return self._vlm_invocations / self._total_documents
    
    @property
    def average_cost_per_document(self) -> float:
        """Calculate average cost per document."""
        if not self.documents:
            return 0.0
        total_cost = sum(doc.total_cost for doc in self.documents.values())
        return total_cost / len(self.documents)
    
    @property
    def average_latency_per_document(self) -> float:
        """Calculate average latency per document."""
        if not self.documents:
            return 0.0
        total_latency = sum(doc.total_latency for doc in self.documents.values())
        return total_latency / len(self.documents)
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all documents."""
        return {
            "total_documents": len(self.documents),
            "average_cost_per_document": self.average_cost_per_document,
            "average_latency_per_document": self.average_latency_per_document,
            "vlm_invocation_rate": self.vlm_invocation_rate,
            "vlm_invocations": self._vlm_invocations,
            "total_cost": sum(doc.total_cost for doc in self.documents.values()),
            "total_latency": sum(doc.total_latency for doc in self.documents.values()),
            "within_cost_target": self.average_cost_per_document < 0.01,
            "within_latency_target": self.average_latency_per_document < 30.0,
            "vlm_rate_acceptable": self.vlm_invocation_rate < 0.10
        }
    
    def to_json(self, filepath: str = None) -> str:
        """Export all tracking data to JSON."""
        data = {
            "summary": self.get_summary(),
            "documents": {
                doc_id: record.to_dict() 
                for doc_id, record in self.documents.items()
            }
        }
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def reset(self):
        """Reset all tracking data."""
        self.documents = {}
        self.current_doc_id = None
        self._vlm_invocations = 0
        self._total_documents = 0


# Global tracker instance
_global_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_tracker():
    """Reset the global tracker."""
    global _global_tracker
    _global_tracker = CostTracker()
