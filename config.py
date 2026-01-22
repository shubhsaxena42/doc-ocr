"""
IDAI Pipeline Configuration

Environment variables, thresholds, cost constants, and mode configurations
for the Intelligent Document AI pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


# ============================================================================
# Environment Configuration
# ============================================================================

# PaddlePaddle environment flags
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_executor'] = '0'

# CUDA configuration
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


class PipelineMode(Enum):
    """Pipeline execution modes."""
    FULL = "full"           # Full pipeline with all tiers including VLM
    CPU_LITE = "cpu-lite"   # CPU-only mode, skips Tier 3 VLM


# ============================================================================
# Threshold Configuration
# ============================================================================

@dataclass
class ThresholdConfig:
    """Threshold values for conflict detection and consensus."""
    
    # Fuzzy matching thresholds
    fuzzy_match_threshold: float = 0.90      # RapidFuzz score threshold
    token_overlap_threshold: float = 0.70    # Token overlap for textual fields
    
    # Numeric conflict detection
    numeric_diff_threshold: float = 0.05     # 5% difference triggers conflict
    
    # Detection verification thresholds
    signature_stroke_density_min: float = 0.05
    signature_stroke_density_max: float = 0.30
    signature_min_components: int = 5         # Minimum connected components
    
    stamp_circularity_iou: float = 0.50       # IoU threshold for ellipse fit
    stamp_max_ocr_words: int = 10             # Reject if too much text in bbox
    
    # Confidence thresholds
    confidence_high: float = 0.95             # High confidence, no adjudication
    confidence_borderline_min: float = 0.60   # Borderline for active learning
    confidence_borderline_max: float = 0.95
    
    # VLM invocation limit
    vlm_max_invocation_rate: float = 0.10     # Must be <10% of documents


# ============================================================================
# Cost Configuration
# ============================================================================

@dataclass
class CostConfig:
    """Cost per component invocation (in USD)."""
    
    paddle_ocr: float = 0.0           # Local, free
    deepseek_ocr: float = 0.0         # Local quantized, free
    yolo_detection: float = 0.0       # Local ONNX, free
    tier1_rules: float = 0.0          # Deterministic, free
    tier2_slm: float = 0.001          # Qwen2.5-1.5B per invocation
    tier3_vlm: float = 0.02           # Qwen2.5-VL-7B per invocation
    
    # Target cost per document
    target_cost_per_doc: float = 0.01


# ============================================================================
# Latency Configuration
# ============================================================================

@dataclass
class LatencyConfig:
    """Target latency per component (in seconds)."""
    
    parallel_ocr: float = 5.0         # Both OCR engines in parallel
    yolo_detection: float = 3.0       # YOLO with batch_size=4
    tier1_rules: float = 0.001        # <1ms
    tier2_slm: float = 5.0            # SLM inference
    tier3_vlm: float = 15.0           # VLM inference
    
    # Target total latency
    target_total_latency: float = 30.0


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model paths and configurations."""
    
    # OCR Models
    paddle_ocr_lang: str = "en"
    paddle_ocr_use_angle_cls: bool = True
    paddle_ocr_use_gpu: bool = True
    
    # YOLO Models (ONNX format)
    yolo_signature_model: str = "models/yolo_signature.onnx"
    yolo_stamp_model: str = "models/yolo_stamp.onnx"
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    
    # SLM (Tier 2)
    slm_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    slm_quantization: str = "4bit"  # NF4 via bitsandbytes
    slm_max_context_chars: int = 200
    
    # VLM (Tier 3)
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_quantization: str = "4bit"  # NF4 via bitsandbytes
    vlm_crop_size: tuple = (200, 200)


# ============================================================================
# Field Configuration
# ============================================================================

@dataclass
class FieldConfig:
    """Configuration for field extraction."""
    
    # Fields to extract
    fields: List[str] = field(default_factory=lambda: [
        "dealer_name",
        "model_name", 
        "horse_power",
        "asset_cost",
        "signature",
        "stamp"
    ])
    
    # Numeric fields (use numeric conflict detection)
    numeric_fields: List[str] = field(default_factory=lambda: [
        "horse_power",
        "asset_cost"
    ])
    
    # Textual fields (use fuzzy match conflict detection)
    textual_fields: List[str] = field(default_factory=lambda: [
        "dealer_name",
        "model_name"
    ])
    
    # Visual fields (binary + bbox)
    visual_fields: List[str] = field(default_factory=lambda: [
        "signature",
        "stamp"
    ])
    
    # Company suffixes to strip for dealer matching
    company_suffixes: List[str] = field(default_factory=lambda: [
        "pvt", "ltd", "private", "limited",
        "traders", "motors", "tractors",
        "agencies", "enterprises", "co",
        "inc", "corp", "llc", "llp"
    ])
    
    # Devanagari digit mapping
    devanagari_digits: Dict[str, str] = field(default_factory=lambda: {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
    })
    
    # Gujarati digit mapping
    gujarati_digits: Dict[str, str] = field(default_factory=lambda: {
        '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
        '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9'
    })


# ============================================================================
# HSV Color Ranges for Stamp Detection
# ============================================================================

@dataclass
class StampColorConfig:
    """HSV color ranges for stamp detection."""
    
    # Red stamps (wraps around in HSV)
    red_lower1: tuple = (0, 100, 100)
    red_upper1: tuple = (10, 255, 255)
    red_lower2: tuple = (160, 100, 100)
    red_upper2: tuple = (180, 255, 255)
    
    # Blue stamps
    blue_lower: tuple = (100, 100, 100)
    blue_upper: tuple = (130, 255, 255)
    
    # Black stamps (low saturation, low value)
    black_lower: tuple = (0, 0, 0)
    black_upper: tuple = (180, 50, 80)


# ============================================================================
# Active Learning Configuration
# ============================================================================

@dataclass
class ActiveLearningConfig:
    """Configuration for active learning loop."""
    
    samples_per_round: int = 30
    high_disagreement_samples: int = 10
    borderline_confidence_samples: int = 10
    rare_layout_samples: int = 10
    
    # Minimum golden set size
    min_golden_set_size: int = 50
    
    # Co-training promotion threshold
    co_training_confidence_threshold: float = 0.95


# ============================================================================
# Global Configuration Instance
# ============================================================================

@dataclass
class PipelineConfig:
    """Main configuration container."""
    
    mode: PipelineMode = PipelineMode.FULL
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    stamp_colors: StampColorConfig = field(default_factory=StampColorConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "output"
    golden_set_path: str = "data/golden_set.json"
    
    def set_mode(self, mode: str):
        """Set pipeline mode from string."""
        if mode == "cpu-lite":
            self.mode = PipelineMode.CPU_LITE
        else:
            self.mode = PipelineMode.FULL
    
    @property
    def use_vlm(self) -> bool:
        """Check if VLM (Tier 3) should be used."""
        return self.mode == PipelineMode.FULL


# Create default config instance
config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the global configuration instance."""
    return config


def set_mode(mode: str):
    """Set the pipeline mode."""
    config.set_mode(mode)
