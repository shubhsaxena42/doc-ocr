"""
VLM Judge Module

Qwen2.5-VL-7B integration for Tier 3 visual adjudication.
Uses 4-bit quantization for efficient inference on image crops.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from PIL import Image


@dataclass
class VLMJudgment:
    """Result from VLM judge."""
    resolved_value: Optional[Any]
    confidence: float
    status: str  # "RESOLVED", "UNCERTAIN", "ERROR"
    reasoning: str = ""
    latency: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "resolved_value": self.resolved_value,
            "confidence": self.confidence,
            "status": self.status,
            "reasoning": self.reasoning,
            "latency": self.latency,
            "metadata": self.metadata
        }


class VLMJudge:
    """
    Vision Language Model judge for Tier 3 adjudication.
    
    Uses Qwen2.5-VL-7B with 4-bit quantization.
    Analyzes image crops to resolve visual ambiguities.
    """
    
    # Prompt templates for different field types
    PROMPTS = {
        "dealer_name": """Look at this cropped region from a document. 
What is the dealer/seller name visible in this image?
If unclear, respond with UNCERTAIN.
Respond with just the name, nothing else.""",
        
        "model_name": """Look at this cropped region from a document.
What is the tractor/vehicle model name visible in this image?
If unclear, respond with UNCERTAIN.
Respond with just the model name, nothing else.""",
        
        "horse_power": """Look at this cropped region from a document.
What is the horse power (HP) value visible in this image?
If unclear, respond with UNCERTAIN.
Respond with just the number, nothing else.""",
        
        "asset_cost": """Look at this cropped region from a document.
What is the price/cost/amount visible in this image?
If unclear, respond with UNCERTAIN.
Respond with just the numeric value, nothing else.""",
        
        "signature": """Look at this cropped region from a document.
Is there a handwritten signature in this image?
Respond with YES or NO.""",
        
        "stamp": """Look at this cropped region from a document.
Is there an official stamp or seal in this image?
Respond with YES or NO.""",
        
        "general": """Look at this cropped region from a document.
What text is visible in this area? The OCR engines disagree:
Option 1: "{value1}"
Option 2: "{value2}"
Which is correct? Respond with just the correct text."""
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        crop_size: Tuple[int, int] = (200, 200),
        use_quantization: bool = True
    ):
        """
        Initialize VLM judge.
        
        Args:
            model_name: HuggingFace model name
            crop_size: Default crop size for disputed regions
            use_quantization: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.crop_size = crop_size
        self.use_quantization = use_quantization
        self.model = None
        self.processor = None
        self._initialized = False
        self._invocation_count = 0
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            import torch
            
            # Configure quantization
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto"
            }
            
            if self.use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print("bitsandbytes not available, using float16")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self._initialized = True
            
        except Exception as e:
            print(f"VLM initialization error: {e}")
            self._initialized = True  # Prevent repeated attempts
    
    @property
    def invocation_count(self) -> int:
        """Get total number of VLM invocations."""
        return self._invocation_count
    
    def judge_crop(
        self,
        image_crop: np.ndarray,
        field_name: str,
        value1: str = "",
        value2: str = ""
    ) -> VLMJudgment:
        """
        Judge a cropped image region.
        
        Args:
            image_crop: Cropped image as numpy array
            field_name: Name of the field being judged
            value1: First OCR value (optional, for comparison)
            value2: Second OCR value (optional, for comparison)
            
        Returns:
            VLMJudgment with resolved value
        """
        start_time = time.time()
        self._invocation_count += 1
        
        self._lazy_init()
        
        if self.model is None:
            # Model not available, use fallback
            return self._fallback_judge(field_name, start_time)
        
        try:
            # Convert numpy to PIL
            if isinstance(image_crop, np.ndarray):
                pil_image = Image.fromarray(image_crop)
            else:
                pil_image = image_crop
            
            # Resize if needed
            if pil_image.size[0] > self.crop_size[0] or pil_image.size[1] > self.crop_size[1]:
                pil_image.thumbnail(self.crop_size, Image.Resampling.LANCZOS)
            
            # Get appropriate prompt
            prompt = self.PROMPTS.get(field_name, self.PROMPTS["general"])
            if "{value1}" in prompt:
                prompt = prompt.format(value1=value1, value2=value2)
            
            # Prepare conversation format for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            )
            
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )[0]
            
            return self._parse_response(response, field_name, value1, value2, start_time)
            
        except Exception as e:
            latency = time.time() - start_time
            return VLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"VLM error: {str(e)}",
                latency=latency
            )
    
    def judge_from_image(
        self,
        image_path: str,
        bbox: List[int],
        field_name: str,
        value1: str = "",
        value2: str = "",
        padding: int = 20
    ) -> VLMJudgment:
        """
        Extract crop from image and judge.
        
        Args:
            image_path: Path to full image
            bbox: Bounding box [x, y, width, height]
            field_name: Name of the field
            value1: First OCR value
            value2: Second OCR value
            padding: Extra pixels around crop
            
        Returns:
            VLMJudgment with resolved value
        """
        import cv2
        
        start_time = time.time()
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="ERROR",
                    reasoning="Could not load image",
                    latency=time.time() - start_time
                )
            
            # Extract crop with padding
            x, y, w, h = bbox
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            return self.judge_crop(crop_rgb, field_name, value1, value2)
            
        except Exception as e:
            return VLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"Crop error: {str(e)}",
                latency=time.time() - start_time
            )
    
    def _parse_response(
        self,
        response: str,
        field_name: str,
        value1: str,
        value2: str,
        start_time: float
    ) -> VLMJudgment:
        """Parse VLM response."""
        latency = time.time() - start_time
        response = response.strip()
        
        # Handle UNCERTAIN
        if "UNCERTAIN" in response.upper() or "UNCLEAR" in response.upper():
            return VLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="UNCERTAIN",
                reasoning="VLM could not determine",
                latency=latency
            )
        
        # Handle YES/NO for signature/stamp
        if field_name in ["signature", "stamp"]:
            is_present = "YES" in response.upper()
            return VLMJudgment(
                resolved_value={"present": is_present},
                confidence=0.85 if is_present else 0.8,
                status="RESOLVED",
                reasoning=response,
                latency=latency
            )
        
        # For other fields, use the response as the value
        resolved = response
        confidence = 0.8
        
        # Boost confidence if matches one of the options
        if value1 and self._fuzzy_match(response, value1):
            resolved = value1
            confidence = 0.9
        elif value2 and self._fuzzy_match(response, value2):
            resolved = value2
            confidence = 0.9
        
        return VLMJudgment(
            resolved_value=resolved,
            confidence=confidence,
            status="RESOLVED",
            reasoning="VLM extraction",
            latency=latency
        )
    
    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """Simple fuzzy match for response comparison."""
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        return t1 == t2 or t1 in t2 or t2 in t1
    
    def _fallback_judge(
        self,
        field_name: str,
        start_time: float
    ) -> VLMJudgment:
        """Fallback when model not available."""
        latency = time.time() - start_time
        return VLMJudgment(
            resolved_value=None,
            confidence=0.0,
            status="UNCERTAIN",
            reasoning="VLM not available",
            latency=latency
        )


# Global instance
_vlm_judge: Optional[VLMJudge] = None


def get_vlm_judge() -> VLMJudge:
    """Get or create the global VLM judge instance."""
    global _vlm_judge
    if _vlm_judge is None:
        _vlm_judge = VLMJudge()
    return _vlm_judge


def judge_visual_conflict(
    image_path: str,
    bbox: List[int],
    field_name: str,
    value1: str = "",
    value2: str = ""
) -> VLMJudgment:
    """
    Convenience function to judge a visual conflict.
    
    Args:
        image_path: Path to the document image
        bbox: Bounding box of disputed region
        field_name: Name of the field
        value1: First OCR value
        value2: Second OCR value
        
    Returns:
        VLMJudgment with resolution
    """
    judge = get_vlm_judge()
    return judge.judge_from_image(image_path, bbox, field_name, value1, value2)


def get_invocation_count() -> int:
    """Get total VLM invocations."""
    if _vlm_judge:
        return _vlm_judge.invocation_count
    return 0
