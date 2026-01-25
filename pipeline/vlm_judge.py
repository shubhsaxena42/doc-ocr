"""
VLM Judge Module

Qwen3-VL-8B integration for Tier 3 visual adjudication.
Uses 8-bit quantization for accurate inference on document images.

Key Design Principles:
1. EXTRACTION with REASONING: Ask VLM to extract fields with JSON output
2. LABEL ANCHORING: Always instruct VLM to find the label first, then read the value
3. HIGH RESOLUTION: Preserve document detail with minimal downsampling (2048px)
4. FULL IMAGE CONTEXT: VLM always sees the full document for better accuracy
5. INFERENCE for UNCLEAR: When image quality is poor, use reasoning to infer values
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
    
    Uses Qwen3-VL-8B with 8-bit quantization for accuracy.
    Analyzes FULL document images to extract and verify fields.
    
    KEY IMPROVEMENT: Uses intelligent EXTRACTION with REASONING.
    Asks VLM to extract fields in JSON format, inferring unclear values.
    """
    
    # VERIFICATION prompts with label anchoring (Issue #2, #6 fixes)
    # These ask VLM to VERIFY between two options, not re-extract
    VERIFICATION_PROMPTS = {
        "dealer_name": """You are verifying OCR results on a tractor/vehicle invoice.

Two OCR engines extracted different values for the DEALER/SELLER NAME:
  Option A: "{value1}"
  Option B: "{value2}"

INSTRUCTIONS:
1. First, locate the LABEL: Look for text like "Dealer:", "Seller:", "From:", or a company letterhead at the top.
2. Then read the VALUE printed next to or below that label.
3. Choose which option (A or B) matches what you see, or say NEITHER if both are wrong.

Respond in this format:
CHOICE: [A or B or NEITHER]
REASON: [brief explanation]""",
        
        "model_name": """You are verifying OCR results on a tractor/vehicle invoice.

Two OCR engines extracted different values for the TRACTOR/VEHICLE MODEL:
  Option A: "{value1}"
  Option B: "{value2}"

INSTRUCTIONS:
1. First, locate the LABEL: Look for "Model:", "Tractor Model:", "Vehicle:", or in a product description.
2. Look for brand names like Mahindra, Massey Ferguson, John Deere, New Holland, Eicher, Kubota, Swaraj, etc.
3. The model should include brand + model number (e.g., "Mahindra 575 DI", "Massey Ferguson 1035").
4. Choose which option (A or B) matches what you see, or say NEITHER if both are wrong.

Respond in this format:
CHOICE: [A or B or NEITHER]
REASON: [brief explanation]""",
        
        "horse_power": """You are verifying OCR results on a tractor/vehicle invoice.

Two OCR engines extracted different values for HORSE POWER:
  Option A: "{value1}" HP
  Option B: "{value2}" HP

INSTRUCTIONS:
1. First, locate the LABEL: Look for "HP", "Horse Power", "H.P.", or "H/P" printed on the document.
2. Read the NUMBER printed directly before or after that label.
3. Typical tractor HP values are between 20-100 HP. Values outside this range are suspicious.
4. Choose which option (A or B) matches what you see, or say NEITHER if both are wrong.

Respond in this format:
CHOICE: [A or B or NEITHER]
REASON: [brief explanation]""",
        
        "asset_cost": """You are verifying OCR results on a tractor/vehicle invoice.

Two OCR engines extracted different values for the TOTAL COST/PRICE:
  Option A: Rs. {value1}
  Option B: Rs. {value2}

INSTRUCTIONS:
1. First, locate the LABEL: Look for "Total", "Grand Total", "Amount", "Price", or "Cost".
2. Read the RUPEE VALUE printed next to that label (may be in lakhs, e.g., "8,50,000" = 850000).
3. Tractor prices typically range from Rs. 3,00,000 to Rs. 15,00,000.
4. Choose which option (A or B) is closer to what you see, or say NEITHER if both are wrong.

Respond in this format:
CHOICE: [A or B or NEITHER]
REASON: [brief explanation]""",
        
        "signature": """You are verifying if a SIGNATURE exists on this tractor invoice.

Two OCR engines disagree about whether a signature is present:
  Option A: {value1}
  Option B: {value2}

INSTRUCTIONS:
1. Look for a handwritten signature, typically near the bottom of the document.
2. Look for labels like "Customer's Signature", "Authorized Signatory", or similar.
3. A signature looks like handwritten cursive or scribbles, not printed text.

Respond in this format:
CHOICE: [A or B]
REASON: [describe what you see - signature present or not]""",
        
        "stamp": """You are verifying if an OFFICIAL STAMP/SEAL exists on this tractor invoice.

Two OCR engines disagree about whether a stamp is present:
  Option A: {value1}
  Option B: {value2}

INSTRUCTIONS:
1. Look for a circular or rectangular stamp, usually blue, red, or purple ink.
2. Official stamps often contain company names, addresses, or "Authorized" text.
3. Stamps may be partially overlapping text or near signatures.

Respond in this format:
CHOICE: [A or B]
REASON: [describe what you see - stamp present or not]"""
    }
    
    # Legacy extraction prompts (kept for fallback)
    PROMPTS = {
        "dealer_name": """Look at this invoice document.
Find the LABEL: "Dealer", "Seller", "From", or company letterhead.
Read the VALUE next to that label.
Respond with just the dealer/company name, nothing else.""",
        
        "model_name": """Look at this invoice document.
Find the LABEL: "Model", "Tractor Model", or "Vehicle".
Read the VALUE next to that label (should include brand + model number).
Respond with just the model name, nothing else.""",
        
        "horse_power": """Look at this invoice document.
Find the LABEL: "HP", "Horse Power", "H.P.", or "H/P".
Read the NUMBER next to that label.
Respond with just the HP number (e.g., 45), nothing else.""",
        
        "asset_cost": """Look at this invoice document.
Find the LABEL: "Total", "Grand Total", "Amount", or "Price".
Read the RUPEE VALUE next to that label.
Respond with just the numeric value (e.g., 850000), nothing else.""",
        
        "signature": """Look at this document.
Is there a HANDWRITTEN SIGNATURE visible? Look near "Customer's Signature" or "Authorized Signatory" labels.
Respond with YES or NO.""",
        
        "stamp": """Look at this document.
Is there an OFFICIAL STAMP or SEAL visible? (circular/rectangular, colored ink)
Respond with YES or NO.""",
        
        "general": """Look at this document region.
The OCR engines disagree:
  Option A: "{value1}"
  Option B: "{value2}"
Which is correct? Respond with just the correct text."""
    }
    
    # Higher resolution limits (Issue #3 fix)
    DEFAULT_MAX_SIZE = 2048  # Was 1024 - preserve more detail
    DEFAULT_CROP_SIZE = (400, 400)  # Was (200, 200) - larger crops for small text
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",  # T4 compatible VLM
        crop_size: Tuple[int, int] = None,
        use_quantization: bool = True,
        quantization_bits: int = 4  # 4-bit for T4's 16GB VRAM
    ):
        """
        Initialize VLM judge.
        
        Args:
            model_name: HuggingFace model name
            crop_size: Default crop size for disputed regions (default: 400x400)
            use_quantization: Whether to use quantization
            quantization_bits: Bits for quantization (4 or 8, default: 8 for accuracy)
        """
        self.model_name = model_name
        self.crop_size = crop_size or self.DEFAULT_CROP_SIZE
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits
        self.model = None
        self.processor = None
        self._initialized = False
        self._invocation_count = 0
    
    def _lazy_init(self):
        """Lazy initialization of model optimized for Kaggle T4 GPU."""
        if self._initialized:
            return
        
        print(f"🔄 Loading VLM model: {self.model_name}...")
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            # Check for bfloat16 support (T4 supports it)
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            print(f"  Using compute dtype: {compute_dtype}")
            
            # Configure model loading for Kaggle
            model_kwargs = {
                "torch_dtype": compute_dtype,
                "device_map": "auto",
                "trust_remote_code": True,  # Required for Qwen models
                "low_cpu_mem_usage": True   # Reduces peak memory during loading
            }
            
            if self.use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    
                    # 8-bit quantization for accuracy (fits in T4's 16GB)
                    # 4-bit available for memory-constrained scenarios
                    if self.quantization_bits == 4:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        print("  Using 4-bit NF4 quantization")
                    else:  # Default to 8-bit for accuracy
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0
                        )
                        print("  Using 8-bit INT8 quantization")
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    print("  Warning: bitsandbytes not available, using native dtype")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            print("✓ VLM model loaded successfully!")
            self._initialized = True
            
        except Exception as e:
            print(f"❌ VLM initialization error: {e}")
            print("Possible solutions:")
            print("  1. pip install qwen-vl-utils")
            print("  2. pip install --upgrade transformers accelerate")
            print("  3. Check if model name is correct on HuggingFace")
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
    
    def judge_crop_from_path(
        self,
        image_path: str,
        field_name: str,
        bbox: List[int],
        padding: float = 0.2
    ) -> VLMJudgment:
        """
        Judge a field using a TIGHT CROP from the image.
        
        KEY IMPROVEMENT: For visual fields like signatures/stamps,
        sending a focused crop (with 20% padding) is more accurate
        than sending the full 2048px image.
        
        Args:
            image_path: Path to the full document image
            field_name: Name of the field (signature, stamp)
            bbox: Bounding box [x1, y1, x2, y2] from YOLO detector
            padding: Padding ratio (0.2 = 20% padding around bbox)
            
        Returns:
            VLMJudgment with resolved value
        """
        import cv2
        
        start_time = time.time()
        self._invocation_count += 1
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="ERROR",
                    reasoning=f"Could not load image: {image_path}",
                    latency=time.time() - start_time
                )
            
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # Add padding
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            # Crop
            crop = image[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Use the existing judge_crop method
            return self.judge_crop(crop_rgb, field_name, "", "")
            
        except Exception as e:
            return VLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"Crop error: {str(e)}",
                latency=time.time() - start_time
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
        Judge a field by viewing the FULL image with VERIFICATION prompts.
        
        IMPROVED: Uses verification mode - asks VLM to choose between OCR values
        instead of re-extracting from scratch.
        
        Args:
            image_path: Path to full image
            bbox: Bounding box (ignored - VLM sees full image for context)
            field_name: Name of the field
            value1: First OCR value (from engine 1)
            value2: Second OCR value (from engine 2)
            padding: Ignored (kept for backwards compatibility)
            
        Returns:
            VLMJudgment with resolved value (one of value1, value2, or new extraction)
        """
        # Use the new verification-based method
        return self.adjudicate_conflict(image_path, field_name, value1, value2)
    
    def adjudicate_conflict(
        self,
        image_path: str,
        field_name: str,
        value1: Any,
        value2: Any
    ) -> VLMJudgment:
        """
        Adjudicate between two OCR values using VERIFICATION mode.
        
        KEY IMPROVEMENT: Instead of asking "What is the value?", we ask
        "Which of these two options (A or B) is correct?"
        
        This leverages the work already done by OCR engines and gives the VLM
        a simpler, more constrained task.
        
        Args:
            image_path: Path to the full document image
            field_name: Name of the field being adjudicated
            value1: First OCR value (Option A)
            value2: Second OCR value (Option B)
            
        Returns:
            VLMJudgment with the chosen value and reasoning
        """
        import cv2
        
        start_time = time.time()
        self._invocation_count += 1
        
        self._lazy_init()
        
        if self.model is None:
            return self._fallback_judge(field_name, start_time)
        
        try:
            # Load FULL image
            image = cv2.imread(image_path)
            if image is None:
                return VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="ERROR",
                    reasoning="Could not load image",
                    latency=time.time() - start_time
                )
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize with HIGHER resolution limit (Issue #3 fix)
            max_dim = self.DEFAULT_MAX_SIZE  # 2048px instead of 1024px
            h, w = image_rgb.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                # Use LANCZOS for higher quality downsampling
                pil_image = Image.fromarray(image_rgb)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                pil_image = Image.fromarray(image_rgb)
            
            # Get VERIFICATION prompt (Issue #2 fix)
            if field_name in self.VERIFICATION_PROMPTS:
                prompt = self.VERIFICATION_PROMPTS[field_name].format(
                    value1=str(value1) if value1 else "NOT_FOUND",
                    value2=str(value2) if value2 else "NOT_FOUND"
                )
            else:
                # Fallback to general verification
                prompt = f"""You are verifying OCR results on a document.

Two OCR engines extracted different values for {field_name}:
  Option A: "{value1}"
  Option B: "{value2}"

Look at the document and decide which option is correct.

Respond in this format:
CHOICE: [A or B or NEITHER]
REASON: [brief explanation]"""
            
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
            
            # Generate with slightly higher temperature for better reasoning
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
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
            
            # Parse verification response
            return self._parse_verification_response(
                response, field_name, value1, value2, start_time
            )
            
        except Exception as e:
            return VLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"VLM adjudication error: {str(e)}",
                latency=time.time() - start_time
            )
    
    def _parse_verification_response(
        self,
        response: str,
        field_name: str,
        value1: Any,
        value2: Any,
        start_time: float
    ) -> VLMJudgment:
        """
        Parse VLM verification response.
        
        Looks for CHOICE: A/B/NEITHER and REASON: ...
        """
        import re
        
        latency = time.time() - start_time
        response_upper = response.upper()
        
        # Extract choice
        choice_match = re.search(r'CHOICE:\s*([ABN]|NEITHER)', response_upper)
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        
        reasoning = reason_match.group(1).strip() if reason_match else response
        
        if choice_match:
            choice = choice_match.group(1).strip()
            
            if choice == 'A':
                return VLMJudgment(
                    resolved_value=value1,
                    confidence=0.9,
                    status="RESOLVED",
                    reasoning=f"VLM chose Option A: {reasoning}",
                    latency=latency,
                    metadata={"choice": "A", "verification_mode": True}
                )
            elif choice == 'B':
                return VLMJudgment(
                    resolved_value=value2,
                    confidence=0.9,
                    status="RESOLVED",
                    reasoning=f"VLM chose Option B: {reasoning}",
                    latency=latency,
                    metadata={"choice": "B", "verification_mode": True}
                )
            else:  # NEITHER or N
                # Both options wrong - return uncertain and let fallback handle
                return VLMJudgment(
                    resolved_value=None,
                    confidence=0.3,
                    status="UNCERTAIN",
                    reasoning=f"VLM rejected both options: {reasoning}",
                    latency=latency,
                    metadata={"choice": "NEITHER", "verification_mode": True}
                )
        
        # Fallback: Check if response contains one of the values
        response_lower = response.lower()
        val1_str = str(value1).lower() if value1 else ""
        val2_str = str(value2).lower() if value2 else ""
        
        if val1_str and val1_str in response_lower:
            return VLMJudgment(
                resolved_value=value1,
                confidence=0.75,
                status="RESOLVED",
                reasoning=f"VLM response contains value1: {reasoning}",
                latency=latency
            )
        elif val2_str and val2_str in response_lower:
            return VLMJudgment(
                resolved_value=value2,
                confidence=0.75,
                status="RESOLVED",
                reasoning=f"VLM response contains value2: {reasoning}",
                latency=latency
            )
        
        # Cannot determine choice
        return VLMJudgment(
            resolved_value=None,
            confidence=0.0,
            status="UNCERTAIN",
            reasoning=f"Could not parse VLM choice: {response[:100]}",
            latency=latency
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
    
    def extract_all_fields(
        self,
        image_path: str,
        fields: List[str] = None
    ) -> Dict[str, VLMJudgment]:
        """
        Extract ALL fields from document image using VLM.
        
        This is called when VLM is invoked for any field - it extracts
        all fields at once from the full image, ignoring previous tier results.
        
        Args:
            image_path: Path to the document image
            fields: List of field names to extract (default: all 6 fields)
            
        Returns:
            Dict mapping field_name to VLMJudgment
        """
        import re
        
        start_time = time.time()
        self._invocation_count += 1
        
        if fields is None:
            fields = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]
        
        self._lazy_init()
        
        results = {}
        
        if self.model is None:
            # Model not available, return uncertain for all fields
            latency = time.time() - start_time
            for field_name in fields:
                results[field_name] = VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning="VLM model not available",
                    latency=latency
                )
            return results
        
        try:
            # Load full image
            pil_image = Image.open(image_path)
            
            # Resize if too large - use HIGHER resolution (Issue #3 fix)
            max_size = self.DEFAULT_MAX_SIZE  # 2048px instead of 1024px
            if pil_image.size[0] > max_size or pil_image.size[1] > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # JSON-based extraction prompt with reasoning for unclear values (Qwen3-VL optimized)
            extraction_prompt = \"\"\"You are an expert document AI extracting information from a tractor/vehicle invoice.

Extract the following fields in JSON format. For each field:
1. FIRST find the LABEL on the document, THEN read the VALUE next to it
2. If a value is unclear due to image quality, use your reasoning to infer the most likely value based on the rest of the document
3. If you cannot determine a value at all, use null

FIELDS TO EXTRACT:
- dealer_name: The company/dealer name (look for letterhead, "Dealer:", "Seller:", "From:")
- model_name: Tractor brand + model (e.g., "Mahindra 575 DI", "Massey Ferguson 1035")
  Common brands: Mahindra, Massey Ferguson, John Deere, New Holland, Eicher, Kubota, Swaraj, Sonalika, Powertrac, TAFE
- horse_power: The HP value as a number (look for "HP", "H.P.", typical range 20-100)
- asset_cost: Total price in rupees as a number (look for "Total", "Grand Total", range 300000-1500000)
- signature: Is there a handwritten signature? (true/false)
- stamp: Is there an official stamp/seal? (true/false)

Respond with ONLY valid JSON in this exact format:
{
  "dealer_name": "value or null",
  "model_name": "value or null", 
  "horse_power": number_or_null,
  "asset_cost": number_or_null,
  "signature": true_or_false,
  "stamp": true_or_false,
  "confidence": "high/medium/low",
  "reasoning": "brief explanation of any inferred values"
}\"\"\"

            # Prepare conversation format for Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": extraction_prompt}
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
                max_new_tokens=300,
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
            
            latency = time.time() - start_time
            
            # Parse the response for each field
            results = self._parse_all_fields_response(response, fields, latency)
            
        except Exception as e:
            latency = time.time() - start_time
            for field_name in fields:
                results[field_name] = VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="ERROR",
                    reasoning=f"VLM extraction error: {str(e)}",
                    latency=latency
                )
        
        return results
    
    def _parse_all_fields_response(
        self,
        response: str,
        fields: List[str],
        latency: float
    ) -> Dict[str, VLMJudgment]:
        """Parse the all-fields VLM extraction response (JSON or legacy text format)."""
        import re
        import json
        
        results = {}
        base_confidence = 0.75  # Default confidence
        
        # Try JSON parsing first (Qwen3-VL optimized)
        try:
            # Extract JSON block if wrapped in markdown
            json_match = re.search(r'```(?:json)?\\s*(\\{.*?\\})\\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\\{[^{}]*"dealer_name"[^{}]*\\}', response, re.DOTALL | re.IGNORECASE)
                json_str = json_match.group(0) if json_match else None
            
            if json_str:
                parsed = json.loads(json_str)
                
                # Get confidence from JSON
                conf_str = parsed.get("confidence", "medium")
                if isinstance(conf_str, str):
                    confidence_map = {"high": 0.90, "medium": 0.75, "low": 0.60}
                    base_confidence = confidence_map.get(conf_str.lower(), 0.75)
                
                reasoning_note = parsed.get("reasoning", "VLM JSON extraction")
                
                for field_name in fields:
                    value = parsed.get(field_name)
                    
                    if value is None or value == "null" or value == "":
                        results[field_name] = VLMJudgment(
                            resolved_value=None,
                            confidence=0.0,
                            status="RESOLVED",
                            reasoning="VLM could not find field in image",
                            latency=latency
                        )
                    elif field_name in ["signature", "stamp"]:
                        is_present = value if isinstance(value, bool) else str(value).upper() == "TRUE"
                        results[field_name] = VLMJudgment(
                            resolved_value={"present": is_present, "bbox": None, "verified": True},
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning=f"VLM extraction: {reasoning_note}",
                            latency=latency
                        )
                    elif field_name == "horse_power":
                        hp_value = float(value) if isinstance(value, (int, float)) else None
                        if hp_value is None and isinstance(value, str):
                            hp_match = re.search(r'(\\d+)', value)
                            hp_value = float(hp_match.group(1)) if hp_match else None
                        results[field_name] = VLMJudgment(
                            resolved_value=hp_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning=f"VLM extraction: {reasoning_note}",
                            latency=latency
                        )
                    elif field_name == "asset_cost":
                        cost_value = float(value) if isinstance(value, (int, float)) else None
                        if cost_value is None and isinstance(value, str):
                            cost_clean = re.sub(r'[^\\d.]', '', value)
                            cost_value = float(cost_clean) if cost_clean else None
                        results[field_name] = VLMJudgment(
                            resolved_value=cost_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning=f"VLM extraction: {reasoning_note}",
                            latency=latency
                        )
                    else:
                        results[field_name] = VLMJudgment(
                            resolved_value=value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning=f"VLM extraction: {reasoning_note}",
                            latency=latency
                        )
                
                return results
                
        except (json.JSONDecodeError, AttributeError):
            pass  # Fall through to legacy text parsing
        
        # Legacy text format parsing (fallback)
        # Extract confidence level
        conf_match = re.search(r'CONFIDENCE:\\s*(\\w+)', response, re.IGNORECASE)
        confidence_str = conf_match.group(1).lower() if conf_match else "medium"
        confidence_map = {"high": 0.90, "medium": 0.75, "low": 0.60}
        base_confidence = confidence_map.get(confidence_str, 0.75)
        
        # Field name mapping for legacy format
        field_patterns = {
            "dealer_name": r'DEALER_NAME:\\s*(.+?)(?:\\n|$)',
            "model_name": r'MODEL_NAME:\\s*(.+?)(?:\\n|$)',
            "horse_power": r'HORSE_POWER:\\s*(.+?)(?:\\n|$)',
            "asset_cost": r'ASSET_COST:\\s*(.+?)(?:\\n|$)',
            "signature": r'SIGNATURE:\\s*(.+?)(?:\\n|$)',
            "stamp": r'STAMP:\\s*(.+?)(?:\\n|$)'
        }
        
        for field_name in fields:
            pattern = field_patterns.get(field_name)
            if not pattern:
                results[field_name] = VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning="Unknown field",
                    latency=latency
                )
                continue
            
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Check for NOT_FOUND
                if "NOT_FOUND" in value.upper() or not value:
                    results[field_name] = VLMJudgment(
                        resolved_value=None,
                        confidence=0.0,
                        status="RESOLVED",
                        reasoning="VLM could not find field in image",
                        latency=latency
                    )
                else:
                    # Process field-specific values
                    if field_name in ["signature", "stamp"]:
                        is_present = "YES" in value.upper()
                        results[field_name] = VLMJudgment(
                            resolved_value={"present": is_present, "bbox": None, "verified": True},
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="VLM document-level extraction",
                            latency=latency
                        )
                    elif field_name == "horse_power":
                        hp_match = re.search(r'(\\d+)', value)
                        hp_value = float(hp_match.group(1)) if hp_match else None
                        results[field_name] = VLMJudgment(
                            resolved_value=hp_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="VLM document-level extraction",
                            latency=latency
                        )
                    elif field_name == "asset_cost":
                        cost_clean = re.sub(r'[^\\d.]', '', value)
                        cost_value = float(cost_clean) if cost_clean else None
                        results[field_name] = VLMJudgment(
                            resolved_value=cost_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="VLM document-level extraction",
                            latency=latency
                        )
                    else:
                        results[field_name] = VLMJudgment(
                            resolved_value=value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="VLM document-level extraction",
                            latency=latency
                        )
            else:
                results[field_name] = VLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning="Field not found in VLM response",
                    latency=latency
                )
        
        return results


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
