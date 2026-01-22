"""
SLM Judge Module

Qwen2.5-Coder-1.5B-Instruct integration for Tier 2 adjudication.
Uses 4-bit quantization via bitsandbytes for efficient inference.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
import re


@dataclass
class SLMJudgment:
    """Result from SLM judge."""
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


class SLMJudge:
    """
    Small Language Model judge for Tier 2 adjudication.
    
    Uses Qwen2.5-Coder-1.5B-Instruct with 4-bit quantization.
    Resolves OCR conflicts using context and reasoning.
    """
    
    # Prompt template for disambiguation
    PROMPT_TEMPLATE = """You are an expert document AI assistant specializing in financial document extraction.

Two OCR engines extracted different values for the same field. Determine the correct value.

Field: {field_name}
OCR Engine 1: "{value1}"
OCR Engine 2: "{value2}"
Context: {context}

Instructions:
1. Analyze both values considering common OCR errors
2. Use the context to determine the most likely correct value
3. If you cannot determine with confidence, respond with "UNCERTAIN"

Respond in this exact format:
ANSWER: [the correct value or UNCERTAIN]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation]"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_context_chars: int = 200,
        use_quantization: bool = True
    ):
        """
        Initialize SLM judge.
        
        Args:
            model_name: HuggingFace model name
            max_context_chars: Maximum context window size
            use_quantization: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
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
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self._initialized = True
            
        except Exception as e:
            print(f"SLM initialization error: {e}")
            self._initialized = True  # Prevent repeated attempts
    
    def judge(
        self,
        field_name: str,
        value1: str,
        value2: str,
        context: str = ""
    ) -> SLMJudgment:
        """
        Judge between two conflicting values.
        
        Args:
            field_name: Name of the field being judged
            value1: First OCR value
            value2: Second OCR value
            context: Document context (truncated to max_context_chars)
            
        Returns:
            SLMJudgment with resolved value or UNCERTAIN status
        """
        start_time = time.time()
        
        self._lazy_init()
        
        if self.model is None:
            # Model not available, use fallback heuristics
            return self._fallback_judge(field_name, value1, value2, context, start_time)
        
        try:
            # Prepare prompt
            truncated_context = context[:self.max_context_chars] if context else "No additional context available"
            
            prompt = self.PROMPT_TEMPLATE.format(
                field_name=field_name,
                value1=value1,
                value2=value2,
                context=truncated_context
            )
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            return self._parse_response(response, field_name, value1, value2, start_time)
            
        except Exception as e:
            latency = time.time() - start_time
            return SLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"Model error: {str(e)}",
                latency=latency
            )
    
    def _parse_response(
        self,
        response: str,
        field_name: str,
        value1: str,
        value2: str,
        start_time: float
    ) -> SLMJudgment:
        """Parse model response to extract judgment."""
        latency = time.time() - start_time
        
        # Extract ANSWER
        answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if not answer_match:
            return SLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="UNCERTAIN",
                reasoning="Could not parse model response",
                latency=latency,
                metadata={"raw_response": response}
            )
        
        answer = answer_match.group(1).strip()
        
        # Check for UNCERTAIN
        if "UNCERTAIN" in answer.upper():
            return SLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="UNCERTAIN",
                reasoning="Model uncertain",
                latency=latency
            )
        
        # Extract CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
        confidence_str = conf_match.group(1).lower() if conf_match else "medium"
        
        confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        confidence = confidence_map.get(confidence_str, 0.7)
        
        # Extract REASONING
        reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else ""
        
        # Determine which value was chosen
        resolved = answer
        if self._values_match(answer, value1):
            resolved = value1
        elif self._values_match(answer, value2):
            resolved = value2
        
        return SLMJudgment(
            resolved_value=resolved,
            confidence=confidence,
            status="RESOLVED",
            reasoning=reasoning,
            latency=latency
        )
    
    def _values_match(self, answer: str, value: str) -> bool:
        """Check if answer matches a value (fuzzy)."""
        answer_clean = re.sub(r'[^\w\s]', '', answer.lower())
        value_clean = re.sub(r'[^\w\s]', '', value.lower())
        
        return (
            answer_clean == value_clean or
            answer_clean in value_clean or
            value_clean in answer_clean
        )
    
    def _fallback_judge(
        self,
        field_name: str,
        value1: str,
        value2: str,
        context: str,
        start_time: float
    ) -> SLMJudgment:
        """
        Fallback judgment when model is not available.
        Uses simple heuristics based on field type.
        """
        latency = time.time() - start_time
        
        # For numeric fields, try to parse and compare
        if field_name in ["horse_power", "asset_cost"]:
            try:
                # Extract numbers
                nums1 = re.findall(r'[\d,]+\.?\d*', value1.replace(',', ''))
                nums2 = re.findall(r'[\d,]+\.?\d*', value2.replace(',', ''))
                
                if nums1 and nums2:
                    n1 = float(nums1[0])
                    n2 = float(nums2[0])
                    
                    # Check which appears in context
                    if context:
                        if str(int(n1)) in context and str(int(n2)) not in context:
                            return SLMJudgment(
                                resolved_value=value1,
                                confidence=0.7,
                                status="RESOLVED",
                                reasoning="Value found in context",
                                latency=latency
                            )
                        elif str(int(n2)) in context and str(int(n1)) not in context:
                            return SLMJudgment(
                                resolved_value=value2,
                                confidence=0.7,
                                status="RESOLVED",
                                reasoning="Value found in context",
                                latency=latency
                            )
            except:
                pass
        
        # For text fields, prefer longer, more complete values
        if len(value1) > len(value2) * 1.5:
            return SLMJudgment(
                resolved_value=value1,
                confidence=0.6,
                status="RESOLVED",
                reasoning="Preferred more complete value",
                latency=latency
            )
        elif len(value2) > len(value1) * 1.5:
            return SLMJudgment(
                resolved_value=value2,
                confidence=0.6,
                status="RESOLVED",
                reasoning="Preferred more complete value",
                latency=latency
            )
        
        # Cannot determine
        return SLMJudgment(
            resolved_value=None,
            confidence=0.0,
            status="UNCERTAIN",
            reasoning="Fallback heuristics inconclusive",
            latency=latency
        )


# Global instance
_slm_judge: Optional[SLMJudge] = None


def get_slm_judge() -> SLMJudge:
    """Get or create the global SLM judge instance."""
    global _slm_judge
    if _slm_judge is None:
        _slm_judge = SLMJudge()
    return _slm_judge


def judge_conflict(
    field_name: str,
    value1: str,
    value2: str,
    context: str = ""
) -> SLMJudgment:
    """
    Convenience function to judge a conflict.
    
    Args:
        field_name: Name of the field
        value1: First value
        value2: Second value
        context: Document context
        
    Returns:
        SLMJudgment with resolution
    """
    judge = get_slm_judge()
    return judge.judge(field_name, value1, value2, context)
