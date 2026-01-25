"""
SLM Judge Module

Qwen2.5-1.5B-Instruct integration for Tier 2 adjudication.
Uses 4-bit quantization via bitsandbytes for efficient inference.

KEY IMPROVEMENTS:
- JSON output format for robust parsing
- Layout-aware context hints
- Fuzzy matching integration
"""

import time
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


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


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from model output.
    
    Handles common issues with smaller models:
    - JSON wrapped in markdown code blocks
    - Extra text before/after JSON
    - Missing quotes, trailing commas
    """
    # Try to find JSON block in markdown
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = json_match.group()
            # Remove trailing commas
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    
    return None


class SLMJudge:
    """
    Small Language Model judge for Tier 2 adjudication.
    
    Uses Qwen2.5-1.5B-Instruct (not Coder variant - better for documents).
    Resolves OCR conflicts using context and reasoning.
    
    KEY IMPROVEMENT: Uses JSON output for robust parsing.
    """
    
    # JSON-based prompt template for robust parsing
    PROMPT_TEMPLATE = '''You are a document extraction expert. Two OCR systems read a tractor invoice and got different values.

Field: {field_name}
OCR 1: "{value1}"
OCR 2: "{value2}"
{layout_hint}
{master_hint}

Analyze both values. Consider:
- Common OCR errors (1/l, 0/O, missing characters)
- Which value looks more like a valid {field_name}
{field_hint}

Respond with ONLY a JSON object (no other text):
{{"answer": "the correct value or null if uncertain", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}'''

    FIELD_HINTS = {
        "dealer_name": "Dealer names are usually company names with words like 'Tractors', 'Motors', 'Agro', 'Pvt Ltd'",
        "model_name": "Model names include brand (Mahindra, Massey, John Deere, etc.) + model number",
        "horse_power": "HP values are typically 20-100 for tractors. Check for digits only.",
        "asset_cost": "Costs are in rupees, typically 3-15 lakh (300000-1500000). Look for numeric values.",
        "signature": "Signature is YES or NO based on presence of handwriting",
        "stamp": "Stamp is YES or NO based on presence of official seal"
    }

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",  # Changed from Qwen-Coder to Llama for better docs
        max_context_chars: int = 1500,  # Increased from 200 per critique recommendation
        use_quantization: bool = True
    ):
        """
        Initialize SLM judge.
        
        Args:
            model_name: HuggingFace model name. Llama-3.2-3B-Instruct recommended for documents.
                       Alternatives: "Qwen/Qwen2.5-1.5B-Instruct", "microsoft/phi-2"
            max_context_chars: Maximum context window size (1500 recommended for invoices)
            use_quantization: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._init_error = None  # Track initialization errors
    
    def _lazy_init(self):
        """Lazy initialization of model with explicit error tracking."""
        if self._initialized:
            return
        
        print(f"Loading SLM model: {self.model_name}...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Configure quantization
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True
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
                    print("Warning: bitsandbytes not available, using float16")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✓ SLM model loaded successfully!")
            self._initialized = True
            
        except Exception as e:
            self._init_error = str(e)
            print(f"ERROR: SLM initialization failed: {e}")
            print("The pipeline will use fallback heuristics (lower accuracy)")
            self._initialized = True  # Prevent repeated attempts
    
    def judge(
        self,
        field_name: str,
        value1: str,
        value2: str,
        context: str = "",
        layout_hint: str = "",
        master_hint: str = ""
    ) -> SLMJudgment:
        """
        Judge between two conflicting values using JSON output format.
        
        Args:
            field_name: Name of the field being judged
            value1: First OCR value
            value2: Second OCR value
            context: Document context
            layout_hint: Spatial layout information (e.g., "Value1 found in header area")
            master_hint: Master-list fuzzy match hint (e.g., "System suggests: 'ABC Motors'")
            
        Returns:
            SLMJudgment with resolved value or UNCERTAIN status
        """
        start_time = time.time()
        
        self._lazy_init()
        
        if self.model is None:
            # Model not available, use fallback heuristics
            return self._fallback_judge(field_name, value1, value2, context, start_time)
        
        try:
            # Prepare layout hint
            if layout_hint:
                layout_section = f"Layout info: {layout_hint}"
            else:
                layout_section = ""
            
            # Prepare master-list hint
            if master_hint:
                master_section = f"Master-list hint: {master_hint}"
            else:
                master_section = ""
            
            # Get field-specific hint
            field_hint = self.FIELD_HINTS.get(field_name, "")
            
            # Format prompt
            prompt = self.PROMPT_TEMPLATE.format(
                field_name=field_name,
                value1=value1,
                value2=value2,
                layout_hint=layout_section,
                master_hint=master_section,
                field_hint=field_hint
            )
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            if prompt in response:
                response = response[len(prompt):].strip()
            
            # Parse JSON response
            return self._parse_json_response(response, field_name, value1, value2, start_time)
            
        except Exception as e:
            latency = time.time() - start_time
            return SLMJudgment(
                resolved_value=None,
                confidence=0.0,
                status="ERROR",
                reasoning=f"Model error: {str(e)}",
                latency=latency
            )
    
    def _parse_json_response(
        self,
        response: str,
        field_name: str,
        value1: str,
        value2: str,
        start_time: float
    ) -> SLMJudgment:
        """
        Parse JSON response from SLM using robust extraction.
        
        Uses the extract_json_from_response function to handle
        malformed output from smaller models.
        """
        latency = time.time() - start_time
        
        # Try to extract JSON
        parsed = extract_json_from_response(response)
        
        if parsed:
            answer = parsed.get("answer")
            confidence = parsed.get("confidence", 0.5)
            reasoning = parsed.get("reasoning", "")
            
            # Normalize confidence to float
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0.5
            
            # Handle null/uncertain answers
            if answer is None or answer == "null" or answer.upper() == "UNCERTAIN":
                return SLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning=reasoning or "SLM could not determine",
                    latency=latency,
                    metadata={"raw_response": response[:200]}
                )
            
            return SLMJudgment(
                resolved_value=answer,
                confidence=min(confidence, 1.0),
                status="RESOLVED",
                reasoning=reasoning,
                latency=latency,
                metadata={"parse_method": "json"}
            )
        
        # Fallback: Try to find answer in response text
        return self._fallback_parse(response, field_name, value1, value2, latency)
    
    def _fallback_parse(
        self,
        response: str,
        field_name: str,
        value1: str,
        value2: str,
        latency: float
    ) -> SLMJudgment:
        """
        Fallback parsing when JSON extraction fails.
        
        Looks for patterns like 'the correct value is...' or 
        checks if one of the original values appears in the response.
        """
        response_lower = response.lower()
        
        # Check if one of the values is mentioned
        v1_in = value1.lower() in response_lower if value1 else False
        v2_in = value2.lower() in response_lower if value2 else False
        
        if v1_in and not v2_in:
            return SLMJudgment(
                resolved_value=value1,
                confidence=0.6,  # Lower confidence for fallback
                status="RESOLVED",
                reasoning="Fallback: Value found in response",
                latency=latency,
                metadata={"parse_method": "fallback_text"}
            )
        elif v2_in and not v1_in:
            return SLMJudgment(
                resolved_value=value2,
                confidence=0.6,
                status="RESOLVED",
                reasoning="Fallback: Value found in response",
                latency=latency,
                metadata={"parse_method": "fallback_text"}
            )
        
        # Could not determine
        return SLMJudgment(
            resolved_value=None,
            confidence=0.0,
            status="UNCERTAIN",
            reasoning="Could not parse model response (JSON extraction failed)",
            latency=latency,
            metadata={"raw_response": response[:300], "parse_method": "failed"}
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
    
    def extract_all_fields(
        self,
        context: str,
        fields: List[str] = None
    ) -> Dict[str, SLMJudgment]:
        """
        Extract ALL fields from document context using SLM.
        
        This is called when SLM is invoked for any field - it extracts
        all fields at once, ignoring previous tier results.
        
        Args:
            context: Full document OCR text
            fields: List of field names to extract (default: all 6 fields)
            
        Returns:
            Dict mapping field_name to SLMJudgment
        """
        start_time = time.time()
        
        if fields is None:
            fields = ["dealer_name", "model_name", "horse_power", "asset_cost", "signature", "stamp"]
        
        self._lazy_init()
        
        results = {}
        
        if self.model is None:
            # Model not available, return uncertain for all fields
            latency = time.time() - start_time
            for field_name in fields:
                results[field_name] = SLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning="SLM model not available",
                    latency=latency
                )
            return results
        
        # Comprehensive extraction prompt
        extraction_prompt = f"""You are an expert document AI assistant specializing in tractor/vehicle invoice extraction.

Extract ALL the following fields from the invoice document text below.
For each field, provide the extracted value or "NOT_FOUND" if not present.

DOCUMENT TEXT:
{context[:1500]}

FIELDS TO EXTRACT:
1. dealer_name - The name of the dealer/seller company
2. model_name - The tractor/vehicle model name (brand + model)
3. horse_power - The HP (horse power) value as a number
4. asset_cost - The total price/cost as a number (in rupees)
5. signature - Is there mention of a signature? (YES/NO)
6. stamp - Is there mention of a stamp/seal? (YES/NO)

Respond in this EXACT format:
DEALER_NAME: [value or NOT_FOUND]
MODEL_NAME: [value or NOT_FOUND]
HORSE_POWER: [value or NOT_FOUND]
ASSET_COST: [value or NOT_FOUND]
SIGNATURE: [YES/NO/NOT_FOUND]
STAMP: [YES/NO/NOT_FOUND]
CONFIDENCE: [high/medium/low]"""

        try:
            inputs = self.tokenizer(extraction_prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency = time.time() - start_time
            
            # Parse the response for each field
            results = self._parse_all_fields_response(response, fields, latency)
            
        except Exception as e:
            latency = time.time() - start_time
            for field_name in fields:
                results[field_name] = SLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="ERROR",
                    reasoning=f"SLM extraction error: {str(e)}",
                    latency=latency
                )
        
        return results
    
    def _parse_all_fields_response(
        self,
        response: str,
        fields: List[str],
        latency: float
    ) -> Dict[str, SLMJudgment]:
        """Parse the all-fields extraction response."""
        results = {}
        
        # Extract confidence level
        conf_match = re.search(r'CONFIDENCE:\s*(\w+)', response, re.IGNORECASE)
        confidence_str = conf_match.group(1).lower() if conf_match else "medium"
        confidence_map = {"high": 0.85, "medium": 0.70, "low": 0.55}
        base_confidence = confidence_map.get(confidence_str, 0.70)
        
        # Field name mapping
        field_patterns = {
            "dealer_name": r'DEALER_NAME:\s*(.+?)(?:\n|$)',
            "model_name": r'MODEL_NAME:\s*(.+?)(?:\n|$)',
            "horse_power": r'HORSE_POWER:\s*(.+?)(?:\n|$)',
            "asset_cost": r'ASSET_COST:\s*(.+?)(?:\n|$)',
            "signature": r'SIGNATURE:\s*(.+?)(?:\n|$)',
            "stamp": r'STAMP:\s*(.+?)(?:\n|$)'
        }
        
        for field_name in fields:
            pattern = field_patterns.get(field_name)
            if not pattern:
                results[field_name] = SLMJudgment(
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
                    results[field_name] = SLMJudgment(
                        resolved_value=None,
                        confidence=0.0,
                        status="RESOLVED",
                        reasoning="SLM could not find field in document",
                        latency=latency
                    )
                else:
                    # Process field-specific values
                    if field_name in ["signature", "stamp"]:
                        is_present = "YES" in value.upper()
                        results[field_name] = SLMJudgment(
                            resolved_value={"present": is_present, "bbox": None, "verified": False},
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="SLM document-level extraction",
                            latency=latency
                        )
                    elif field_name == "horse_power":
                        # Extract numeric HP value
                        hp_match = re.search(r'(\d+)', value)
                        hp_value = float(hp_match.group(1)) if hp_match else None
                        results[field_name] = SLMJudgment(
                            resolved_value=hp_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="SLM document-level extraction",
                            latency=latency
                        )
                    elif field_name == "asset_cost":
                        # Extract numeric cost value
                        cost_clean = re.sub(r'[^\d.]', '', value)
                        cost_value = float(cost_clean) if cost_clean else None
                        results[field_name] = SLMJudgment(
                            resolved_value=cost_value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="SLM document-level extraction",
                            latency=latency
                        )
                    else:
                        results[field_name] = SLMJudgment(
                            resolved_value=value,
                            confidence=base_confidence,
                            status="RESOLVED",
                            reasoning="SLM document-level extraction",
                            latency=latency
                        )
            else:
                results[field_name] = SLMJudgment(
                    resolved_value=None,
                    confidence=0.0,
                    status="UNCERTAIN",
                    reasoning="Field not found in SLM response",
                    latency=latency
                )
        
        return results


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
