"""
Field Parser Module

RapidFuzz-based matching for dealer names, numeric normalization for
asset_cost and horse_power, and model name extraction.

Handles:
- Devanagari and Gujarati digit conversion
- Currency symbol stripping (₹, $, Rs.)
- Numeric range handling (450000-480000 → median)
- Company suffix stripping for better matching
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz, process


# ============================================================================
# Digit Mappings
# ============================================================================

DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

GUJARATI_DIGITS = {
    '૦': '0', '૧': '1', '૨': '2', '૩': '3', '૪': '4',
    '૫': '5', '૬': '6', '૭': '7', '૮': '8', '૯': '9'
}

# Combined native digit mapping
NATIVE_DIGITS = {**DEVANAGARI_DIGITS, **GUJARATI_DIGITS}

# Company suffixes to strip
COMPANY_SUFFIXES = [
    r'\bpvt\.?\b', r'\bprivate\b', r'\bltd\.?\b', r'\blimited\b',
    r'\btraders\b', r'\bmotors\b', r'\btractors\b', r'\bagencies\b',
    r'\benterprises\b', r'\bco\.?\b', r'\binc\.?\b', r'\bcorp\.?\b',
    r'\bllc\b', r'\bllp\b', r'\b&\b', r'\band\b'
]

# Currency symbols and patterns
CURRENCY_PATTERNS = [
    r'₹\s*', r'\$\s*', r'Rs\.?\s*', r'INR\s*', r'USD\s*',
    r'rupees?\s*', r'/-'
]

# Tractor brand patterns
TRACTOR_BRANDS = [
    'mahindra', 'john deere', 'eicher', 'massey ferguson', 'swaraj',
    'sonalika', 'tafe', 'farmtrac', 'new holland', 'kubota', 'escorts',
    'preet', 'force', 'captain', 'vst', 'indo farm', 'ace', 'powertrac'
]


@dataclass
class ParseResult:
    """Result from a field parser."""
    raw_value: str
    parsed_value: Any
    confidence: float
    matched: bool = False
    match_score: float = 0.0
    matched_to: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "raw_value": self.raw_value,
            "parsed_value": self.parsed_value,
            "confidence": self.confidence,
            "matched": self.matched,
            "match_score": self.match_score,
            "matched_to": self.matched_to,
            "metadata": self.metadata
        }


def normalize_native_digits(text: str) -> str:
    """Convert Devanagari and Gujarati digits to Arabic numerals."""
    for native, arabic in NATIVE_DIGITS.items():
        text = text.replace(native, arabic)
    return text


def strip_currency(text: str) -> str:
    """Remove currency symbols and patterns from text."""
    result = text
    for pattern in CURRENCY_PATTERNS:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    return result.strip()


def strip_company_suffixes(text: str) -> str:
    """Remove common company suffixes from dealer names."""
    result = text.lower()
    for pattern in COMPANY_SUFFIXES:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def normalize_numeric(text: str) -> Tuple[Optional[float], float]:
    """
    Normalize a numeric string to a float value.
    
    Handles:
    - Native digits (Devanagari, Gujarati)
    - Commas as thousand separators
    - Currency symbols
    - Ranges (450000-480000 → median)
    - Lakhs notation (4.5 lakh → 450000)
    
    Args:
        text: Raw text containing a number
        
    Returns:
        Tuple of (parsed_value, confidence)
    """
    if not text:
        return None, 0.0
    
    # Normalize native digits
    text = normalize_native_digits(text)
    
    # Strip currency
    text = strip_currency(text)
    
    # Remove commas (thousand separators)
    text = text.replace(',', '')
    
    # Handle lakhs notation
    lakh_match = re.search(r'(\d+\.?\d*)\s*(?:lakh|lac|l)', text, re.IGNORECASE)
    if lakh_match:
        value = float(lakh_match.group(1)) * 100000
        return value, 0.9
    
    # Handle crore notation
    crore_match = re.search(r'(\d+\.?\d*)\s*(?:crore|cr)', text, re.IGNORECASE)
    if crore_match:
        value = float(crore_match.group(1)) * 10000000
        return value, 0.9
    
    # Handle ranges (e.g., 450000-480000)
    range_match = re.search(r'(\d+\.?\d*)\s*[-–—to]+\s*(\d+\.?\d*)', text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        median = (low + high) / 2
        return median, 0.85  # Lower confidence for ranges
    
    # Extract plain number
    number_match = re.search(r'(\d+\.?\d*)', text)
    if number_match:
        value = float(number_match.group(1))
        return value, 0.95
    
    return None, 0.0


def parse_dealer_name(
    text: str,
    master_list: List[str],
    threshold: float = 90.0
) -> ParseResult:
    """
    Parse and match dealer name against master list.
    
    Uses RapidFuzz token_set_ratio for fuzzy matching.
    Strips company suffixes before matching.
    
    Args:
        text: Raw extracted text
        master_list: List of known dealer names
        threshold: Minimum match score (0-100)
        
    Returns:
        ParseResult with match information
    """
    if not text or not master_list:
        return ParseResult(
            raw_value=text or "",
            parsed_value=None,
            confidence=0.0,
            matched=False
        )
    
    # Normalize the input
    normalized = strip_company_suffixes(text)
    
    # Normalize master list for comparison
    normalized_master = [strip_company_suffixes(name) for name in master_list]
    
    # Create mapping back to original names
    master_map = {strip_company_suffixes(name): name for name in master_list}
    
    # Find best match using token_set_ratio
    result = process.extractOne(
        normalized,
        normalized_master,
        scorer=fuzz.token_set_ratio
    )
    
    if result is None:
        return ParseResult(
            raw_value=text,
            parsed_value=text,
            confidence=0.3,
            matched=False
        )
    
    match_text, score, _ = result
    
    if score >= threshold:
        original_name = master_map.get(match_text, match_text)
        return ParseResult(
            raw_value=text,
            parsed_value=original_name,
            confidence=score / 100.0,
            matched=True,
            match_score=score,
            matched_to=original_name
        )
    else:
        return ParseResult(
            raw_value=text,
            parsed_value=text,
            confidence=score / 100.0 * 0.5,  # Lower confidence for unmatched
            matched=False,
            match_score=score
        )


def parse_model_name(text: str) -> ParseResult:
    """
    Parse tractor model name from text.
    
    Looks for patterns like:
    - Brand + number (e.g., "Mahindra 575 DI")
    - HP model patterns (e.g., "45 HP")
    
    Args:
        text: Raw extracted text
        
    Returns:
        ParseResult with model information
    """
    if not text:
        return ParseResult(
            raw_value="",
            parsed_value=None,
            confidence=0.0
        )
    
    text_lower = text.lower()
    confidence = 0.5
    model_name = text.strip()
    brand_found = None
    
    # Look for known tractor brands
    for brand in TRACTOR_BRANDS:
        if brand in text_lower:
            brand_found = brand.title()
            confidence = 0.8
            
            # Try to extract model number after brand
            pattern = rf'{brand}\s*([A-Z0-9\-\s]+(?:DI|HP|XT|XL|TURBO)?)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model_name = f"{brand_found} {match.group(1).strip()}"
                confidence = 0.9
            break
    
    # Extract HP/BHP if present
    hp_match = re.search(r'(\d+)\s*(?:HP|BHP|H\.P\.)', text, re.IGNORECASE)
    if hp_match:
        hp_value = hp_match.group(1)
        if brand_found and hp_value not in model_name:
            model_name = f"{model_name} {hp_value}HP"
        confidence = max(confidence, 0.85)
    
    return ParseResult(
        raw_value=text,
        parsed_value=model_name,
        confidence=confidence,
        metadata={"brand": brand_found}
    )


def parse_horse_power(text: str) -> ParseResult:
    """
    Parse horse power value from text.
    
    Handles patterns like:
    - "45 HP", "45HP", "45 H.P."
    - "45 BHP"
    - Ranges: "40-50 HP" (returns median)
    
    Args:
        text: Raw extracted text
        
    Returns:
        ParseResult with HP value
    """
    if not text:
        return ParseResult(
            raw_value="",
            parsed_value=None,
            confidence=0.0
        )
    
    # Normalize native digits
    text = normalize_native_digits(text)
    
    # Look for HP range
    range_match = re.search(
        r'(\d+)\s*[-–—to]+\s*(\d+)\s*(?:HP|BHP|H\.?P\.?)', 
        text, 
        re.IGNORECASE
    )
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        median = (low + high) / 2
        return ParseResult(
            raw_value=text,
            parsed_value=median,
            confidence=0.85,
            metadata={"range": [low, high]}
        )
    
    # Look for single HP value
    hp_match = re.search(r'(\d+)\s*(?:HP|BHP|H\.?P\.?)', text, re.IGNORECASE)
    if hp_match:
        hp_value = int(hp_match.group(1))
        return ParseResult(
            raw_value=text,
            parsed_value=hp_value,
            confidence=0.95
        )
    
    # Try to find any number as fallback
    num_match = re.search(r'(\d+)', text)
    if num_match:
        value = int(num_match.group(1))
        # Only accept reasonable HP values (10-200)
        if 10 <= value <= 200:
            return ParseResult(
                raw_value=text,
                parsed_value=value,
                confidence=0.5,  # Low confidence without HP suffix
                metadata={"extracted_without_suffix": True}
            )
    
    return ParseResult(
        raw_value=text,
        parsed_value=None,
        confidence=0.0
    )


def parse_asset_cost(text: str) -> ParseResult:
    """
    Parse asset cost (price) from text.
    
    Handles:
    - Currency symbols (₹, $, Rs.)
    - Comma separators
    - Lakhs/Crores notation
    - Ranges (returns median)
    - Native digits
    
    Args:
        text: Raw extracted text
        
    Returns:
        ParseResult with cost value
    """
    if not text:
        return ParseResult(
            raw_value="",
            parsed_value=None,
            confidence=0.0
        )
    
    value, confidence = normalize_numeric(text)
    
    if value is not None:
        # Validate reasonable tractor price range (50,000 - 50,00,000 INR)
        if 50000 <= value <= 5000000:
            return ParseResult(
                raw_value=text,
                parsed_value=value,
                confidence=confidence
            )
        elif value > 0:
            # Accept but with lower confidence
            return ParseResult(
                raw_value=text,
                parsed_value=value,
                confidence=confidence * 0.7,
                metadata={"out_of_typical_range": True}
            )
    
    return ParseResult(
        raw_value=text,
        parsed_value=None,
        confidence=0.0
    )


def extract_field_from_text(
    full_text: str,
    field_type: str,
    master_list: List[str] = None
) -> ParseResult:
    """
    Extract a specific field from full document text.
    
    Uses heuristics and patterns to locate field values.
    
    Args:
        full_text: Full OCR text from document
        field_type: Type of field to extract
        master_list: Master list for dealer matching
        
    Returns:
        ParseResult with extracted value
    """
    if field_type == "dealer_name":
        # Look for dealer-related keywords
        patterns = [
            r'(?:dealer|seller|vendor|from|supplier)[:\s]+([^\n]+)',
            r'(?:name of dealer|dealer name)[:\s]+([^\n]+)',
            r'(?:sold by|authorized dealer)[:\s]+([^\n]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                if master_list:
                    return parse_dealer_name(raw, master_list)
                return ParseResult(
                    raw_value=raw,
                    parsed_value=raw,
                    confidence=0.7
                )
    
    elif field_type == "model_name":
        # Look for model-related keywords
        patterns = [
            r'(?:model|tractor model|vehicle model)[:\s]+([^\n]+)',
            r'(?:description|product)[:\s]+([^\n]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return parse_model_name(match.group(1).strip())
        
        # Fallback: look for brand patterns anywhere
        return parse_model_name(full_text)
    
    elif field_type == "horse_power":
        # Look for HP anywhere in text
        return parse_horse_power(full_text)
    
    elif field_type == "asset_cost":
        # Look for price patterns
        patterns = [
            r'(?:total|amount|price|cost|value)[:\s]*([₹$]?\s*[\d,\.]+(?:\s*(?:lakh|lac|crore|cr))?)',
            r'(?:invoice value|asset cost|asset value)[:\s]*([₹$]?\s*[\d,\.]+)',
            r'(?:grand total|net amount)[:\s]*([₹$]?\s*[\d,\.]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return parse_asset_cost(match.group(1).strip())
    
    return ParseResult(
        raw_value="",
        parsed_value=None,
        confidence=0.0
    )


def compute_token_overlap(text1: str, text2: str) -> float:
    """
    Compute token overlap ratio between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap ratio (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


def compute_fuzzy_score(text1: str, text2: str) -> float:
    """
    Compute fuzzy match score between two texts.
    
    Uses token_set_ratio for best partial matching.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Match score (0-100)
    """
    if not text1 or not text2:
        return 0.0
    
    return fuzz.token_set_ratio(text1.lower(), text2.lower())


def compute_numeric_diff(value1: float, value2: float) -> float:
    """
    Compute relative difference between two numeric values.
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Relative difference (0-1+)
    """
    if value1 == 0 and value2 == 0:
        return 0.0
    
    avg = (abs(value1) + abs(value2)) / 2
    if avg == 0:
        return 1.0
    
    return abs(value1 - value2) / avg
