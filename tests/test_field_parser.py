"""
Tests for Field Parser Module

Tests numeric normalization, fuzzy matching, and suffix stripping.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.field_parser import (
    normalize_native_digits,
    strip_currency,
    strip_company_suffixes,
    normalize_numeric,
    parse_dealer_name,
    parse_model_name,
    parse_horse_power,
    parse_asset_cost,
    compute_fuzzy_score,
    compute_token_overlap
)


class TestNativeDigits:
    """Tests for native digit conversion."""
    
    def test_devanagari_digits(self):
        """Test Devanagari to Arabic conversion."""
        assert normalize_native_digits("१२३") == "123"
        assert normalize_native_digits("४५०००") == "45000"
        assert normalize_native_digits("₹१,२३,४५६") == "₹1,23,456"
    
    def test_gujarati_digits(self):
        """Test Gujarati to Arabic conversion."""
        assert normalize_native_digits("૧૨૩") == "123"
        assert normalize_native_digits("૪૫૦૦૦") == "45000"
    
    def test_mixed_digits(self):
        """Test mixed native and Arabic digits."""
        assert normalize_native_digits("१23૪5") == "12345"
    
    def test_no_native_digits(self):
        """Test string without native digits."""
        assert normalize_native_digits("12345") == "12345"
        assert normalize_native_digits("Hello") == "Hello"


class TestCurrencyStripping:
    """Tests for currency symbol removal."""
    
    def test_rupee_symbol(self):
        """Test ₹ symbol stripping."""
        assert strip_currency("₹12,345") == "12,345"
        assert strip_currency("₹ 1,23,456") == "1,23,456"
    
    def test_rs_notation(self):
        """Test Rs. notation stripping."""
        assert strip_currency("Rs. 45000") == "45000"
        assert strip_currency("Rs 45000") == "45000"
    
    def test_dollar_symbol(self):
        """Test $ symbol stripping."""
        assert strip_currency("$1,234") == "1,234"
        assert strip_currency("$ 1234") == "1234"
    
    def test_inr_notation(self):
        """Test INR notation stripping."""
        assert strip_currency("INR 45000") == "45000"
    
    def test_trailing_slash(self):
        """Test trailing /- removal."""
        assert strip_currency("45000/-") == "45000"


class TestNumericNormalization:
    """Tests for numeric value normalization."""
    
    def test_simple_number(self):
        """Test simple number extraction."""
        value, conf = normalize_numeric("45000")
        assert value == 45000.0
        assert conf > 0.9
    
    def test_comma_separated(self):
        """Test comma as thousand separator."""
        value, conf = normalize_numeric("4,50,000")
        assert value == 450000.0
    
    def test_with_currency(self):
        """Test with currency symbol."""
        value, conf = normalize_numeric("₹4,50,000")
        assert value == 450000.0
    
    def test_lakh_notation(self):
        """Test lakh notation."""
        value, conf = normalize_numeric("4.5 lakh")
        assert value == 450000.0
        
        value, conf = normalize_numeric("4.5 lac")
        assert value == 450000.0
    
    def test_range(self):
        """Test range handling (returns median)."""
        value, conf = normalize_numeric("450000-480000")
        assert value == 465000.0  # Median
        assert conf < 0.9  # Lower confidence for ranges
    
    def test_native_digits(self):
        """Test with native digits."""
        value, conf = normalize_numeric("₹४,५०,०००")
        assert value == 450000.0


class TestCompanySuffixStripping:
    """Tests for company suffix removal."""
    
    def test_pvt_ltd(self):
        """Test Pvt. Ltd. removal."""
        assert strip_company_suffixes("ABC Motors Pvt. Ltd.") == "abc"
        assert strip_company_suffixes("XYZ Traders Pvt Ltd") == "xyz"
    
    def test_various_suffixes(self):
        """Test various company suffixes."""
        assert strip_company_suffixes("Dealers & Enterprises") == "dealers"
        assert strip_company_suffixes("Tractors Agency") == "agency"


class TestDealerNameParsing:
    """Tests for dealer name matching."""
    
    def test_exact_match(self):
        """Test exact dealer name match."""
        master_list = ["ABC Motors", "XYZ Traders", "Delta Tractors"]
        result = parse_dealer_name("ABC Motors", master_list)
        assert result.matched is True
        assert result.parsed_value == "ABC Motors"
    
    def test_fuzzy_match(self):
        """Test fuzzy dealer name match."""
        master_list = ["ABC Motors Pvt Ltd", "XYZ Traders"]
        result = parse_dealer_name("ABC Motor", master_list)
        assert result.matched is True or result.match_score > 80
    
    def test_no_match(self):
        """Test no match scenario."""
        master_list = ["ABC Motors", "XYZ Traders"]
        result = parse_dealer_name("Completely Different Name", master_list)
        assert result.matched is False


class TestHorsePowerParsing:
    """Tests for horse power extraction."""
    
    def test_hp_suffix(self):
        """Test HP suffix parsing."""
        result = parse_horse_power("45 HP")
        assert result.parsed_value == 45
        assert result.confidence > 0.9
    
    def test_bhp_suffix(self):
        """Test BHP suffix parsing."""
        result = parse_horse_power("50 BHP")
        assert result.parsed_value == 50
    
    def test_hp_range(self):
        """Test HP range (returns median)."""
        result = parse_horse_power("40-50 HP")
        assert result.parsed_value == 45.0
    
    def test_hp_without_suffix(self):
        """Test HP extraction without suffix (lower confidence)."""
        result = parse_horse_power("45")
        assert result.parsed_value == 45
        assert result.confidence < 0.8  # Lower confidence


class TestAssetCostParsing:
    """Tests for asset cost extraction."""
    
    def test_simple_cost(self):
        """Test simple cost parsing."""
        result = parse_asset_cost("450000")
        assert result.parsed_value == 450000.0
    
    def test_cost_with_currency(self):
        """Test cost with currency."""
        result = parse_asset_cost("₹4,50,000")
        assert result.parsed_value == 450000.0
    
    def test_cost_in_lakhs(self):
        """Test cost in lakhs notation."""
        result = parse_asset_cost("4.5 Lakh")
        assert result.parsed_value == 450000.0
    
    def test_out_of_range_cost(self):
        """Test out of typical range cost."""
        result = parse_asset_cost("100")  # Too low for tractor
        assert result.confidence < 0.7


class TestFuzzyMatching:
    """Tests for fuzzy matching utilities."""
    
    def test_exact_match(self):
        """Test exact string match."""
        score = compute_fuzzy_score("ABC Motors", "ABC Motors")
        assert score == 100
    
    def test_partial_match(self):
        """Test partial string match."""
        score = compute_fuzzy_score("ABC Motors", "ABC Motor")
        assert score > 80
    
    def test_no_match(self):
        """Test no match."""
        score = compute_fuzzy_score("ABC", "XYZ")
        assert score < 50


class TestTokenOverlap:
    """Tests for token overlap computation."""
    
    def test_full_overlap(self):
        """Test full token overlap."""
        overlap = compute_token_overlap("abc def ghi", "abc def ghi")
        assert overlap == 1.0
    
    def test_partial_overlap(self):
        """Test partial token overlap."""
        overlap = compute_token_overlap("abc def", "abc xyz")
        assert 0.2 < overlap < 0.6
    
    def test_no_overlap(self):
        """Test no token overlap."""
        overlap = compute_token_overlap("abc", "xyz")
        assert overlap == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
