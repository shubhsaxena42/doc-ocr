# Ablation Study Results

## Overview

This document records ablation experiments measuring the contribution of each pipeline component to overall accuracy.

## Experimental Setup

- **Dataset**: Golden set with 50+ manually verified documents
- **Metrics**: Document-Level Accuracy (DLA), Field mAP, Cost, Latency
- **Baseline**: OCR-only extraction without adjudication

---

## Results

### Configuration Comparison

| Configuration | DLA | Cost/Doc | Latency/Doc | Notes |
|--------------|-----|----------|-------------|-------|
| 1. OCR-only baseline | _TBD_ | $0.00 | ~5s | No adjudication |
| 2. + Tier 1 rules | _TBD_ | $0.00 | ~5s | + Normalization, Jaccard |
| 3. + Tier 2 SLM | _TBD_ | ~$0.001 | ~10s | + Qwen2.5-1.5B |
| 4. + Tier 3 VLM | _TBD_ | ~$0.005 | ~15s | + Qwen2.5-VL-7B |
| 5. + Calibration | _TBD_ | ~$0.005 | ~15s | + Isotonic regression |

### Field-Level Breakdown

| Field | OCR-only | +T1 Rules | +T2 SLM | +T3 VLM | +Calibration |
|-------|----------|-----------|---------|---------|--------------|
| dealer_name | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| model_name | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| horse_power | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| asset_cost | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| signature | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| stamp | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## Key Findings

### 1. OCR-only Baseline
- **Observation**: _To be filled after experiments_
- **Bottleneck**: _Identify main error sources_

### 2. Tier 1 Rules Impact
- **Improvement**: _Expected 5-10% DLA gain_
- **Key contributions**:
  - Native digit normalization (Devanagari/Gujarati)
  - Currency symbol stripping
  - Range median calculation

### 3. Tier 2 SLM Impact
- **Improvement**: _Expected 3-5% DLA gain_
- **Conflict resolution rate**: _% of conflicts resolved_
- **Escalation rate**: _% requiring Tier 3_

### 4. Tier 3 VLM Impact
- **Improvement**: _Expected 1-3% DLA gain_
- **Invocation rate**: Must stay <10%
- **Best for**: Handwriting, overlapping elements

### 5. Calibration Impact
- **Confidence accuracy**: Before vs after
- **ECE improvement**: _Expected Calibration Error reduction_

---

## Error Analysis

### Most Common Error Categories

1. **OCR_Mismatch**: _%_
2. **SLM_Uncertain**: _%_
3. **VLM_Error**: _%_
4. **Handwriting_Noise**: _%_
5. **Stamp_Overlap**: _%_

### Per-Language Accuracy

| Language | DLA | Sample Size |
|----------|-----|-------------|
| English | _TBD_ | _TBD_ |
| Hindi | _TBD_ | _TBD_ |
| Gujarati | _TBD_ | _TBD_ |
| Mixed | _TBD_ | _TBD_ |

---

## Recommendations

1. _To be filled based on findings_
2. _Areas for improvement_
3. _Cost optimization opportunities_

---

## How to Run Ablations

```bash
# 1. OCR-only (no adjudication)
python executable.py --input train --output results_ocr_only.json --mode cpu-lite

# 2-5. Full pipeline variations require code modifications
# See config.py for threshold adjustments

# Evaluate each configuration
python evaluate.py --predictions results_ocr_only.json --golden data/golden_set.json
```

---

*Last updated: [DATE]*
