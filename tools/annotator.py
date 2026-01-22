"""
Streamlit Annotator Tool

Manual annotation UI for creating the golden set.
Samples documents based on:
- 10 highest OCR disagreement
- 10 borderline confidence (0.6-0.95)
- 10 rarest layout clusters

Usage:
    streamlit run tools/annotator.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Check if streamlit is available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Run: pip install streamlit")


def load_results(results_path: str) -> dict:
    """Load extraction results."""
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {"results": []}


def load_golden_set(golden_path: str) -> dict:
    """Load existing golden set."""
    if Path(golden_path).exists():
        with open(golden_path, 'r') as f:
            return json.load(f)
    return {
        "_schema_version": "1.0",
        "_description": "Golden set for IDAI pipeline calibration",
        "documents": []
    }


def save_golden_set(golden_set: dict, golden_path: str):
    """Save golden set to file."""
    with open(golden_path, 'w') as f:
        json.dump(golden_set, f, indent=2)


def get_annotated_ids(golden_set: dict) -> set:
    """Get set of already annotated document IDs."""
    return {doc["doc_id"] for doc in golden_set.get("documents", [])}


def sample_documents(results: list, n_samples: int = 30) -> list:
    """
    Sample documents for annotation based on:
    - 10 highest OCR disagreement
    - 10 borderline confidence (0.6-0.95)
    - 10 random for diversity
    """
    # Sort by confidence to find borderline cases
    sorted_by_conf = sorted(
        results,
        key=lambda x: x.get("document_confidence", 0)
    )
    
    # Find borderline (0.6-0.95)
    borderline = [
        r for r in results
        if 0.6 <= r.get("document_confidence", 0) <= 0.95
    ][:10]
    
    # Find low confidence (potential disagreement)
    low_conf = sorted_by_conf[:10]
    
    # Random sample for diversity
    import random
    remaining = [r for r in results if r not in borderline and r not in low_conf]
    random_sample = random.sample(remaining, min(10, len(remaining)))
    
    # Combine and deduplicate
    sampled = []
    seen_ids = set()
    for doc in borderline + low_conf + random_sample:
        doc_id = doc.get("doc_id", "")
        if doc_id not in seen_ids:
            sampled.append(doc)
            seen_ids.add(doc_id)
    
    return sampled[:n_samples]


def run_annotator():
    """Run the Streamlit annotator app."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is required to run the annotator")
        return
    
    st.set_page_config(
        page_title="IDAI Golden Set Annotator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 IDAI Golden Set Annotator")
    st.markdown("Annotate documents to build the golden validation set for calibration.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value="data",
        help="Directory containing images and golden_set.json"
    )
    
    results_path = st.sidebar.text_input(
        "Results File",
        value="results.json",
        help="Path to extraction results for pre-filling"
    )
    
    golden_path = os.path.join(data_dir, "golden_set.json")
    
    # Load data
    results_data = load_results(results_path)
    golden_set = load_golden_set(golden_path)
    annotated_ids = get_annotated_ids(golden_set)
    
    # Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Statistics")
    st.sidebar.metric("Golden Set Size", len(golden_set.get("documents", [])))
    st.sidebar.metric("Target Minimum", 50)
    
    progress = len(golden_set.get("documents", [])) / 50
    st.sidebar.progress(min(progress, 1.0))
    
    if progress >= 1.0:
        st.sidebar.success("✅ Minimum reached!")
    else:
        st.sidebar.info(f"Need {50 - len(golden_set.get('documents', []))} more documents")
    
    # Main content
    results = results_data.get("results", [])
    
    # Filter unannotated
    unannotated = [r for r in results if r.get("doc_id") not in annotated_ids]
    
    if not unannotated:
        st.warning("No unannotated documents found. Upload results first.")
        
        # Allow manual image selection
        st.markdown("### Or annotate images directly:")
        image_dir = st.text_input("Image Directory", value="train")
        
        if Path(image_dir).exists():
            images = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
            if images:
                selected_img = st.selectbox(
                    "Select Image",
                    options=[str(p) for p in images[:100]]
                )
                if selected_img:
                    st.image(selected_img, use_column_width=True)
                    
                    # Annotation form
                    with st.form("manual_annotation"):
                        doc_id = Path(selected_img).stem
                        
                        dealer_name = st.text_input("Dealer Name")
                        model_name = st.text_input("Model Name")
                        horse_power = st.number_input("Horse Power", min_value=0, max_value=500)
                        asset_cost = st.number_input("Asset Cost", min_value=0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            signature_present = st.checkbox("Signature Present")
                        with col2:
                            stamp_present = st.checkbox("Stamp Present")
                        
                        submitted = st.form_submit_button("Save Annotation")
                        
                        if submitted:
                            annotation = {
                                "doc_id": doc_id,
                                "image_path": selected_img,
                                "ground_truth": {
                                    "dealer_name": dealer_name if dealer_name else None,
                                    "model_name": model_name if model_name else None,
                                    "horse_power": horse_power if horse_power > 0 else None,
                                    "asset_cost": asset_cost if asset_cost > 0 else None,
                                    "signature": {"present": signature_present},
                                    "stamp": {"present": stamp_present}
                                },
                                "layout_cluster": 0,
                                "annotated_at": datetime.now().isoformat()
                            }
                            
                            golden_set["documents"].append(annotation)
                            save_golden_set(golden_set, golden_path)
                            st.success(f"Saved annotation for {doc_id}")
                            st.experimental_rerun()
        return
    
    # Sample documents
    sampled = sample_documents(unannotated)
    
    st.markdown(f"### Documents to Annotate ({len(sampled)} sampled)")
    
    # Document selector
    doc_options = [f"{d['doc_id']} (conf: {d.get('document_confidence', 0):.2f})" for d in sampled]
    selected_idx = st.selectbox("Select Document", range(len(doc_options)), format_func=lambda x: doc_options[x])
    
    selected_doc = sampled[selected_idx]
    
    # Display image
    image_path = selected_doc.get("image_path", "")
    if Path(image_path).exists():
        st.image(image_path, use_column_width=True)
    else:
        st.warning(f"Image not found: {image_path}")
    
    # Show pre-extracted values
    st.markdown("### Pre-extracted Values (from pipeline)")
    fields = selected_doc.get("fields", {})
    
    col1, col2 = st.columns(2)
    with col1:
        for field in ["dealer_name", "model_name", "horse_power"]:
            if field in fields:
                st.text(f"{field}: {fields[field].get('value', 'N/A')}")
    with col2:
        for field in ["asset_cost", "signature", "stamp"]:
            if field in fields:
                st.text(f"{field}: {fields[field].get('value', 'N/A')}")
    
    # Annotation form
    st.markdown("### Correct Values (Ground Truth)")
    
    with st.form("annotation_form"):
        dealer_name = st.text_input(
            "Dealer Name",
            value=str(fields.get("dealer_name", {}).get("value", "") or "")
        )
        
        model_name = st.text_input(
            "Model Name",
            value=str(fields.get("model_name", {}).get("value", "") or "")
        )
        
        hp_default = fields.get("horse_power", {}).get("value", 0)
        horse_power = st.number_input(
            "Horse Power",
            min_value=0,
            max_value=500,
            value=int(hp_default) if hp_default else 0
        )
        
        cost_default = fields.get("asset_cost", {}).get("value", 0)
        asset_cost = st.number_input(
            "Asset Cost (₹)",
            min_value=0,
            value=int(cost_default) if cost_default else 0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            sig_default = fields.get("signature", {}).get("value", {})
            signature_present = st.checkbox(
                "Signature Present",
                value=sig_default.get("present", False) if isinstance(sig_default, dict) else bool(sig_default)
            )
        with col2:
            stamp_default = fields.get("stamp", {}).get("value", {})
            stamp_present = st.checkbox(
                "Stamp Present",
                value=stamp_default.get("present", False) if isinstance(stamp_default, dict) else bool(stamp_default)
            )
        
        layout_cluster = st.number_input("Layout Cluster", min_value=0, max_value=10, value=0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            submitted = st.form_submit_button("💾 Save Annotation", type="primary")
        with col2:
            skip = st.form_submit_button("⏭️ Skip Document")
        with col3:
            unclear = st.form_submit_button("❓ Mark Unclear")
        
        if submitted:
            annotation = {
                "doc_id": selected_doc["doc_id"],
                "image_path": image_path,
                "ground_truth": {
                    "dealer_name": dealer_name if dealer_name else None,
                    "model_name": model_name if model_name else None,
                    "horse_power": horse_power if horse_power > 0 else None,
                    "asset_cost": asset_cost if asset_cost > 0 else None,
                    "signature": {"present": signature_present, "bbox": None},
                    "stamp": {"present": stamp_present, "bbox": None}
                },
                "layout_cluster": layout_cluster,
                "annotated_at": datetime.now().isoformat()
            }
            
            golden_set["documents"].append(annotation)
            save_golden_set(golden_set, golden_path)
            st.success(f"✅ Saved annotation for {selected_doc['doc_id']}")
            st.experimental_rerun()
        
        if unclear:
            annotation = {
                "doc_id": selected_doc["doc_id"],
                "image_path": image_path,
                "ground_truth": None,
                "layout_cluster": layout_cluster,
                "annotated_at": datetime.now().isoformat(),
                "status": "unclear"
            }
            golden_set["documents"].append(annotation)
            save_golden_set(golden_set, golden_path)
            st.warning(f"⚠️ Marked {selected_doc['doc_id']} as unclear")
            st.experimental_rerun()


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_annotator()
    else:
        print("To run the annotator, install streamlit: pip install streamlit")
        print("Then run: streamlit run tools/annotator.py")
