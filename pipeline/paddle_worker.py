
import os
import sys
import json
import logging
import warnings

# Suppress warnings and logs
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_executor"] = "0"
os.environ["FLAGS_use_mkl"] = "0"
os.environ["MKLDNN_DISABLE"] = "1"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

# Configure logging to stderr so stdout remains clean for JSON
logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

def run_paddle_worker(image_path):
    try:
        from paddleocr import PaddleOCR
        import torch
    except ImportError:
        print(json.dumps({"error": "PaddleOCR not installed", "results": []}))
        return

    try:
        # Determine device
        has_gpu = torch.cuda.is_available()
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            use_textline_orientation=True,
            lang='en',
            use_gpu=has_gpu,
            show_log=False,
            enable_mkldnn=False
        )
        
        # Run inference
        ocr_results = ocr.ocr(str(image_path), cls=True)
        
        results = []
        if ocr_results and ocr_results[0]:
            for line in ocr_results[0]:
                if line and len(line) >= 2:
                    # line format: [[[x1,y1],[x2,y2],...], ("text", conf)]
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    results.append({
                        "text": text,
                        "bbox_points": bbox_points,
                        "confidence": confidence
                    })
        
        # Output JSON to stdout
        print(json.dumps({"success": True, "results": results}))
        
    except Exception as e:
        # Output error to stdout as JSON
        print(json.dumps({"success": False, "error": str(e), "results": []}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
        
    image_path = sys.argv[1]
    run_paddle_worker(image_path)
