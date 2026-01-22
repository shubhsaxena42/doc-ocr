#!/usr/bin/env python3
"""
IDAI Pipeline - Tractor Loan Invoice Extraction

Main executable entry point for the Intelligent Document AI pipeline.
Extracts 6 key fields from tractor loan invoices with ≥95% DLA target.

Usage:
    python executable.py --input /path/to/images --output results.json --mode full
    python executable.py --input /path/to/images --output results.json --mode cpu-lite
    python executable.py --input /path/to/image.png --output results.json --golden-set data/golden_set.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Set environment variables before imports
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_executor'] = '0'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IDAI Pipeline - Tractor Loan Invoice Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a directory of images
    python executable.py --input ./train --output results.json --mode full
    
    # Process a single image in cpu-lite mode (no VLM)
    python executable.py --input invoice.png --output results.json --mode cpu-lite
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input image file or directory containing images'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='results.json',
        help='Path to output JSON file (default: results.json)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'cpu-lite'],
        default='full',
        help='Pipeline mode: "full" includes Tier 3 VLM, "cpu-lite" skips VLM (default: full)'
    )
    

    
    parser.add_argument(
        '--dealer-list', '-d',
        default=None,
        help='Path to JSON file with dealer master list'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    parser.add_argument(
        '--cost-report',
        default=None,
        help='Path to save detailed cost report'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Import pipeline components
    try:
        from pipeline.main_pipeline import create_processor, DocumentProcessor
        from pipeline.cost_tracker import get_tracker, reset_tracker
    except ImportError as e:
        print(f"Error importing pipeline modules: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Reset tracker for fresh run
    reset_tracker()
    
    # Create processor
    if args.verbose:
        print(f"Initializing pipeline in '{args.mode}' mode...")
    
    processor = create_processor(
        mode=args.mode,
        golden_set_path=None,  # No golden set calibration
        dealer_list_path=args.dealer_list
    )
    
    # Process input
    results = []
    start_time = datetime.now()
    
    if input_path.is_file():
        # Single file
        if args.verbose:
            print(f"Processing: {input_path}")
        result = processor.process_document(str(input_path))
        results.append(result)
    else:
        # Directory
        extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in '{args.input}'")
            sys.exit(1)
        
        if args.verbose:
            print(f"Found {len(image_files)} images to process")
        
        results = processor.process_batch(
            [str(f) for f in image_files],
            show_progress=not args.no_progress
        )
    
    end_time = datetime.now()
    
    # Get summary from tracker
    tracker = get_tracker()
    summary = tracker.get_summary()
    
    # Prepare output - simplified format
    # For single document, output just the document result
    # For multiple documents, output a list of results
    if len(results) == 1:
        output_data = results[0].to_dict()
    else:
        output_data = [r.to_dict() for r in results]
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if args.verbose:
        print(f"\nResults saved to: {output_path}")
    
    # Save cost report if requested
    if args.cost_report:
        cost_path = Path(args.cost_report)
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        tracker.to_json(str(cost_path))
        if args.verbose:
            print(f"Cost report saved to: {cost_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    # Calculate success/failure from results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"Documents processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Average cost/doc: ${summary['average_cost_per_document']:.4f}")
    print(f"Average latency/doc: {summary['average_latency_per_document']:.2f}s")
    print(f"VLM invocation rate: {summary['vlm_invocation_rate']*100:.1f}%")
    print()
    
    # Check targets
    targets_met = []
    if summary["within_cost_target"]:
        targets_met.append("✓ Cost target (<$0.01/doc)")
    else:
        targets_met.append("✗ Cost target (<$0.01/doc)")
    
    if summary["within_latency_target"]:
        targets_met.append("✓ Latency target (<30s/doc)")
    else:
        targets_met.append("✗ Latency target (<30s/doc)")
    
    if summary["vlm_rate_acceptable"]:
        targets_met.append("✓ VLM rate target (<10%)")
    else:
        targets_met.append("✗ VLM rate target (<10%)")
    
    print("Target Status:")
    for target in targets_met:
        print(f"  {target}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
