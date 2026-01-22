"""
Synthetic Data Generator

Generates synthetic training samples for signature/stamp detection.
Creates 100 samples with:
- Random rotations (±15°)
- Brightness/contrast variations
- Occlusions
- Different stamp colors (red, blue, black)
- Different stamp shapes (circular, rectangular)

Usage:
    python tools/synthetic_generator.py --output data/synthetic --count 100
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV and Pillow required. Run: pip install opencv-python Pillow")


def create_signature(
    width: int = 200,
    height: int = 80,
    complexity: int = 5
) -> np.ndarray:
    """
    Generate a synthetic signature.
    
    Args:
        width: Image width
        height: Image height
        complexity: Number of curves in signature
        
    Returns:
        RGBA numpy array with signature
    """
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Generate signature-like curves
    points = [(0, height // 2)]
    
    for i in range(complexity):
        x = int((i + 1) * width / (complexity + 1))
        y = height // 2 + random.randint(-height // 3, height // 3)
        points.append((x, y))
    
    points.append((width, height // 2 + random.randint(-height // 4, height // 4)))
    
    # Draw smooth curves
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        # Add control points for bezier-like curve
        steps = 20
        for t in range(steps):
            t1 = t / steps
            t2 = (t + 1) / steps
            
            px1 = int(x1 + (x2 - x1) * t1)
            py1 = int(y1 + (y2 - y1) * t1 + np.sin(t1 * np.pi) * random.randint(-10, 10))
            px2 = int(x1 + (x2 - x1) * t2)
            py2 = int(y1 + (y2 - y1) * t2 + np.sin(t2 * np.pi) * random.randint(-10, 10))
            
            line_width = random.randint(1, 3)
            draw.line([(px1, py1), (px2, py2)], fill=(0, 0, 0, 255), width=line_width)
    
    # Add some flourishes
    for _ in range(random.randint(1, 3)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(5, 15)
        draw.arc([(x-r, y-r), (x+r, y+r)], 0, random.randint(90, 270), fill=(0, 0, 0, 255))
    
    return np.array(img)


def create_stamp(
    size: int = 100,
    shape: str = "circular",
    color: str = "red",
    text: str = "APPROVED"
) -> np.ndarray:
    """
    Generate a synthetic stamp.
    
    Args:
        size: Stamp size (diameter for circular)
        shape: "circular" or "rectangular"
        color: "red", "blue", or "black"
        text: Text to put on stamp
        
    Returns:
        RGBA numpy array with stamp
    """
    color_map = {
        "red": (180, 30, 30, 200),
        "blue": (30, 30, 180, 200),
        "black": (30, 30, 30, 200)
    }
    stamp_color = color_map.get(color, color_map["red"])
    
    if shape == "circular":
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Outer circle
        draw.ellipse(
            [(5, 5), (size-5, size-5)],
            outline=stamp_color,
            width=3
        )
        
        # Inner circle
        draw.ellipse(
            [(15, 15), (size-15, size-15)],
            outline=stamp_color,
            width=2
        )
        
        # Add text (simplified - would need font in production)
        # Draw text approximation with lines
        center_y = size // 2
        for i, char in enumerate(text[:8]):
            x = 20 + i * (size - 40) // 8
            draw.text((x, center_y - 5), char, fill=stamp_color)
        
    else:  # rectangular
        width = int(size * 1.5)
        height = size
        img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Border
        draw.rectangle(
            [(5, 5), (width-5, height-5)],
            outline=stamp_color,
            width=3
        )
        
        # Inner border
        draw.rectangle(
            [(10, 10), (width-10, height-10)],
            outline=stamp_color,
            width=1
        )
        
        # Text
        draw.text((width//4, height//3), text[:10], fill=stamp_color)
    
    return np.array(img)


def apply_augmentations(
    image: np.ndarray,
    rotation: float = 0,
    brightness: float = 1.0,
    contrast: float = 1.0,
    add_noise: bool = False
) -> np.ndarray:
    """
    Apply augmentations to an image.
    
    Args:
        image: Input image as numpy array
        rotation: Rotation angle in degrees
        brightness: Brightness multiplier
        contrast: Contrast multiplier
        add_noise: Whether to add noise
        
    Returns:
        Augmented image
    """
    if not CV2_AVAILABLE:
        return image
    
    result = image.copy()
    
    # Rotation
    if rotation != 0:
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        result = cv2.warpAffine(result, matrix, (w, h), borderValue=(255, 255, 255, 0))
    
    # Brightness and contrast (only for RGB channels)
    if len(result.shape) == 3 and result.shape[2] >= 3:
        rgb = result[:, :, :3].astype(np.float32)
        rgb = rgb * contrast + (brightness - 1) * 127
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        result[:, :, :3] = rgb
    
    # Noise
    if add_noise:
        noise = np.random.normal(0, 10, result[:, :, :3].shape).astype(np.int16)
        noisy = result[:, :, :3].astype(np.int16) + noise
        result[:, :, :3] = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return result


def create_occlusion(
    image: np.ndarray,
    occlusion_ratio: float = 0.2
) -> np.ndarray:
    """
    Add random occlusion to image.
    
    Args:
        image: Input image
        occlusion_ratio: Ratio of image to occlude
        
    Returns:
        Image with occlusion
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    # Random rectangle occlusion
    occ_w = int(w * occlusion_ratio)
    occ_h = int(h * occlusion_ratio)
    
    x = random.randint(0, w - occ_w)
    y = random.randint(0, h - occ_h)
    
    # Semi-transparent gray occlusion
    result[y:y+occ_h, x:x+occ_w, :3] = 200
    if result.shape[2] == 4:
        result[y:y+occ_h, x:x+occ_w, 3] = 255
    
    return result


def generate_synthetic_dataset(
    output_dir: str,
    count: int = 100,
    include_signatures: bool = True,
    include_stamps: bool = True
) -> List[Dict]:
    """
    Generate synthetic dataset for training.
    
    Args:
        output_dir: Directory to save images
        count: Number of samples to generate
        include_signatures: Whether to include signatures
        include_stamps: Whether to include stamps
        
    Returns:
        List of sample metadata dicts
    """
    if not CV2_AVAILABLE:
        print("OpenCV required for synthetic generation")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    for i in range(count):
        # Create base document (white background)
        doc_width = 800
        doc_height = 600
        document = np.ones((doc_height, doc_width, 4), dtype=np.uint8) * 255
        document[:, :, 3] = 255
        
        # Add some document texture
        for _ in range(random.randint(5, 15)):
            x = random.randint(50, doc_width - 100)
            y = random.randint(50, doc_height - 100)
            w = random.randint(100, 300)
            # Simulate text lines
            document[y:y+2, x:x+w, :3] = random.randint(100, 150)
        
        annotations = {
            "image_id": f"synthetic_{i:04d}",
            "signature": None,
            "stamp": None
        }
        
        # Add signature
        if include_signatures and random.random() > 0.3:
            sig = create_signature(
                width=random.randint(150, 250),
                height=random.randint(60, 100),
                complexity=random.randint(3, 7)
            )
            
            # Apply augmentations
            rotation = random.uniform(-15, 15)
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.9, 1.1)
            
            sig = apply_augmentations(
                sig, rotation, brightness, contrast,
                add_noise=random.random() > 0.7
            )
            
            # Occasional occlusion
            if random.random() > 0.8:
                sig = create_occlusion(sig, random.uniform(0.1, 0.3))
            
            # Place on document
            sig_x = random.randint(50, doc_width - sig.shape[1] - 50)
            sig_y = random.randint(doc_height // 2, doc_height - sig.shape[0] - 50)
            
            # Blend signature onto document
            for c in range(3):
                alpha = sig[:, :, 3] / 255.0
                h, w = sig.shape[:2]
                document[sig_y:sig_y+h, sig_x:sig_x+w, c] = (
                    document[sig_y:sig_y+h, sig_x:sig_x+w, c] * (1 - alpha) +
                    sig[:, :, c] * alpha
                ).astype(np.uint8)
            
            annotations["signature"] = {
                "present": True,
                "bbox": [sig_x, sig_y, sig.shape[1], sig.shape[0]]
            }
        
        # Add stamp
        if include_stamps and random.random() > 0.4:
            stamp_shape = random.choice(["circular", "rectangular"])
            stamp_color = random.choice(["red", "blue", "black"])
            stamp_text = random.choice(["APPROVED", "VERIFIED", "PAID", "ORIGINAL"])
            
            stamp = create_stamp(
                size=random.randint(80, 120),
                shape=stamp_shape,
                color=stamp_color,
                text=stamp_text
            )
            
            # Apply augmentations
            rotation = random.uniform(-15, 15)
            brightness = random.uniform(0.8, 1.2)
            
            stamp = apply_augmentations(stamp, rotation, brightness)
            
            # Place on document (usually bottom or corner)
            if random.random() > 0.5:
                stamp_x = random.randint(doc_width - stamp.shape[1] - 100, doc_width - stamp.shape[1] - 20)
            else:
                stamp_x = random.randint(20, 200)
            stamp_y = random.randint(doc_height - stamp.shape[0] - 150, doc_height - stamp.shape[0] - 20)
            
            # Blend stamp onto document
            for c in range(3):
                alpha = stamp[:, :, 3] / 255.0
                h, w = stamp.shape[:2]
                document[stamp_y:stamp_y+h, stamp_x:stamp_x+w, c] = (
                    document[stamp_y:stamp_y+h, stamp_x:stamp_x+w, c] * (1 - alpha) +
                    stamp[:, :, c] * alpha
                ).astype(np.uint8)
            
            annotations["stamp"] = {
                "present": True,
                "bbox": [stamp_x, stamp_y, stamp.shape[1], stamp.shape[0]],
                "color": stamp_color,
                "shape": stamp_shape
            }
        
        # Save image
        image_filename = f"synthetic_{i:04d}.png"
        image_path = output_path / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(document, cv2.COLOR_RGBA2BGR))
        
        annotations["image_path"] = str(image_path)
        samples.append(annotations)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{count} samples")
    
    # Save annotations
    annotations_path = output_path / "annotations.json"
    with open(annotations_path, 'w') as f:
        json.dump({"samples": samples}, f, indent=2)
    
    print(f"\nGenerated {count} synthetic samples in {output_dir}")
    print(f"Annotations saved to {annotations_path}")
    
    return samples


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic signature/stamp data"
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/synthetic',
        help='Output directory (default: data/synthetic)'
    )
    
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    
    parser.add_argument(
        '--no-signatures',
        action='store_true',
        help='Skip signature generation'
    )
    
    parser.add_argument(
        '--no-stamps',
        action='store_true',
        help='Skip stamp generation'
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        output_dir=args.output,
        count=args.count,
        include_signatures=not args.no_signatures,
        include_stamps=not args.no_stamps
    )


if __name__ == "__main__":
    main()
