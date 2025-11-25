"""
Image inspection tool - displays image and basic analysis
Helps understand why OCR is failing
"""

from pathlib import Path
from PIL import Image, ImageStat, ImageDraw, ImageFont
import numpy as np


def analyze_image(image_path: str):
    """Analyze image content and characteristics."""
    print("=" * 70)
    print("IMAGE INSPECTION TOOL")
    print("=" * 70 + "\n")
    
    print(f"Image: {Path(image_path).name}\n")
    
    with Image.open(image_path) as img:
        # Basic info
        print("BASIC INFORMATION:")
        print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
        print(f"  Mode: {img.mode}")
        print(f"  Format: {img.format}")
        
        # Convert to array for analysis
        img_array = np.array(img)
        
        # Color analysis
        print("\nCOLOR ANALYSIS:")
        if len(img_array.shape) == 3:  # Color image
            print(f"  Channels: {img_array.shape[2]}")
            print(f"  Mean RGB: ({img_array[:,:,0].mean():.1f}, "
                  f"{img_array[:,:,1].mean():.1f}, "
                  f"{img_array[:,:,2].mean():.1f})")
        else:
            print(f"  Grayscale")
            print(f"  Mean value: {img_array.mean():.1f}")
        
        # Statistics
        stat = ImageStat.Stat(img)
        print("\nSTATISTICS:")
        print(f"  Mean: {stat.mean}")
        print(f"  Std Dev: {stat.stddev}")
        print(f"  Extrema: {stat.extrema}")
        
        # Brightness analysis
        if img.mode == 'RGB':
            grayscale = img.convert('L')
            brightness = np.array(grayscale).mean()
        else:
            brightness = img_array.mean()
        
        print(f"\nBRIGHTNESS:")
        print(f"  Average: {brightness:.1f}/255")
        if brightness < 50:
            print(f"  ⚠ Very dark image")
        elif brightness > 200:
            print(f"  ⚠ Very bright image")
        else:
            print(f"  ✓ Normal brightness")
        
        # Contrast analysis
        if img.mode == 'RGB':
            contrast = np.array(img.convert('L')).std()
        else:
            contrast = img_array.std()
        
        print(f"\nCONTRAST:")
        print(f"  Std deviation: {contrast:.1f}")
        if contrast < 20:
            print(f"  ⚠ Very low contrast - text may not be visible")
        elif contrast < 40:
            print(f"  ⚠ Low contrast - may need enhancement")
        else:
            print(f"  ✓ Adequate contrast")
        
        # Edge detection (simple)
        if img.mode == 'RGB':
            edges = img.convert('L')
        else:
            edges = img
        
        edges_array = np.array(edges)
        edge_count = np.sum(np.abs(np.diff(edges_array, axis=0)) > 30)
        edge_count += np.sum(np.abs(np.diff(edges_array, axis=1)) > 30)
        edge_density = edge_count / (img.size[0] * img.size[1])
        
        print(f"\nEDGE DETECTION:")
        print(f"  Edge density: {edge_density:.4f}")
        if edge_density < 0.01:
            print(f"  ⚠ Very few edges - may be blank or very blurry")
        elif edge_density < 0.05:
            print(f"  ⚠ Low edge count - may lack detail")
        else:
            print(f"  ✓ Contains edges/details")
        
        # Check if image is mostly uniform
        std_threshold = 15
        is_uniform = contrast < std_threshold
        
        print(f"\nCONTENT ASSESSMENT:")
        if is_uniform:
            print(f"  ⚠ Image appears mostly uniform/blank")
            print(f"  ⚠ May not contain readable text")
        else:
            print(f"  ✓ Image has variation/content")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70 + "\n")
        
        if is_uniform:
            print("❌ This image appears to be mostly blank or uniform")
            print("   The model is likely returning empty because there is no text.")
            print("\n   Possible reasons:")
            print("   1. Image is actually blank/empty")
            print("   2. Text is same color as background (no contrast)")
            print("   3. Image is severely overexposed or underexposed")
            print("   4. Wrong image file loaded")
        elif contrast < 40:
            print("⚠ Low contrast detected")
            print("   Try:")
            print("   1. Enhance contrast before OCR")
            print("   2. Adjust brightness levels")
            print("   3. Use image preprocessing")
        elif brightness < 50 or brightness > 200:
            print("⚠ Brightness issue detected")
            print("   Try:")
            print("   1. Adjust image exposure")
            print("   2. Use histogram equalization")
            print("   3. Try different scanning settings")
        else:
            print("✓ Image properties look OK for OCR")
            print("   If OCR still fails:")
            print("   1. Text may be in unsupported script/language")
            print("   2. Text may be too small or degraded")
            print("   3. Try different OCR model")
        
        # Save analysis visualization
        save_analysis_visualization(img, image_path)


def save_analysis_visualization(img: Image.Image, original_path: str):
    """Save a visualization with analysis overlays."""
    from PIL import ImageFilter
    
    test_dir = Path(__file__).parent
    output_dir = test_dir / "image_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with original and processed versions
    width, height = img.size
    
    # 1. Original
    vis_width = 800
    if width > vis_width:
        ratio = vis_width / width
        vis_size = (vis_width, int(height * ratio))
        img_resized = img.resize(vis_size, Image.Resampling.LANCZOS)
    else:
        img_resized = img.copy()
    
    # 2. Grayscale
    gray = img_resized.convert('L')
    
    # 3. High contrast
    from PIL import ImageEnhance
    high_contrast = ImageEnhance.Contrast(img_resized).enhance(3.0)
    
    # 4. Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Save each
    base_name = Path(original_path).stem
    img_resized.save(output_dir / f"{base_name}_original.jpg", quality=90)
    gray.save(output_dir / f"{base_name}_grayscale.jpg", quality=90)
    high_contrast.save(output_dir / f"{base_name}_contrast.jpg", quality=90)
    edges.save(output_dir / f"{base_name}_edges.jpg", quality=90)
    
    print(f"\n📊 Analysis images saved to: {output_dir}")
    print(f"   - {base_name}_original.jpg")
    print(f"   - {base_name}_grayscale.jpg")
    print(f"   - {base_name}_contrast.jpg")
    print(f"   - {base_name}_edges.jpg")
    print("\n   👁️  Visually inspect these to understand image content")


def main():
    """Run image inspection."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Target the problematic image
    image_path = project_root / "data" / "images" / "1d21248b-9dda-4e67-972b-70aa96e35eee.jpeg"
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        print("\nUpdate the image_path variable in the script.")
        return
    
    analyze_image(str(image_path))


if __name__ == "__main__":
    main()
