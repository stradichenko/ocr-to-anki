"""
Enhanced test script for Ollama OCR with image preprocessing
Applies sharpening and contrast enhancement before OCR
"""

import os
import json
import base64
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import io


def get_image_info(image_path: str) -> dict:
    """Get image metadata."""
    with Image.open(image_path) as img:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        return {
            'width': img.size[0],
            'height': img.size[1],
            'format': img.format,
            'mode': img.mode,
            'size_mb': round(file_size_mb, 4)
        }


def enhance_image(image_path: str, max_width: int = 400, 
                  contrast_factor: float = 2.0, 
                  sharpness_factor: float = 2.0,
                  save_enhanced: bool = False,
                  output_path: str = None) -> str:
    """
    Enhance image with contrast and sharpening, then resize.
    Returns base64 encoded image.
    
    Args:
        image_path: Path to image file
        max_width: Maximum width in pixels (default: 400)
        contrast_factor: Contrast multiplier (default: 2.0, 1.0 = no change)
        sharpness_factor: Sharpness multiplier (default: 2.0, 1.0 = no change)
        save_enhanced: Save enhanced image to file (default: False)
        output_path: Path to save enhanced image (default: None)
    
    Returns:
        Base64 encoded enhanced image
    """
    with Image.open(image_path) as img:
        original_size = img.size
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        print(f"  Applying enhancements...")
        
        # Step 1: Increase contrast
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(contrast_factor)
        print(f"  - Contrast: {contrast_factor}x")
        
        # Step 2: Sharpen image
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(sharpness_factor)
        print(f"  - Sharpness: {sharpness_factor}x")
        
        # Step 3: Apply unsharp mask for additional sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        print(f"  - Unsharp mask applied")
        
        # Step 4: Resize if needed
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  - Resized: {original_size} → {new_size}")
        
        # Save enhanced image if requested
        if save_enhanced and output_path:
            img.save(output_path, format='JPEG', quality=95)
            print(f"  - Saved enhanced image to: {output_path}")
        
        # Save to bytes and encode
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def enhanced_ollama_ocr(image_path: str, language: str = "french", 
                        timeout: int = 300,
                        contrast: float = 2.0,
                        sharpness: float = 2.0,
                        save_enhanced: bool = False,
                        enhanced_output_path: str = None) -> dict:
    """
    Perform OCR with enhanced image preprocessing.
    
    Args:
        image_path: Path to image file
        language: Language to expect (default: french)
        timeout: Timeout in seconds (default: 300)
        contrast: Contrast enhancement factor (default: 2.0)
        sharpness: Sharpness enhancement factor (default: 2.0)
        save_enhanced: Save enhanced image for inspection (default: False)
        enhanced_output_path: Path to save enhanced image
    
    Returns:
        Dictionary with OCR results
    """
    # More explicit prompt
    prompt = f"""You are looking at an image with {language} text.
Your task is to read ALL the text you can see in this image.
Extract every single word, even if handwritten or unclear.
Output ONLY the words you find, one word per line.
Do not add any explanations, just list the words."""
    
    # Get image info
    print(f"Reading image: {image_path}")
    image_info = get_image_info(image_path)
    print(f"  Original: [{image_info['width']}, {image_info['height']}] {image_info['format']} {image_info['size_mb']}MB")
    
    # Enhance and encode image
    print("  Processing and enhancing image...")
    image_base64 = enhance_image(
        image_path, 
        max_width=400, 
        contrast_factor=contrast, 
        sharpness_factor=sharpness,
        save_enhanced=save_enhanced,
        output_path=enhanced_output_path
    )
    encoded_size_mb = len(image_base64) / (1024 * 1024)
    print(f"  Encoded size: {encoded_size_mb:.4f}MB")
    
    # Prepare request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.0,  # Zero temperature for deterministic output
            "num_predict": 1000   # Allow more tokens for response
        }
    }
    
    print(f"\nSending request to Ollama...")
    print(f"  Model: qwen3-vl:2b")
    print(f"  Timeout: {timeout} seconds")
    print(f"  Waiting for response...\n")
    
    try:
        import time
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=timeout)
        
        elapsed = time.time() - start_time
        print(f"  Response received in {elapsed:.2f} seconds")
        
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', '')
        
        # Extract words
        words = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*'):
                words.extend(line.split())
        
        return {
            'success': True,
            'image_path': image_path,
            'image_info': image_info,
            'enhancement': {
                'contrast': contrast,
                'sharpness': sharpness
            },
            'language': language,
            'prompt': prompt,
            'raw_response': response_text,
            'words': words,
            'word_count': len(words),
            'processing_time': round(elapsed, 2)
        }
    
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': f'Request timed out after {timeout} seconds',
            'image_path': image_path,
            'image_info': image_info,
            'enhancement': {
                'contrast': contrast,
                'sharpness': sharpness
            }
        }
    except requests.exceptions.ConnectionError as e:
        return {
            'success': False,
            'error': f'Connection error: {str(e)}. Is Ollama running? Try: ollama serve',
            'image_path': image_path,
            'image_info': image_info
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'image_path': image_path,
            'image_info': image_info
        }


def main():
    """Run enhanced OCR test."""
    print("=" * 60)
    print("ENHANCED OLLAMA OCR TEST")
    print("With Contrast and Sharpness Enhancement")
    print("=" * 60 + "\n")
    
    # Find handwritten.jpeg
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    possible_paths = [
        project_root / "data" / "images" / "handwritten.jpeg",
        project_root / "tests" / "handwritten.jpeg",
        project_root / "handwritten.jpeg",
    ]
    
    image_path = None
    for path in possible_paths:
        if path.exists():
            image_path = str(path)
            break
    
    if not image_path:
        print("ERROR: handwritten.jpeg not found in:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    # Create directory for enhanced images
    enhanced_dir = test_dir / "enhanced_images"
    enhanced_dir.mkdir(exist_ok=True)
    
    # Test with different enhancement levels
    enhancement_presets = [
        {'name': 'Moderate', 'contrast': 1.8, 'sharpness': 1.5},
        {'name': 'Strong', 'contrast': 2.5, 'sharpness': 2.0},
        {'name': 'Aggressive', 'contrast': 3.0, 'sharpness': 2.5},
    ]
    
    print("Testing multiple enhancement levels:\n")
    print("Enhanced images will be saved to tests/enhanced_images/\n")
    all_results = []
    
    for i, preset in enumerate(enhancement_presets, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}/{len(enhancement_presets)}: {preset['name']} Enhancement")
        print(f"Contrast: {preset['contrast']}x, Sharpness: {preset['sharpness']}x")
        print('=' * 60 + '\n')
        
        # Path for saving enhanced image
        enhanced_path = enhanced_dir / f"enhanced_{preset['name'].lower()}.jpg"
        
        result = enhanced_ollama_ocr(
            image_path, 
            language="french", 
            timeout=300,
            contrast=preset['contrast'],
            sharpness=preset['sharpness'],
            save_enhanced=True,
            enhanced_output_path=str(enhanced_path)
        )
        
        result['preset_name'] = preset['name']
        result['enhanced_image_path'] = str(enhanced_path)
        all_results.append(result)
        
        # Display results
        print("\n=== Results ===")
        if result['success']:
            print(f"✓ Success!")
            print(f"Processing time: {result['processing_time']}s")
            print(f"Words found: {result['word_count']}")
            if result['words']:
                print(f"\nWords extracted:")
                for word in result['words'][:10]:
                    print(f"  - {word}")
                if result['word_count'] > 10:
                    print(f"  ... and {result['word_count'] - 10} more")
            else:
                print("\n⚠ Model returned empty response - check raw_response in JSON")
                if result['raw_response']:
                    print(f"Raw response preview: {result['raw_response'][:100]}...")
        else:
            print(f"✗ Failed: {result['error']}")
        
        print()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60 + "\n")
    
    successful = [r for r in all_results if r['success']]
    
    if successful:
        print("Enhancement Level | Words Found | Processing Time")
        print("-" * 60)
        for r in all_results:
            if r['success']:
                print(f"{r['preset_name']:17s} | {r['word_count']:11d} | {r['processing_time']:6.2f}s")
        
        # Best result
        best = max(successful, key=lambda x: x['word_count'])
        print(f"\n✓ Best result: {best['preset_name']} with {best['word_count']} words")
    else:
        print("✗ All enhancement levels failed")
    
    # Save all results
    output_file = test_dir / "test_enhanced_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image_path': image_path,
            'tests_run': len(all_results),
            'successful': len(successful),
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60 + "\n")
    
    if successful:
        if all(r['word_count'] == 0 for r in successful):
            print("⚠ Model processed images but returned no words!")
            print("\nPossible issues:")
            print("  1. Model doesn't understand the image content")
            print("  2. Handwriting is too difficult for this model")
            print("  3. Image language/script not in model's training data")
            print("  4. Model needs different prompting strategy")
            print("\nNext steps:")
            print("  1. Check enhanced images in tests/enhanced_images/")
            print("  2. Try with a printed text image first")
            print("  3. Try different model: ollama pull llava:7b")
            print("  4. Test with simpler/clearer image")
            print("  5. Check raw_response in test_enhanced_results.json")
        else:
            print("Image enhancement improved OCR results!")
            print("\nTo use best settings in production:")
            best_contrast = best['enhancement']['contrast']
            best_sharpness = best['enhancement']['sharpness']
            print(f"  - Contrast factor: {best_contrast}")
            print(f"  - Sharpness factor: {best_sharpness}")
            print("\nApply these in your preprocessing pipeline.")
    else:
        print("Enhancement didn't resolve timeout issues.")
        print("\nTry:")
        print("  1. Further reduce image size (max_width=200)")
        print("  2. Process image in smaller sections")
        print("  3. Use different model (llava:7b)")
    
    print(f"\nCheck enhanced images at: {enhanced_dir}")


if __name__ == "__main__":
    main()
