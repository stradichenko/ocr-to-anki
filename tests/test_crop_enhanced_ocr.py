"""
Combined test: Crop image into sections AND apply enhancements
This should give the best OCR results by:
1. Reducing image complexity (cropping)
2. Improving text clarity (enhancement)
"""

import os
import json
import base64
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import io
import time


def enhance_and_crop_sections(image_path: str, sections: int = 6,
                               crop_width: int = 400,
                               contrast: float = 2.0,
                               sharpness: float = 1.5,
                               save_sections: bool = False,
                               output_dir: str = None) -> list:
    """
    Crop image into sections and apply enhancements to each.
    
    Args:
        image_path: Path to source image
        sections: Number of horizontal sections (default: 6)
        crop_width: Target width for each section (default: 400)
        contrast: Contrast enhancement factor (default: 2.0)
        sharpness: Sharpness enhancement factor (default: 1.5)
        save_sections: Save enhanced sections to disk (default: False)
        output_dir: Directory to save sections
    
    Returns:
        List of dictionaries with section data
    """
    with Image.open(image_path) as img:
        width, height = img.size
        section_height = height // sections
        
        print(f"Original image: {width}x{height}")
        print(f"Creating {sections} sections, each ~{section_height}px height")
        print(f"Enhancement: Contrast {contrast}x, Sharpness {sharpness}x\n")
        
        cropped_sections = []
        
        for i in range(sections):
            # Calculate crop box
            top = i * section_height
            bottom = (i + 1) * section_height if i < sections - 1 else height
            box = (0, top, width, bottom)
            
            # Crop section
            section = img.crop(box)
            
            # Convert to RGB if necessary
            if section.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', section.size, (255, 255, 255))
                rgb_img.paste(section, mask=section.split()[-1] if section.mode in ('RGBA', 'LA') else None)
                section = rgb_img
            
            # Apply enhancements BEFORE resizing
            contrast_enhancer = ImageEnhance.Contrast(section)
            section = contrast_enhancer.enhance(contrast)
            
            sharpness_enhancer = ImageEnhance.Sharpness(section)
            section = sharpness_enhancer.enhance(sharpness)
            
            section = section.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Now resize
            if section.size[0] > crop_width:
                ratio = crop_width / section.size[0]
                new_size = (crop_width, int(section.size[1] * ratio))
                section = section.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save section if requested
            if save_sections and output_dir:
                section_path = Path(output_dir) / f"section_{i+1:02d}.jpg"
                section.save(section_path, format='JPEG', quality=95)
            
            # Encode to base64
            buffer = io.BytesIO()
            section.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            
            cropped_sections.append({
                'section_num': i + 1,
                'original_box': box,
                'dimensions': section.size,
                'size_kb': len(encoded) / 1024,
                'encoded': encoded
            })
            
            print(f"Section {i+1:02d}: {section.size[0]}x{section.size[1]:3d}, {len(encoded)/1024:6.2f}KB")
        
        return cropped_sections


def test_ocr_on_section(section_data: dict, language: str = "french", 
                        timeout: int = 60) -> dict:
    """Test OCR on an enhanced cropped section."""
    url = "http://localhost:11434/api/generate"
    
    # More explicit prompt
    prompt = f"""Read all text in this image section. The text is in {language}.
Extract every word you can see, even if partially visible.
List only the words, one per line. No explanations."""
    
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": prompt,
        "images": [section_data['encoded']],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300
        }
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        response_text = result.get('response', '').strip()
        
        # Extract words
        words = []
        for line in response_text.split('\n'):
            line = line.strip()
            # Filter out common non-word responses
            if line and not any(line.startswith(x) for x in ['#', '*', '-', 'Section', 'Image', 'Text']):
                words.extend(line.split())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                unique_words.append(word)
        
        return {
            'success': True,
            'section': section_data['section_num'],
            'time': round(elapsed, 2),
            'raw_response': response_text,
            'words': unique_words,
            'word_count': len(unique_words)
        }
    
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'section': section_data['section_num'],
            'error': f'Timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'success': False,
            'section': section_data['section_num'],
            'error': str(e)
        }


def main():
    """Run combined crop + enhancement test."""
    print("=" * 70)
    print("ENHANCED CROPPED SECTION OCR TEST")
    print("Combines cropping + enhancement for optimal results")
    print("=" * 70 + "\n")
    
    # Setup paths
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    image_path = project_root / "data" / "images" / "handwritten.jpeg"
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Create output directory for sections
    sections_dir = test_dir / "enhanced_sections"
    sections_dir.mkdir(exist_ok=True)
    
    # Test different enhancement levels
    enhancement_configs = [
        {'name': 'Light', 'contrast': 1.5, 'sharpness': 1.2, 'sections': 8},
        {'name': 'Moderate', 'contrast': 2.0, 'sharpness': 1.5, 'sections': 8},
        {'name': 'Strong', 'contrast': 2.5, 'sharpness': 2.0, 'sections': 8},
    ]
    
    all_test_results = []
    
    for config in enhancement_configs:
        print(f"\n{'=' * 70}")
        print(f"Test: {config['name']} Enhancement")
        print(f"Contrast: {config['contrast']}x, Sharpness: {config['sharpness']}x")
        print(f"Sections: {config['sections']}")
        print('=' * 70 + '\n')
        
        # Create sections directory for this config
        config_dir = sections_dir / config['name'].lower()
        config_dir.mkdir(exist_ok=True)
        
        # Crop and enhance
        print("Step 1: Cropping and enhancing sections\n")
        sections = enhance_and_crop_sections(
            str(image_path),
            sections=config['sections'],
            crop_width=400,
            contrast=config['contrast'],
            sharpness=config['sharpness'],
            save_sections=True,
            output_dir=str(config_dir)
        )
        
        # Test OCR on each section
        print(f"\nStep 2: Running OCR on {len(sections)} sections\n")
        results = []
        
        for section in sections:
            section_num = section['section_num']
            print(f"Testing section {section_num:02d}/{len(sections)}... ", end='', flush=True)
            
            result = test_ocr_on_section(section, language="french", timeout=90)
            results.append(result)
            
            if result['success']:
                print(f"✓ {result['time']:5.2f}s, {result['word_count']} words")
            else:
                print(f"✗ {result['error']}")
        
        # Summarize this config
        successful = [r for r in results if r['success']]
        total_words = sum(r['word_count'] for r in successful)
        avg_time = sum(r['time'] for r in successful) / len(successful) if successful else 0
        
        print(f"\nConfig '{config['name']}' summary:")
        print(f"  Success: {len(successful)}/{len(results)}")
        print(f"  Total words: {total_words}")
        print(f"  Avg time: {avg_time:.2f}s")
        print(f"  Sections saved: {config_dir}")
        
        all_test_results.append({
            'config': config,
            'results': results,
            'summary': {
                'successful': len(successful),
                'total': len(results),
                'total_words': total_words,
                'avg_time': avg_time
            }
        })
    
    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON ACROSS ENHANCEMENT LEVELS")
    print("=" * 70 + "\n")
    
    print(f"{'Enhancement':<15} | {'Success':<10} | {'Words':<8} | {'Avg Time':<10}")
    print("-" * 70)
    for test in all_test_results:
        config = test['config']
        summary = test['summary']
        print(f"{config['name']:<15} | "
              f"{summary['successful']}/{summary['total']:<8} | "
              f"{summary['total_words']:<8} | "
              f"{summary['avg_time']:5.2f}s")
    
    # Find best config
    best_test = max(all_test_results, key=lambda x: x['summary']['total_words'])
    best_config = best_test['config']
    
    print(f"\n✓ Best: {best_config['name']} with {best_test['summary']['total_words']} words")
    
    # Show all extracted words from best config
    print(f"\n=== All Words from Best Config ('{best_config['name']}') ===\n")
    for result in best_test['results']:
        if result['success'] and result['words']:
            print(f"Section {result['section']:02d}:")
            for word in result['words']:
                print(f"  - {word}")
    
    # Save results
    output_file = test_dir / "crop_enhanced_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image_path': str(image_path),
            'tests': all_test_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70 + "\n")
    
    if best_test['summary']['total_words'] > 0:
        print(f"✓ Successfully extracted {best_test['summary']['total_words']} words!")
        print(f"\nBest settings:")
        print(f"  Enhancement: {best_config['name']}")
        print(f"  Contrast: {best_config['contrast']}x")
        print(f"  Sharpness: {best_config['sharpness']}x")
        print(f"  Sections: {best_config['sections']}")
        print(f"\nTo implement in production:")
        print(f"  1. Split images into {best_config['sections']} sections")
        print(f"  2. Apply contrast={best_config['contrast']}, sharpness={best_config['sharpness']}")
        print(f"  3. Process each section individually")
        print(f"  4. Combine results")
        print(f"\nCheck enhanced sections at: {sections_dir}")
    else:
        print("✗ No words extracted even with enhancements + cropping")
        print("\nThis suggests:")
        print("  1. The model qwen3-vl:2b cannot handle this type of content")
        print("  2. Try different model: ollama pull llava:7b")
        print("  3. Test with clearer/simpler image first")
        print("  4. The handwriting may be too difficult")


if __name__ == "__main__":
    main()
