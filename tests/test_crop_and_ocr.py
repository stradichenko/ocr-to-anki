"""
Test script that crops a large image into smaller sections
and tests OCR on each section to find the issue
"""

import os
import json
import base64
import requests
from pathlib import Path
from PIL import Image
import io
import time


def crop_image_into_sections(image_path: str, sections: int = 4) -> list:
    """
    Crop image into N equal sections (top to bottom).
    
    Args:
        image_path: Path to image
        sections: Number of sections to create (default: 4)
    
    Returns:
        List of base64 encoded image sections
    """
    with Image.open(image_path) as img:
        width, height = img.size
        section_height = height // sections
        
        print(f"Original image: {width}x{height}")
        print(f"Creating {sections} sections of ~{section_height}px height each\n")
        
        cropped_sections = []
        
        for i in range(sections):
            # Calculate crop box
            top = i * section_height
            bottom = (i + 1) * section_height if i < sections - 1 else height
            box = (0, top, width, bottom)
            
            # Crop and resize if needed
            section = img.crop(box)
            
            # Resize to max 200px width for much faster processing
            if section.size[0] > 200:
                ratio = 200 / section.size[0]
                new_size = (200, int(section.size[1] * ratio))
                section = section.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if section.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', section.size, (255, 255, 255))
                rgb_img.paste(section, mask=section.split()[-1] if section.mode in ('RGBA', 'LA') else None)
                section = rgb_img
            
            # Encode to base64
            buffer = io.BytesIO()
            section.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            
            cropped_sections.append({
                'section_num': i + 1,
                'dimensions': section.size,
                'size_kb': len(encoded) / 1024,
                'encoded': encoded
            })
            
            print(f"Section {i+1}: {section.size[0]}x{section.size[1]}, {len(encoded)/1024:.2f}KB")
        
        return cropped_sections


def test_ocr_on_section(section_data: dict, timeout: int = 60) -> dict:
    """Test OCR on a single cropped section."""
    url = "http://localhost:11434/api/generate"
    
    # Very simple prompt
    prompt = "Extract all visible text from this image. List only the words, one per line."
    
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": prompt,
        "images": [section_data['encoded']],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200
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
            if line and not line.startswith('#') and not line.startswith('*'):
                words.extend(line.split())
        
        return {
            'success': True,
            'section': section_data['section_num'],
            'time': round(elapsed, 2),
            'response': response_text,
            'words': words,
            'word_count': len(words)
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
    """Run cropped section tests."""
    print("=" * 60)
    print("CROPPED SECTION OCR TEST")
    print("=" * 60 + "\n")
    
    # Find image
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    image_path = project_root / "data" / "images" / "handwritten.jpeg"
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Crop into sections
    print("Step 1: Cropping image into sections\n")
    sections = crop_image_into_sections(str(image_path), sections=6)
    
    # Test each section
    print("\n" + "=" * 60)
    print("Step 2: Testing OCR on each section")
    print("=" * 60 + "\n")
    
    results = []
    for section in sections:
        section_num = section['section_num']
        print(f"Testing section {section_num}/{len(sections)}...")
        print(f"  Size: {section['dimensions']}, {section['size_kb']:.2f}KB")
        
        result = test_ocr_on_section(section, timeout=90)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Success in {result['time']}s")
            print(f"  Words found: {result['word_count']}")
            if result['words']:
                print(f"  Sample: {', '.join(result['words'][:5])}")
        else:
            print(f"  ✗ Failed: {result['error']}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60 + "\n")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful sections: {len(successful)}/{len(results)}")
    print(f"Failed sections: {len(failed)}/{len(results)}")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        total_words = sum(r['word_count'] for r in successful)
        print(f"\nAverage processing time: {avg_time:.2f}s")
        print(f"Total words extracted: {total_words}")
        
        print("\n=== All extracted words ===")
        for r in successful:
            if r['words']:
                print(f"\nSection {r['section']}:")
                for word in r['words']:
                    print(f"  - {word}")
    
    if failed:
        print("\n=== Failed sections ===")
        for r in failed:
            print(f"Section {r['section']}: {r['error']}")
    
    # Save results
    output_file = test_dir / "cropped_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image_path': str(image_path),
            'total_sections': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60 + "\n")
    
    if len(successful) == len(results):
        print("✓ All sections processed successfully!")
        print("\nThe issue is the full image size. Solutions:")
        print("  1. Process image in sections (implement chunking)")
        print("  2. Reduce image resolution more aggressively")
        print("  3. Use a more powerful model or hardware")
    elif len(successful) > 0:
        print(f"Partial success ({len(successful)}/{len(results)} sections)")
        print("\nSome sections work, others don't. This suggests:")
        print("  1. Complex sections with dense text cause timeouts")
        print("  2. Image preprocessing might help (enhance contrast)")
        print("  3. Try even smaller sections")
    else:
        print("✗ All sections failed")
        print("\nEven small sections timeout. This suggests:")
        print("  1. Model/server issue - restart Ollama")
        print("  2. Try different model (llava:7b, minicpm-v)")
        print("  3. Check system resources")


if __name__ == "__main__":
    main()
