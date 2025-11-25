"""
Debug script for images that return empty OCR results
Tests different prompts and parameters to find what works
"""

import json
import base64
import requests
import time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import io


def get_image_base64(image_path: str, max_width: int = 600, enhance: bool = False) -> str:
    """Encode image with optional enhancement."""
    with Image.open(image_path) as img:
        # Convert to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img
        
        # Optional enhancement
        if enhance:
            img = ImageEnhance.Contrast(img).enhance(2.5)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        
        # Resize
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Encode
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def test_prompt(image_base64: str, prompt: str, name: str, timeout: int = 120) -> dict:
    """Test a specific prompt configuration."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300
        }
    }
    
    print(f"\nTesting: {name}")
    print(f"  Prompt length: {len(prompt)} chars")
    
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        response_text = result.get('response', '').strip()
        
        # Try to extract words
        words = []
        if response_text:
            # Try JSON first
            try:
                import json as json_lib
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    words = json_lib.loads(response_text[json_start:json_end])
            except:
                # Fallback to text splitting
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line and not any(x in line.lower() for x in ['image', 'text', 'see', 'the', 'is']):
                        words.extend(line.split())
        
        print(f"  ✓ Completed in {elapsed:.2f}s")
        print(f"  Response length: {len(response_text)} chars")
        print(f"  Words found: {len(words)}")
        if words:
            print(f"  Sample: {', '.join(words[:5])}")
        
        return {
            'name': name,
            'success': True,
            'time': elapsed,
            'response': response_text,
            'words': words,
            'word_count': len(words)
        }
    
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return {
            'name': name,
            'success': False,
            'error': str(e)
        }


def main():
    """Run diagnostic tests on problematic image."""
    print("=" * 70)
    print("IMAGE OCR DEBUG TOOL")
    print("=" * 70)
    
    # Get image path
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Target the problematic image
    image_path = project_root / "data" / "images" / "1d21248b-9dda-4e67-972b-70aa96e35eee.jpeg"
    
    if not image_path.exists():
        print(f"\nERROR: Image not found: {image_path}")
        print("Update the image_path variable in the script.")
        return
    
    print(f"\nTarget image: {image_path.name}")
    
    # Load image info
    with Image.open(image_path) as img:
        print(f"Image size: {img.size[0]}x{img.size[1]}")
        print(f"Mode: {img.mode}")
        print(f"Format: {img.format}")
    
    # Test different configurations
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 70)
    
    configs = [
        {
            'name': 'Standard (600px, no enhancement)',
            'width': 600,
            'enhance': False,
            'prompt': 'Extract all visible text from this image. Return only a JSON array of words: ["word1", "word2"]. If no text, return []'
        },
        {
            'name': 'Minimal Prompt',
            'width': 600,
            'enhance': False,
            'prompt': 'List all text you see. One word per line.'
        },
        {
            'name': 'Ultra Simple',
            'width': 600,
            'enhance': False,
            'prompt': 'What words are in this image?'
        },
        {
            'name': 'Enhanced Image (contrast + sharpness)',
            'width': 600,
            'enhance': True,
            'prompt': 'Extract all visible text from this image. Return only a JSON array of words: ["word1", "word2"]'
        },
        {
            'name': 'Smaller Resolution (400px)',
            'width': 400,
            'enhance': False,
            'prompt': 'Extract all text. Return JSON array of words.'
        },
        {
            'name': 'Larger Resolution (800px)',
            'width': 800,
            'enhance': False,
            'prompt': 'Extract all text. Return JSON array of words.'
        },
    ]
    
    results = []
    
    for config in configs:
        image_b64 = get_image_base64(
            str(image_path),
            max_width=config['width'],
            enhance=config['enhance']
        )
        
        result = test_prompt(
            image_b64,
            config['prompt'],
            config['name'],
            timeout=150
        )
        results.append(result)
        
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    
    successful = [r for r in results if r.get('success')]
    with_words = [r for r in successful if r.get('word_count', 0) > 0]
    
    print(f"Tests run: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"With words extracted: {len(with_words)}")
    
    if with_words:
        print("\n✓ Configurations that extracted words:")
        for r in with_words:
            print(f"  - {r['name']}: {r['word_count']} words in {r['time']:.1f}s")
        
        best = max(with_words, key=lambda x: x['word_count'])
        print(f"\n✓ Best configuration: {best['name']}")
        print(f"  Words: {', '.join(best['words'][:10])}")
        if best['word_count'] > 10:
            print(f"  ... and {best['word_count'] - 10} more")
    else:
        print("\n✗ No configuration successfully extracted words")
        print("\nPossible reasons:")
        print("  1. Image contains no readable text")
        print("  2. Text is too blurry/small/distorted")
        print("  3. Model doesn't recognize the language/script")
        print("  4. Image is primarily graphical with minimal text")
    
    # Save results
    output_file = test_dir / "image_debug_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image': str(image_path),
            'tests': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Recommendations
    if with_words:
        best = max(with_words, key=lambda x: x['word_count'])
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70 + "\n")
        print(f"Use these settings for this type of image:")
        print(f"  - Max width: {[c['width'] for c in configs if c['name'] == best['name']][0]}px")
        print(f"  - Enhancement: {[c['enhance'] for c in configs if c['name'] == best['name']][0]}")
        print(f"  - Prompt style: Simplified")


if __name__ == "__main__":
    main()
