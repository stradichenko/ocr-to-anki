"""
Simple test script for Ollama OCR
Tests with a single image (handwritten.jpeg) in French
"""

import os
import json
import base64
import requests
from pathlib import Path
from PIL import Image
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


def resize_image_if_needed(image_path: str, max_width: int = 400) -> str:
    """
    Resize image if it's too large to avoid timeouts.
    Returns base64 encoded image.
    """
    with Image.open(image_path) as img:
        original_size = img.size
        
        # Only resize if width exceeds max_width
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  Resized from {original_size} to {new_size}")
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        # Save to bytes and encode
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def simple_ollama_ocr(image_path: str, language: str = "french", timeout: int = 300) -> dict:
    """
    Perform simple OCR test with Ollama.
    
    Args:
        image_path: Path to image file
        language: Language to expect (default: french)
        timeout: Timeout in seconds (default: 300)
    
    Returns:
        Dictionary with OCR results
    """
    # Even simpler prompt
    prompt = f"Extract all text from this image. The text is in {language}. List only the words, one per line."
    
    # Get image info
    print(f"Reading image: {image_path}")
    image_info = get_image_info(image_path)
    print(f"  Original: [{image_info['width']}, {image_info['height']}] {image_info['format']} {image_info['size_mb']}MB")
    
    # Resize and encode image
    print("  Processing image...")
    image_base64 = resize_image_if_needed(image_path, max_width=400)
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
            "temperature": 0.1,  # Lower temperature for more focused output
            "num_predict": 500   # Limit response length
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
        
        # Extract words (simple split by lines/spaces)
        words = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*'):
                words.extend(line.split())
        
        return {
            'success': True,
            'image_path': image_path,
            'image_info': image_info,
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
            'image_info': image_info
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
    """Run simple test."""
    print("=== Simple Ollama OCR Test ===\n")
    
    # Find handwritten.jpeg
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Check multiple possible locations
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
    
    # Run OCR with 5 minute timeout
    result = simple_ollama_ocr(image_path, language="french", timeout=300)
    
    # Display results
    print("\n=== Results ===")
    if result['success']:
        print(f"✓ Success!")
        print(f"Language: {result['language']}")
        print(f"Processing time: {result['processing_time']}s")
        print(f"Words found: {result['word_count']}")
        print(f"\nWords extracted:")
        for word in result['words']:
            print(f"  - {word}")
        print(f"\n=== Raw Response ===")
        print(result['raw_response'])
    else:
        print(f"✗ Failed: {result['error']}")
        if 'image_info' in result:
            print(f"\nImage info:")
            for key, value in result['image_info'].items():
                print(f"  {key}: {value}")
    
    # Save result
    output_file = test_dir / "test_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull result saved to: {output_file}")


if __name__ == "__main__":
    main()
