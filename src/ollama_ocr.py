"""
Ollama OCR Script
Uses local Ollama model (qwen3-vl:2b) to perform OCR on images.
Configurable via settings.yaml for text type, scope, and language.
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any
import yaml
import requests
from datetime import datetime
from PIL import Image
import io
import time


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_image_shape(image_path: str) -> tuple:
    """
    Get image dimensions as (width, height, channels).
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (width, height, channels, format, file_size_mb)
    """
    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format if img.format else "Unknown"
        
        # Determine number of channels
        if img.mode == 'RGB':
            channels = 3
        elif img.mode == 'RGBA':
            channels = 4
        elif img.mode == 'L':  # Grayscale
            channels = 1
        else:
            channels = len(img.getbands())
    
    # Get file size in MB
    file_size_bytes = os.path.getsize(image_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return (width, height, channels, img_format, file_size_mb)


def resize_image_for_ocr(image_path: str, max_width: int = 800) -> str:
    """
    Resize image and return base64 encoded string.
    Reduces image size to avoid timeouts while maintaining readability.
    
    Args:
        image_path: Path to image file
        max_width: Maximum width in pixels (default: 800)
    
    Returns:
        Base64 encoded image string
    """
    with Image.open(image_path) as img:
        original_size = img.size
        
        # Resize if needed
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  Resized: {original_size} → {new_size}")
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
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


def build_ocr_prompt(config: Dict[str, Any]) -> str:
    """Build OCR prompt based on configuration settings."""
    text_type = config['ollama_ocr']['text_type']
    analysis_scope = config['ollama_ocr']['analysis_scope']
    highlight_color = config['ollama_ocr']['highlight_color']
    language = config['ollama_ocr']['language']
    
    prompt_parts = []
    
    # Text type specification
    if text_type == "detect":
        prompt_parts.append("Analyze this image and detect whether the text is handwritten or printed.")
    elif text_type == "handwritten":
        prompt_parts.append("This image contains handwritten text.")
    elif text_type == "printed":
        prompt_parts.append("This image contains printed text.")
    
    # Analysis scope specification
    if analysis_scope == "highlighted":
        prompt_parts.append(f"Focus only on words that are highlighted in {highlight_color}.")
    else:
        prompt_parts.append("Analyze all text in the entire image.")
    
    # Language specification
    if language == "detect":
        prompt_parts.append("Detect the language automatically.")
    else:
        prompt_parts.append(f"The text is in {language}.")
    
    # Output format
    prompt_parts.append("Extract all text and provide a list of words detected.")
    prompt_parts.append("Output format: Return a JSON array of words, for example: [\"word1\", \"word2\", \"word3\"]")
    
    return " ".join(prompt_parts)


def perform_ocr(image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform OCR on an image using Ollama model.
    
    Args:
        image_path: Path to the image file
        config: Configuration dictionary
    
    Returns:
        Dictionary containing OCR results and metadata
    """
    ollama_config = config['ollama_ocr']
    url = f"{ollama_config['url']}/api/generate"
    
    # Get and print image shape
    try:
        width, height, channels, img_format, file_size_mb = get_image_shape(image_path)
        print(f"  Image: [{width}, {height}, {channels}] {img_format} {file_size_mb:.4f}MB")
    except Exception as e:
        print(f"  Warning: Could not read image info: {e}")
        width = height = channels = None
        img_format = "Unknown"
        file_size_mb = 0
    
    # Resize and encode image (more aggressive resizing)
    max_width = ollama_config.get('max_image_width', 600)
    image_base64 = resize_image_for_ocr(image_path, max_width=max_width)
    encoded_size_mb = len(image_base64) / (1024 * 1024)
    print(f"  Encoded: {encoded_size_mb:.4f}MB")
    
    # Build prompt
    prompt = build_ocr_prompt(config)
    
    # Prepare request payload
    payload = {
        "model": ollama_config['model'],
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    
    # Make request with timing
    print(f"  Processing...")
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=ollama_config['timeout']
        )
        
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        
        result = response.json()
        
        # Extract words from response
        response_text = result.get('response', '')
        words = extract_words_from_response(response_text)
        
        result_dict = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'config': {
                'text_type': ollama_config['text_type'],
                'analysis_scope': ollama_config['analysis_scope'],
                'highlight_color': ollama_config['highlight_color'],
                'language': ollama_config['language']
            },
            'prompt': prompt,
            'raw_response': response_text,
            'words': words,
            'word_count': len(words)
        }
        
        # Add image metadata if available
        if width and height:
            result_dict['image_info'] = {
                'width': width,
                'height': height,
                'channels': channels,
                'format': img_format,
                'size_mb': round(file_size_mb, 4)
            }
        
        return result_dict
    
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        
        result_dict = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'error': str(e),
            'words': []
        }
        
        # Add image metadata even on error
        if width and height:
            result_dict['image_info'] = {
                'width': width,
                'height': height,
                'channels': channels,
                'format': img_format,
                'size_mb': round(file_size_mb, 4)
            }
        
        return result_dict


def extract_words_from_response(response_text: str) -> List[str]:
    """
    Extract words list from model response.
    Tries to parse JSON array, falls back to text splitting.
    """
    try:
        # Try to find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start != -1 and end > start:
            json_str = response_text[start:end]
            words = json.loads(json_str)
            if isinstance(words, list):
                return [str(w).strip() for w in words if w]
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: split by whitespace and common delimiters
    words = []
    for line in response_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('{') and not line.startswith('['):
            words.extend(line.split())
    
    return words


def process_images(config: Dict[str, Any], input_dir: str = None) -> List[Dict[str, Any]]:
    """
    Process all images in input directory.
    
    Args:
        config: Configuration dictionary
        input_dir: Override input directory (defaults to config setting)
    
    Returns:
        List of OCR results for all images
    """
    if input_dir is None:
        input_dir = config['ocr']['input_dir']
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return []
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
        result = perform_ocr(str(image_file), config)
        results.append(result)
        
        if 'error' in result:
            print(f"  ✗ Error after {result['processing_time']}s: {result['error']}")
        else:
            print(f"  ✓ Completed in {result['processing_time']}s")
            print(f"  Prompt used: {result['prompt']}")
            print(f"  Words found: {result['word_count']}")
            
            # Display words found
            if result['words']:
                print(f"  Extracted words:")
                for word in result['words']:
                    print(f"    - {word}")
            else:
                print(f"  (No words extracted)")
    
    return results


def save_results(results: List[Dict[str, Any]], config: Dict[str, Any]):
    """Save OCR results to output directory."""
    output_dir = Path(config['ollama_ocr']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    for result in results:
        image_name = Path(result['image_path']).stem
        output_file = output_dir / f"{image_name}_ocr.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_file = output_dir / f"ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        'total_images': len(results),
        'successful': len([r for r in results if 'error' not in r]),
        'failed': len([r for r in results if 'error' in r]),
        'total_words': sum(r.get('word_count', 0) for r in results),
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    
    print("=== Ollama OCR ===")
    print(f"Model: {config['ollama_ocr']['model']}")
    print(f"Text Type: {config['ollama_ocr']['text_type']}")
    print(f"Analysis Scope: {config['ollama_ocr']['analysis_scope']}")
    print(f"Language: {config['ollama_ocr']['language']}")
    print()
    
    # Process images
    results = process_images(config)
    
    # Save results if configured
    if results and config['ollama_ocr']['save_response']:
        save_results(results, config)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total images processed: {len(results)}")
    print(f"Total words found: {sum(r.get('word_count', 0) for r in results)}")


if __name__ == "__main__":
    main()
