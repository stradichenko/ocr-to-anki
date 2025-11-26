"""
Simple Ollama OCR Test Script
Tests any Ollama vision model on a single image.
Usage: python src/test_model_ocr.py [MODEL] [IMAGE_PATH]
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
import io
import time


def resize_image_for_ocr(image_path: str, max_width: int = 800) -> str:
    """Resize image and return base64 encoded string."""
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


def get_image_info(image_path: str):
    """Get basic image information."""
    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format if img.format else "Unknown"
        channels = 3 if img.mode == 'RGB' else (4 if img.mode == 'RGBA' else 1)
    
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    return width, height, channels, img_format, file_size_mb


def check_model_available(url: str, model: str) -> bool:
    """Check if the specified model is available in Ollama."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        available_models = [m['name'] for m in data.get('models', [])]
        
        print("Available models:")
        for m in available_models:
            marker = "✓" if m == model else " "
            print(f"  {marker} {m}")
        print()
        
        return model in available_models
    except Exception as e:
        print(f"⚠️ Could not check available models: {e}\n")
        return True  # Proceed anyway


def verify_model_health(url: str, model: str) -> bool:
    """
    Try to verify model is loadable by making a simple test request.
    For vision models, this is skipped since they require images.
    
    Returns:
        True if model appears healthy, False otherwise
    """
    print(f"Verifying model health for '{model}'...")
    
    # Vision models that are known to require images
    vision_models = ['gemma3', 'qwen', 'llava', 'deepseek-ocr', 'minicpm', 'cogvlm']
    
    # Skip health check for vision models
    if any(vm in model.lower() for vm in vision_models):
        print("✓ Vision model detected, skipping text-only health check\n")
        return True
    
    try:
        # Make a minimal request to test if model loads
        test_payload = {
            "model": model,
            "prompt": "test",
            "stream": False,
            "options": {
                "num_predict": 1
            }
        }
        
        response = requests.post(
            f"{url}/api/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ Model appears healthy\n")
            return True
        else:
            print(f"⚠️ Model health check failed: HTTP {response.status_code}\n")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Model health check failed: {e}\n")
        return False


def get_ocr_prompt(style: str = "simple") -> str:
    """
    Get OCR prompt based on style preference.
    
    Args:
        style: Prompt style - 'simple', 'detailed', 'conversational', or 'json'
    
    Returns:
        Prompt string
    """
    if style == "simple":
        return """List all words visible in this image, one per line."""
    
    elif style == "conversational":
        # Most natural - mimics interactive chat
        return """Can you do OCR of this image? Please extract all visible text."""
    
    elif style == "detailed":
        return """Carefully read and transcribe all visible text from this image.
        
Pay close attention to:
- Spelling and accents
- Handwritten characters
- Similar-looking letters (e, o, a)

List each word on a separate line."""
    
    elif style == "json":
        return """Extract all visible text from this image as a JSON array of words.

Return ONLY a JSON array like: ["word1", "word2", "word3"]

If no text is visible, return: []"""
    
    else:
        return """Can you do OCR of this image? Please extract all visible text."""


def test_deepseek_ocr(image_path: str, url: str = "http://localhost:11434", 
                      model: str = "deepseek-ocr:latest", max_width: int = 800,
                      prompt_style: str = "simple", quiet: bool = False):
    """
    Test OCR on a single image using any Ollama vision model.
    
    Args:
        image_path: Path to image file
        url: Ollama server URL
        model: Model name to use
        max_width: Maximum image width for resizing
        prompt_style: Prompt style ('simple', 'detailed', or 'json')
        quiet: If True, only output words (for piping)
    """
    if not quiet:
        print("=== Ollama Vision OCR Test ===")
        print(f"Model: {model}")
        print(f"Image: {image_path}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if model is available
    model_available = check_model_available(url, model)
    if not model_available:
        if not quiet:
            print(f"⚠️ Warning: Model '{model}' not found in available models.")
            print(f"You may need to pull it first: ollama pull {model}\n")
        return []
    
    # Verify model health (skip if quiet mode)
    if not quiet and not verify_model_health(url, model):
        print(f"❌ Model '{model}' failed health check. The model may be corrupted.")
        print(f"\nRecommended fix:")
        print(f"1. Remove the corrupted model: ollama rm {model}")
        print(f"2. Re-pull the model: ollama pull {model}")
        print(f"3. Run this test again\n")
        return []
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    # Get image info
    try:
        width, height, channels, img_format, file_size_mb = get_image_info(image_path)
        print(f"Image info: [{width}, {height}, {channels}] {img_format} {file_size_mb:.4f}MB")
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return
    
    # Resize and encode image
    print("Encoding image...")
    try:
        image_base64 = resize_image_for_ocr(image_path, max_width=max_width)
        encoded_size_mb = len(image_base64) / (1024 * 1024)
        print(f"Encoded size: {encoded_size_mb:.4f}MB")
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return
    
    # Build prompt based on style
    prompt = get_ocr_prompt(prompt_style)
    
    print(f"Prompt style: {prompt_style}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Estimated text prompt tokens: ~{len(prompt) // 4}")
    print()
    
    # Prepare request
    api_url = f"{url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 512,  # Reduced to prevent runaway repetition
            "temperature": 0.1,  # Slightly higher for more natural output
            "top_k": 10,  # Increased from 1 to allow more variety
            "top_p": 0.9,  # Reduced from 1 for more focused output
            "repeat_penalty": 1.2,  # Increased to strongly discourage repetition
            "stop": ["\n\n\n", "</s>", "---"],  # Different stop sequences
            "num_gpu": -1,
            "num_batch": 1024
        }
    }
    
    print("Model parameters:")
    print(f"  num_ctx: {payload['options']['num_ctx']}")
    print(f"  num_predict: {payload['options']['num_predict']}")
    print(f"  temperature: {payload['options']['temperature']}")
    print(f"  top_k: {payload['options']['top_k']}")
    print(f"  top_p: {payload['options']['top_p']}")
    print(f"  repeat_penalty: {payload['options']['repeat_penalty']}")
    print()
    
    # Make request - increased timeout to 10 minutes
    print("Processing with Ollama Vision OCR...")
    print("(This may take several minutes for larger models...)")
    start_time = time.time()
    
    try:
        response = requests.post(api_url, json=payload, timeout=600)  # 10 minutes
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        if not quiet:
            print(f"✓ Completed in {elapsed_time:.2f}s\n")
        
        # Check if model hit token limit
        done_reason = result.get('done_reason', '')
        if done_reason == 'length':
            print(f"⚠️ Warning: Model hit token limit (done_reason: {done_reason})")
            print("  Consider reducing num_predict or simplifying the prompt\n")
        
        # Extract response
        response_text = result.get('response', '')
        thinking_text = result.get('thinking', '')
        
        # Show thinking if available
        if thinking_text:
            print("=== Model Thinking ===")
            print(thinking_text)
            print()
        
        # Show raw response
        print("=== Raw Response ===")
        print(f"Response type: {type(response_text)}")
        print(f"Response length: {len(response_text)} characters")
        print(f"Response repr (first 200 chars): {repr(response_text[:200])}")
        print()
        print("Full response:")
        print(response_text)
        print()
        
        # Try to extract words - handle both JSON and plain text formats
        words = []
        
        # First try JSON format
        try:
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            
            if start != -1 and end > start:
                json_str = response_text[start:end]
                print(f"Found JSON array, extracting...")
                print(f"Extracted JSON string (first 200 chars): {json_str[:200]}")
                words = json.loads(json_str)
                if isinstance(words, list):
                    words = [str(w).strip() for w in words if w]
                    print(f"Parsed {len(words)} words from JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Could not parse as JSON: {e}")
        
        # Fallback: parse as line-separated words
        if not words or done_reason == 'length':
            print(f"Attempting line-by-line extraction...")
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Skip common intro/explanation lines
                skip_patterns = [
                    "here's", "here is", "the words", "visible in", 
                    "from the image", "list of", "following",
                    "---", "===", "```"
                ]
                if any(pattern in line.lower() for pattern in skip_patterns):
                    continue
                
                # Remove common prefixes (bullets, numbers, markdown)
                # Handle: *, -, •, 1., 1), etc.
                line = line.lstrip('*-•→►▪▫ ')  # Remove bullet points
                line = line.lstrip('0123456789.)> ')  # Remove numbering
                line = line.strip()  # Clean up again
                
                # Skip if now empty or too short
                if not line or len(line) < 2:
                    continue
                    
                # If line contains multiple words separated by spaces, split them
                # But be careful with phrases like "l'amour" or hyphenated words
                if ' ' in line and not any(c in line for c in ["'", "-", "'"]):
                    # Multiple separate words on one line
                    for word in line.split():
                        word = word.strip()
                        if word and len(word) > 1:
                            words.append(word)
                else:
                    # Single word or phrase
                    words.append(line)
            
            print(f"Extracted {len(words)} words from line-separated format")
            
            # Show a sample of extracted words if verbose
            if words and len(words) <= 10:
                print(f"Sample extracted words: {', '.join(words[:10])}")
            elif words:
                print(f"First 10 extracted words: {', '.join(words[:10])}")
        
        # Detect repetitive output (sign of model failure)
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = len(words) / unique_words if unique_words > 0 else 0
            if repetition_ratio > 5:
                print(f"\n⚠️ Warning: High repetition detected (ratio: {repetition_ratio:.1f})")
                print(f"  Unique words: {unique_words} / Total: {len(words)}")
                print(f"  This suggests the model struggled with the task")
                # Deduplicate
                words = list(dict.fromkeys(words))  # Preserve order while removing dupes
                print(f"  After deduplication: {len(words)} words\n")
        
        print()
        
        # Show extracted words
        print("=== Extracted Words ===")
        if words:
            print(f"Found {len(words)} words:")
            for i, word in enumerate(words, 1):
                print(f"  {i}. {word}")
        else:
            print("(No words extracted)")
        print()
        
        # Show detailed model metadata
        print("=== Model Metadata ===")
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Done: {result.get('done', 'N/A')}")
        print(f"Done reason: {done_reason if done_reason else 'N/A'}")
        print(f"Created at: {result.get('created_at', 'N/A')}")
        print()
        
        # Token information
        if result.get('prompt_eval_count'):
            prompt_tokens = result.get('prompt_eval_count')
            text_prompt_tokens = len(prompt) // 4  # Rough estimate
            image_tokens = prompt_tokens - text_prompt_tokens
            print("Token breakdown:")
            print(f"  Total prompt tokens: {prompt_tokens}")
            print(f"  Text prompt tokens (est): ~{text_prompt_tokens}")
            print(f"  Image tokens (est): ~{image_tokens}")
        
        if result.get('eval_count'):
            print(f"  Response tokens: {result.get('eval_count')}")
        
        if result.get('prompt_eval_count') and result.get('eval_count'):
            total_tokens = result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
            print(f"  Total tokens (prompt + response): {total_tokens}")
        print()
        
        # Duration information (high precision)
        print("Duration breakdown:")
        if result.get('total_duration'):
            total_s = result.get('total_duration') / 1_000_000_000
            print(f"  Total duration: {total_s:.6f}s")
        if result.get('load_duration'):
            load_s = result.get('load_duration') / 1_000_000_000
            print(f"  Load duration: {load_s:.6f}s")
        if result.get('prompt_eval_duration'):
            prompt_eval_s = result.get('prompt_eval_duration') / 1_000_000_000
            print(f"  Prompt eval duration: {prompt_eval_s:.6f}s")
            if result.get('prompt_eval_count'):
                tokens_per_sec = result.get('prompt_eval_count') / prompt_eval_s
                print(f"    ({tokens_per_sec:.2f} tokens/sec)")
        if result.get('eval_duration'):
            eval_s = result.get('eval_duration') / 1_000_000_000
            print(f"  Eval duration: {eval_s:.6f}s")
            if result.get('eval_count'):
                tokens_per_sec = result.get('eval_count') / eval_s
                print(f"    ({tokens_per_sec:.2f} tokens/sec)")
        print()
        
        # Debug: show all response fields
        print("=== All Response Fields ===")
        for key in result.keys():
            val = result[key]
            if key not in ['response', 'thinking']:  # Already shown above
                print(f"  {key}: {val if not isinstance(val, str) or len(str(val)) < 100 else str(val)[:100] + '...'}")
        print()
        
        # Save result
        output_file = Path("output") / f"model_test_{model.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        result_dict = {
            'image_path': image_path,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'prompt': prompt,
            'raw_response': response_text,
            'thinking': thinking_text if thinking_text else None,
            'words': words,
            'word_count': len(words),
            'model_info': {
                'model': result.get('model'),
                'done': result.get('done'),
                'created_at': result.get('created_at'),
                'total_duration_s': result.get('total_duration', 0) / 1_000_000_000 if result.get('total_duration') else None,
                'load_duration_s': result.get('load_duration', 0) / 1_000_000_000 if result.get('load_duration') else None,
                'prompt_eval_count': result.get('prompt_eval_count'),
                'prompt_eval_duration_s': result.get('prompt_eval_duration', 0) / 1_000_000_000 if result.get('prompt_eval_duration') else None,
                'eval_count': result.get('eval_count'),
                'eval_duration_s': result.get('eval_duration', 0) / 1_000_000_000 if result.get('eval_duration') else None,
            },
            'image_info': {
                'width': width,
                'height': height,
                'channels': channels,
                'format': img_format,
                'size_mb': round(file_size_mb, 4),
                'encoded_size_mb': round(encoded_size_mb, 4)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Result saved to: {output_file}")
        
    except requests.exceptions.RequestException as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error after {elapsed_time:.2f}s: {e}\n")
        
        # Try to get more error details
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"\nServer error details:")
                print(f"  Status: {e.response.status_code}")
                print(f"  Error: {error_data.get('error', 'No error message')}")
                
                # Save error details
                error_file = Path("output") / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                error_file.parent.mkdir(exist_ok=True)
                
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'model': model,
                        'image_path': image_path,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'status_code': e.response.status_code,
                        'server_response': error_data
                    }, f, indent=2)
                
                print(f"\n✓ Error details saved to: {error_file}")
            except:
                print(f"\nRaw error response: {e.response.text[:500]}")
        
        print("\nTroubleshooting steps:")
        print("1. Check if Ollama is running: curl http://localhost:11434/api/tags")
        print(f"2. Verify model is pulled: ollama pull {model}")
        print(f"3. Try running the model directly: ollama run {model}")
        print("4. Check Ollama logs for more details")
        
        # Add specific guidance for corrupted models
        if "unable to load model" in str(e).lower() or "sha256" in str(e).lower():
            print("\n⚠️ This looks like a corrupted model error.")
            print("Recommended fix:")
            print(f"  1. Remove the corrupted model: ollama rm {model}")
            print(f"  2. Re-pull the model: ollama pull {model}")
            print(f"  3. Run this test again")


def main():
    """Main function."""
    import sys
    
    # Default values
    default_model = "gemma2:2b"
    default_image = "data/images/handwritten.jpeg"
    default_prompt_style = "conversational"
    
    # Parse command line arguments
    model = default_model
    image_path = default_image
    prompt_style = default_prompt_style
    quiet = False
    words_only = False
    
    # Handle flags
    args = []
    for arg in sys.argv[1:]:
        if arg in ['-q', '--quiet']:
            quiet = True
        elif arg in ['-w', '--words-only']:
            words_only = True
        elif arg not in ['-h', '--help', 'help']:
            args.append(arg)
    
    if len(args) > 0:
        model = args[0]
    
    if len(args) > 1:
        image_path = args[1]
    
    if len(args) > 2:
        prompt_style = args[2]
    
    # Show usage if help requested
    if sys.argv[1] if len(sys.argv) > 1 else "" in ['-h', '--help', 'help']:
        print("Usage: python src/test_model_ocr.py [OPTIONS] [MODEL] [IMAGE_PATH] [PROMPT_STYLE]")
        print()
        print("Options:")
        print("  -q, --quiet       Suppress verbose output")
        print("  -w, --words-only  Output only words (one per line) for piping")
        print()
        print("Arguments:")
        print("  MODEL         Ollama model to use (default: gemma2:2b)")
        print("                Examples: gemma3:4b, deepseek-ocr:latest, qwen2-vl:2b")
        print("  IMAGE_PATH    Path to image file (default: data/images/handwritten.jpeg)")
        print("  PROMPT_STYLE  Prompt style (default: conversational)")
        print("                Options: conversational, simple, detailed, json")
        print()
        print("Examples:")
        print("  # Normal usage")
        print("  python src/test_model_ocr.py gemma3:4b data/images/test.jpg")
        print()
        print("  # Quiet mode (less output)")
        print("  python src/test_model_ocr.py -q gemma3:4b data/images/test.jpg")
        print()
        print("  # Pipe to ocr_to_json.py")
        print("  python src/test_model_ocr.py -w gemma3:4b data/images/test.jpg | python src/ocr_to_json.py -o notes.json")
        print()
        print("  # Complete pipeline")
        print("  python src/test_model_ocr.py --words-only gemma3:4b image.jpg | python src/ocr_to_json.py --pretty | python src/import_to_anki.py")
        return
    
    # Run OCR
    words = test_deepseek_ocr(
        image_path=image_path,
        url="http://localhost:11434",
        model=model,
        max_width=800,
        prompt_style=prompt_style,
        quiet=quiet or words_only
    )
    
    # Output words for piping
    if words_only and words:
        for word in words:
            print(word)


if __name__ == "__main__":
    main()
