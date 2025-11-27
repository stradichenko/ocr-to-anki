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
import sys


class TeeOutput:
    """Captures stdout/stderr and writes to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


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
    
    # Use conversational style - more natural for models
    prompt = "Can you do OCR of this image? "
    
    # Add specific instructions based on config
    if text_type == "handwritten":
        prompt += "This is handwritten text. "
    elif text_type == "printed":
        prompt += "This is printed/typed text. "
    
    if analysis_scope == "highlighted":
        prompt += f"Please extract only the text highlighted in {highlight_color}. "
    else:
        prompt += "Please extract all visible text. "
    
    if language != "detect":
        prompt += f"The text is in {language}. "
    
    # Request structured output for easier parsing
    prompt += "\nPlease list each word on a separate line, or provide a JSON array of words."
    
    return prompt


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
    
    verbose = ollama_config.get('verbose_logging', False)
    save_raw_request = ollama_config.get('save_raw_request', False)
    save_model_info = ollama_config.get('save_model_info', False)
    
    # Get and print image shape
    try:
        width, height, channels, img_format, file_size_mb = get_image_shape(image_path)
        print(f"  Image: [{width}, {height}, {channels}] {img_format} {file_size_mb:.4f}MB")
    except Exception as e:
        print(f"  Warning: Could not read image info: {e}")
        width = height = channels = None
        img_format = "Unknown"
        file_size_mb = 0
    
    # Resize and encode image
    max_width = ollama_config.get('max_image_width', 600)
    image_base64 = resize_image_for_ocr(image_path, max_width=max_width)
    encoded_size_mb = len(image_base64) / (1024 * 1024)
    print(f"  Encoded: {encoded_size_mb:.4f}MB")
    
    # Build prompt
    prompt = build_ocr_prompt(config)
    
    if verbose:
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Image size: {len(image_base64)} bytes")
        # Estimate text prompt tokens (rough estimate: ~4 chars per token)
        estimated_prompt_tokens = len(prompt) // 4
        print(f"  Estimated text prompt tokens: ~{estimated_prompt_tokens}")
    
    # tune model parameters for OCR
    payload = {
        "model": ollama_config['model'],
        "prompt": prompt,
        "images": [image_base64],
        "stream": False, # iterative output not needed for OCR
        "options": {
            "num_ctx": 4096,  # Context window: prompt + response
            "num_predict": 2048,  # Increased max tokens - allow model to complete response
            "presence_penalty": 0.0,  # Penalizes tokens based on whether they've appeared at all (regardless of frequency). Discourage repetition.
            "temperature": 0.0,  # Lower temperature for more consistent output
            "top_k": 1,  # Top-K sampling: limits token pool (range: 1-100, default: 40). Lower means less random.
            "top_p": 1,  # Nucleus sampling: cumulative probability threshold (range: 0-1, default: 0.9). More means diverse output.
            "repeat_penalty": 1.05,  # Penalize token repetition (range: 0-2, default: 1.0, >1 reduces repetition)
            "stop": ["]\n\n", "</s>", "\n\n\n"],  # Multiple stop sequences to prevent rambling
            "num_gpu": -1,  # Use all available GPUs (RANGE: -1 to 8, default: -1)
            "num_batch": 1024  # Increase batch size for processing larger inputs efficiently (default: 512, Range: 1-2048)
            #"thinking": False,  # Enable 'thinking' field for debugging"
            #"reasoning": False  # Enable 'reasoning' field for debugging"
            #"enable_thinking": False  # Disables thinking mode
        }
    }
    
    # Add verbose flag to get more model info
    if verbose:
        payload["verbose"] = True
    
    # Make request with timing
    print(f"  Processing...")
    start_time = time.time()
    
    try:
        if verbose:
            print(f"  Sending request to {url}")
            print(f"  Model: {ollama_config['model']}")
        
        response = requests.post(
            url,
            json=payload,
            timeout=ollama_config['timeout']
        )
        
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        
        result = response.json()
        
        # DEBUG: Print the entire result structure
        if verbose:
            print(f"  Full result keys: {list(result.keys())}")
            print(f"  Full result (first 1000 chars): {str(result)[:1000]}")
            
            # Check what's actually in 'response' field
            if 'response' in result:
                resp = result['response']
                print(f"  'response' type: {type(resp)}")
                print(f"  'response' length: {len(resp) if resp else 0}")
                print(f"  'response' repr: {repr(resp)[:1000]}")
                
                # Check for non-printable characters
                if resp:
                    printable_chars = sum(1 for c in resp if c.isprintable())
                    total_chars = len(resp)
                    print(f"  Printable characters: {printable_chars}/{total_chars}")
                    
                    # Show hex dump of first 100 bytes
                    print(f"  First 100 bytes (hex): {resp[:100].encode('utf-8').hex()}")
        
        # Log model metadata if available
        if verbose and save_model_info:
            model_info = {
                'model': result.get('model'),
                'created_at': result.get('created_at'),
                'done': result.get('done'),
                'total_duration_s': result.get('total_duration', 0) / 1_000_000_000 if result.get('total_duration') else None,
                'load_duration_s': result.get('load_duration', 0) / 1_000_000_000 if result.get('load_duration') else None,
                'prompt_eval_count': result.get('prompt_eval_count'),
                'prompt_eval_duration_s': result.get('prompt_eval_duration', 0) / 1_000_000_000 if result.get('prompt_eval_duration') else None,
                'eval_count': result.get('eval_count'),
                'eval_duration_s': result.get('eval_duration', 0) / 1_000_000_000 if result.get('eval_duration') else None,
            }
            print(f"  Model info: {json.dumps(model_info, indent=2)}")
        
        # Extract words from response
        response_text = result.get('response', '')
        thinking_text = result.get('thinking', '')
        
        # Show thinking if verbose
        if verbose and thinking_text:
            print(f"  Model thinking (first 200 chars): {thinking_text[:200]}...")
            print(f"  Thinking length: {len(thinking_text)} characters")
        
        # DEBUG: Check all possible response fields
        if verbose:
            for key in ['response', 'text', 'output', 'content', 'message', 'choices', 'thinking']:
                if key in result:
                    val = result[key]
                    print(f"  Found key '{key}': type={type(val)}, len={len(str(val))}, preview={repr(str(val)[:100])}")
        
        # Debugging output
        print(f"  Response type: {type(response_text)}")
        print(f"  Response repr: {repr(response_text[:200])}")
        print(f"  Response bytes: {response_text.encode('utf-8')[:200]}")
        
        # Calculate image tokens
        if verbose and result.get('prompt_eval_count'):
            prompt_tokens = result.get('prompt_eval_count')
            text_prompt_tokens = len(prompt) // 4  # Rough estimate
            image_tokens = prompt_tokens - text_prompt_tokens
            print(f"  Prompt tokens breakdown:")
            print(f"    - Total prompt tokens: {prompt_tokens}")
            print(f"    - Text prompt tokens (est): ~{text_prompt_tokens}")
            print(f"    - Image tokens (est): ~{image_tokens}")
        
        words = extract_words_from_response(response_text)
        
        if verbose:
            print(f"  Response length: {len(response_text)} characters")
            print(f"  Words extracted: {len(words)}")
            if result.get('eval_count'):
                print(f"  Tokens generated: {result.get('eval_count')}")
            if result.get('prompt_eval_count'):
                print(f"  Prompt tokens: {result.get('prompt_eval_count')}")
            
            # Display special tokens and image tokens if available
            if result.get('prompt_eval_count') and result.get('eval_count'):
                total_tokens = result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                print(f"  Total tokens: {total_tokens}")
            
            # Display durations in seconds with high precision
            if result.get('total_duration'):
                total_s = result.get('total_duration') / 1_000_000_000
                print(f"  Total duration: {total_s:.6f}s")
            if result.get('load_duration'):
                load_s = result.get('load_duration') / 1_000_000_000
                print(f"  Load duration: {load_s:.6f}s")
            if result.get('prompt_eval_duration'):
                prompt_eval_s = result.get('prompt_eval_duration') / 1_000_000_000
                print(f"  Prompt eval duration: {prompt_eval_s:.6f}s")
            if result.get('eval_duration'):
                eval_s = result.get('eval_duration') / 1_000_000_000
                print(f"  Eval duration: {eval_s:.6f}s")
            
            # Show response preview if no words extracted
            if not words and response_text:
                print(f"  Response preview: {response_text[:200]}...")
        
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
            'thinking': thinking_text if thinking_text else None,  # ADD THIS
            'words': words,
            'word_count': len(words)
        }
        
        # Add model metadata if available
        if save_model_info:
            result_dict['model_info'] = {
                'model': result.get('model'),
                'total_duration_s': result.get('total_duration', 0) / 1_000_000_000 if result.get('total_duration') else None,
                'load_duration_s': result.get('load_duration', 0) / 1_000_000_000 if result.get('load_duration') else None,
                'prompt_eval_count': result.get('prompt_eval_count'),
                'prompt_eval_duration_s': result.get('prompt_eval_duration', 0) / 1_000_000_000 if result.get('prompt_eval_duration') else None,
                'eval_count': result.get('eval_count'),
                'eval_duration_s': result.get('eval_duration', 0) / 1_000_000_000 if result.get('eval_duration') else None,
            }
        
        # Save raw request if enabled
        if save_raw_request:
            result_dict['raw_request'] = {
                'url': url,
                'model': payload['model'],
                'prompt': payload['prompt'],
                'image_size_bytes': len(image_base64),
                'options': payload.get('options', {})
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
        
        if verbose:
            print(f"  Request failed after {elapsed_time:.2f}s")
            print(f"  Error type: {type(e).__name__}")
        
        result_dict = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'error': str(e),
            'error_type': type(e).__name__,
            'words': []
        }
        
        # Save raw request on error if enabled
        if save_raw_request:
            result_dict['raw_request'] = {
                'url': url,
                'model': payload['model'],
                'prompt': payload['prompt'],
                'image_size_bytes': len(image_base64),
                'options': payload.get('options', {})
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
    if not response_text or not response_text.strip():
        return []
    
    try:
        # Try to find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start != -1 and end > start:
            json_str = response_text[start:end]
            words = json.loads(json_str)
            if isinstance(words, list):
                # Clean and filter words from JSON
                cleaned_words = []
                for w in words:
                    if w:
                        cleaned = clean_word(str(w))
                        if cleaned:
                            cleaned_words.append(cleaned)
                return cleaned_words
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: aggressive text extraction
    words = []
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and common response patterns
        if not line:
            continue
        
        # Skip lines that look like explanations or markdown
        skip_patterns = [
            'the image', 'i see', 'i can see', 'this is', 'there is', 'there are',
            'the text', 'words are', 'contains', 'appears to', 'seems to',
            '#', '*', '**', '```', 'json', 'array', '---', '===',
            'note:', 'warning:', 'important:'
        ]
        
        if any(pattern in line.lower() for pattern in skip_patterns):
            continue
        
        # Skip lines that are pure JSON/code syntax
        if line in ['{', '}', '[', ']', ',']:
            continue
        
        # Extract actual words
        import re
        # Split by whitespace and common delimiters
        extracted = re.findall(r"[^\s,;|]+", line)
        
        for word in extracted:
            cleaned = clean_word(word)
            if cleaned:
                words.append(cleaned)
    
    return words


def clean_word(word: str) -> str:
    """
    Clean a word by removing unwanted punctuation and validating.
    
    Args:
        word: Raw word string
    
    Returns:
        Cleaned word or empty string if invalid
    """
    if not word:
        return ""
    
    # Remove leading/trailing quotes, brackets, parentheses
    word = word.strip('",[](){}*_-+=<>/')
    
    # Remove leading/trailing punctuation but preserve internal punctuation
    # Keep: apostrophes, hyphens, periods (for abbreviations)
    # Remove: leading/trailing ? ! . , : ;
    word = word.strip('?!.,;:')
    
    # Skip if word is now empty
    if not word:
        return ""
    
    # Skip if word is only punctuation
    if all(c in '?!.,;:+-=*/\\|@#$%^&*()[]{}' for c in word):
        return ""
    
    # Skip very short "words" that are likely noise (but keep 1-2 char words that have letters)
    if len(word) <= 2 and not any(c.isalpha() for c in word):
        return ""
    
    return word


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
    
    # Setup logging to file
    output_dir = Path(config['ollama_ocr']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Redirect stdout to both console and file
    tee = TeeOutput(log_file)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    verbose = config['ollama_ocr'].get('verbose_logging', False)
    
    try:
        print("=== Ollama OCR ===")
        print(f"Log file: {log_file}")
        print(f"Model: {config['ollama_ocr']['model']}")
        print(f"Text Type: {config['ollama_ocr']['text_type']}")
        print(f"Analysis Scope: {config['ollama_ocr']['analysis_scope']}")
        print(f"Language: {config['ollama_ocr']['language']}")
        print(f"Verbose logging: {verbose}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        if verbose:
            successful = [r for r in results if 'error' not in r]
            if successful:
                avg_time = sum(r['processing_time'] for r in successful) / len(successful)
                print(f"Average processing time: {avg_time:.6f}s")
                
                # If model_info available, show token stats
                with_tokens = [r for r in successful if 'model_info' in r and r['model_info'].get('eval_count')]
                if with_tokens:
                    avg_tokens = sum(r['model_info']['eval_count'] for r in with_tokens) / len(with_tokens)
                    print(f"Average tokens generated: {avg_tokens:.2f}")
                    
                    # Show average prompt tokens if available
                    with_prompt_tokens = [r for r in with_tokens if r['model_info'].get('prompt_eval_count')]
                    if with_prompt_tokens:
                        avg_prompt_tokens = sum(r['model_info']['prompt_eval_count'] for r in with_prompt_tokens) / len(with_prompt_tokens)
                        print(f"Average prompt tokens: {avg_prompt_tokens:.2f}")
        
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n✓ Log saved to: {log_file}")
    
    finally:
        # Restore stdout and close log file
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
