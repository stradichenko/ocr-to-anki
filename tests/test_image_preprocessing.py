"""
Image Preprocessing Test for OCR
Tests various image preprocessing combinations to evaluate OCR performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import base64

from ollama_ocr import load_config, perform_ocr


class ImagePreprocessor:
    """Handles various image preprocessing methods."""
    
    @staticmethod
    def resize_image(img: Image.Image, max_width: int = 800) -> Image.Image:
        """Resize image to max_width."""
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img.copy()
    
    @staticmethod
    def increase_contrast(img: Image.Image, factor: float = 2.0) -> Image.Image:
        """Increase image contrast."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def to_grayscale(img: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        return img.convert('L').convert('RGB')
    
    @staticmethod
    def detect_edges(img: Image.Image) -> Image.Image:
        """Apply edge detection using Canny algorithm."""
        # Convert PIL to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert back to RGB PIL Image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    @staticmethod
    def save_temp_image(img: Image.Image, temp_path: Path, name: str) -> str:
        """Save preprocessed image temporarily."""
        temp_path.mkdir(parents=True, exist_ok=True)
        file_path = temp_path / name
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img
        
        img.save(file_path, 'JPEG', quality=95)
        return str(file_path)


def generate_preprocessing_combinations() -> List[Tuple[str, List[str]]]:
    """
    Generate all preprocessing combinations to test.
    
    Returns:
        List of (combination_name, [preprocessing_steps])
    """
    combinations = [
        ("original", []),
        ("resized", ["resize"]),
        ("contrast", ["contrast"]),
        ("grayscale", ["grayscale"]),
        ("edges", ["edges"]),
        ("resized_contrast", ["resize", "contrast"]),
        ("resized_grayscale", ["resize", "grayscale"]),
        ("resized_edges", ["resize", "edges"]),
        ("contrast_grayscale", ["contrast", "grayscale"]),
        ("contrast_edges", ["contrast", "edges"]),
        ("grayscale_edges", ["grayscale", "edges"]),
        ("resized_contrast_grayscale", ["resize", "contrast", "grayscale"]),
        ("resized_contrast_edges", ["resize", "contrast", "edges"]),
    ]
    return combinations


def apply_preprocessing(image_path: str, steps: List[str], temp_dir: Path) -> str:
    """
    Apply preprocessing steps to an image.
    
    Args:
        image_path: Path to original image
        steps: List of preprocessing step names
        temp_dir: Directory to save temporary files
    
    Returns:
        Path to preprocessed image
    """
    if not steps:
        return image_path
    
    img = Image.open(image_path)
    processor = ImagePreprocessor()
    
    # Apply each preprocessing step in order
    for step in steps:
        if step == "resize":
            img = processor.resize_image(img, max_width=800)
        elif step == "contrast":
            img = processor.increase_contrast(img, factor=2.0)
        elif step == "grayscale":
            img = processor.to_grayscale(img)
        elif step == "edges":
            img = processor.detect_edges(img)
    
    # Save with descriptive name
    original_name = Path(image_path).stem
    combo_name = "_".join(steps) if steps else "original"
    temp_name = f"{original_name}_{combo_name}.jpg"
    
    return processor.save_temp_image(img, temp_dir, temp_name)


def build_test_ocr_prompt() -> str:
    """Build a simpler, more direct OCR prompt for testing."""
    prompt = """Extract all visible text from this image.

Return your response as a simple JSON array of words.
Example: ["word1", "word2", "word3"]

If there is no text, return: []

Do not include explanations. Only return the JSON array."""
    
    return prompt


def run_preprocessing_tests(config: Dict[str, Any], test_images: List[str]) -> List[Dict[str, Any]]:
    """
    Run OCR tests on all image and preprocessing combinations.
    
    Args:
        config: Configuration dictionary
        test_images: List of image paths to test
    
    Returns:
        List of test results
    """
    # Override config for testing - increase limits
    test_config = config.copy()
    test_config['ollama_ocr'] = config['ollama_ocr'].copy()
    
    # Increase timeout and token limits for thorough testing
    test_config['ollama_ocr']['timeout'] = 6000  # 10 minutes per image
    test_config['ollama_ocr']['max_image_width'] = 800  # Higher resolution
    test_config['ollama_ocr']['verbose_logging'] = True  # Always verbose for tests
    test_config['ollama_ocr']['save_model_info'] = True  # Always save model info
    
    combinations = generate_preprocessing_combinations()
    temp_dir = Path("data/temp_preprocessed")
    results = []
    
    total_tests = len(test_images) * len(combinations)
    current_test = 0
    
    print(f"\n{'='*80}")
    print(f"Running {total_tests} tests ({len(test_images)} images × {len(combinations)} preprocessing combinations)")
    print(f"Test Configuration:")
    print(f"  Timeout: {test_config['ollama_ocr']['timeout']}s")
    print(f"  Max Image Width: {test_config['ollama_ocr']['max_image_width']}px")
    print(f"  Model: {test_config['ollama_ocr']['model']}")
    print(f"  Using simplified test prompt")
    print(f"{'='*80}\n")
    
    for image_path in test_images:
        image_name = Path(image_path).name
        print(f"\n{'─'*80}")
        print(f"Testing image: {image_name}")
        print(f"{'─'*80}")
        
        for combo_name, steps in combinations:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] {combo_name}")
            print(f"  Steps: {' → '.join(steps) if steps else 'none'}")
            
            try:
                # Apply preprocessing
                preprocessed_path = apply_preprocessing(image_path, steps, temp_dir)
                
                # Run OCR with modified config
                ocr_result = perform_ocr_test(preprocessed_path, test_config)
                
                # Compile result
                result = {
                    'original_image': image_name,
                    'preprocessing': combo_name,
                    'preprocessing_steps': steps,
                    'preprocessed_path': preprocessed_path,
                    'word_count': ocr_result.get('word_count', 0),
                    'words': ocr_result.get('words', []),
                    'processing_time': ocr_result.get('processing_time', 0),
                    'error': ocr_result.get('error'),
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'model_info' in ocr_result:
                    result['model_info'] = ocr_result['model_info']
                
                # Add raw response for debugging
                if 'raw_response' in ocr_result:
                    result['raw_response'] = ocr_result['raw_response']
                    result['raw_response_preview'] = ocr_result['raw_response'][:500]
                
                results.append(result)
                
                print(f"  ✓ Words: {result['word_count']}, Time: {result['processing_time']}s")
                if result['words']:
                    print(f"  Words: {', '.join(result['words'][:10])}{'...' if len(result['words']) > 10 else ''}")
                else:
                    # Debug: show why no words were extracted
                    if 'raw_response' in ocr_result and ocr_result['raw_response']:
                        preview = ocr_result['raw_response'][:300].replace('\n', ' ')
                        print(f"  ⚠ No words extracted. Response preview: {preview}...")
                
            except Exception as e:
                import traceback
                print(f"  ✗ Error: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                results.append({
                    'original_image': image_name,
                    'preprocessing': combo_name,
                    'preprocessing_steps': steps,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'word_count': 0,
                    'words': [],
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat()
                })
    
    return results


def perform_ocr_test(image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform OCR for testing with simplified prompt and better debugging.
    
    Args:
        image_path: Path to the image file
        config: Configuration dictionary
    
    Returns:
        Dictionary containing OCR results and metadata
    """
    import requests
    import time
    from ollama_ocr import get_image_shape, resize_image_for_ocr, extract_words_from_response
    
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
    
    # Resize and encode image
    max_width = ollama_config.get('max_image_width', 800)
    image_base64 = resize_image_for_ocr(image_path, max_width=max_width)
    encoded_size_mb = len(image_base64) / (1024 * 1024)
    print(f"  Encoded: {encoded_size_mb:.4f}MB")
    
    # Use simplified test prompt
    prompt = build_test_ocr_prompt()
    print(f"  Prompt length: {len(prompt)} characters")
    
    # Prepare request payload
    payload = {
        "model": ollama_config['model'],
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 5000,
            "temperature": 0.1,
            "top_k": 40,
            "top_p": 0.9,
        }
    }
    
    print(f"  Processing...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=ollama_config['timeout'])
        elapsed_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        # Get response text
        response_text = result.get('response', '')
        
        # Debug: Print raw response info
        print(f"  Response bytes: {len(response_text.encode('utf-8'))}")
        print(f"  Response chars: {len(response_text)}")
        print(f"  Tokens generated: {result.get('eval_count', 0)}")
        print(f"  Total duration: {result.get('total_duration', 0) / 1_000_000:.0f}ms")
        
        # Show first 300 chars of response
        if response_text:
            preview = response_text[:300].replace('\n', '\\n')
            print(f"  Response start: {preview}...")
        else:
            print(f"  ⚠ Empty response from model!")
        
        # Extract words
        words = extract_words_from_response(response_text)
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'prompt': prompt,
            'raw_response': response_text,
            'words': words,
            'word_count': len(words),
            'model_info': {
                'model': result.get('model'),
                'total_duration_ms': result.get('total_duration', 0) / 1_000_000 if result.get('total_duration') else None,
                'prompt_eval_count': result.get('prompt_eval_count'),
                'eval_count': result.get('eval_count'),
            }
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ✗ Request failed: {e}")
        
        return {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(elapsed_time, 2),
            'error': str(e),
            'words': [],
            'word_count': 0
        }


def generate_statistical_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistical analysis report from test results.
    
    Args:
        results: List of test results
    
    Returns:
        Statistical report dictionary
    """
    from collections import defaultdict
    import statistics
    
    # Group results by image and by preprocessing method
    by_image = defaultdict(list)
    by_preprocessing = defaultdict(list)
    
    for result in results:
        if not result.get('error'):
            by_image[result['original_image']].append(result)
            by_preprocessing[result['preprocessing']].append(result)
    
    # Calculate statistics for each preprocessing method
    preprocessing_stats = {}
    for method, method_results in by_preprocessing.items():
        word_counts = [r['word_count'] for r in method_results]
        processing_times = [r['processing_time'] for r in method_results]
        
        preprocessing_stats[method] = {
            'samples': len(method_results),
            'word_count': {
                'mean': statistics.mean(word_counts) if word_counts else 0,
                'median': statistics.median(word_counts) if word_counts else 0,
                'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
                'total': sum(word_counts)
            },
            'processing_time': {
                'mean': statistics.mean(processing_times) if processing_times else 0,
                'median': statistics.median(processing_times) if processing_times else 0,
                'stdev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0,
                'total': sum(processing_times)
            }
        }
    
    # Calculate statistics for each image
    image_stats = {}
    for image, image_results in by_image.items():
        word_counts = [r['word_count'] for r in image_results]
        processing_times = [r['processing_time'] for r in image_results]
        
        # Find best preprocessing method for this image
        best_method = max(image_results, key=lambda x: x['word_count'])
        
        image_stats[image] = {
            'samples': len(image_results),
            'word_count': {
                'mean': statistics.mean(word_counts) if word_counts else 0,
                'median': statistics.median(word_counts) if word_counts else 0,
                'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0,
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
            },
            'processing_time': {
                'mean': statistics.mean(processing_times) if processing_times else 0,
            },
            'best_method': {
                'name': best_method['preprocessing'],
                'word_count': best_method['word_count'],
                'words': best_method['words']
            }
        }
    
    # Overall statistics
    all_word_counts = [r['word_count'] for r in results if not r.get('error')]
    all_times = [r['processing_time'] for r in results if not r.get('error')]
    
    overall_stats = {
        'total_tests': len(results),
        'successful_tests': len(all_word_counts),
        'failed_tests': len(results) - len(all_word_counts),
        'total_words_extracted': sum(all_word_counts),
        'total_processing_time': sum(all_times),
        'average_word_count': statistics.mean(all_word_counts) if all_word_counts else 0,
        'average_processing_time': statistics.mean(all_times) if all_times else 0
    }
    
    # Find best overall method
    best_overall = max(preprocessing_stats.items(), 
                      key=lambda x: x[1]['word_count']['mean'])
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall': overall_stats,
        'by_preprocessing_method': preprocessing_stats,
        'by_image': image_stats,
        'best_preprocessing_method': {
            'name': best_overall[0],
            'mean_word_count': best_overall[1]['word_count']['mean'],
            'mean_processing_time': best_overall[1]['processing_time']['mean']
        }
    }
    
    return report


def print_report(report: Dict[str, Any]):
    """Print formatted statistical report to console."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*80)
    
    # Overall Statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    overall = report['overall']
    print(f"Total Tests:           {overall['total_tests']}")
    print(f"Successful:            {overall['successful_tests']}")
    print(f"Failed:                {overall['failed_tests']}")
    print(f"Total Words Extracted: {overall['total_words_extracted']}")
    print(f"Total Processing Time: {overall['total_processing_time']:.2f}s")
    print(f"Avg Words per Test:    {overall['average_word_count']:.2f}")
    print(f"Avg Time per Test:     {overall['average_processing_time']:.2f}s")
    
    # Best Method
    print("\n🏆 BEST PREPROCESSING METHOD")
    print("-" * 80)
    best = report['best_preprocessing_method']
    print(f"Method:            {best['name']}")
    print(f"Avg Word Count:    {best['mean_word_count']:.2f}")
    print(f"Avg Process Time:  {best['mean_processing_time']:.2f}s")
    
    # By Preprocessing Method
    print("\n📈 RESULTS BY PREPROCESSING METHOD")
    print("-" * 80)
    methods = sorted(report['by_preprocessing_method'].items(),
                    key=lambda x: x[1]['word_count']['mean'],
                    reverse=True)
    
    for method_name, stats in methods:
        print(f"\n{method_name.upper()}")
        print(f"  Words:    Mean={stats['word_count']['mean']:.2f}, "
              f"Median={stats['word_count']['median']:.2f}, "
              f"Std={stats['word_count']['stdev']:.2f}")
        print(f"            Min={stats['word_count']['min']}, "
              f"Max={stats['word_count']['max']}, "
              f"Total={stats['word_count']['total']}")
        print(f"  Time:     Mean={stats['processing_time']['mean']:.2f}s, "
              f"Median={stats['processing_time']['median']:.2f}s")
    
    # By Image
    print("\n📷 RESULTS BY IMAGE")
    print("-" * 80)
    for image_name, stats in report['by_image'].items():
        print(f"\n{image_name}")
        print(f"  Words:       Mean={stats['word_count']['mean']:.2f}, "
              f"Min={stats['word_count']['min']}, "
              f"Max={stats['word_count']['max']}")
        print(f"  Time:        Avg={stats['processing_time']['mean']:.2f}s")
        print(f"  Best Method: {stats['best_method']['name']} "
              f"({stats['best_method']['word_count']} words)")
        if stats['best_method']['words']:
            words_preview = ', '.join(stats['best_method']['words'][:8])
            if len(stats['best_method']['words']) > 8:
                words_preview += '...'
            print(f"  Words:       {words_preview}")


def main():
    """Main test execution."""
    # Load configuration
    config = load_config()
    
    # Define test images
    test_images = [
        "data/images/1d21248b-9dda-4e67-972b-70aa96e35eee.jpeg",
        "data/images/handwritten.jpeg",
        "data/images/orange_purple.jpeg"
    ]
    
    # Verify images exist
    missing = [img for img in test_images if not Path(img).exists()]
    if missing:
        print(f"Error: Missing images: {missing}")
        return
    
    print("="*80)
    print("IMAGE PREPROCESSING TEST FOR OCR")
    print("="*80)
    print(f"Model: {config['ollama_ocr']['model']}")
    print(f"Test Images: {len(test_images)}")
    print(f"Preprocessing Methods: {len(generate_preprocessing_combinations())}")
    
    # Run tests
    results = run_preprocessing_tests(config, test_images)
    
    # Generate statistical report
    report = generate_statistical_report(results)
    
    # Print report
    print_report(report)
    
    # Save results
    output_dir = Path(config['ollama_ocr']['output_dir']) / "preprocessing_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = output_dir / f"test_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save statistical report
    report_file = output_dir / f"statistical_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("📁 FILES SAVED")
    print("="*80)
    print(f"Detailed Results: {results_file}")
    print(f"Statistical Report: {report_file}")
    print(f"Preprocessed Images: data/temp_preprocessed/")
    print("="*80)


if __name__ == "__main__":
    main()
