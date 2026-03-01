#!/usr/bin/env python3
"""
CLI script to perform OCR on an image or batch of images using Tesseract.
"""

import argparse
import sys
from pathlib import Path
import yaml

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Please install: pip install pytesseract pillow")
    sys.exit(1)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}


def perform_ocr(image_path: str, language: str = None, config: dict = None) -> str:
    """
    Perform OCR on the given image.
    
    Args:
        image_path: Path to the image file
        language: Language code for OCR (e.g., 'eng', 'jpn', 'spa')
        config: Configuration dictionary
    
    Returns:
        Extracted text from the image
    """
    if config is None:
        config = {}
    
    ocr_config = config.get('ocr', {})
    
    # Set tesseract command if specified
    tesseract_cmd = ocr_config.get('tesseract_cmd', 'tesseract')
    if tesseract_cmd != 'tesseract':
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    # Use provided language or default from config
    if language is None:
        language = ocr_config.get('default_language', 'eng')
    
    # Build tesseract config string
    psm = ocr_config.get('psm', 3)
    oem = ocr_config.get('oem', 3)
    additional_config = ocr_config.get('config', '')
    tesseract_config = f'--psm {psm} --oem {oem} {additional_config}'.strip()
    
    try:
        # Load and process image
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language, config=tesseract_config)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")


def get_image_files(directory: Path, recursive: bool = False) -> list:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
    
    Returns:
        List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
    
    if recursive:
        image_files = [f for f in directory.rglob('*') if f.suffix.lower() in image_extensions]
    else:
        image_files = [f for f in directory.glob('*') if f.suffix.lower() in image_extensions]
    
    return sorted(image_files)


def process_batch(input_dir: Path, language: str = None, config: dict = None, 
                  verbose: bool = False, save_output: bool = False, recursive: bool = False):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory path
        language: Language code for OCR
        config: Configuration dictionary
        verbose: Enable verbose output
        save_output: Save OCR results to text files
        recursive: Process subdirectories recursively
    """
    ocr_config = config.get('ocr', {})
    output_dir = None
    
    if save_output:
        output_dir = Path(ocr_config.get('output_dir', 'data/processed_images'))
        output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = get_image_files(input_dir, recursive)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    print("-" * 50)
    
    for idx, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            text = perform_ocr(str(image_path), language, config)
            
            if verbose:
                print(f"Extracted {len(text)} characters")
            
            if save_output and text:
                output_file = output_dir / f"{image_path.stem}.txt"
                output_file.write_text(text, encoding='utf-8')
                if verbose:
                    print(f"Saved to: {output_file}")
            
            if not save_output:
                print(f"\n--- {image_path.name} ---")
                print(text if text else "No text detected")
        
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}", file=sys.stderr)
            continue
    
    print("\n" + "-" * 50)
    print(f"Batch processing complete: {len(image_files)} image(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Perform OCR on an image or batch of images using Tesseract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python ocr_image.py image.png
  python ocr_image.py image.jpg -l jpn
  
  # Process all images from input folder (from config)
  python ocr_image.py --batch
  python ocr_image.py --batch -l jpn --save
  python ocr_image.py --batch --recursive
  
  # Process images from custom folder
  python ocr_image.py --batch --input-dir custom/folder
        """
    )
    
    parser.add_argument(
        'image',
        type=str,
        nargs='?',
        help='Path to the image file (required unless --batch is used)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all images from the input directory specified in config'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Override input directory from config (use with --batch)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process subdirectories recursively (use with --batch)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save OCR results to text files in output directory (use with --batch)'
    )
    
    parser.add_argument(
        '-l', '--language',
        type=str,
        default=None,
        help='Language code for OCR (e.g., eng, jpn, spa, eng+jpn for multiple). Defaults to config setting.'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    ocr_config = config.get('ocr', {})
    
    # Batch processing mode
    if args.batch:
        input_dir = Path(args.input_dir) if args.input_dir else Path(ocr_config.get('input_dir', 'data/images'))
        
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        
        if not input_dir.is_dir():
            print(f"Error: Path is not a directory: {input_dir}", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            lang = args.language or ocr_config.get('default_language', 'eng')
            print(f"Batch processing mode")
            print(f"Input directory: {input_dir}")
            print(f"Language: {lang}")
            print(f"Recursive: {args.recursive}")
            print(f"Save output: {args.save}")
        
        try:
            process_batch(input_dir, args.language, config, args.verbose, args.save, args.recursive)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Single image mode
    else:
        if not args.image:
            parser.error("the following arguments are required: image (or use --batch)")
        
        # Validate image file exists
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        
        if not image_path.is_file():
            print(f"Error: Path is not a file: {args.image}", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            lang = args.language or ocr_config.get('default_language', 'eng')
            print(f"Processing image: {args.image}")
            print(f"Language: {lang}")
            print("-" * 50)
        
        try:
            # Perform OCR
            text = perform_ocr(str(image_path), args.language, config)
            
            # Print results
            if text:
                print(text)
            else:
                print("No text detected in image.")
            
            if args.verbose:
                print("-" * 50)
                print(f"Characters extracted: {len(text)}")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
