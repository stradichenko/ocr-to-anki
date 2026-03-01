#!/usr/bin/env env python3
"""
Image preprocessing script for OCR optimization using FFmpeg.
Applies various filters to improve text recognition for LLM-based OCR.
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml
import argparse


def load_settings():
    """Load settings from YAML configuration file."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        settings = yaml.safe_load(f)
    return settings.get('image_preprocessing', {})


def get_ffmpeg_filter_chain(settings):
    """Build FFmpeg filter chain based on settings."""
    filters = []
    
    # Grayscale conversion
    if settings.get('grayscale', True):
        filters.append("format=gray")
    
    # Contrast and brightness
    if settings.get('enhance_contrast', True):
        contrast = settings.get('contrast', 1.8)
        brightness = settings.get('brightness', 0.05)
        filters.append(f"eq=contrast={contrast}:brightness={brightness}")
    
    # Noise reduction
    if settings.get('denoise', True):
        denoise_strength = settings.get('denoise_strength', 'medium')
        denoise_params = {
            'light': '2:1:3:2',
            'medium': '4:3:6:4.5',
            'heavy': '6:5:8:6'
        }
        filters.append(f"hqdn3d={denoise_params.get(denoise_strength, '4:3:6:4.5')}")
    
    # Sharpening
    if settings.get('sharpen', True):
        sharpen_strength = settings.get('sharpen_strength', 1.2)
        filters.append(f"unsharp=5:5:{sharpen_strength}:5:5:0.0")
    
    # Scaling
    if settings.get('scale', True):
        scale_width = settings.get('scale_width', 2048)
        filters.append(f"scale={scale_width}:-1")
    
    # Rotation (deskew)
    if settings.get('rotate', False):
        rotation = settings.get('rotation_degrees', 0)
        if rotation != 0:
            filters.append(f"rotate={rotation}*PI/180")
    
    # Threshold/Binarization
    if settings.get('binarize', False):
        threshold = settings.get('threshold_value', 0.5)
        filters.append(f"threshold")
    
    return ",".join(filters)


def preprocess_image(input_path, output_path, settings, verbose=False):
    """
    Preprocess image using FFmpeg with configured filters.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        settings: Dictionary with preprocessing settings
        verbose: Print FFmpeg output
    """
    filter_chain = get_ffmpeg_filter_chain(settings)
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', filter_chain,
    ]
    
    # Output quality
    quality = settings.get('output_quality', 2)
    cmd.extend(['-q:v', str(quality)])
    
    # Output format
    output_format = settings.get('output_format', 'png')
    if output_format == 'png':
        cmd.extend(['-c:v', 'png'])
    
    # Overwrite output
    cmd.extend(['-y', str(output_path)])
    
    if verbose:
        print(f"Running FFmpeg command:")
        print(" ".join(cmd))
        print(f"\nFilter chain: {filter_chain}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if verbose:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        if verbose:
            print(f"FFmpeg stderr: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.", file=sys.stderr)
        return False


def process_directory(input_dir, output_dir, settings, verbose=False):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    images = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(images)} images...")
    success_count = 0
    
    for img_file in images:
        output_file = output_path / f"processed_{img_file.stem}.png"
        print(f"Processing: {img_file.name} -> {output_file.name}")
        
        if preprocess_image(img_file, output_file, settings, verbose):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(images)} images processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess images for OCR using FFmpeg'
    )
    parser.add_argument(
        'input',
        help='Input image file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file or directory (default: processed_<input>)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed FFmpeg output'
    )
    parser.add_argument(
        '--config',
        help='Path to custom settings.yaml file'
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings()
    
    # Determine input type
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Process
    if input_path.is_file():
        # Single file processing
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"processed_{input_path.name}"
        
        print(f"Processing: {input_path}")
        if preprocess_image(input_path, output_path, settings, args.verbose):
            print(f"Success! Output saved to: {output_path}")
        else:
            print("Processing failed", file=sys.stderr)
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory processing
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path.parent / f"processed_{input_path.name}"
        
        process_directory(input_path, output_dir, settings, args.verbose)
    
    else:
        print(f"Error: {args.input} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()