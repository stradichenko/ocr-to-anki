#!/usr/bin/env python3
"""
Vision OCR using llama-mtmd-cli with LOCAL files.
NO downloads, NO HuggingFace - uses cached local model + mmproj.
NO tesseract, NO easyOCR - pure Gemma 3 vision.
"""

import subprocess
import sys
import os
from pathlib import Path
from PIL import Image
import time


def ocr_with_local_files(
    image_path: str,
    prompt: str = "What text do you see in this image? Extract all visible text.",
    model_path: str = None,
    mmproj_path: str = None
) -> str:
    """
    Use llama-mtmd-cli with LOCAL model and mmproj files.
    NO downloads, fully offline.
    
    Args:
        image_path: Path to image
        prompt: OCR prompt
        model_path: Path to local GGUF model
        mmproj_path: Path to local mmproj file
    """
    # Get default paths
    models_dir = Path.home() / '.cache' / 'llama.cpp' / 'models'
    model_path = model_path or (models_dir / 'gemma-3-4b-it-q4_0.gguf')
    mmproj_path = mmproj_path or (models_dir / 'mmproj-model-f16-4B.gguf')
    
    print(f"=== Gemma 3 Vision OCR (Local Files) ===")
    print(f"Model: {model_path}")
    print(f"Projector: {mmproj_path}")
    print()
    
    # Check files exist
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print()
        print("Download with:")
        print("  ./scripts/setup-llama-cpp.sh")
        sys.exit(1)
    
    if not Path(mmproj_path).exists():
        print(f"❌ Vision projector not found: {mmproj_path}")
        print()
        print("Download with:")
        print("  ./scripts/setup-llama-cpp.sh")
        sys.exit(1)
    
    # Check image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Get image info
    img = Image.open(image_path)
    print(f"Image: {image_path}")
    print(f"Size: {img.size[0]}x{img.size[1]}, Mode: {img.mode}")
    img.close()
    print()
    
    # Build command using LOCAL files
    cmd = [
        'llama-mtmd-cli',
        '-m', str(model_path),
        '--mmproj', str(mmproj_path),
        '--image', image_path,
        '-p', prompt,
        '-n', '256',
        '--temp', '0.1',
        '--threads', '4',
        '--seed', '42',
    ]
    
    print("🔒 Using LOCAL files (fully offline)")
    print(f"Command: {' '.join(cmd[:6])}...")
    print()
    print("🔄 Processing image with Gemma 3 vision...")
    print("   ⏱️  Performance estimates (CPU):")
    print("      • Image encoding: ~50-60 minutes")
    print("      • Text generation: ~4 minutes")
    print("      • Total: ~60 minutes per image")
    print()
    print("   Speed improvements:")
    print("      • Resize image to 448x448: ~15 minutes")
    print("      • Use GPU (if available): ~2-5 minutes")
    print()
    
    try:
        start = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours
        )
        
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"❌ Command failed with code {result.returncode}")
            print()
            if result.stderr:
                print("Error output:")
                print(result.stderr[:1000])
            if result.stdout:
                print("\nStdout:")
                print(result.stdout[:1000])
            return ""
        
        # Extract generated text
        output = result.stdout.strip()
        
        # Remove system messages
        lines = output.split('\n')
        clean_lines = []
        skip_prefixes = ['build:', 'system', 'llama', 'clip', 'load_backend', '[', 'main:', 'image']
        
        for line in lines:
            # Skip if line starts with system prefix
            if any(line.strip().startswith(prefix) for prefix in skip_prefixes):
                continue
            # Skip if it's the prompt echo
            if prompt in line:
                continue
            # Skip performance metrics
            if 'ms' in line and ('per token' in line or 'eval time' in line):
                continue
            if line.strip():
                clean_lines.append(line)
        
        result_text = '\n'.join(clean_lines)
        
        print(f"✅ Processed in {elapsed/60:.1f} minutes")
        print()
        
        return result_text
    
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 2 hours")
        print()
        print("Troubleshooting:")
        print("  1. Try smaller image (resize to 448x448)")
        print("  2. Use GPU if available")
        print("  3. Check if model/mmproj files are corrupted")
        raise
    
    except FileNotFoundError:
        print("❌ llama-mtmd-cli not found!")
        print()
        print("Make sure you're in Nix environment:")
        print("  nix develop")
        print("  which llama-mtmd-cli")
        raise


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OCR with Gemma 3 vision using local files - NO tesseract, fully offline'
    )
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--prompt', 
                       default='What text do you see in this image? List every word.',
                       help='OCR prompt')
    parser.add_argument('--model', help='Path to GGUF model (optional)')
    parser.add_argument('--mmproj', help='Path to mmproj file (optional)')
    
    args = parser.parse_args()
    
    try:
        result = ocr_with_local_files(
            args.image,
            prompt=args.prompt,
            model_path=args.model,
            mmproj_path=args.mmproj
        )
        
        print("=" * 70)
        print("OCR RESULT")
        print("=" * 70)
        print(result if result else "(No text detected or model hallucinated)")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
