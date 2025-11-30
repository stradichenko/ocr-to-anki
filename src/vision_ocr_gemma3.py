#!/usr/bin/env python3
"""
Vision OCR using llama-mtmd-cli (Multi-Modal CLI for Gemma 3 vision).
NO tesseract, NO easyOCR - pure vision model OCR.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import io


def test_text_only(cli_binary: str, model_path: str, mmproj_path: str = None) -> bool:
    """Test text-only generation first to ensure model works."""
    print("Testing text-only generation first...")
    
    # llama-mtmd-cli REQUIRES --mmproj even for text
    cmd = [
        cli_binary,
        '-m', str(model_path),
        '--mmproj', str(mmproj_path),
        '-p', 'Hello!',  # Simpler prompt
        '-n', '5',       # Fewer tokens
        '--temp', '0.1',
        '--seed', '42',  # Reproducible
    ]
    
    try:
        print(f"Command: {' '.join(cmd)}")
        print("Loading model (this can take 30-60s on first run)...")
        
        start = time.time()
        # Give it more time for initial model loading
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes for initial load
        )
        elapsed = time.time() - start
        
        # Show output even if failed
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")
        
        if result.returncode != 0:
            print(f"❌ Command failed with code {result.returncode}")
            return False
        
        print(f"✅ Text generation worked! ({elapsed:.1f}s)")
        
        # Extract just the generated text (after the prompt)
        output = result.stdout.strip()
        if 'Hello!' in output:
            generated = output.split('Hello!', 1)[-1].strip()
            print(f"Generated: {generated[:100]}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Text generation timed out (model may be too slow)")
        print("\nTips:")
        print("  • First run is slower (model loading)")
        print("  • Try running directly: llama-mtmd-cli -m MODEL --mmproj MMPROJ -p 'Hi' -n 5")
        print("  • Consider using llama-server instead (stays loaded)")
        return False
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False


def ocr_with_gemma3_cli(
    image_path: str,
    prompt: str = "What text do you see?",
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    debug: bool = True,
    timeout: int = 600
) -> str:
    """
    Use llama-mtmd-cli for vision OCR.
    Note: llama-mtmd-cli REQUIRES --mmproj for ALL operations.
    """
    # Check for llama-mtmd-cli
    cli_binary = None
    for binary in ['llama-mtmd-cli', 'llama-gemma3-cli']:
        try:
            subprocess.run([binary, '--version'], 
                          capture_output=True, check=True, timeout=2)
            cli_binary = binary
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    if not cli_binary:
        raise RuntimeError(
            "llama-mtmd-cli not found!\n\n"
            "Build it with:\n"
            "  chmod +x scripts/build-llama-gemma3-cli.sh\n"
            "  ./scripts/build-llama-gemma3-cli.sh"
        )
    
    # Check image exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Get image size
    with Image.open(image_path) as img:
        print(f"Image info: {img.size[0]}x{img.size[1]}, mode={img.mode}")
    
    # Get default paths
    models_dir = Path.home() / '.cache' / 'llama.cpp' / 'models'
    default_model = models_dir / 'gemma-3-4b-it-q4_0.gguf'
    default_mmproj = models_dir / 'mmproj-model-f16-4B.gguf'
    
    model_path = Path(model_path) if model_path else default_model
    mmproj_path = Path(mmproj_path) if mmproj_path else default_mmproj
    
    # Verify files
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not mmproj_path.exists():
        raise FileNotFoundError(f"Vision projector not found: {mmproj_path}")
    
    print(f"\n=== Vision OCR Configuration ===")
    print(f"Binary: {cli_binary}")
    print(f"Model: {model_path.name} ({model_path.stat().st_size/(1024**3):.2f} GB)")
    print(f"Projector: {mmproj_path.name} ({mmproj_path.stat().st_size/(1024**3):.2f} GB)")
    print(f"Image: {image_path}")
    print(f"Timeout: {timeout}s")
    print()
    
    # Test text-only first with mmproj
    if not test_text_only(cli_binary, model_path, mmproj_path):
        print("\n⚠️  Text generation failed/slow. Trying vision anyway...")
    
    print(f"\nNow testing vision with prompt: {prompt}")
    print("=" * 50)
    
    # Build vision command - simpler format
    cmd = [
        cli_binary,
        '-m', str(model_path),
        '--mmproj', str(mmproj_path),
        '--image', image_path,
        '-p', prompt,
        '-n', '128',  # Reduced tokens
        '--temp', '0.1',
        '--seed', '42',
    ]
    
    if debug:
        print(f"Debug command: {' '.join(cmd)}")
    
    print(f"\n🔄 Processing image with vision model...")
    print(f"⏱️  This may take 2-5 minutes on CPU...")
    print(f"    (First run is slower due to model loading)")
    
    try:
        start = time.time()
        
        # Run with longer timeout for vision
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"\n❌ Command failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:1000]}")
            return ""
        
        print(f"\n✅ Vision processing complete in {elapsed:.1f}s")
        
        # Extract the generated text
        output = result.stdout.strip()
        
        # Remove the prompt echo if present
        if prompt in output:
            parts = output.split(prompt, 1)
            if len(parts) > 1:
                output = parts[1].strip()
        
        # Clean up system messages
        lines = output.split('\n')
        clean_lines = []
        for line in lines:
            # Skip system messages
            if line.startswith('[') or line.startswith('llama') or line.startswith('clip'):
                continue
            if 'build:' in line or 'system:' in line:
                continue
            if line.strip():
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  Timeout after {timeout}s")
        print("\nThis is normal for vision on CPU. Solutions:")
        print("1. Use llama-server instead (stays loaded):")
        print("   python src/llama_cpp_server.py")
        print("2. Use smaller images")
        print("3. Use text-only model with Tesseract OCR")
        raise
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def main():
    """CLI interface for vision OCR."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR using Gemma 3 vision - NO tesseract')
    parser.add_argument('image', nargs='?', help='Path to image file')
    parser.add_argument('--prompt', default='What text do you see in this image?',
                       help='Custom OCR prompt')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--test-cli', action='store_true', help='Just test the CLI')
    
    args = parser.parse_args()
    
    if args.test_cli:
        # Quick test of llama-mtmd-cli
        print("Testing llama-mtmd-cli...")
        models_dir = Path.home() / '.cache' / 'llama.cpp' / 'models'
        model = models_dir / 'gemma-3-4b-it-q4_0.gguf'
        mmproj = models_dir / 'mmproj-model-f16-4B.gguf'
        
        if test_text_only('llama-mtmd-cli', model, mmproj):
            print("\n✅ CLI works!")
        else:
            print("\n❌ CLI not working properly")
            print("\nAlternative: Use llama-server instead:")
            print("  python src/llama_cpp_server.py")
        
        sys.exit(0)
    
    if not args.image:
        parser.error("Image path required (or use --test-cli)")
    
    # Get paths
    models_dir = Path.home() / '.cache' / 'llama.cpp' / 'models'
    model_path = args.model or (models_dir / 'gemma-3-4b-it-q4_0.gguf')
    mmproj_path = args.mmproj or (models_dir / 'mmproj-model-f16-4B.gguf')
    
    try:
        result = ocr_with_gemma3_cli(
            args.image,
            prompt=args.prompt,
            model_path=model_path,
            mmproj_path=mmproj_path,
            debug=args.debug,
            timeout=args.timeout
        )
        
        print("\n" + "=" * 70)
        print("OCR RESULT")
        print("=" * 70)
        print(result if result else "(No text detected)")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Recommendation: Use llama-server instead")
        print("   It stays loaded in memory and is much faster:")
        print("   python src/llama_cpp_server.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
