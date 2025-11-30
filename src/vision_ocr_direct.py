#!/usr/bin/env python3
"""
Direct vision OCR using llama-cli (bypasses server issues).
"""

import subprocess
import base64
import tempfile
from pathlib import Path
from PIL import Image
import io


def ocr_with_llamacli(image_path: str) -> str:
    """
    Use llama-cli directly for vision OCR.
    This bypasses llama-server compatibility issues.
    """
    models_dir = Path.home() / '.cache' / 'llama.cpp' / 'models'
    model_path = models_dir / 'gemma-3-4b-it-q4_0.gguf'
    mmproj_path = models_dir / 'mmproj-model-f16-4B.gguf'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not mmproj_path.exists():
        raise FileNotFoundError(f"Vision projector not found: {mmproj_path}")
    
    # Prepare prompt
    prompt = "Extract all visible text from this image. List each word or phrase."
    
    # Run llama-cli with vision
    cmd = [
        'llama-cli',
        '--model', str(model_path),
        '--mmproj', str(mmproj_path),
        '--image', image_path,
        '--prompt', prompt,
        '--n-predict', '256',
        '--temp', '0.1',
        '--threads', '4',
    ]
    
    print(f"Running llama-cli with vision...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            check=True
        )
        
        # Extract just the generated text (after the prompt)
        output = result.stdout
        
        # llama-cli includes the prompt in output, remove it
        if prompt in output:
            output = output.split(prompt, 1)[1]
        
        return output.strip()
    
    except subprocess.CalledProcessError as e:
        print(f"❌ llama-cli failed:")
        print(f"   stdout: {e.stdout[:200]}")
        print(f"   stderr: {e.stderr[:200]}")
        raise RuntimeError(f"llama-cli vision failed: {e.stderr}")
    
    except FileNotFoundError:
        raise RuntimeError(
            "llama-cli not found. Make sure you're in the Nix environment:\n"
            "  nix develop"
        )


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/vision_ocr_direct.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    try:
        text = ocr_with_llamacli(image_path)
        print("\n" + "="*50)
        print("OCR Result:")
        print("="*50)
        print(text)
        print("="*50)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
