#!/usr/bin/env python3
"""
Vision OCR using llama.cpp Gemma 3 4B (fully offline).
"""

import sys
import base64
from pathlib import Path
from PIL import Image
import io
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from llama_cpp_server import LlamaCppServer


def optimize_image(image_path: str, max_size: int = 512) -> str:
    """Optimize image for faster processing."""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def extract_text(image_path: str) -> dict:
    """Extract text from image using vision model."""
    
    with LlamaCppServer(verbose=True) as server:
        if not server.has_vision:
            raise RuntimeError(
                f"Vision projector not found at: {server.mmproj_path}\n"
                f"Run: ./scripts/setup-llama-cpp.sh"
            )
        
        print(f"Processing: {image_path}")
        image_base64 = optimize_image(image_path, max_size=512)
        
        result = server.generate(
            prompt="Extract all visible text. List each word.",
            image_data=image_base64,
            max_tokens=256,
            temperature=0.1,
            timeout=60
        )
        
        return {
            'image': image_path,
            'text': result['content'],
            'timestamp': datetime.now().isoformat()
        }


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/vision_ocr.py <image_path>")
        sys.exit(1)
    
    result = extract_text(sys.argv[1])
    
    # Save result
    output_dir = Path("data/ocr_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{Path(result['image']).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Result saved: {output_file}")
    print(f"\nExtracted text:\n{result['text']}")


if __name__ == '__main__':
    main()
