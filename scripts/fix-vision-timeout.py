#!/usr/bin/env python3
"""
Fix vision timeout issues by optimizing image handling and request format.
"""

import sys
import base64
import requests
import json
import time
from pathlib import Path
from PIL import Image
import io

def optimize_image(image_path: str, max_size: int = 512) -> str:
    """Optimize image for faster vision processing."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (vision models work better with smaller images)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save as JPEG with compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=70, optimize=True)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')

def test_vision_optimized():
    """Test vision with optimized settings to avoid timeouts."""
    
    print("Optimized Vision Test for Gemma 3")
    print("=" * 50)
    
    # 1. Check server
    try:
        props = requests.get("http://127.0.0.1:8080/props", timeout=5).json()
        if not props.get("modalities", {}).get("vision", False):
            print("❌ Vision not enabled on server!")
            return
        print("✅ Server has vision enabled")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return
    
    # 2. Prepare optimized image
    image_path = Path("data/images/handwritten.jpeg")
    if not image_path.exists():
        print(f"Creating test image at {image_path}...")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a simple test image
        img = Image.new('RGB', (200, 100), color='white')
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test\nVision\nOCR", fill='black')
        img.save(image_path)
    
    print(f"📷 Optimizing image: {image_path}")
    
    # Test different image sizes to find what works
    for max_size in [256, 512, 768]:
        print(f"\nTest with {max_size}x{max_size} max size:")
        print("-" * 40)
        
        image_base64 = optimize_image(str(image_path), max_size=max_size)
        print(f"  Image size: {len(image_base64)/1024:.1f} KB (base64)")
        
        # Use the simplest possible prompt
        request_data = {
            "prompt": f"<bos><start_of_turn>user\n<start_of_image>\nOCR<end_of_turn>\n<start_of_turn>model\n",
            "image_data": [{"data": image_base64}],
            "n_predict": 50,  # Limit output for faster response
            "temperature": 0.1,
            "stop": ["<end_of_turn>"],
            "cache_prompt": False
        }
        
        print(f"  Sending request...")
        start_time = time.time()
        
        try:
            # Use streaming to get partial results faster
            response = requests.post(
                "http://127.0.0.1:8080/completion",
                json=request_data,
                timeout=30,  # Shorter timeout
                stream=False
            )
            
            if response.status_code == 200:
                result = response.json()
                elapsed = time.time() - start_time
                content = result.get("content", "")
                
                print(f"  ✅ Response in {elapsed:.1f}s")
                print(f"  Result: {content[:100]}...")
                
                # If we got a response, this size works!
                print(f"\n🎉 SUCCESS! Use {max_size}x{max_size} images for vision")
                return max_size
            else:
                print(f"  ❌ Error {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"  ⏱️ Timeout after 30s - image too large")
            continue
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("Recommendations:")
    print("1. Use smaller images (256x256 or 512x512)")
    print("2. Keep prompts short")
    print("3. Limit output tokens")
    print("4. Consider batch processing for multiple images")

if __name__ == "__main__":
    test_vision_optimized()
