#!/usr/bin/env python3
"""
Test Gemma 3 vision capabilities with the proper endpoint.
"""

import sys
import base64
import requests
import json
from pathlib import Path

def test_vision_properly():
    """Test vision using the confirmed working endpoint."""
    
    print("Testing Gemma 3 Vision with Proper Configuration")
    print("=" * 50)
    
    # 1. Verify server is running
    try:
        props = requests.get("http://127.0.0.1:8080/props").json()
        has_vision = props.get("modalities", {}).get("vision", False)
        print(f"✅ Server running with vision={'enabled' if has_vision else 'disabled'}")
        
        if not has_vision:
            print("❌ Vision not enabled! Check --mmproj flag")
            return
    except:
        print("❌ Server not running. Start with:")
        print("   python src/llama_cpp_server.py")
        return
    
    # 2. Load test image
    image_path = Path("data/images/handwritten.jpeg")
    if not image_path.exists():
        print(f"Creating test image...")
        # Create a simple test image with PIL
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test OCR\nHello World\nGemma Vision", fill='black')
        image_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(image_path)
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"📷 Image: {image_path}")
    print(f"   Size: {len(image_base64)/1024:.1f} KB (base64)")
    print()
    
    # 3. Test with /v1/chat/completions (OpenAI format)
    print("Test 1: Chat Completions Endpoint (OpenAI format)")
    print("-" * 50)
    
    request_data = {
        "model": "/home/gespitia/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text do you see in this image? Please list every word."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json=request_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Response: {content[:200]}")
            
            # Check if it's actually reading the image
            if any(word in content.lower() for word in ["see", "image", "text", "shows"]):
                print("   → Response references visual content")
            else:
                print("   → Response might be generic")
        else:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print()
    
    # 4. Alternative: Try raw completion with image embedding
    print("Test 2: Direct Completion with Image Token")
    print("-" * 50)
    
    # Use the exact format from the chat template
    prompt = f"""<bos><start_of_turn>user
<start_of_image>
What text is written in this image?<end_of_turn>
<start_of_turn>model
"""
    
    request_data = {
        "prompt": prompt,
        "image_data": [{"data": image_base64, "id": 1}],  # Try with id field
        "n_predict": 200,
        "temperature": 0.1,
        "stop": ["<end_of_turn>"]
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8080/completion",
            json=request_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", "")
            print(f"✅ Response: {content[:200]}")
        else:
            print(f"❌ Error {response.status_code}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_vision_properly()
