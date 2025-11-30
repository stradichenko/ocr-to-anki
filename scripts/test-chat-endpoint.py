#!/usr/bin/env python3
"""Test the chat completions endpoint with image."""

import json
import base64
import requests
from pathlib import Path

def test_chat_with_image():
    """Test /v1/chat/completions endpoint."""
    
    # Load image
    image_path = Path("data/images/handwritten.jpeg")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print("Testing /v1/chat/completions endpoint")
    print("=====================================")
    
    # Format as OpenAI-style request
    request_data = {
        "model": "gemma-3-4b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print("Sending multimodal chat request...")
    try:
        response = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            json=request_data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Response: {content}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_chat_with_image()
