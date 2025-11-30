#!/usr/bin/env python3
"""
Minimal test to diagnose vision model issues.
Saves detailed logs and extracts key information.
"""

import json
import base64
import requests
import time
from pathlib import Path
from datetime import datetime

def test_vision(image_path: str = "data/images/handwritten.jpeg", verbose: bool = True):
    """Test vision with detailed logging."""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"vision_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "image": str(image_path),
        "tests": []
    }
    
    # Check server
    print("1. Checking server health...")
    try:
        health = requests.get("http://127.0.0.1:8080/health", timeout=5).json()
        print(f"   ✅ Server status: {health.get('status')}")
        results["server_health"] = health
    except Exception as e:
        print(f"   ❌ Server not responding: {e}")
        results["server_health"] = {"error": str(e)}
        return
    
    # Get server properties
    print("2. Getting server properties...")
    try:
        props = requests.get("http://127.0.0.1:8080/props", timeout=5).json()
        vision_enabled = props.get("vision", False)
        print(f"   Vision support: {'✅ Yes' if vision_enabled else '❌ No'}")
        results["server_props"] = {
            "vision": vision_enabled,
            "multimodal": props.get("multimodal", False),
            "has_clip": "clip" in str(props).lower()
        }
    except Exception as e:
        print(f"   ❌ Could not get properties: {e}")
        results["server_props"] = {"error": str(e)}
    
    # Load and encode image
    print(f"3. Loading image: {image_path}")
    if not Path(image_path).exists():
        print(f"   ❌ Image not found!")
        return
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"   Image size: {len(image_base64) / 1024:.1f} KB (base64)")
    results["image_size_kb"] = len(image_base64) / 1024
    
    # Test prompts - simple and direct
    test_prompts = [
        ("simple", "Can you read this?"),
        ("ocr", "Please do OCR of this image."),
        ("direct", "What text is in the image?"),
    ]
    
    for test_name, prompt_text in test_prompts:
        print(f"\n4. Test '{test_name}': {prompt_text}")
        
        # Format request
        formatted_prompt = f"<bos><start_of_turn>user\n<start_of_image>\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        
        request_data = {
            "prompt": formatted_prompt,
            "image_data": [{"data": image_base64}],
            "n_predict": 200,
            "temperature": 0.1,
            "stop": ["<end_of_turn>"],
            "cache_prompt": False
        }
        
        # Send request
        print("   Sending request...")
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://127.0.0.1:8080/completion",
                json=request_data,
                timeout=120
            )
            response_data = response.json()
            elapsed = time.time() - start_time
            
            content = response_data.get("content", "")
            tokens_eval = response_data.get("tokens_evaluated", 0)
            tokens_pred = response_data.get("tokens_predicted", 0)
            
            print(f"   Response ({elapsed:.1f}s): {content[:100]}...")
            print(f"   Tokens: eval={tokens_eval}, pred={tokens_pred}")
            
            # Save test result
            test_result = {
                "test_name": test_name,
                "prompt": prompt_text,
                "response": content,
                "elapsed_seconds": elapsed,
                "tokens_evaluated": tokens_eval,
                "tokens_predicted": tokens_pred,
                "response_full": response_data
            }
            results["tests"].append(test_result)
            
            # Check if response indicates vision is working
            if any(word in content.lower() for word in ["i see", "the text", "image shows", "written"]):
                print("   📊 Response suggests vision processing")
            elif any(word in content.lower() for word in ["can't see", "no image", "provide", "upload"]):
                print("   ⚠️  Response suggests no image received")
            else:
                print("   ❓ Response is ambiguous")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
            results["tests"].append({
                "test_name": test_name,
                "error": str(e)
            })
    
    # Save results
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {log_file}")
    
    # Analyze results
    print("\n=== Analysis ===")
    successful_tests = [t for t in results["tests"] if "response" in t]
    
    if successful_tests:
        # Check if responses are all similar (hallucination pattern)
        responses = [t["response"] for t in successful_tests]
        if len(set(responses)) == 1:
            print("⚠️  All responses identical - likely hallucinating")
        else:
            print("✅ Responses vary - model is processing input")
        
        # Check for vision indicators
        vision_words = ["see", "image", "text", "written", "shows", "displays"]
        has_vision_words = any(
            any(word in t["response"].lower() for word in vision_words)
            for t in successful_tests
        )
        
        if has_vision_words:
            print("✅ Responses reference visual content")
        else:
            print("⚠️  Responses don't reference visual content")
    
    print("\nTo examine full results:")
    print(f"  cat {log_file} | python -m json.tool")

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/images/handwritten.jpeg"
    test_vision(image_path)
