"""
Diagnostic script to check Ollama health and model availability
"""

import requests
import json
import time


def check_ollama_server():
    """Check if Ollama server is running."""
    print("=== Checking Ollama Server ===\n")
    
    url = "http://localhost:11434/api/tags"
    
    try:
        print(f"Connecting to {url}...")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        models = data.get('models', [])
        
        print("✓ Ollama server is running\n")
        print(f"Available models ({len(models)}):")
        for model in models:
            print(f"  - {model.get('name')} (size: {model.get('size', 0) / (1024**3):.2f}GB)")
        
        return True, models
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama server")
        print("\nTry:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Or check if it's running: ps aux | grep ollama")
        return False, []
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, []


def test_simple_text_generation():
    """Test simple text generation without images."""
    print("\n=== Testing Simple Text Generation ===\n")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": "Say hello in one word",
        "stream": False,
        "options": {
            "num_predict": 10
        }
    }
    
    try:
        print("Sending simple text prompt...")
        print(f"Model: qwen3-vl:2b")
        print(f"Timeout: 30 seconds\n")
        
        start = time.time()
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Response received in {elapsed:.2f} seconds")
        print(f"Response: {result.get('response', '')}")
        print(f"Model loaded: {result.get('model', '')}")
        
        return True
    
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        print("\nThis suggests the model is not loaded or responding")
        print("Try: ollama run qwen3-vl:2b")
        return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_vision_with_tiny_image():
    """Test vision model with a tiny generated image."""
    print("\n=== Testing Vision with Tiny Image ===\n")
    
    try:
        from PIL import Image, ImageDraw
        import io
        import base64
        
        # Create a tiny 50x50 white image with black text "Hi"
        img = Image.new('RGB', (50, 50), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 15), "Hi", fill='black')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print(f"Created test image: 50x50 pixels")
        print(f"Encoded size: {len(img_base64) / 1024:.2f}KB\n")
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen3-vl:2b",
            "prompt": "What text do you see?",
            "images": [img_base64],
            "stream": False,
            "options": {
                "num_predict": 20
            }
        }
        
        print("Sending vision request...")
        print(f"Timeout: 60 seconds\n")
        
        start = time.time()
        response = requests.post(url, json=payload, timeout=60)
        elapsed = time.time() - start
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Response received in {elapsed:.2f} seconds")
        print(f"Response: {result.get('response', '')}")
        
        return True
    
    except requests.exceptions.Timeout:
        print("✗ Vision request timed out")
        print("\nThe model cannot process images in reasonable time")
        print("\nPossible issues:")
        print("  1. Model not properly loaded")
        print("  2. Insufficient system resources (RAM/CPU)")
        print("  3. Model corruption")
        print("\nTry:")
        print("  ollama pull qwen3-vl:2b  # Re-download model")
        return False
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_system_resources():
    """Check basic system resources."""
    print("\n=== System Resources ===\n")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.available / (1024**3):.2f}GB available / {mem.total / (1024**3):.2f}GB total")
        print(f"CPU: {psutil.cpu_count()} cores, {psutil.cpu_percent(interval=1)}% usage")
        
        if mem.available / (1024**3) < 2:
            print("\n⚠ Warning: Low available RAM (< 2GB)")
            print("  Vision models require significant memory")
        
    except ImportError:
        print("psutil not installed, skipping resource check")
        print("Install with: pip install psutil")


def main():
    """Run all diagnostics."""
    print("=" * 60)
    print("OLLAMA HEALTH CHECK")
    print("=" * 60 + "\n")
    
    # Check server
    server_ok, models = check_ollama_server()
    
    if not server_ok:
        return
    
    # Check if qwen3-vl:2b is available
    model_found = any('qwen3-vl:2b' in m.get('name', '') for m in models)
    
    if not model_found:
        print("\n⚠ Warning: qwen3-vl:2b not found in available models")
        print("Install with: ollama pull qwen3-vl:2b")
        return
    
    # Test text generation
    text_ok = test_simple_text_generation()
    
    if not text_ok:
        print("\n⚠ Basic text generation failed - model may not be loaded")
        return
    
    # Test vision
    vision_ok = test_vision_with_tiny_image()
    
    if not vision_ok:
        print("\n⚠ Vision processing failed")
    
    # System resources
    check_system_resources()
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60 + "\n")
    
    if text_ok and not vision_ok:
        print("Text generation works but vision processing fails.")
        print("\nThis indicates the model has issues with image processing.")
        print("\nActions to try:")
        print("  1. Re-download model: ollama pull qwen3-vl:2b")
        print("  2. Try different model: ollama pull llava:7b")
        print("  3. Check Ollama logs for errors")
        print("  4. Restart Ollama service")
    elif text_ok and vision_ok:
        print("✓ All tests passed!")
        print("\nYour handwritten.jpeg image might be too complex.")
        print("Try:")
        print("  1. Use a simpler image with less text")
        print("  2. Crop to just one word/line")
        print("  3. Increase timeout to 10+ minutes")
    else:
        print("Basic functionality is not working.")
        print("Check Ollama installation and service status.")


if __name__ == "__main__":
    main()
