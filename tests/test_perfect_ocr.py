"""
Barebones test for OCR using gemma3:4b on handwritten.jpeg
Outputs only words, one per line, for easy piping.
"""

import requests
import base64
from PIL import Image
import io
import sys


def encode_image(image_path: str, max_width: int = 800) -> str:
    """Encode image to base64, resizing if needed."""
    with Image.open(image_path) as img:
        if img.size[0] > max_width:
            ratio = max_width / img.size[0]
            new_size = (max_width, int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')


def extract_words(response_text: str) -> list:
    """Extract words from response."""
    words = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip intro lines
        skip = ["here's", "here is", "ocr", "extracted", "from the image", "using"]
        if any(p in line.lower() for p in skip):
            continue
        
        # Remove quotes
        line = line.strip('"\'')
        
        # Remove bullets and numbering
        line = line.lstrip('*-•→►▪▫0123456789.)> ')
        line = line.strip()
        
        if len(line) < 2:
            continue
        
        # Split on commas
        if ',' in line:
            for word in line.split(','):
                word = word.strip()
                if word and len(word) >= 2:
                    words.append(word)
        # Split on spaces (but preserve apostrophes)
        elif ' ' in line and "'" not in line and "-" not in line:
            for word in line.split():
                word = word.strip()
                if word and len(word) > 1:
                    words.append(word)
        else:
            words.append(line)
    
    return words


def main():
    # Configuration
    model = "gemma3:4b"
    image_path = "data/images/handwritten.jpeg"
    prompt = "Can you do OCR of this image? Please extract all visible text."
    
    # Encode image
    image_base64 = encode_image(image_path, max_width=800)
    
    # Payload
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 512,
            "temperature": 0.1,
            "top_k": 10,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "stop": ["\n\n\n", "</s>", "---"],
            "num_gpu": -1,
            "num_batch": 1024
        }
    }
    
    # Request
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=600
    )
    
    result = response.json()
    response_text = result.get('response', '')
    
    # Extract and print words
    words = extract_words(response_text)
    
    # Output: one word per line
    for word in words:
        print(word)


if __name__ == "__main__":
    main()
