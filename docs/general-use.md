# OCR to Anki - Usage Guide

## Tesseract OCR Pipeline

### Basic Tesseract Usage

```python
# Option 1: Pipe directly from OCR
python src/tesseract_ocr_image.py image.png | python src/ocr_to_json.py --pretty -o data/new_notes.json

# Option 2: Two-step process
python src/tesseract_ocr_image.py image.png > ocr_output.txt
python src/ocr_to_json.py -i ocr_output.txt -o data/new_notes.json --pretty

# Option 3: With additional tags
python src/tesseract_ocr_image.py image.png | python src/ocr_to_json.py --tag vocabulary --tag chapter1 -o notes.json

# Option 4: Custom separator (for comma-separated terms)
echo "犬,猫,本" | python src/ocr_to_json.py -s "," --pretty
```

Take into account that tesseract better handle text if the language is suggested with the option `-l` and say... `eng`, `jpn` or `eng+jpn`.

---

## Ollama OCR with Vision Model

### Overview

The Ollama OCR script (`src/ollama_ocr.py`) uses a local vision-language model (qwen3-vl:2b) to perform intelligent OCR on images. Unlike traditional Tesseract OCR, this approach uses AI to understand context, handle various text types, and provide more flexible analysis options.

### How It Works

1. **Configuration Loading**: Reads settings from `config/settings.yaml` to determine OCR behavior
2. **Prompt Generation**: Builds a custom prompt based on your configuration (text type, scope, language)
3. **Image Encoding**: Converts images to base64 format for API transmission
4. **Model Inference**: Sends the image and prompt to the local Ollama server
5. **Response Parsing**: Extracts words from the model's response (attempts JSON parsing first, falls back to text splitting)
6. **Results Storage**: Saves individual results and summary files as JSON

### Configuration Options

All settings are in `config/settings.yaml` under the `ollama_ocr` section:

#### Text Type (`text_type`)
- `detect`: Let the model determine if text is handwritten or printed
- `handwritten`: Optimize for handwritten text recognition
- `printed`: Optimize for printed/typed text recognition

#### Analysis Scope (`analysis_scope`)
- `whole`: Analyze all text in the entire image
- `highlighted`: Focus only on highlighted text (uses `highlight_color` setting)

#### Highlight Color (`highlight_color`)
- Options: `yellow`, `orange`, `blue`, `green`, `purple`, `red`
- Only relevant when `analysis_scope` is set to `highlighted`

#### Language (`language`)
- `detect`: Automatic language detection
- Specific language: `english`, `japanese`, `spanish`, `french`, etc.

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull the Model**:
   ```bash
   ollama pull qwen3-vl:2b
   ```
3. **Verify Ollama is Running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Usage Examples

#### Example 1: Basic OCR on All Images

Configuration in `settings.yaml`:
```yaml
ollama_ocr:
  text_type: "detect"
  analysis_scope: "whole"
  language: "detect"
```

Run:
```bash
python src/ollama_ocr.py
```

This will:
- Process all images in `data/images/`
- Auto-detect text type and language
- Extract all visible text
- Save results to `data/ollama_ocr_results/`

#### Example 2: Extract Only Yellow Highlighted Text

Configuration in `settings.yaml`:
```yaml
ollama_ocr:
  text_type: "printed"
  analysis_scope: "highlighted"
  highlight_color: "yellow"
  language: "english"
```

Run:
```bash
python src/ollama_ocr.py
```

This prompts the model to focus only on yellow-highlighted English text.

#### Example 3: Handwritten Japanese Notes

Configuration in `settings.yaml`:
```yaml
ollama_ocr:
  text_type: "handwritten"
  analysis_scope: "whole"
  language: "japanese"
```

Run:
```bash
python src/ollama_ocr.py
```

Optimized for recognizing handwritten Japanese characters.

### Output Structure

#### Individual Result Files
Location: `data/ollama_ocr_results/{image_name}_ocr.json`

```json
{
  "image_path": "/path/to/image.png",
  "timestamp": "2024-01-15T10:30:45.123456",
  "config": {
    "text_type": "detect",
    "analysis_scope": "whole",
    "highlight_color": "yellow",
    "language": "detect"
  },
  "prompt": "Full prompt sent to model...",
  "raw_response": "Model's complete response...",
  "words": ["word1", "word2", "word3"],
  "word_count": 3
}
```

#### Summary File
Location: `data/ollama_ocr_results/ocr_summary_YYYYMMDD_HHMMSS.json`

```json
{
  "total_images": 5,
  "successful": 4,
  "failed": 1,
  "total_words": 127,
  "results": [/* array of all individual results */]
}
```

### Integration with Anki Pipeline

After running Ollama OCR, you can pipe the results to create Anki notes:

```bash
# Extract words from OCR results
cat data/ollama_ocr_results/image_ocr.json | jq -r '.words[]' | python src/ocr_to_json.py --pretty -o data/anki_notes.json

# Or combine multiple results
jq -r '.results[].words[]' data/ollama_ocr_results/ocr_summary_*.json | python src/ocr_to_json.py -o notes.json
```

### Tips for Best Results

1. **Image Quality**: Higher resolution images (1500+ pixels wide) work better
2. **Aggressive Resizing**: For timeout issues, reduce `max_image_width` to 400-600px in settings
3. **Contrast**: Ensure good contrast between text and background
4. **Model Selection**: `qwen3-vl:2b` is fast but smaller; try larger models for better accuracy
5. **Timeout**: Increase `timeout` setting for large images or slower machines
6. **Language Specificity**: Specifying the exact language often improves accuracy vs. auto-detect
7. **Highlighted Text**: Works best when highlights are bright and saturated
8. **Batch Processing**: Process similar images with the same configuration for consistency
9. **Test with Sections**: Use `test_crop_and_ocr.py` to process images in smaller chunks

### Configuration for Faster Processing

If experiencing timeouts, adjust these settings in `config/settings.yaml`:

```yaml
ollama_ocr:
  timeout: 360  # Increase to 6 minutes
  max_image_width: 400  # Reduce from 600 to 400 for faster processing
```

**Trade-off**: Lower resolution may reduce accuracy but significantly speeds up processing.

---

## Chunking Strategy for Large Images

For best results with handwritten or complex images, use a chunking approach:

### Why Chunking Works

1. **Smaller context** = faster processing (15-30s vs 300s+)
2. **Better focus** = model sees less noise per request
3. **No timeouts** = sections process reliably
4. **Enhanced sections** = better text clarity

### Recommended Workflow

````bash
# 1. Test to find optimal settings
python tests/test_crop_enhanced_ocr.py

# 2. Use best settings in production
# Example: 8 sections, Moderate enhancement (2.0x contrast, 1.5x sharpness)
````

### Implementation Example

```python
from PIL import Image, ImageEnhance, ImageFilter

def process_image_in_chunks(image_path, sections=8):
    """Process image by splitting into enhanced sections."""
    results = []
    
    with Image.open(image_path) as img:
        height = img.size[1]
        section_height = height // sections
        
        for i in range(sections):
            # Crop section
            top = i * section_height
            bottom = (i + 1) * section_height if i < sections - 1 else height
            section = img.crop((0, top, img.size[0], bottom))
            
            # Enhance
            section = ImageEnhance.Contrast(section).enhance(2.0)
            section = ImageEnhance.Sharpness(section).enhance(1.5)
            section = section.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            
            # Resize to 400px width
            if section.size[0] > 400:
                ratio = 400 / section.size[0]
                new_size = (400, int(section.size[1] * ratio))
                section = section.resize(new_size, Image.Resampling.LANCZOS)
            
            # Send to OCR...
            words = perform_ocr_on_section(section)
            results.extend(words)
    
    return results
```

### Performance Comparison

| Approach | Processing Time | Words Extracted | Timeout Risk |
|----------|----------------|-----------------|--------------|
| Full image (no enhancement) | 300s+ (timeout) | 0 | High |
| Full image (enhanced) | 60-180s | 0-5 | Medium |
| Chunked (8 sections) | 120-240s total | 20-100+ | Low |
| Chunked + Enhanced | 120-240s total | 30-150+ | Very Low |

**Recommendation**: Always use chunked + enhanced approach for handwritten or complex images.

### Troubleshooting

**"Connection refused" error**:
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is accessible

**Model not found**:
- Pull the model: `ollama pull qwen3-vl:2b`
- Verify with: `ollama list`

**Poor word extraction**:
- Check `raw_response` in output JSON to see what model returned
- Try adjusting `text_type` or `language` settings
- Consider using a larger model for better accuracy

**Timeout errors**:
- Increase `timeout` in settings.yaml
- Try a smaller model or reduce image size
- Check system resources (CPU/RAM usage)

### Comparison: Ollama vs Tesseract

| Feature | Tesseract | Ollama (Vision Model) |
|---------|-----------|----------------------|
| Speed | Fast | Slower (model inference) |
| Accuracy | Good for clean, printed text | Better for complex/handwritten text |
| Context Understanding | None | Yes (understands highlighting, layout) |
| Language Support | Extensive (100+ languages) | Depends on model training |
| Setup | Simple | Requires Ollama + model download |
| Offline | Yes | Yes (local model) |
| Resource Usage | Low | Higher (GPU recommended) |
| Cost | Free | Free (local) |

**When to use each**:
- **Tesseract**: Clean scanned documents, known language, batch processing
- **Ollama**: Handwritten notes, highlighted text, mixed content, context-aware extraction