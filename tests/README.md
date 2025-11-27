# Tests

## Test Scripts Overview

### 1. Ollama Health Check (`test_ollama_health.py`)

Verifies Ollama server and model are working properly.

```bash
python tests/test_ollama_health.py
```

**Checks:**
- Server connectivity
- Model availability
- Text generation
- Vision processing
- System resources

### 2. Cropped Section Test (`test_crop_and_ocr.py`)

Crops large images into smaller sections and tests each individually.

```bash
python tests/test_crop_and_ocr.py
```

**What it does:**
- Splits image into 6 horizontal sections
- Resizes each to max 400px width
- Tests OCR on each section with 90s timeout
- Shows which sections work and which fail

**When to use:**
- When full image times out
- To identify problematic image regions
- To test if chunking strategy would work

**Output:**
- Shows processing time per section
- Lists all extracted words by section
- Saves results to `cropped_test_results.json`

### 3. Simple OCR Test (`test_ollama_ocr.py`)

Tests OCR on full image with basic settings.

```bash
python tests/test_ollama_ocr.py
```

**Use after** health check and cropping test pass.

### 4. Enhanced OCR Test (`test_ollama_ocr_enhanced.py`)

Tests OCR with image preprocessing (contrast and sharpening enhancement).

```bash
python tests/test_ollama_ocr_enhanced.py
```

**What it does:**
- Tests 3 enhancement levels: Moderate, Strong, Aggressive
- Applies contrast increase (1.8x to 3.0x)
- Applies sharpening (1.5x to 2.5x)
- Uses unsharp mask for additional clarity
- Compares results across enhancement levels
- Identifies best enhancement settings

**Enhancement levels:**
- **Moderate**: Contrast 1.8x, Sharpness 1.5x (safe baseline)
- **Strong**: Contrast 2.5x, Sharpness 2.0x (recommended)
- **Aggressive**: Contrast 3.0x, Sharpness 2.5x (maximum enhancement)

**Output:**
- Comparison table showing words found per enhancement level
- Identifies best performing enhancement
- Saves all results to `test_enhanced_results.json`
- Provides recommendations for production settings

**When to use:**
- When basic OCR has poor accuracy
- For low-contrast or blurry images
- To find optimal preprocessing settings
- Before implementing production pipeline

**Trade-offs:**
- Higher enhancement = better text clarity
- Too much enhancement = potential artifacts
- Slightly larger file size (quality=95)

### 5. Crop + Enhancement Test (`test_crop_enhanced_ocr.py`)

**Best approach**: Combines cropping AND enhancement for optimal OCR results.

```bash
python tests/test_crop_enhanced_ocr.py
```

**What it does:**
- Tests 3 enhancement levels (Light, Moderate, Strong)
- Crops image into 8 horizontal sections for each level
- Applies enhancements to each section BEFORE OCR
- Compares results across all configurations
- Saves enhanced sections for inspection

**Why this works better:**
- **Smaller chunks** = faster processing, less timeout risk
- **Enhanced sections** = clearer text, better accuracy
- **Multiple configs** = finds optimal settings automatically

**Enhancement levels:**
- **Light**: Contrast 1.5x, Sharpness 1.2x (minimal, preserves detail)
- **Moderate**: Contrast 2.0x, Sharpness 1.5x (balanced)
- **Strong**: Contrast 2.5x, Sharpness 2.0x (maximum clarity)

**Output:**
- Saves all enhanced sections to `tests/enhanced_sections/{level}/`
- Comparison table of words found per level
- Complete word list from best configuration
- JSON results with all details

**When to use:**
- **Always** - this is the recommended approach
- When other tests fail or give poor results
- To find optimal preprocessing settings
- For production pipeline configuration

**Expected results:**
- 20-100+ words depending on image content
- Processing time: 15-30s per section
- Best config identified automatically

### 6. Gemma Vocabulary Enricher (`test_gemma_enricher.py`)

**Automatically enrich Anki vocabulary** with definitions, examples, and smart tags using Gemma AI.

```bash
python tests/test_gemma_enricher.py
```

**What it does:**
- Reads vocabulary JSON file (format: `[{"front": "word"}, ...]`)
- Auto-detects language of each word
- Detects part(s) of speech (noun, verb, adjective, etc.)
- Generates definition in target language
- Creates 2 example phrases in target language
- Adds smart tags (language::french, pos::noun, etc.)
- Saves enriched vocabulary ready for Anki import

**Input format** (`data/vocabulary_input.json`):
```json
[
  {"front": "bonjour"},
  {"front": "cat"},
  {"front": "correr", "tags": ["chapter::1"]}
]
```

**Output format** (`data/enriched_vocabulary/enriched_vocabulary_TIMESTAMP.json`):
```json
[
  {
    "front": "bonjour",
    "back": "A French greeting meaning 'hello' or 'good day'",
    "examples": [
      "Bonjour, comment allez-vous?",
      "She said bonjour to everyone at the party."
    ],
    "tags": ["language::french", "pos::noun", "pos::interjection"]
  }
]
```

**Configuration** (in `config/settings.yaml`):
```yaml
gemma_enricher:
  model: "gemma3:1b"
  definition_language: "english"  # Language for definitions
  examples_language: "english"    # Language for examples
  add_language_tags: true         # Auto-detect word language
  add_pos_tags: true              # Detect part of speech
```

**Prerequisites:**
```bash
# Install Gemma model
ollama pull gemma3:1b

# Verify it's available
ollama list
```

**When to use:**
- After OCR extraction to add context to words
- To prepare vocabulary lists for Anki
- To auto-tag and organize vocabulary by language/POS
- When creating flashcards from word lists

**Performance:**
- Processing time: ~5-10s per word
- Batch processing with delays to avoid overwhelming model
- Configurable batch size (default: 5 words)

**Features:**
- **Smart language detection**: Automatically identifies word language
- **POS tagging**: Adds grammatical category tags
- **Bilingual support**: Definitions and examples in different languages
- **Preserves existing tags**: Keeps any tags from input
- **Error handling**: Gracefully handles timeouts and errors
- **Sample creation**: Auto-creates sample input if none exists

### 7. Image Inspection Tool (`test_image_inspect.py`)

**Diagnose why OCR returns empty results** by analyzing image properties.

```bash
python tests/test_image_inspect.py
```

**What it analyzes:**
- Image dimensions and format
- Color distribution and brightness
- Contrast levels
- Edge detection (content density)
- Whether image is blank/uniform

**Output:**
- Detailed statistics about image properties
- Visual warning for problematic characteristics
- Saves analysis visualizations:
  - `_original.jpg` - Resized view
  - `_grayscale.jpg` - B&W conversion
  - `_contrast.jpg` - High contrast version
  - `_edges.jpg` - Edge detection

**When to use:**
- When OCR returns empty/blank responses
- To verify image actually contains text
- Before spending time on OCR debugging
- To check image quality issues

**Common findings:**
- **Blank image**: Uniform color, no edges detected
- **Low contrast**: Text same color as background
- **Overexposed**: Brightness > 200, washed out
- **Underexposed**: Brightness < 50, too dark

### 8. Image Debug Tool (`test_image_debug.py`)

**Test multiple OCR configurations** on problematic images.

```bash
python tests/test_image_debug.py
```

Tests 6 different approaches to find what works best.

**Use after** `test_image_inspect.py` confirms image has content.

### 9. Gemma3:4b Vocabulary Enricher (`test_gemma_vocabulary_enricher.py`)

**Automatically enrich Anki vocabulary** with definitions and examples using Gemma3:4b AI.

```bash
python tests/test_gemma_vocabulary_enricher.py
```

**What it does:**
- Reads notes.json file (output from ocr_to_json.py)
- Generates clear definitions in configured language (from settings.yaml)
- Creates 2 natural example sentences in configured language
- Uses conversational prompts for better Gemma responses
- Adds enrichment metadata tags
- Saves detailed logs for debugging

**Prerequisites:**
```bash
# Install Gemma3:4b model (larger, more capable)
ollama pull gemma3:4b

# Verify it's available
ollama list | grep gemma3:4b
```

**Input format** (`notes.json`):
```json
{
  "notes": [
    {
      "fields": {"Front": "coûta", "Back": ""},
      "tags": ["ocr"]
    }
  ]
}
```

**Output format** (`tests/enriched_vocabulary/enriched_notes_TIMESTAMP.json`):
```json
{
  "notes": [
    {
      "fields": {
        "Front": "coûta",
        "Back": "Definition: Past tense of 'coûter', meaning to cost or require payment.\n\nExamples:\n1. Cette voiture m'a coûté très cher.\n2. Le voyage nous a coûté mille euros."
      },
      "tags": ["ocr", "gemma-enriched", "def-lang:english", "ex-lang:english"]
    }
  ]
}
```

**Features:**
- **Conversational prompts**: Better responses from Gemma
- **Batch processing**: Processes words in groups with pauses
- **Detailed logging**: Every request/response saved to logs/
- **Progress tracking**: Real-time progress with percentage
- **Error handling**: Gracefully handles timeouts and failures
- **Language configuration**: Uses settings from config/settings.yaml

**Configuration** (in `config/settings.yaml`):
```yaml
gemma_enricher:
  definition_language: "english"  # Language for definitions
  examples_language: "english"    # Language for example sentences
```

**When to use:**
- After OCR extraction to add context to vocabulary
- To prepare flashcards with complete information
- When you need definitions and usage examples
- For language learning material preparation

**Performance:**
- Processing time: ~10-15s per word
- Includes 2s pause between definition and examples
- 5s pause between batches (default: 3 words)
- Test mode processes first 5 words only

**Logs and debugging:**
- Main log: `logs/gemma_enrichment_TIMESTAMP/enrichment.log`
- Per-word logs: `logs/gemma_enrichment_TIMESTAMP/[word]_definition.json`
- Per-word logs: `logs/gemma_enrichment_TIMESTAMP/[word]_examples.json`
- Summary: `logs/gemma_enrichment_TIMESTAMP/summary.txt`

**Example workflow:**
```bash
# 1. Extract text from image
python src/ocr_image.py image.jpg > extracted.txt

# 2. Convert to Anki notes
python src/ocr_to_json.py -i extracted.txt -o notes.json

# 3. Enrich with definitions and examples
python tests/test_gemma_vocabulary_enricher.py

# 4. Import enriched notes to Anki
python src/anki_importer.py tests/enriched_vocabulary/enriched_notes_*.json
```
