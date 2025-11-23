"""
Gemma Vocabulary Enricher Test
Uses Ollama gemma3:1b model to enrich Anki vocabulary cards with:
- Definitions in target language
- Example phrases
- Auto-detected language and part-of-speech tags
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any
import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_language(word: str, config: Dict[str, Any]) -> str:
    """
    Detect the language of a word using Gemma.
    
    Args:
        word: Word to detect language for
        config: Configuration dictionary
    
    Returns:
        Detected language name
    """
    gemma_config = config['gemma_enricher']
    url = f"{gemma_config['url']}/api/generate"
    
    prompt = f"""What language is this word: "{word}"?
Reply with ONLY the language name in lowercase (e.g., english, spanish, french, japanese, german, etc.).
No explanations, just the language name."""
    
    payload = {
        "model": gemma_config['model'],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 10
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        language = result.get('response', '').strip().lower()
        # Clean up common variations
        language = language.replace('language', '').replace(':', '').strip()
        return language if language else "unknown"
    except Exception as e:
        print(f"  Warning: Could not detect language for '{word}': {e}")
        return "unknown"


def detect_pos(word: str, language: str, config: Dict[str, Any]) -> List[str]:
    """
    Detect part(s) of speech for a word.
    
    Args:
        word: Word to analyze
        language: Language of the word
        config: Configuration dictionary
    
    Returns:
        List of POS tags (e.g., ["noun"], ["verb", "noun"])
    """
    gemma_config = config['gemma_enricher']
    url = f"{gemma_config['url']}/api/generate"
    
    prompt = f"""What part(s) of speech is this {language} word: "{word}"?
Reply with ONLY the part(s) of speech in lowercase English, comma-separated if multiple.
Options: noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection
Example responses: "noun" or "noun, verb" or "adjective"
No explanations, just the part(s) of speech."""
    
    payload = {
        "model": gemma_config['model'],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 20
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        pos_text = result.get('response', '').strip().lower()
        
        # Parse comma-separated POS tags
        pos_tags = [tag.strip() for tag in pos_text.split(',')]
        # Filter valid POS tags
        valid_pos = ['noun', 'verb', 'adjective', 'adverb', 'pronoun', 
                     'preposition', 'conjunction', 'interjection']
        pos_tags = [tag for tag in pos_tags if tag in valid_pos]
        
        return pos_tags if pos_tags else ["unknown"]
    except Exception as e:
        print(f"  Warning: Could not detect POS for '{word}': {e}")
        return ["unknown"]


def enrich_vocabulary_item(item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a single vocabulary item with definition, examples, and tags.
    
    Args:
        item: Vocabulary item with at least a 'front' field
        config: Configuration dictionary
    
    Returns:
        Enriched vocabulary item
    """
    gemma_config = config['gemma_enricher']
    url = f"{gemma_config['url']}/api/generate"
    
    word = item.get('front', '')
    if not word:
        return item
    
    print(f"\nEnriching: {word}")
    
    # Detect language if enabled
    language = "unknown"
    if gemma_config['add_language_tags']:
        print(f"  Detecting language...")
        language = detect_language(word, config)
        print(f"  Language: {language}")
    
    # Detect part of speech if enabled
    pos_tags = []
    if gemma_config['add_pos_tags']:
        print(f"  Detecting part of speech...")
        pos_tags = detect_pos(word, language, config)
        print(f"  POS: {', '.join(pos_tags)}")
    
    # Generate definition and examples
    def_lang = gemma_config['definition_language']
    ex_lang = gemma_config['examples_language']
    
    prompt = f"""For the word "{word}" (a {language} word):

1. Provide a clear definition in {def_lang}.
2. Provide 2 example phrases or sentences using this word in {ex_lang}.

Format your response as JSON:
{{
  "definition": "the definition here",
  "examples": ["example 1", "example 2"]
}}

Only return the JSON, no other text."""
    
    payload = {
        "model": gemma_config['model'],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 300
        }
    }
    
    print(f"  Generating definition and examples...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=gemma_config['timeout'])
        elapsed = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        response_text = result.get('response', '')
        
        # Try to parse JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                definition = data.get('definition', '')
                examples = data.get('examples', [])
            else:
                # Fallback parsing
                lines = [l.strip() for l in response_text.split('\n') if l.strip()]
                definition = lines[0] if lines else ''
                examples = lines[1:3] if len(lines) > 1 else []
        except json.JSONDecodeError:
            # Fallback: use first lines
            lines = [l.strip() for l in response_text.split('\n') if l.strip()]
            definition = lines[0] if lines else ''
            examples = lines[1:3] if len(lines) > 1 else []
        
        # Build enriched item
        enriched = {
            'front': word,
            'back': definition if definition else '(No definition generated)',
            'examples': examples if examples else [],
            'tags': []
        }
        
        # Add language tag
        if language != "unknown":
            enriched['tags'].append(f"language::{language}")
        
        # Add POS tags
        for pos in pos_tags:
            if pos != "unknown":
                enriched['tags'].append(f"pos::{pos}")
        
        # Preserve any existing tags from input
        if 'tags' in item:
            enriched['tags'].extend([t for t in item['tags'] if t not in enriched['tags']])
        
        print(f"  ✓ Completed in {elapsed:.2f}s")
        print(f"  Definition: {definition[:60]}{'...' if len(definition) > 60 else ''}")
        print(f"  Examples: {len(examples)}")
        print(f"  Tags: {', '.join(enriched['tags'])}")
        
        return enriched
        
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout after {gemma_config['timeout']}s")
        return {**item, 'back': '(Timeout)', 'tags': item.get('tags', [])}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {**item, 'back': f'(Error: {e})', 'tags': item.get('tags', [])}


def process_vocabulary_file(input_file: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a vocabulary JSON file and enrich all items.
    
    Args:
        input_file: Path to input JSON file
        config: Configuration dictionary
    
    Returns:
        List of enriched vocabulary items
    """
    # Load input file
    with open(input_file, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    
    if not isinstance(vocabulary, list):
        raise ValueError("Input file must contain a JSON array of vocabulary items")
    
    print(f"Loaded {len(vocabulary)} vocabulary items from {input_file}")
    
    enriched_items = []
    gemma_config = config['gemma_enricher']
    batch_size = gemma_config['batch_size']
    
    for i, item in enumerate(vocabulary, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(vocabulary)}")
        print('='*60)
        
        enriched = enrich_vocabulary_item(item, config)
        enriched_items.append(enriched)
        
        # Small delay between batches to avoid overwhelming the model
        if i % batch_size == 0 and i < len(vocabulary):
            print(f"\n  Batch complete. Brief pause...")
            time.sleep(2)
    
    return enriched_items


def save_enriched_vocabulary(items: List[Dict[str, Any]], config: Dict[str, Any]):
    """Save enriched vocabulary to output directory."""
    from datetime import datetime
    
    gemma_config = config['gemma_enricher']
    output_dir = Path(gemma_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"enriched_vocabulary_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Enriched vocabulary saved to: {output_file}")
    print('='*60)
    
    # Print summary
    total_words = len(items)
    with_definitions = len([i for i in items if i.get('back') and not i['back'].startswith('(')])
    with_examples = len([i for i in items if i.get('examples')])
    
    print(f"\nSummary:")
    print(f"  Total words: {total_words}")
    print(f"  With definitions: {with_definitions}")
    print(f"  With examples: {with_examples}")
    print(f"  Success rate: {with_definitions/total_words*100:.1f}%")


def create_sample_input():
    """Create a sample input file for testing."""
    sample_data = [
        {"front": "bonjour"},
        {"front": "cat"},
        {"front": "correr"},
        {"front": "本", "tags": ["from::book"]},
        {"front": "schön"}
    ]
    
    output_file = Path("data/vocabulary_input.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Sample input created: {output_file}")
    return str(output_file)


def main():
    """Main test execution."""
    print("=" * 60)
    print("GEMMA VOCABULARY ENRICHER TEST")
    print("=" * 60 + "\n")
    
    # Load config
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    config_path = project_root / "config" / "settings.yaml"
    
    config = load_config(str(config_path))
    gemma_config = config['gemma_enricher']
    
    print(f"Configuration:")
    print(f"  Model: {gemma_config['model']}")
    print(f"  Definition language: {gemma_config['definition_language']}")
    print(f"  Examples language: {gemma_config['examples_language']}")
    print(f"  Add language tags: {gemma_config['add_language_tags']}")
    print(f"  Add POS tags: {gemma_config['add_pos_tags']}")
    print()
    
    # Check for input file or create sample
    input_file = project_root / gemma_config['input_file']
    
    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        print("Creating sample input file with test vocabulary...\n")
        input_file = create_sample_input()
    
    # Process vocabulary
    enriched = process_vocabulary_file(str(input_file), config)
    
    # Save results
    save_enriched_vocabulary(enriched, config)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
