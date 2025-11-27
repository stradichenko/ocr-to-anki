#!/usr/bin/env python3
"""
CLI script to convert OCR output into Anki-compatible JSON format.
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
import re


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults", file=sys.stderr)
        return {}


def parse_terms(text: str) -> list[str]:
    """
    Parse text into individual terms (words).
    Handles hyphenation at line breaks, apostrophes, and comma-separated words.
    
    Args:
        text: Input text containing terms
    
    Returns:
        List of individual terms
    """
    # First, handle hyphenated words at line breaks (word- \n word -> word)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Handle apostrophes that might be split
    text = re.sub(r"(\w+)'\s+(\w+)", r"\1'\2", text)
    
    # Split on whitespace, newlines, and commas
    # This regex finds words, preserving internal hyphens and apostrophes
    words = re.findall(r"\b[\w'-]+\b", text)
    
    # Filter and clean
    terms = []
    for word in words:
        # Remove leading/trailing hyphens, apostrophes, and quotes
        word = word.strip("'-\"")
        
        # Keep words that are at least 2 characters
        if len(word) >= 2:
            terms.append(word)
    
    # Remove duplicates while preserving order (case-insensitive)
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)
    
    return unique_terms


def create_anki_notes(terms: list[str], config: dict) -> dict:
    """
    Create Anki notes structure from terms.
    
    Args:
        terms: List of terms to convert
        config: Configuration dictionary
    
    Returns:
        Dictionary with settings and notes in Anki format
    """
    import_defaults = config.get('import_defaults', {})
    
    settings = {
        "defaultDeck": import_defaults.get('deck', 'Default'),
        "defaultModel": import_defaults.get('model', 'Basic'),
        "batchSize": import_defaults.get('batch_size', 10)
    }
    
    notes = []
    allow_duplicates = import_defaults.get('allow_duplicates', False)
    
    for term in terms:
        note = {
            "fields": {
                "Front": term,
                "Back": ""  # Empty back field to be filled later
            },
            "tags": ["ocr"],
            "allowDuplicate": allow_duplicates
        }
        notes.append(note)
    
    return {
        "settings": settings,
        "notes": notes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert OCR output to Anki-compatible JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From OCR output file
  python src/ocr_to_json.py -i ocr_output.txt -o notes.json
  
  # From stdin (pipe from ocr_image.py)
  python src/ocr_image.py image.png | python src/ocr_to_json.py -o notes.json
  
  # Direct text input
  echo -e "犬\\n猫\\n本" | python src/ocr_to_json.py
  
  # With minimum word length filter
  python src/ocr_to_json.py -i terms.txt --min-length 3 -o notes.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input file containing OCR output (default: read from stdin)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file (default: print to stdout)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=2,
        help='Minimum word length to include (default: 2)'
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        action='append',
        default=None,
        help='Additional tags to add to all notes (can be used multiple times)'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output'
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Read from stdin
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse terms
    terms = parse_terms(text)
    
    # Apply minimum length filter
    if args.min_length > 2:
        terms = [t for t in terms if len(t) >= args.min_length]
    
    if not terms:
        print("Warning: No terms found in input", file=sys.stderr)
    
    # Create Anki notes structure
    anki_data = create_anki_notes(terms, config)
    
    # Add custom tags if specified
    if args.tag:
        for note in anki_data['notes']:
            note['tags'].extend(args.tag)
    
    # Format JSON
    if args.pretty:
        json_output = json.dumps(anki_data, indent=2, ensure_ascii=False)
    else:
        json_output = json.dumps(anki_data, ensure_ascii=False)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"Created {len(terms)} notes in {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == '__main__':
    main()
