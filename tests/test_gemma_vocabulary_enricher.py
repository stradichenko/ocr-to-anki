"""
Test Gemma3:4b vocabulary enrichment for Anki notes.
Enriches vocabulary with definitions and example phrases.
"""

import json
import yaml
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys


class GemmaVocabularyEnricher:
    """Enriches vocabulary using Gemma3:4b model."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize enricher with configuration."""
        self.config = self._load_config(config_path)
        self.gemma_config = self.config.get('gemma_enricher', {})
        
        # Model settings
        self.model = "gemma3:4b"  # Using 4b version as requested
        self.url = self.gemma_config.get('url', 'http://localhost:11434')
        self.timeout = self.gemma_config.get('timeout', 60)
        
        # Language settings
        self.definition_language = self.gemma_config.get('definition_language', 'english')
        self.examples_language = self.gemma_config.get('examples_language', 'english')
        
        # Setup logging
        self.log_dir = Path("logs") / f"gemma_enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Statistics
        self.stats = {
            'total_words': 0,
            'definitions_generated': 0,
            'examples_generated': 0,
            'translations_made': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config not found at {config_path}, using defaults")
            return {}
    
    def _setup_logging(self):
        """Setup detailed logging."""
        # File handler for all logs
        log_file = self.log_dir / "enrichment.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('GemmaEnricher')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _save_request_log(self, word: str, prompt_type: str, prompt: str, response: dict):
        """Save detailed request/response for debugging."""
        log_file = self.log_dir / f"{word}_{prompt_type}.json"
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'word': word,
            'prompt_type': prompt_type,
            'prompt': prompt,
            'model': self.model,
            'response': response,
            'languages': {
                'definition': self.definition_language,
                'examples': self.examples_language
            }
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def detect_language(self, word: str) -> str:
        """
        Detect the language of a word.
        
        Args:
            word: The word to analyze
            
        Returns:
            Detected language name
        """
        prompt = f"""I need to know what language this word is from: "{word}"

Just tell me the language name (like English, French, Spanish, etc). Nothing else."""
        
        self.logger.debug(f"Detecting language for '{word}'...")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10,
                    "top_k": 5
                }
            }
            
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            language = result.get('response', '').strip().lower()
            
            # Clean up common additions
            language = language.replace('.', '').replace('!', '').replace('the word is ', '')
            language = language.replace('this is ', '').replace('it is ', '')
            
            self.logger.debug(f"Detected language: {language}")
            return language
            
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return "unknown"
    
    def translate_word(self, word: str, target_language: str, source_language: str = None) -> str:
        """
        Translate a word to the target language if needed.
        
        Args:
            word: The word to translate
            target_language: Target language for translation
            source_language: Source language (will detect if not provided)
            
        Returns:
            Translated word or original if translation not needed/failed
        """
        # Detect source language if not provided
        if not source_language:
            source_language = self.detect_language(word)
        
        # If already in target language, no translation needed
        if source_language.lower() == target_language.lower():
            self.logger.debug(f"Word '{word}' already in {target_language}, no translation needed")
            return word
        
        # Conversational translation prompt
        prompt = f"""Hey! I need your help with a translation. Can you translate the {source_language} word "{word}" to {target_language}?

Just give me the translation, nothing else. If it has multiple meanings, give me the most common one."""
        
        self.logger.info(f"Translating '{word}' from {source_language} to {target_language}...")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 20,
                    "top_k": 10
                }
            }
            
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            translation = result.get('response', '').strip()
            
            # Clean up the translation
            translation = translation.strip('"\'.,!').split('\n')[0]
            
            if translation and translation.lower() != word.lower():
                self.logger.info(f"✓ Translated: {word} → {translation}")
                self.stats['translations_made'] += 1
                
                # Save translation log
                self._save_request_log(word, "translation", prompt, result)
                return translation
            else:
                self.logger.warning(f"Translation failed or same as original")
                return word
                
        except Exception as e:
            self.logger.error(f"Error translating '{word}': {e}")
            return word
    
    def get_definition(self, word: str, translated_word: str = None) -> Optional[str]:
        """
        Get definition for a word using conversational prompt.
        
        Args:
            word: The original word
            translated_word: Translated version if different from original
            
        Returns:
            Definition string or None if error
        """
        # Use translated word for definition if available
        term_to_define = translated_word if translated_word else word
        
        # Include original word context if translated
        if translated_word and translated_word != word:
            prompt = f"""Hey! I'm learning vocabulary and need your help. Can you give me a clear, simple definition of the word "{term_to_define}" in {self.definition_language}? 

This is the {self.definition_language} translation of "{word}".

Important: The ENTIRE definition must be in {self.definition_language} only. Do not mix in English or other languages.

Just the definition please, nothing else. Keep it concise but informative, like a dictionary entry."""
        else:
            prompt = f"""Hey! I'm learning vocabulary and need your help. Can you give me a clear, simple definition of the word "{term_to_define}" in {self.definition_language}? 

Important: The ENTIRE definition must be in {self.definition_language} only. Do not use English words or phrases unless absolutely necessary (like proper nouns).

Just the definition please, nothing else. Keep it concise but informative, like a dictionary entry."""
        
        self.logger.info(f"Getting definition for '{term_to_define}'...")
        self.logger.debug(f"Prompt: {prompt[:100]}...")
        
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower for factual definitions
                    "num_predict": 250,  # Definitions shouldn't be too long
                    "top_k": 20,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            elapsed = time.time() - start_time
            
            definition = result.get('response', '').strip()
            
            # Clean up common prefixes that Gemma might add
            prefixes_to_remove = [
                "Here's the definition:",
                "Definition:",
                "The definition is:",
                f"{term_to_define} means:",
                f"{term_to_define}:",
                "Sure!",
                "Of course!",
                "In Italian:",
                f"In {self.definition_language}:",
                f"In {self.definition_language.capitalize()}:",
            ]
            
            for prefix in prefixes_to_remove:
                if definition.lower().startswith(prefix.lower()):
                    definition = definition[len(prefix):].strip()
            
            self.logger.info(f"✓ Definition generated in {elapsed:.2f}s")
            self.logger.debug(f"Definition: {definition[:100]}...")
            
            # Save request log
            self._save_request_log(word, "definition", prompt, result)
            
            self.stats['definitions_generated'] += 1
            return definition
            
        except Exception as e:
            self.logger.error(f"✗ Error getting definition for '{term_to_define}': {e}")
            self.stats['errors'] += 1
            return None
    
    def get_examples(self, word: str, translated_word: str = None) -> List[str]:
        """
        Get example sentences for a word using conversational prompt.
        
        Args:
            word: The original word
            translated_word: Translated version if different from original
            
        Returns:
            List of example sentences
        """
        # Determine which word to use for examples based on target language
        source_lang = self.detect_language(word)
        
        # If examples should be in the same language as the original word
        if source_lang.lower() == self.examples_language.lower():
            term_for_examples = word
            context = ""
        # If examples should be in a different language, use translation
        elif translated_word and translated_word != word:
            term_for_examples = translated_word
            context = f" (the {self.examples_language} translation of '{word}')"
        else:
            # Need to translate for examples
            term_for_examples = self.translate_word(word, self.examples_language, source_lang)
            context = f" (the {self.examples_language} translation of '{word}')" if term_for_examples != word else ""
        
        # Conversational prompt for natural examples
        prompt = f"""I'm studying the word "{term_for_examples}"{context} and need some help with examples. Could you create 2 simple, natural sentences using this word? The sentences should be in {self.examples_language}.

Please write just the 2 sentences, one per line."""
        
        self.logger.info(f"Getting examples for '{term_for_examples}'...")
        self.logger.debug(f"Prompt: {prompt[:100]}...")
        
        try:
            start_time = time.time()
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Reduced from 0.9 for more controlled output
                    "num_predict": 300,
                    "top_k": 40,
                    "top_p": 0.95
                }
            }
            
            response = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            elapsed = time.time() - start_time
            
            response_text = result.get('response', '').strip()
            
            # Parse examples from response
            examples = []
            for line in response_text.split('\n'):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Remove numbering and bullets
                line = line.lstrip('1234567890.)- •*')
                line = line.strip()
                
                # Skip meta text and introductions in multiple languages
                skip_phrases = [
                    'here are', 'example', 'sentence', 'sure', 'of course',
                    'ecco', 'le frasi', 'le due frasi', 'frasi:', 
                    'examples:', 'esempi:', 'voici', 'voilà',
                    'here you go', 'certainly', 'absolutely'
                ]
                if any(phrase in line.lower() for phrase in skip_phrases):
                    continue
                
                # Skip lines that are too short or just punctuation
                if line and len(line) > 10 and not all(c in '.,!?:;' for c in line):
                    examples.append(line)
                
                if len(examples) >= 2:  # We only want 2 examples
                    break
            
            self.logger.info(f"✓ Examples generated in {elapsed:.2f}s")
            self.logger.debug(f"Examples: {examples}")
            
            # Save request log
            self._save_request_log(word, "examples", prompt, result)
            
            self.stats['examples_generated'] += 1
            return examples[:2]  # Ensure max 2 examples
            
        except Exception as e:
            self.logger.error(f"✗ Error getting examples for '{term_for_examples}': {e}")
            self.stats['errors'] += 1
            return []
    
    def enrich_note(self, note: dict) -> dict:
        """
        Enrich a single note with definition and examples.
        
        Args:
            note: Anki note dictionary
            
        Returns:
            Enriched note dictionary
        """
        word = note['fields']['Front']
        self.stats['total_words'] += 1
        
        print(f"\n{'='*60}")
        print(f"Processing word {self.stats['total_words']}: {word}")
        print(f"{'='*60}")
        
        # Detect source language
        print(f"\n🔍 Detecting language...")
        source_language = self.detect_language(word)
        print(f"   Source language: {source_language}")
        
        # Translate if needed for definition
        translated_for_def = None
        if source_language.lower() != self.definition_language.lower():
            print(f"\n🌐 Translating to {self.definition_language} for definition...")
            translated_for_def = self.translate_word(word, self.definition_language, source_language)
            if translated_for_def != word:
                print(f"   Translation: {word} → {translated_for_def}")
        
        # Small delay to avoid overwhelming the model
        time.sleep(2)
        
        # Get definition
        print(f"\n📖 Generating definition in {self.definition_language}...")
        definition = self.get_definition(word, translated_for_def)
        
        # Small delay to avoid overwhelming the model
        time.sleep(2)
        
        # Get examples (will handle translation internally if needed)
        print(f"\n💡 Generating examples in {self.examples_language}...")
        examples = self.get_examples(word, translated_for_def)
        
        # Build back field
        back_parts = []
        
        # Add translation if it was done
        if translated_for_def and translated_for_def != word:
            back_parts.append(f"Translation ({self.definition_language}): {translated_for_def}")
        
        if definition:
            back_parts.append(f"Definition: {definition}")
            print(f"\n✓ Definition: {definition[:100]}{'...' if len(definition) > 100 else ''}")
        
        if examples:
            back_parts.append("\nExamples:")
            for i, example in enumerate(examples, 1):
                back_parts.append(f"{i}. {example}")
                print(f"✓ Example {i}: {example[:80]}{'...' if len(example) > 80 else ''}")
        
        # Update note
        enriched_note = note.copy()
        enriched_note['fields']['Back'] = '\n'.join(back_parts) if back_parts else ""
        
        # Add enrichment metadata tags
        if 'tags' not in enriched_note:
            enriched_note['tags'] = []
        
        # Add enrichment indicator tag
        enriched_note['tags'].append('gemma-enriched')
        
        # Add detected source language as a tag (e.g., "language::french")
        source_lang_tag = f'language::{source_language.lower()}'
        enriched_note['tags'].append(source_lang_tag)
        
        # Add metadata about enrichment languages
        enriched_note['tags'].append(f'def-lang:{self.definition_language}')
        enriched_note['tags'].append(f'ex-lang:{self.examples_language}')
        
        return enriched_note
    
    def enrich_notes(self, notes: List[dict], batch_size: int = 3) -> List[dict]:
        """
        Enrich multiple notes with batching and progress tracking.
        
        Args:
            notes: List of Anki notes
            batch_size: Number of notes to process before pause
            
        Returns:
            List of enriched notes
        """
        enriched_notes = []
        total_notes = len(notes)
        
        print(f"\n🚀 Starting enrichment of {total_notes} notes")
        print(f"   Definition language: {self.definition_language}")
        print(f"   Examples language: {self.examples_language}")
        print(f"   Model: {self.model}")
        print(f"   Batch size: {batch_size}")
        
        for i, note in enumerate(notes):
            # Progress indicator
            progress = (i / total_notes) * 100
            print(f"\n[{i+1}/{total_notes}] Progress: {progress:.1f}%")
            
            # Enrich the note
            enriched_note = self.enrich_note(note)
            enriched_notes.append(enriched_note)
            
            # Batch pause to avoid overwhelming the model
            if (i + 1) % batch_size == 0 and i < total_notes - 1:
                print(f"\n⏸ Batch complete, pausing for 5 seconds...")
                time.sleep(5)
        
        return enriched_notes
    
    def print_summary(self):
        """Print enrichment summary."""
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n{'='*60}")
        print(f"ENRICHMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total words processed: {self.stats['total_words']}")
        print(f"Definitions generated: {self.stats['definitions_generated']}")
        print(f"Examples generated: {self.stats['examples_generated']}")
        print(f"Translations made: {self.stats['translations_made']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Success rate: {((self.stats['definitions_generated'] + self.stats['examples_generated']) / (self.stats['total_words'] * 2) * 100):.1f}%")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average time per word: {elapsed/self.stats['total_words']:.1f}s")
        print(f"\nLogs saved to: {self.log_dir}")


def main():
    """Run vocabulary enrichment test."""
    print("=" * 70)
    print("GEMMA3:4B VOCABULARY ENRICHMENT TEST")
    print("=" * 70)
    
    # Check if input file exists
    input_file = Path("notes.json")
    if not input_file.exists():
        print(f"\n❌ Error: Input file not found: {input_file}")
        print("Please ensure notes.json exists in the project root")
        return
    
    # Load notes
    print(f"\n📄 Loading notes from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    notes = data.get('notes', [])
    settings = data.get('settings', {})
    
    print(f"✓ Loaded {len(notes)} notes")
    
    # Limit for testing (process first 5 words to test)
    test_limit = 5
    print(f"\n⚠️ Testing mode: Processing only first {test_limit} words")
    test_notes = notes[:test_limit]
    
    # Initialize enricher
    enricher = GemmaVocabularyEnricher()
    
    # Check if Gemma is available
    print(f"\n🔍 Checking Gemma3:4b availability...")
    try:
        response = requests.get(f"{enricher.url}/api/tags", timeout=5)
        models = [m['name'] for m in response.json().get('models', [])]
        
        if 'gemma3:4b' not in models:
            print(f"⚠️ Warning: gemma3:4b not found in available models")
            print(f"Available models: {', '.join(models)}")
            print(f"\nTo install: ollama pull gemma3:4b")
            
            # Check for alternative models
            if 'gemma2:2b' in models or 'gemma:2b' in models:
                print("\n💡 Alternative: You have a smaller Gemma model available.")
                print("   The test will continue but results may vary.")
        else:
            print("✓ Gemma3:4b is available")
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        print("Is Ollama running? Try: ollama serve")
        return
    
    # Enrich notes
    print(f"\n🎯 Starting enrichment process...")
    enriched_notes = enricher.enrich_notes(test_notes, batch_size=3)
    
    # Save enriched notes
    output_dir = Path("tests") / "enriched_vocabulary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"enriched_notes_{timestamp}.json"
    
    output_data = {
        'settings': settings,
        'notes': enriched_notes,
        'enrichment_info': {
            'model': enricher.model,
            'definition_language': enricher.definition_language,
            'examples_language': enricher.examples_language,
            'timestamp': timestamp,
            'total_processed': len(enriched_notes)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Enriched notes saved to: {output_file}")
    
    # Print sample of enriched content
    print(f"\n📝 Sample enriched notes:")
    for i, note in enumerate(enriched_notes[:3]):
        print(f"\n{i+1}. {note['fields']['Front']}")
        print(f"   {note['fields']['Back'][:200]}...")
    
    # Print summary
    enricher.print_summary()
    
    # Save summary to file
    summary_file = enricher.log_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Enrichment Summary - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total words: {enricher.stats['total_words']}\n")
        f.write(f"Definitions: {enricher.stats['definitions_generated']}\n")
        f.write(f"Examples: {enricher.stats['examples_generated']}\n")
        f.write(f"Errors: {enricher.stats['errors']}\n")
        f.write(f"Time: {time.time() - enricher.stats['start_time']:.1f}s\n")
    
    print(f"\n✅ Test complete! Check {output_file} for results")
    print(f"📊 Detailed logs available in: {enricher.log_dir}")


if __name__ == "__main__":
    main()
