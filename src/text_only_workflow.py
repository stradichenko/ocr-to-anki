"""
Text-only vocabulary enrichment workflow.
Input: Plain text file with vocabulary words
Output: Enriched Anki-ready flashcards
"""

from llama_cpp_server import LlamaCppServer
import json

def enrich_vocabulary(words: list[str]) -> list[dict]:
    """Enrich vocabulary words with definitions and examples."""
    flashcards = []
    
    with LlamaCppServer() as server:
        for word in words:
            print(f"Processing: {word}")
            
            # Get definition
            definition_result = server.generate(
                prompt=f'Define "{word}" in simple English.',
                max_tokens=100,
                temperature=0.1
            )
            
            # Get examples
            examples_result = server.generate(
                prompt=f'Create 2 example sentences using "{word}".',
                max_tokens=150,
                temperature=0.7
            )
            
            flashcards.append({
                'word': word,
                'definition': definition_result['content'],
                'examples': examples_result['content'],
            })
    
    return flashcards

# Usage:
words = ['bonjour', 'merci', 'au revoir']
enriched = enrich_vocabulary(words)
print(json.dumps(enriched, indent=2))
