#!/usr/bin/env python3
"""
Script to import notes from JSON file to Anki via AnkiConnect.
Usage: python anki_importer.py <json_file>
"""

import sys
import json
import yaml
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path


class AnkiConnectError(Exception):
    """Exception raised when AnkiConnect returns an error."""
    pass


class AnkiImporter:
    def __init__(self, config_path: Optional[Path] = None):
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        
        self.config = self._load_config(config_path)
        
        # Set connection parameters from config
        anki_config = self.config.get("anki_connect", {})
        self.url = anki_config.get("url", "http://localhost:8765")
        self.version = anki_config.get("version", 6)
        self.timeout = anki_config.get("timeout", 10)
        
        # Load import defaults
        self.import_defaults = self.config.get("import_defaults", {})
        
        # Load logging settings
        self.logging = self.config.get("logging", {})
        self.verbose = self.logging.get("verbose", True)
        self.show_progress = self.logging.get("show_progress", True)
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            if self.verbose:
                print(f"⚠ Config file not found: {config_path}, using defaults")
            return {}
        except yaml.YAMLError as e:
            print(f"⚠ Error parsing config file: {e}, using defaults")
            return {}
    
    def invoke(self, action: str, params: Dict[str, Any]) -> Any:
        """Send request to AnkiConnect API."""
        payload = {
            "action": action,
            "version": self.version,
            "params": params
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            raise AnkiConnectError(
                "Could not connect to AnkiConnect. "
                "Make sure Anki is running with AnkiConnect installed."
            )
        except requests.exceptions.RequestException as e:
            raise AnkiConnectError(f"Request failed: {e}")
        
        if result.get("error"):
            raise AnkiConnectError(f"AnkiConnect error: {result['error']}")
        
        return result.get("result")
    
    def check_connection(self) -> bool:
        """Check if AnkiConnect is available."""
        try:
            version = self.invoke("version", {})
            print(f"✓ Connected to AnkiConnect (version {version})")
            return True
        except AnkiConnectError as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def create_deck(self, deck_name: str) -> None:
        """Create a deck if it doesn't exist."""
        try:
            self.invoke("createDeck", {"deck": deck_name})
            print(f"✓ Deck '{deck_name}' ready")
        except AnkiConnectError:
            pass  # Deck might already exist
    
    def add_note(self, note: Dict[str, Any]) -> Optional[int]:
        """Add a single note to Anki."""
        try:
            note_id = self.invoke("addNote", {"note": note})
            return note_id
        except AnkiConnectError as e:
            print(f"✗ Failed to add note: {e}")
            return None
    
    def add_notes(self, notes: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Add multiple notes to Anki in batch."""
        try:
            note_ids = self.invoke("addNotes", {"notes": notes})
            return note_ids
        except AnkiConnectError as e:
            print(f"✗ Failed to add notes: {e}")
            return [None] * len(notes)
    
    def import_from_json(self, json_path: Path) -> Dict[str, int]:
        """Import notes from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract notes and settings (JSON can override config defaults)
        notes = data.get("notes", [])
        settings = data.get("settings", {})
        
        # Use config defaults, but allow JSON to override
        default_deck = settings.get("defaultDeck", self.import_defaults.get("deck", "Default"))
        default_model = settings.get("defaultModel", self.import_defaults.get("model", "Basic"))
        batch_size = settings.get("batchSize", self.import_defaults.get("batch_size", 10))
        allow_duplicates = settings.get("allowDuplicates", self.import_defaults.get("allow_duplicates", False))
        duplicate_scope = settings.get("duplicateScope", self.import_defaults.get("duplicate_scope", "deck"))
        
        if not notes:
            print("No notes found in JSON file")
            return {"total": 0, "success": 0, "failed": 0}
        
        if self.verbose:
            print(f"\nImporting {len(notes)} notes...")
            print(f"  Default deck: {default_deck}")
            print(f"  Default model: {default_model}")
            print(f"  Batch size: {batch_size}")
        
        # Create deck if needed
        self.create_deck(default_deck)
        
        # Process notes in batches
        success_count = 0
        failed_count = 0
        
        for i in range(0, len(notes), batch_size):
            batch = notes[i:i + batch_size]
            
            # Prepare notes with defaults
            prepared_notes = []
            for note in batch:
                prepared_note = {
                    "deckName": note.get("deckName", default_deck),
                    "modelName": note.get("modelName", default_model),
                    "fields": note["fields"],
                    "tags": note.get("tags", []),
                    "options": {
                        "allowDuplicate": note.get("allowDuplicate", allow_duplicates),
                        "duplicateScope": duplicate_scope
                    }
                }
                
                # Add media if present
                for media_type in ["audio", "video", "picture"]:
                    if media_type in note:
                        prepared_note[media_type] = note[media_type]
                
                prepared_notes.append(prepared_note)
            
            # Add batch
            results = self.add_notes(prepared_notes)
            
            for j, note_id in enumerate(results):
                if note_id:
                    success_count += 1
                    if self.show_progress:
                        fields = batch[j]["fields"]
                        front = list(fields.values())[0][:50]
                        print(f"  ✓ Added: {front}... (ID: {note_id})")
                else:
                    failed_count += 1
                    if self.show_progress:
                        fields = batch[j]["fields"]
                        front = list(fields.values())[0][:50]
                        print(f"  ✗ Failed: {front}...")
        
        return {
            "total": len(notes),
            "success": success_count,
            "failed": failed_count
        }


def main():
    if len(sys.argv) != 2:
        print("Usage: python anki_importer.py <json_file>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: File '{json_path}' not found")
        sys.exit(1)
    
    # Create importer with default config path
    importer = AnkiImporter()
    
    # Check connection
    if not importer.check_connection():
        sys.exit(1)
    
    # Import notes
    results = importer.import_from_json(json_path)
    
    # Print summary
    print("\n" + "="*50)
    print("Import Summary:")
    print(f"  Total notes: {results['total']}")
    print(f"  Successfully added: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print("="*50)


if __name__ == "__main__":
    main()