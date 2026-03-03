# OCR to Anki — Flutter App

Cross-platform GUI for the OCR to Anki pipeline.

## Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux    | OK     | Requires NixOS dev shell for build |
| macOS    | OK     | Requires Xcode |
| Windows  | OK     | Requires Visual Studio Build Tools |
| Android  | OK     | Requires Android SDK |

## Architecture

- **State management:** Riverpod
- **Local database:** Drift (SQLite)
- **Routing:** Named routes (MaterialApp)
- **Theming:** Material 3 with `colorSchemeSeed: deepOrange`

## Screens

| Screen | Route | Purpose |
|--------|-------|---------|
| Home | `/` | Context selection (handwritten/highlighted), colour picker, image upload |
| Processing | `/processing` | Progress indicator with phase steps (crop → OCR → enrich) |
| Review | `/review` | Edit cards, toggle selection, export to Anki or JSON |
| Settings | `/settings` | Inference mode, languages, AnkiConnect, highlight detection, LLM params |
| History | `/history` | Past sessions with word lists and export status |

## Services

### InferenceService
Talks to the Python FastAPI backend (remote mode). Future: embedded on-device
inference via llama.cpp FFI.

### HighlightDetector
Pure-Dart port of the Python `highlight_cropper.py`. Detects 6 highlight colours
(yellow, orange, red, green, blue, purple) using HSV colour space analysis with
morphological cleanup and connected-component labelling.

### AnkiExportService
- **AnkiConnect** (desktop): HTTP JSON-RPC to localhost:8765
- **JSON export**: Anki-importable JSON file

## Development

```bash
# Get dependencies
flutter pub get

# Generate drift database code
flutter pub run build_runner build --delete-conflicting-outputs

# Run analysis
flutter analyze

# Run tests
flutter test

# Run the app
flutter run -d linux
```
