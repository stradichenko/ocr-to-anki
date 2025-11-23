# ocr-to-anki


```
ocr-to-anki/
├── flake.nix                  # Main Nix flake: devShell, packages, apps
├── flake.lock                 # Auto-generated lock file for Nix reproducibility
│
├── nix/                       # All Nix-related modules (keeps flake.nix clean)
│   ├── devshell.nix           # Defines development environment (Python, GTK, tools)
│   ├── python-env.nix         # Python environment (if using Nix for Python deps)
│   ├── overlays.nix           # Optional overlays or custom derivations
│   └── packages.nix           # Definitions for building the app as a Nix package
│
├── src/                       # ***ALL application code lives here***
│   ├── app_core/              # Core logic (independent from UI)
│   │   ├── __init__.py
│   │   ├── engine.py          # Example: processing logic
│   │   └── utils.py           # Shared utilities
│   │
│   ├── app_ui/                # GUI code (GTK)
│   │   ├── __init__.py
│   │   ├── main_window.py     # Main GTK window
│   │   ├── widgets/           # Custom widgets
│   │   │   ├── __init__.py
│   │   │   └── progress_panel.py
│   │   ├── dialogs/           # Popups, configuration dialogs
│   │   │   ├── __init__.py
│   │   │   └── settings_dialog.py
│   │   └── ui/                # GTK Builder XML/UI definitions
│   │       ├── main_window.ui
│   │       └── styles.css
│   │
│   ├── cli/                   # CLI commands the user can run
│   │   ├── __init__.py
│   │   └── main.py            # Defines commands: myapp analyze / export / etc.
│   │
│   ├── __main__.py            # Entry point for `python -m src` or `nix run`
│   └── config.py              # Centralized Python config loader
│
├── resources/                 # Files shipped with the app
│   ├── icons/                 # PNG/SVG icons
│   └── sample_data/           # Optional bundled data
│
├── logs/                      # ***Dev-only logs*** (ignored by git)
│   └── .keep                  # Empty file to keep folder in repo
│
├── tests/                     # Automated tests
│   ├── unit/                  # Tests pure logic (app_core)
│   │   └── test_engine.py
│   ├── integration/           # UI + core interaction tests
│   └── ui/                    # Optional: automated GTK tests
│
├── scripts/                   # Dev/maintainer scripts (NOT shipped to users)
│   ├── format.sh              # Code formatting helper
│   ├── update.sh              # Update Nix flake inputs
│   ├── run-dev.sh             # Run app with dev paths enabled
│   └── generate-docs.sh       # Build documentation
│
├── bash/                      # Runtime bash scripts (if part of the app)
│   ├── __init__.txt           # Documentation for scripts
│   ├── helper.sh              # Script used by the Python app via subprocess
│   └── collect_info.sh        # Example: gather system information
│
├── docker/                    # All Docker-related files
│   ├── Dockerfile             # Build the app image
│   └── compose.yml            # Optional docker-compose setup
│
├── docs/                      # Documentation for users/devs
│   ├── installation.md
│   ├── architecture.md
│   └── ui-design.md           # Describe GTK layout & structure
│
├── .gitignore
└── README.md                  # Project overview
```
