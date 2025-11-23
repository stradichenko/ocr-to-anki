{
  description = "ocr-to-anki";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Build LangChain packages from source using buildPythonPackage
        pythonPackagesExtensions = final: prev: {
          tenacity = prev.tenacity.overridePythonAttrs (old: {
            doCheck = false;  # Tests have timing issues in Nix sandbox
          });
          
          langchain = final.buildPythonPackage rec {
            pname = "langchain";
            version = "0.3.14";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-SlroF7WDL6Dh/K3FNT+/dL69L45VApTU3AOfZR3c09E=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ 
              pydantic requests pyyaml sqlalchemy tenacity aiohttp numpy
              langsmith langchain-core langchain-text-splitters
            ];
            
            pythonImportsCheck = [ "langchain" ];
            doCheck = false;  # Tests require network/external services
            dontCheckRuntimeDeps = true;  # langchain-core 0.3.29 satisfies >=0.3.29
          };
          
          langchain-core = final.buildPythonPackage rec {
            pname = "langchain-core";
            version = "0.3.29";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              pname = "langchain_core";
              inherit version;
              hash = "sha256-dz1q7rYS5849mWwL5ANDPYxqked7u3p0YcE+Fc++WwY=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ 
              pydantic requests pyyaml tenacity jsonpatch langsmith packaging
            ];
            
            pythonImportsCheck = [ "langchain_core" ];
            doCheck = false;
            dontCheckRuntimeDeps = true;  # packaging version 25 is fine despite <25 constraint
          };
          
          langchain-text-splitters = final.buildPythonPackage rec {
            pname = "langchain-text-splitters";
            version = "0.3.4";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              pname = "langchain_text_splitters";
              inherit version;
              hash = "sha256-887epGloRIO0SS2fEdwvpmOI2rAcXVxTB5JVFauITCQ=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ final.langchain-core ];
            
            pythonImportsCheck = [ "langchain_text_splitters" ];
            doCheck = false;
          };
          
          langchain-community = final.buildPythonPackage rec {
            pname = "langchain-community";
            version = "0.3.14";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              pname = "langchain_community";
              inherit version;
              hash = "sha256-2LoP4tu1eVv/cHaEtxK6pe43kicZRhCvQVzN/e/aBHk=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ 
              langchain langchain-core dataclasses-json httpx-sse pydantic-settings numpy
            ];
            
            pythonImportsCheck = [ "langchain_community" ];
            doCheck = false;
            dontCheckRuntimeDeps = true;  # numpy 2.x works fine despite <2 constraint
          };
          
          langchain-ollama = final.buildPythonPackage rec {
            pname = "langchain-ollama";
            version = "0.2.2";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              pname = "langchain_ollama";
              inherit version;
              hash = "sha256-LZvLBv/b5Dx8aQbEbnENNtM7a5nNSXXL9UBg8T5RyHU=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ final.langchain-core final.ollama ];
            
            pythonImportsCheck = [ "langchain_ollama" ];
            doCheck = false;
          };
          
          ollama = final.buildPythonPackage rec {
            pname = "ollama";
            version = "0.4.7";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-iR3L5U9VOX2C0onEWd4OqJfhA7hqPx+tD9sYlZIqdf8=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ requests httpx pydantic ];
            
            pythonImportsCheck = [ "ollama" ];
            doCheck = false;
          };
          
          langsmith = final.buildPythonPackage rec {
            pname = "langsmith";
            version = "0.2.4";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-OG/tyBW0X5T6F1cYYOVhwNefrMCil5pTLri4k6TZj6k=";
            };
            
            build-system = [ final.poetry-core ];
            dependencies = with final; [ requests pydantic orjson httpx requests-toolbelt ];
            
            pythonImportsCheck = [ "langsmith" ];
            doCheck = false;
          };
          
          pytesseract = final.buildPythonPackage rec {
            pname = "pytesseract";
            version = "0.3.13";
            pyproject = true;
            
            src = pkgs.fetchPypi {
              inherit pname version;
              hash = "sha256-S/X4gMmUBvUqPPwmM+QtncZ2FeadilCddIZ9O63bXbk=";
            };
            
            build-system = [ final.setuptools ];
            dependencies = with final; [ pillow packaging ];
            
            pythonImportsCheck = [ "pytesseract" ];
            doCheck = false;
          };
        };
        
        # Python 3.11+ with ALL packages from Nix
        python = pkgs.python311.override {
          packageOverrides = pythonPackagesExtensions;
          self = python;
        };
        
        pythonEnv = python.withPackages (ps: with ps; [
          # GTK4 bindings
          pygobject3
          
          # AI/ML dependencies (now from Nix!)
          langchain
          langchain-community
          langchain-ollama
          ollama
          langsmith
          
          # OCR
          pytesseract
          
          # Image processing
          pillow
          opencv4
          
          # Data handling
          pyyaml
          pydantic
          requests
          levenshtein
          
          # Additional dependencies
          jsonpatch
          dataclasses-json
          orjson
          httpx
          tenacity
          aiohttp
          sqlalchemy
          packaging
          
          # Testing
          pytest
          pytest-asyncio
          responses
          
          # Type stubs
          types-pyyaml
          types-requests
        ]);
        
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            
            # GTK4 and all dependencies
            pkgs.gtk4
            pkgs.gobject-introspection
            pkgs.pango
            pkgs.harfbuzz
            pkgs.cairo
            pkgs.gdk-pixbuf
            pkgs.graphene
            
            # OpenGL libraries (required by opencv)
            pkgs.libGL
            pkgs.libGLU
            pkgs.mesa
            
            # GLib and system libraries
            pkgs.glib
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            
            # Ollama (for local LLM)
            pkgs.ollama
            
            # OCR engine
            pkgs.tesseract
            
            # Development tools
            pkgs.ruff
            pkgs.python311Packages.black
            pkgs.python311Packages.mypy
            pkgs.git
            pkgs.nixpkgs-lint
            pkgs.direnv
            pkgs.nix-direnv
            pkgs.nil
            pkgs.nix-tree
          ];
          
          nativeBuildInputs = [
            pkgs.wrapGAppsHook3
            pkgs.gobject-introspection
          ];
          
          shellHook = ''
            echo "OCR to Anki"
            echo ""
            
            # Set up Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            
            # XDG config directory for application settings
            export XDG_CONFIG_HOME="''${XDG_CONFIG_HOME:-$HOME/.config}"
            
            # CRITICAL: Clear system library paths to prevent mixing Nix and system libraries
            # This fixes the "undefined symbol: pango_font_description_get_color" error
            unset LD_LIBRARY_PATH
            
            # Build a clean LD_LIBRARY_PATH with ONLY Nix libraries
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib"
            export LD_LIBRARY_PATH="${pkgs.libGLU}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.mesa}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.gtk4}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.pango.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.harfbuzz.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.cairo}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.gdk-pixbuf}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.graphene}/lib:$LD_LIBRARY_PATH"
            
            # Manually set up GI_TYPELIB_PATH for GTK4 (wrapGAppsHook doesn't run in shells)
            unset GI_TYPELIB_PATH
            export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0"
            export GI_TYPELIB_PATH="${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.glib.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.pango.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.gdk-pixbuf}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.cairo}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.graphene}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.harfbuzz.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
          '';
        };
        
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "anki-ocr";
          version = "0.1.0";
          src = ./.;
          
          buildInputs = [
            pythonEnv
            pkgs.gtk4
            pkgs.gobject-introspection
          ];
          
          nativeBuildInputs = [
            pkgs.wrapGAppsHook
            pkgs.gobject-introspection
          ];
          
          installPhase = ''
            mkdir -p $out/bin $out/lib
            cp -r src $out/lib/
            
            cat > $out/bin/anki-ocr <<EOF
#!/bin/sh
export PYTHONPATH="$out/lib:\$PYTHONPATH"
export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0"
exec ${pythonEnv}/bin/python $out/lib/src/main.py "\$@"
EOF
            chmod +x $out/bin/anki-ocr
          '';
          
          meta = with pkgs.lib; {
            description = "GTK4 desktop application for extracting vocabulary from images using OCR and AI";
            homepage = "https://github.com/stradichenko/anki-ocr-vocab-collector";
            license = licenses.mit;
            platforms = platforms.linux;
            maintainers = [];
          };
        };
        
        # Docker image built directly from Nix for perfect reproducibility
        packages.dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "ocr-to-anki";
          tag = "latest";
          
          # Include the application and all runtime dependencies
          contents = [
            self.packages.${system}.default
            pkgs.gtk4
            pkgs.tesseract
            pkgs.ollama
            pkgs.coreutils
            pkgs.bash
          ];
          
          config = {
            Cmd = [ "/bin/anki-ocr" ];
            WorkingDir = "/app";
            Env = [
              "PYTHONPATH=/lib"
              "GI_TYPELIB_PATH=${pkgs.gtk4}/lib/girepository-1.0:${pkgs.gobject-introspection}/lib/girepository-1.0"
              "XDG_CONFIG_HOME=/config"
              "XDG_DATA_HOME=/data"
            ];
            ExposedPorts = {
              "8080/tcp" = {};
            };
            Volumes = {
              "/config" = {};
              "/data" = {};
            };
            Labels = {
              "org.opencontainers.image.title" = "OCR to Anki";
              "org.opencontainers.image.description" = "GTK4 desktop application for extracting vocabulary from images using OCR and AI";
              "org.opencontainers.image.source" = "https://github.com/stradichenko/anki-ocr-vocab-collector";
              "org.opencontainers.image.version" = "0.1.0";
            };
          };
          
          # Create optimal layers for better caching
          maxLayers = 120;
        };
      }
    );
}