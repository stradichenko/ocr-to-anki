{
  description = "ocr-to-anki";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Allow unfree packages (for CUDA if needed)
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;  # Enable CUDA support globally
          };
        };
        
        # Build LangChain packages from source using buildPythonPackage
        pythonPackagesExtensions = final: prev: {
          tenacity = prev.tenacity.overridePythonAttrs (old: {
            doCheck = false;
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
            doCheck = false;
            dontCheckRuntimeDeps = true;
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
            dontCheckRuntimeDeps = true;
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
            dontCheckRuntimeDeps = true;
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
        
        # Python 3.11+ with packages from Nix
        python = pkgs.python311.override {
          packageOverrides = pythonPackagesExtensions;
          self = python;
        };
        
        pythonEnv = python.withPackages (ps: with ps; [
          # GTK4 bindings
          pygobject3
          
          # AI/ML dependencies
          langchain
          langchain-community
          langchain-ollama
          ollama
          langsmith
          
          # HuggingFace Hub (for model downloads) - ALREADY THERE!
          huggingface-hub
          
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
        
        # Build llama-gemma3-cli from source (proper Nix derivation)
        llama-gemma3-cli = pkgs.stdenv.mkDerivation {
          pname = "llama-gemma3-cli";
          version = "b6981-647b960";  # Match llama-server version
          
          src = pkgs.fetchFromGitHub {
            owner = "ggerganov";
            repo = "llama.cpp";
            rev = "647b960";
            sha256 = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";  # Will fail, update with real hash
          };
          
          nativeBuildInputs = with pkgs; [ cmake pkg-config ];
          
          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DGGML_LLAMAFILE=ON"
            "-DGGML_OPENMP=ON"
            "-DGGML_NATIVE=OFF"
            "-DLLAMA_CURL=OFF"  # Disable curl dependency
          ];
          
          buildPhase = ''
            runHook preBuild
            cmake --build . --config Release --target llama-gemma3-cli -j$NIX_BUILD_CORES
            runHook postBuild
          '';
          
          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            cp bin/llama-gemma3-cli $out/bin/
            runHook postInstall
          '';
          
          meta = with pkgs.lib; {
            description = "Gemma 3 vision CLI for llama.cpp";
            homepage = "https://github.com/ggerganov/llama.cpp";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
        
        # Build llama-mtmd-cli from source (FIXED VERSION)
        llama-mtmd-cli = pkgs.stdenv.mkDerivation rec {
          pname = "llama-mtmd-cli";
          version = "6981-647b960";
          
          src = pkgs.fetchFromGitHub {
            owner = "ggerganov";
            repo = "llama.cpp";
            rev = "647b960cf5ec5497c7d3e2c3d4eb3b7ce5be34d2";
            sha256 = "sha256-0000000000000000000000000000000000000000000=";
          };
          
          nativeBuildInputs = with pkgs; [ cmake pkg-config ];
          
          buildInputs = with pkgs; [ ];
          
          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DGGML_LLAMAFILE=ON"
            "-DGGML_OPENMP=ON"
            "-DGGML_NATIVE=OFF"
            "-DLLAMA_CURL=OFF"
            "-DBUILD_SHARED_LIBS=OFF"  # Static linking to avoid library issues
          ];
          
          buildPhase = ''
            runHook preBuild
            cmake -B build .
            cmake --build build --config Release --target llama-mtmd-cli -j$NIX_BUILD_CORES
            runHook postBuild
          '';
          
          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            cp build/bin/llama-mtmd-cli $out/bin/ || cp build/llama-mtmd-cli $out/bin/
            chmod +x $out/bin/llama-mtmd-cli
            runHook postInstall
          '';
          
          meta = with pkgs.lib; {
            description = "Multi-modal CLI for llama.cpp (Gemma 3 vision support)";
            homepage = "https://github.com/ggerganov/llama.cpp";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
        
        # Intel OneAPI for SYCL (manual overlay since not in nixpkgs)
        intelOneAPIStub = pkgs.stdenv.mkDerivation {
          name = "intel-oneapi-stub";
          version = "2025.0";
          
          # This is a stub that checks for system OneAPI
          phases = [ "installPhase" ];
          
          installPhase = ''
            mkdir -p $out/bin
            
            # Create wrapper scripts that look for system OneAPI
            cat > $out/bin/check-oneapi <<'EOF'
#!/usr/bin/env bash
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
if [ -d "$ONEAPI_ROOT" ]; then
  echo "Intel OneAPI found at: $ONEAPI_ROOT"
  exit 0
else
  echo "Intel OneAPI not found at: $ONEAPI_ROOT"
  echo "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
  exit 1
fi
EOF
            chmod +x $out/bin/check-oneapi
            
            # Create icx/icpx wrappers that source OneAPI environment
            for compiler in icx icpx; do
              cat > $out/bin/$compiler <<EOF
#!/usr/bin/env bash
ONEAPI_ROOT="\''${ONEAPI_ROOT:-/opt/intel/oneapi}"
if [ -f "\$ONEAPI_ROOT/setvars.sh" ]; then
  source "\$ONEAPI_ROOT/setvars.sh" >/dev/null 2>&1
  exec "\$ONEAPI_ROOT/compiler/latest/bin/$compiler" "\$@"
else
  echo "Error: Intel OneAPI not found at \$ONEAPI_ROOT" >&2
  echo "Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html" >&2
  exit 1
fi
EOF
              chmod +x $out/bin/$compiler
            done
          '';
          
          meta = {
            description = "Intel OneAPI stub for SYCL builds";
            longDescription = ''
              This package provides wrapper scripts for Intel OneAPI compilers.
              Requires Intel OneAPI to be installed at /opt/intel/oneapi.
            '';
          };
        };
        
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
            
            # llama.cpp (includes llama-server and llama-cli)
            pkgs.llama-cpp
            
            # Build dependencies for multi-backend builds
            pkgs.cmake
            pkgs.pkg-config
            pkgs.git
            
            # Vulkan support (for GPU acceleration)
            pkgs.vulkan-headers
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.shaderc  # Provides glslc compiler
            
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
          
          shellHook = ''
            echo "OCR to Anki"
            echo ""
            
            # Set up Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            
            # XDG config directory for application settings
            export XDG_CONFIG_HOME="''${XDG_CONFIG_HOME:-$HOME/.config}"
            
            # llama.cpp model directory
            export LLAMA_CPP_MODELS="''${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
            mkdir -p "$LLAMA_CPP_MODELS"
            
            # CRITICAL: Clear system library paths to prevent mixing Nix and system libraries
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
            
            # Manually set up GI_TYPELIB_PATH for GTK4
            unset GI_TYPELIB_PATH
            export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0"
            export GI_TYPELIB_PATH="${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.glib.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.pango.out}/lib:girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.gdk-pixbuf}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.cairo}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.graphene}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.harfbuzz.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            
            # Detect hardware and provide guidance
            echo "Hardware Detection:"
            if command -v nvidia-smi >/dev/null 2>&1; then
              echo "  ✅ NVIDIA GPU detected"
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "     (GPU info unavailable)"
              echo "  → llama.cpp will use GPU acceleration"
              
              # Check for CUDA compiler
              if command -v nvcc >/dev/null 2>&1; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                echo "  → CUDA compiler: $CUDA_VERSION"
              else
                echo "  ⚠️  CUDA compiler not in Nix environment"
                echo "     For CUDA builds, use: nix develop --impure .#cuda"
              fi
            elif [[ "$OSTYPE" == "darwin"* ]]; then
              echo "  🍎 Apple Silicon detected"
              echo "  → llama.cpp has Metal support"
            else
              echo "  💻 CPU-only mode"
              echo "  → llama.cpp will run on CPU (slower but works)"
            fi
            echo ""
            
            echo "AI Backend Options:"
            echo "  1. llama.cpp (fully offline, recommended) - ./scripts/setup-llama-cpp.sh"
            echo "  2. Ollama (requires internet) - olloma pull gemma3:4b"
            echo ""
            
            echo "llama.cpp status:"
            if [ -f "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0.gguf" ]; then
              echo "  ✅ Gemma 3 4B model found"
              MODEL_SIZE=$(du -h "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0.gguf" 2>/dev/null | cut -f1)
              echo "  📦 Model size: $MODEL_SIZE"
              echo "  📍 Location: $LLAMA_CPP_MODELS"
            else
              echo "  ⚠️  Gemma 3 4B model not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            
            # Check HuggingFace authentication
            echo ""
            echo "HuggingFace Hub:"
            if command -v huggingface-cli >/dev/null 2>&1; then
              if huggingface-cli whoami >/dev/null 2>&1; then
                HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
                echo "  ✅ Logged in as: $HF_USER"
              else
                echo "  ⚠️  Not logged in"
                echo "  To download Gemma models, login with:"
                echo "    huggingface-cli login"
              fi
            else
              echo "  ⚠️  huggingface-cli not found (included in environment)"
            fi
            echo ""
            
            # llama-mtmd-cli setup
            LLAMAMTMD_BIN="$HOME/.local/bin/llama-mtmd-cli"
            
            echo "llama.cpp tools:"
            echo "  • llama-server: ✅ (from nixpkgs)"
            echo "  • llama-cli: ✅ (from nixpkgs)"
            
            # Check for Vulkan support
            if command -v vulkaninfo >/dev/null 2>&1; then
              echo "  • Vulkan: ✅ (GPU backend available)"
              VULKAN_DEVICES=$(vulkaninfo --summary 2>/dev/null | grep "GPU" | wc -l)
              if [ "$VULKAN_DEVICES" -gt 0 ]; then
                echo "    Devices: $VULKAN_DEVICES GPU(s) detected"
              fi
            else
              echo "  • Vulkan: ⚠️  (not available)"
            fi
            
            if command -v llama-mtmd-cli >/dev/null 2>&1; then
              # Check if it's actually working
              if llama-mtmd-cli --version >/dev/null 2>&1; then
                echo "  • llama-mtmd-cli: ✅ (manually built, working)"
              else
                echo "  • llama-mtmd-cli: ⚠️  (found but has library issues)"
                echo "    Rebuild with: ./scripts/build-llama-gemma3-cli.sh"
              fi
            else
              echo "  • llama-mtmd-cli: ❌ (not found)"
              echo "    Build with: ./scripts/build-llama-mtmd-multibackend.sh"
              echo "    Available backends:"
              echo "      --vulkan  (works in Nix, recommended)"
              echo "      --cuda    (requires system CUDA, use outside Nix)"
              echo "      --all     (auto-detect all available)"
            fi
            echo ""
          '';
        };
        
        # CUDA-enabled development shell (separate)
        devShells.cuda = pkgs.mkShell {
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
            
            # llama.cpp (includes llama-server and llama-cli)
            pkgs.llama-cpp
            
            # Build dependencies for multi-backend builds
            pkgs.cmake
            pkgs.pkg-config
            pkgs.git
            
            # Vulkan support (for GPU acceleration)
            pkgs.vulkan-headers
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.shaderc  # Provides glslc compiler
            
            # CUDA toolkit (large download!)
            pkgs.cudatoolkit
            pkgs.cudaPackages.cuda_nvcc
          ];
          
          shellHook = ''
            echo "OCR to Anki (CUDA-enabled environment)"
            echo ""
            
            # Export CUDA paths
            export CUDA_PATH="${pkgs.cudatoolkit}"
            export CUDA_HOME="${pkgs.cudatoolkit}"
            export PATH="${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH"
            
            echo "CUDA Environment:"
            echo "  CUDA_PATH: $CUDA_PATH"
            if command -v nvcc >/dev/null 2>&1; then
              echo "  ✅ nvcc: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
            else
              echo "  ❌ nvcc: not found"
            fi
            echo ""
            
            # Set up Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            
            # XDG config directory for application settings
            export XDG_CONFIG_HOME="''${XDG_CONFIG_HOME:-$HOME/.config}"
            
            # llama.cpp model directory
            export LLAMA_CPP_MODELS="''${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
            mkdir -p "$LLAMA_CPP_MODELS"
            
            # CRITICAL: Clear system library paths to prevent mixing Nix and system libraries
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
            
            # Manually set up GI_TYPELIB_PATH for GTK4
            unset GI_TYPELIB_PATH
            export GI_TYPELIB_PATH="${pkgs.gtk4}/lib/girepository-1.0"
            export GI_TYPELIB_PATH="${pkgs.gobject-introspection}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.glib.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.pango.out}/lib:girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.gdk-pixbuf}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.cairo}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.graphene}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            export GI_TYPELIB_PATH="${pkgs.harfbuzz.out}/lib/girepository-1.0:$GI_TYPELIB_PATH"
            
            # Detect hardware and provide guidance
            echo "Hardware Detection:"
            if command -v nvidia-smi >/dev/null 2>&1; then
              echo "  ✅ NVIDIA GPU detected"
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "     (GPU info unavailable)"
              echo "  → llama.cpp will use GPU acceleration"
              
              # Check for CUDA compiler
              if command -v nvcc >/dev/null 2>&1; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                echo "  → CUDA compiler: $CUDA_VERSION"
              else
                echo "  ⚠️  CUDA compiler not in Nix environment"
                echo "     For CUDA builds, use: nix develop --impure .#cuda"
              fi
            elif [[ "$OSTYPE" == "darwin"* ]]; then
              echo "  🍎 Apple Silicon detected"
              echo "  → llama.cpp has Metal support"
            else
              echo "  💻 CPU-only mode"
              echo "  → llama.cpp will run on CPU (slower but works)"
            fi
            echo ""
            
            echo "AI Backend Options:"
            echo "  1. llama.cpp (fully offline, recommended) - ./scripts/setup-llama-cpp.sh"
            echo "  2. Ollama (requires internet) - olloma pull gemma3:4b"
            echo ""
            
            echo "llama.cpp status:"
            if [ -f "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0.gguf" ]; then
              echo "  ✅ Gemma 3 4B model found"
              MODEL_SIZE=$(du -h "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0.gguf" 2>/dev/null | cut -f1)
              echo "  📦 Model size: $MODEL_SIZE"
              echo "  📍 Location: $LLAMA_CPP_MODELS"
            else
              echo "  ⚠️  Gemma 3 4B model not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            
            # Check HuggingFace authentication
            echo ""
            echo "HuggingFace Hub:"
            if command -v huggingface-cli >/dev/null 2>&1; then
              if huggingface-cli whoami >/dev/null 2>&1; then
                HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
                echo "  ✅ Logged in as: $HF_USER"
              else
                echo "  ⚠️  Not logged in"
                echo "  To download Gemma models, login with:"
                echo "    huggingface-cli login"
              fi
            else
              echo "  ⚠️  huggingface-cli not found (included in environment)"
            fi
            echo ""
            
            # llama-mtmd-cli setup
            LLAMAMTMD_BIN="$HOME/.local/bin/llama-mtmd-cli"
            
            echo "llama.cpp tools:"
            echo "  • llama-server: ✅ (from nixpkgs)"
            echo "  • llama-cli: ✅ (from nixpkgs)"
            
            # Check for Vulkan support
            if command -v vulkaninfo >/dev/null 2>&1; then
              echo "  • Vulkan: ✅ (GPU backend available)"
              VULKAN_DEVICES=$(vulkaninfo --summary 2>/dev/null | grep "GPU" | wc -l)
              if [ "$VULKAN_DEVICES" -gt 0 ]; then
                echo "    Devices: $VULKAN_DEVICES GPU(s) detected"
              fi
            else
              echo "  • Vulkan: ⚠️  (not available)"
            fi
            
            if command -v llama-mtmd-cli >/dev/null 2>&1; then
              # Check if it's actually working
              if llama-mtmd-cli --version >/dev/null 2>&1; then
                echo "  • llama-mtmd-cli: ✅ (manually built, working)"
              else
                echo "  • llama-mtmd-cli: ⚠️  (found but has library issues)"
                echo "    Rebuild with: ./scripts/build-llama-gemma3-cli.sh"
              fi
            else
              echo "  • llama-mtmd-cli: ❌ (not found)"
              echo "    Build with: ./scripts/build-llama-mtmd-multibackend.sh"
              echo "    Available backends:"
              echo "      --vulkan  (works in Nix, recommended)"
              echo "      --cuda    (requires system CUDA, use outside Nix)"
              echo "      --all     (auto-detect all available)"
            fi
            echo ""
          '';
        };
        
        # SYCL-enabled development shell (Intel OneAPI)
        devShells.sycl = pkgs.mkShell {
          packages = [
            pythonEnv
            
            # All existing packages
            pkgs.gtk4
            pkgs.gobject-introspection
            pkgs.pango
            pkgs.harfbuzz
            pkgs.cairo
            pkgs.gdk-pixbuf
            pkgs.graphene
            pkgs.libGL
            pkgs.libGLU
            pkgs.mesa
            pkgs.glib
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.llama-cpp
            
            # Build tools
            pkgs.cmake
            pkgs.pkg-config
            pkgs.git
            
            # Vulkan (works alongside SYCL)
            pkgs.vulkan-headers
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.shaderc
            
            # Intel OneAPI stub
            intelOneAPIStub
          ];
          
          shellHook = ''
            echo "OCR to Anki (Intel SYCL environment)"
            echo ""
            
            # Set up Intel OneAPI environment
            export ONEAPI_ROOT="''${ONEAPI_ROOT:-/opt/intel/oneapi}"
            
            # Check for Intel OneAPI
            if [ -d "$ONEAPI_ROOT" ]; then
              echo "Intel OneAPI Configuration:"
              echo "  ✅ Found at: $ONEAPI_ROOT"
              
              # Source OneAPI environment
              if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
                source "$ONEAPI_ROOT/setvars.sh" >/dev/null 2>&1
                
                # Verify compilers
                if command -v icx >/dev/null 2>&1; then
                  echo "  ✅ icx: $(icx --version | head -1)"
                fi
                if command -v icpx >/dev/null 2>&1; then
                  echo "  ✅ icpx: $(icpx --version | head -1)"
                fi
                
                # OneAPI provides its own Level Zero
                echo "  ✅ Level Zero: Provided by Intel OneAPI"
              else
                echo "  ⚠️  setvars.sh not found"
              fi
            else
              echo "Intel OneAPI: ❌ Not installed"
              echo ""
              echo "To enable SYCL support:"
              echo "  1. Download Intel OneAPI Base Toolkit:"
              echo "     https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
              echo ""
              echo "  2. Install to /opt/intel/oneapi (or set ONEAPI_ROOT)"
              echo ""
              echo "  3. Re-enter this shell:"
              echo "     exit && nix develop --impure .#sycl"
              echo ""
            fi
            echo ""
            
            # Set up Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            export XDG_CONFIG_HOME="''${XDG_CONFIG_HOME:-$HOME/.config}"
            export LLAMA_CPP_MODELS="''${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
            mkdir -p "$LLAMA_CPP_MODELS"
            
            # Build clean LD_LIBRARY_PATH with Nix libraries
            # Note: OneAPI's setvars.sh will add its own paths
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib"
            export LD_LIBRARY_PATH="${pkgs.libGLU}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.mesa}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            
            echo "Build llama-mtmd-cli with SYCL:"
            echo "  ./scripts/build-llama-mtmd-multibackend.sh --sycl"
            echo ""
            echo "Or build with multiple backends:"
            echo "  ./scripts/build-llama-mtmd-multibackend.sh --sycl --vulkan"
            echo ""
            
            echo "Hardware Detection:"
            # Check for Intel GPU
            if lspci 2>/dev/null | grep -i "vga\|3d" | grep -i intel >/dev/null; then
              echo "  ✅ Intel GPU detected"
              lspci | grep -i "vga\|3d" | grep -i intel
            else
              echo "  ⚠️  No Intel GPU detected"
              echo "     SYCL also works on Intel CPUs with AVX-512"
            fi
            echo ""
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
          
          contents = [
            self.packages.${system}.default
            pkgs.gtk4
            pkgs.tesseract
            pkgs.llama-cpp
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
              "org.opencontainers.image.source" = "https://github.com/stradichenko/anki-ocr-vocab-collector";              "org.opencontainers.image.version" = "0.1.0";
            };
          };
          
          maxLayers = 120;
        };
      }
    );
}