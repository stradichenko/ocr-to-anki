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
        
        # Python package overrides
        pythonPackagesExtensions = final: prev: {
          tenacity = prev.tenacity.overridePythonAttrs (old: {
            doCheck = false;
          });
        };
        
        # Python 3.11+ with packages from Nix
        python = pkgs.python311.override {
          packageOverrides = pythonPackagesExtensions;
          self = python;
        };
        
        pythonEnv = python.withPackages (ps: with ps; [
          # GTK4 bindings
          pygobject3
          
          # Image processing
          opencv4
          numpy
          
          # Data handling
          pyyaml
          pydantic
          requests
          
          # Testing
          pytest
          pytest-asyncio
          responses
          
          # API server
          fastapi
          uvicorn
          python-multipart
          
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
            echo "OCR to Anki (fully offline)"
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
              echo "  [OK] NVIDIA GPU detected"
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "     (GPU info unavailable)"
              echo "  → llama.cpp will use GPU acceleration"
              
              # Check for CUDA compiler
              if command -v nvcc >/dev/null 2>&1; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                echo "  → CUDA compiler: $CUDA_VERSION"
              else
                echo "  [WARN] CUDA compiler not in Nix environment"
                echo "     For CUDA builds, use: nix develop --impure .#cuda"
              fi
            elif [[ "$OSTYPE" == "darwin"* ]]; then
              echo "  [macOS] Apple Silicon detected"
              echo "  → llama.cpp has Metal support"
            else
              echo "  [CPU] CPU-only mode"
              echo "  → llama.cpp will run on CPU (slower but works)"
            fi
            echo ""
            
            echo "llama.cpp status (fully offline):"
            if [ -f "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" ]; then
              echo "  [OK] Gemma 3 4B model found"
              MODEL_SIZE=$(du -h "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" 2>/dev/null | cut -f1)
              echo "  :: Model size: $MODEL_SIZE"
              echo "     Location: $LLAMA_CPP_MODELS"
            else
              echo "  [WARN] Gemma 3 4B model not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            
            if [ -f "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" ]; then
              MMPROJ_SIZE=$(du -h "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" 2>/dev/null | cut -f1)
              echo "  [OK] Vision projector found ($MMPROJ_SIZE)"
            else
              echo "  [WARN] Vision projector not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            echo ""
            
            # llama-mtmd-cli setup
            LLAMAMTMD_BIN="$HOME/.local/bin/llama-mtmd-cli"
            
            echo "llama.cpp tools:"
            echo "  • llama-server: [OK] (from nixpkgs)"
            echo "  • llama-cli: [OK] (from nixpkgs)"
            
            # Check for Vulkan support
            if command -v vulkaninfo >/dev/null 2>&1; then
              echo "  • Vulkan: [OK] (GPU backend available)"
              VULKAN_DEVICES=$(vulkaninfo --summary 2>/dev/null | grep "GPU" | wc -l)
              if [ "$VULKAN_DEVICES" -gt 0 ]; then
                echo "    Devices: $VULKAN_DEVICES GPU(s) detected"
              fi
            else
              echo "  • Vulkan: [WARN] (not available)"
            fi
            
            if command -v llama-mtmd-cli >/dev/null 2>&1; then
              # Check if it's actually working
              if llama-mtmd-cli --version >/dev/null 2>&1; then
                echo "  • llama-mtmd-cli: [OK] (manually built, working)"
              else
                echo "  • llama-mtmd-cli: [WARN] (found but has library issues)"
                echo "    Rebuild with: ./scripts/build-llama-gemma3-cli.sh"
              fi
            else
              echo "  • llama-mtmd-cli: [ERR] (not found)"
              echo "    Build with: ./scripts/build-llama-mtmd-multibackend.sh"
              echo "    Available backends:"
              echo "      --vulkan  (works in Nix, recommended)"
              echo "      --cuda    (requires system CUDA, use outside Nix)"
              echo "      --all     (auto-detect all available)"
            fi
            echo ""
          '';
        };
        
        # Flutter development shell (for building the GUI app)
        # Works on Linux (GTK3) and macOS (Cocoa) via platform detection
        devShells.flutter = pkgs.mkShell {
          packages = [
            pkgs.flutter
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            pkgs.clang
            pkgs.sqlite
            pkgs.zlib
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            # GTK3 and its dependencies (Flutter Linux uses GTK3)
            pkgs.gtk3
            pkgs.glib
            pkgs.pcre2
            pkgs.libepoxy
            pkgs.harfbuzz
            pkgs.pango
            pkgs.cairo
            pkgs.gdk-pixbuf
            pkgs.atk
            
            # X11 / Wayland
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXrandr
            pkgs.xorg.libXi
            pkgs.xorg.libXext
            pkgs.xorg.libXfixes
            pkgs.xorg.libXinerama
            pkgs.xorg.libXdamage
            pkgs.xorg.libXcomposite
            pkgs.wayland
            pkgs.libxkbcommon
            
            # GL
            pkgs.libGL
            pkgs.mesa
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Cocoa
            pkgs.darwin.apple_sdk.frameworks.CoreVideo
            pkgs.darwin.apple_sdk.frameworks.IOKit
          ];
          
          shellHook = ''
            # Prevent Nix cmake wrapper from interfering with Flutter's cmake
            unset cmakeFlags
            unset CMAKE_INSTALL_PREFIX
            
            # Use clang as the C/C++ compiler (Flutter expects it)
            export CC=clang
            export CXX=clang++
            
            # Disable Flutter analytics
            flutter config --no-analytics 2>/dev/null || true
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            # Clear system paths to prevent NixOS library mixing
            unset LD_LIBRARY_PATH
            
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib"
            export LD_LIBRARY_PATH="${pkgs.mesa}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.libepoxy}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.gtk3}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.pcre2}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.sqlite.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.harfbuzz}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.pango.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.cairo}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.gdk-pixbuf}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.atk}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.wayland}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.libxkbcommon}/lib:$LD_LIBRARY_PATH"
          '' + ''
            
            PLATFORM="unknown"
            BUILD_CMD="flutter build linux --release"
            if [[ "$OSTYPE" == "darwin"* ]]; then
              PLATFORM="macOS"
              BUILD_CMD="flutter build macos --release"
            elif [[ "$OSTYPE" == "linux"* ]]; then
              PLATFORM="Linux"
              BUILD_CMD="flutter build linux --release"
            fi
            
            echo "╔══════════════════════════════════════════════════╗"
            echo "║       Flutter Dev Shell ($PLATFORM)               ║"
            echo "╚══════════════════════════════════════════════════╝"
            echo ""
            echo "  Flutter: $(flutter --version 2>/dev/null | head -1)"
            echo ""
            echo "  Run app:       cd app && flutter run -d $([[ "$OSTYPE" == "darwin"* ]] && echo "macos" || echo "linux")"
            echo "  Build release: cd app && $BUILD_CMD"
            echo "  Bundle:        ./scripts/build-flutter.sh"
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
              echo "  [OK] nvcc: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
            else
              echo "  [ERR] nvcc: not found"
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
              echo "  [OK] NVIDIA GPU detected"
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "     (GPU info unavailable)"
              echo "  → llama.cpp will use GPU acceleration"
              
              # Check for CUDA compiler
              if command -v nvcc >/dev/null 2>&1; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
                echo "  → CUDA compiler: $CUDA_VERSION"
              else
                echo "  [WARN] CUDA compiler not in Nix environment"
                echo "     For CUDA builds, use: nix develop --impure .#cuda"
              fi
            elif [[ "$OSTYPE" == "darwin"* ]]; then
              echo "  [macOS] Apple Silicon detected"
              echo "  → llama.cpp has Metal support"
            else
              echo "  [CPU] CPU-only mode"
              echo "  → llama.cpp will run on CPU (slower but works)"
            fi
            echo ""
            
            echo "llama.cpp status (fully offline):"
            if [ -f "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" ]; then
              echo "  [OK] Gemma 3 4B model found"
              MODEL_SIZE=$(du -h "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" 2>/dev/null | cut -f1)
              echo "  :: Model size: $MODEL_SIZE"
              echo "     Location: $LLAMA_CPP_MODELS"
            else
              echo "  [WARN] Gemma 3 4B model not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            
            if [ -f "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" ]; then
              MMPROJ_SIZE=$(du -h "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" 2>/dev/null | cut -f1)
              echo "  [OK] Vision projector found ($MMPROJ_SIZE)"
            else
              echo "  [WARN] Vision projector not found"
              echo "  Run: ./scripts/setup-llama-cpp.sh"
            fi
            echo ""
            
            # llama-mtmd-cli setup
            LLAMAMTMD_BIN="$HOME/.local/bin/llama-mtmd-cli"
            
            echo "llama.cpp tools:"
            echo "  • llama-server: [OK] (from nixpkgs)"
            echo "  • llama-cli: [OK] (from nixpkgs)"
            
            # Check for Vulkan support
            if command -v vulkaninfo >/dev/null 2>&1; then
              echo "  • Vulkan: [OK] (GPU backend available)"
              VULKAN_DEVICES=$(vulkaninfo --summary 2>/dev/null | grep "GPU" | wc -l)
              if [ "$VULKAN_DEVICES" -gt 0 ]; then
                echo "    Devices: $VULKAN_DEVICES GPU(s) detected"
              fi
            else
              echo "  • Vulkan: [WARN] (not available)"
            fi
            
            if command -v llama-mtmd-cli >/dev/null 2>&1; then
              if llama-mtmd-cli --version >/dev/null 2>&1; then
                echo "  • llama-mtmd-cli: [OK] (working)"
              else
                echo "  • llama-mtmd-cli: [WARN] (found but has library issues)"
                echo "    Rebuild with: ./scripts/build-llama-mtmd-vulkan.sh"
              fi
            else
              echo "  • llama-mtmd-cli: [ERR] (not found)"
              echo "    Build with: ./scripts/build-llama-mtmd-vulkan.sh"
            fi
            echo ""
          '';
        };
        
        # SYCL/OpenCL-enabled development shell (Intel GPU backends)
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
            
            # Vulkan (works alongside SYCL/OpenCL)
            pkgs.vulkan-headers
            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.shaderc
            
            # Intel GPU compute: Level Zero + compute runtime
            pkgs.level-zero                         # oneAPI Level Zero loader + headers
            pkgs.intel-compute-runtime-legacy1       # GPU driver for Gen8/Gen9/Gen11 (10th gen and older)
            
            # OpenCL (alternative GPU backend, simpler than SYCL)
            pkgs.opencl-headers                     # Khronos OpenCL headers
            pkgs.ocl-icd                            # OpenCL ICD loader
            pkgs.clinfo                             # OpenCL device info tool
            
            # Development tools
            pkgs.ruff
            pkgs.python311Packages.black
          ];
          
          shellHook = ''
            echo "╔══════════════════════════════════════════════════╗"
            echo "║  OCR to Anki -- Intel GPU environment             ║"
            echo "╚══════════════════════════════════════════════════╝"
            echo ""
            
            # Set up Python path
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            export XDG_CONFIG_HOME="''${XDG_CONFIG_HOME:-$HOME/.config}"
            export LLAMA_CPP_MODELS="''${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
            mkdir -p "$LLAMA_CPP_MODELS"
            
            # CRITICAL: Clear system library paths to prevent mixing Nix and system libraries
            unset LD_LIBRARY_PATH
            
            # Build clean LD_LIBRARY_PATH with Nix libraries
            export LD_LIBRARY_PATH="${pkgs.libGL}/lib"
            export LD_LIBRARY_PATH="${pkgs.libGLU}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.mesa}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            
            # Intel GPU driver: make sure Level Zero can find the compute runtime
            export LD_LIBRARY_PATH="${pkgs.level-zero}/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.intel-compute-runtime-legacy1}/lib:$LD_LIBRARY_PATH"
            
            # OpenCL ICD loader needs to find Intel's ICD file
            export OCL_ICD_VENDORS="${pkgs.intel-compute-runtime-legacy1}/etc/OpenCL/vendors"
            export LD_LIBRARY_PATH="${pkgs.ocl-icd}/lib:$LD_LIBRARY_PATH"
            
            # Intel OneAPI environment (for SYCL backend)
            export ONEAPI_ROOT="''${ONEAPI_ROOT:-/opt/intel/oneapi}"
            
            # ── Hardware Detection ──────────────────────────────
            echo "Hardware:"
            if [ -f /sys/class/drm/card1/device/uevent ]; then
              PCI_ID=$(grep PCI_ID /sys/class/drm/card1/device/uevent 2>/dev/null | cut -d= -f2)
              DRIVER=$(grep DRIVER /sys/class/drm/card1/device/uevent 2>/dev/null | cut -d= -f2)
              echo "  Intel GPU: $PCI_ID (driver: $DRIVER)"
            elif [ -f /sys/class/drm/card0/device/uevent ]; then
              PCI_ID=$(grep PCI_ID /sys/class/drm/card0/device/uevent 2>/dev/null | cut -d= -f2)
              DRIVER=$(grep DRIVER /sys/class/drm/card0/device/uevent 2>/dev/null | cut -d= -f2)
              echo "  Intel GPU: $PCI_ID (driver: $DRIVER)"
            fi
            echo ""
            
            # ── OpenCL Status ───────────────────────────────────
            echo "OpenCL backend (recommended for Gen9/Gen11):"
            if command -v clinfo >/dev/null 2>&1; then
              CL_DEVS=$(clinfo -l 2>/dev/null | grep -c "Device" || echo "0")
              if [ "$CL_DEVS" -gt 0 ]; then
                echo "  [OK] $CL_DEVS OpenCL device(s) found"
                clinfo -l 2>/dev/null | grep "Device" | sed 's/^/     /'
              else
                echo "  [WARN] No OpenCL devices (intel-compute-runtime may need NixOS hardware.opengl)"
              fi
            else
              echo "  [WARN] clinfo not available"
            fi
            echo "  Build: ./scripts/build-llama-mtmd-opencl.sh"
            echo ""
            
            # ── SYCL Status ────────────────────────────────────
            echo "SYCL backend (Gen11+, requires Intel oneAPI):"
            if [ -d "$ONEAPI_ROOT" ] && [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
              echo "  [OK] oneAPI found at: $ONEAPI_ROOT"
              echo "  Sourcing setvars.sh..."
              source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || true
              
              if command -v icpx >/dev/null 2>&1; then
                echo "  [OK] icpx: $(icpx --version 2>&1 | head -1)"
              fi
              if command -v sycl-ls >/dev/null 2>&1; then
                echo "  SYCL devices:"
                sycl-ls 2>/dev/null | grep -i "Intel" | head -3 | sed 's/^/     /'
              fi
              echo "  Build: ./scripts/build-llama-mtmd-sycl.sh"
            else
              echo "  [ERR] oneAPI not installed at $ONEAPI_ROOT"
              echo "  Install: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
              echo "  On NixOS: steam-run bash -c 'sudo bash ./l_BaseKit_p_*.sh -a --silent --eula accept'"
            fi
            echo ""
            
            # ── Model Status ───────────────────────────────────
            echo "Models:"
            if [ -f "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" ]; then
              echo "  [OK] $(du -h "$LLAMA_CPP_MODELS/gemma-3-4b-it-q4_0_s.gguf" | cut -f1) gemma-3-4b-it-q4_0_s.gguf"
            else
              echo "  [WARN] Model not found -- run: ./scripts/setup-llama-cpp.sh"
            fi
            if [ -f "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" ]; then
              echo "  [OK] $(du -h "$LLAMA_CPP_MODELS/mmproj-model-f16-4B.gguf" | cut -f1) mmproj-model-f16-4B.gguf"
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
            pkgs.wrapGAppsHook3
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
        
        # ── Bundled Python backend (PyInstaller-free, Nix-wrapped) ──
        # Creates a self-contained backend binary that bundles Python + deps
        packages.backend = pkgs.stdenv.mkDerivation {
          pname = "ocr-to-anki-backend";
          version = "0.1.0";
          src = ./.;
          
          buildInputs = [ pythonEnv ];
          nativeBuildInputs = [ pkgs.makeWrapper ];
          
          installPhase = ''
            mkdir -p $out/lib/ocr-to-anki $out/bin
            
            # Copy the Python source and config
            cp -r src $out/lib/ocr-to-anki/
            cp -r config $out/lib/ocr-to-anki/
            cp requirements.txt $out/lib/ocr-to-anki/
            
            # Create a wrapper script that launches the FastAPI backend
            makeWrapper ${pythonEnv}/bin/python $out/bin/ocr-to-anki-backend \
              --add-flags "-m uvicorn src.api.app:app --host 0.0.0.0 --port 8000" \
              --chdir "$out/lib/ocr-to-anki" \
              --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.llama-cpp ]}" \
              --set PYTHONPATH "$out/lib/ocr-to-anki"
          '';
          
          meta = with pkgs.lib; {
            description = "Bundled FastAPI backend for OCR to Anki";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
        
        # ── Flutter Linux desktop build ─────────────────────────────
        # Builds the Flutter app as a release Linux bundle.
        # Usage: nix build .#flutter-app
        packages.flutter-app = pkgs.stdenv.mkDerivation {
          pname = "ocr-to-anki-flutter";
          version = "0.1.0";
          src = ./app;
          
          nativeBuildInputs = with pkgs; [
            flutter
            cmake
            ninja
            pkg-config
            clang
            makeWrapper
            wrapGAppsHook3
          ];
          
          buildInputs = with pkgs; [
            # GTK3 (Flutter Linux uses GTK3)
            gtk3
            glib
            pcre2
            libepoxy
            harfbuzz
            pango
            cairo
            gdk-pixbuf
            atk
            
            # X11 / Wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXrandr
            xorg.libXi
            xorg.libXext
            xorg.libXfixes
            xorg.libXinerama
            xorg.libXdamage
            xorg.libXcomposite
            wayland
            libxkbcommon
            
            # GL
            libGL
            mesa
            
            # Other
            sqlite
            zlib
          ];
          
          # Flutter needs HOME and a writable pub cache
          FLUTTER_ROOT = "${pkgs.flutter}";
          
          configurePhase = ''
            runHook preConfigure
            
            export HOME=$TMPDIR
            export PUB_CACHE=$TMPDIR/.pub-cache
            export CC=clang
            export CXX=clang++
            
            # Prevent Nix cmake wrapper from interfering with Flutter's cmake
            unset cmakeFlags
            unset CMAKE_INSTALL_PREFIX
            
            # Disable Flutter analytics
            flutter config --no-analytics 2>/dev/null || true
            flutter pub get --offline 2>/dev/null || flutter pub get
            
            runHook postConfigure
          '';
          
          buildPhase = ''
            runHook preBuild
            flutter build linux --release
            runHook postBuild
          '';
          
          installPhase = ''
            runHook preInstall
            
            mkdir -p $out/opt/ocr-to-anki
            cp -r build/linux/*/release/bundle/* $out/opt/ocr-to-anki/
            
            mkdir -p $out/bin
            makeWrapper $out/opt/ocr-to-anki/ocr_to_anki $out/bin/ocr-to-anki \
              --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath [
                pkgs.gtk3
                pkgs.glib
                pkgs.libepoxy
                pkgs.libGL
                pkgs.mesa
                pkgs.sqlite
                pkgs.harfbuzz
                pkgs.pango
                pkgs.cairo
                pkgs.gdk-pixbuf
                pkgs.wayland
                pkgs.libxkbcommon
              ]}"
            
            # Install desktop entry
            mkdir -p $out/share/applications
            cat > $out/share/applications/ocr-to-anki.desktop <<EOF
[Desktop Entry]
Name=OCR to Anki
Comment=Cross-platform OCR to Anki flashcard generator
Exec=$out/bin/ocr-to-anki
Icon=ocr-to-anki
Type=Application
Categories=Education;Utility;
EOF
            
            runHook postInstall
          '';
          
          meta = with pkgs.lib; {
            description = "OCR to Anki – Flutter desktop app for Linux";
            homepage = "https://github.com/stradichenko/anki-ocr-vocab-collector";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };
        
        # ── Full distribution bundle (Flutter app + Python backend) ──
        # Usage: nix build .#bundle
        packages.bundle = pkgs.stdenv.mkDerivation {
          pname = "ocr-to-anki-bundle";
          version = "0.1.0";
          
          dontUnpack = true;
          
          nativeBuildInputs = [ pkgs.makeWrapper ];
          
          installPhase = ''
            mkdir -p $out/bin $out/opt/ocr-to-anki
            
            # Link Flutter app
            ln -s ${self.packages.${system}.flutter-app}/opt/ocr-to-anki/* $out/opt/ocr-to-anki/
            
            # Link backend
            ln -s ${self.packages.${system}.backend}/lib $out/opt/ocr-to-anki/backend-lib
            ln -s ${self.packages.${system}.backend}/bin/ocr-to-anki-backend $out/opt/ocr-to-anki/
            
            # Create a launcher that starts backend then opens the GUI
            cat > $out/bin/ocr-to-anki <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "''${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$SCRIPT_DIR/opt/ocr-to-anki/ocr-to-anki-backend"
GUI="$SCRIPT_DIR/opt/ocr-to-anki/ocr_to_anki"

cleanup() { kill "$BACKEND_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Start backend in background
"$BACKEND" &
BACKEND_PID=$!

# Wait for backend health
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  sleep 0.5
done

# Launch Flutter GUI (foreground)
exec "$GUI" "$@"
LAUNCHER
            chmod +x $out/bin/ocr-to-anki
          '';
          
          meta = with pkgs.lib; {
            description = "OCR to Anki – complete bundle (GUI + backend)";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };
        
        # Docker image built directly from Nix for perfect reproducibility
        packages.dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "ocr-to-anki";
          tag = "latest";
          
          contents = [
            self.packages.${system}.default
            pkgs.gtk4
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