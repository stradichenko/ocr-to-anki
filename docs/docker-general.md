# Docker Integration with Nix Flakes for Perfect Reproducibility

## Overview

The best approach for this codebase is to **build Docker images directly from Nix** using `dockerTools.buildLayeredImage`. This ensures perfect reproducibility since the same Nix derivation produces both your development environment and Docker images.

## Why Use Docker with Nix Flakes?

While Nix Flakes already provide reproducible development environments, Docker adds:

- **Deployment Portability**: Run the application on any system without Nix installed
- **CI/CD Integration**: Most CI systems work seamlessly with Docker
- **Production Consistency**: Ship the exact environment you developed in
- **Simplified Distribution**: Share pre-built images instead of requiring users to build from Nix

## Why Build Docker Images with Nix?

**Perfect Reproducibility**: Same Nix derivation creates both dev environment and production Docker images, eliminating "works on my machine" issues.

## Implementation Guide

Add to your `flake.nix`:

```nix
{
    outputs = { self, nixpkgs }: {
        packages.x86_64-linux.dockerImage = 
            nixpkgs.legacyPackages.x86_64-linux.dockerTools.buildLayeredImage {
                name = "ocr-to-anki";
                tag = "latest";
                contents = [ self.packages.x86_64-linux.default ];
                config = {
                    Cmd = [ "/bin/ocr-to-anki" ];
                    WorkingDir = "/app";
                };
            };
    };
}
```

Build with: `nix build .#dockerImage && docker load < result`

## Building the Docker Image

### Method 1: Pure Nix Build (Recommended for Reproducibility)

```bash
# Build the Docker image using Nix
nix build .#dockerImage

# Load the image into Docker
docker load < result

# Run the container
docker run -it ocr-to-anki:latest
```

### Method 2: Traditional Dockerfile

```bash
# Build using Docker
docker build -f docker/dockerfile -t ocr-to-anki:latest .

# Run with compose
cd docker && docker-compose up -d
```

## Usage Examples

### Run with GUI (X11 forwarding)

```bash
xhost +local:docker
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.config/ocr-to-anki:/config \
  ocr-to-anki:latest
```

### Run in CLI mode

```bash
docker run -it --rm \
  -v $(pwd)/images:/images:ro \
  -v $(pwd)/output:/data \
  ocr-to-anki:latest --cli --input /images
```

### With Ollama integration

```bash
docker-compose -f docker/compose.yml up -d
```

## Why This Approach Works

1. **Nix Provides True Reproducibility**: The same `flake.lock` always produces identical builds
2. **Layered Images Optimize Size**: `buildLayeredImage` creates efficient layer caching
3. **No Version Drift**: All dependencies come from pinned Nix packages
4. **Cross-Platform**: Build once, run anywhere Docker runs
5. **Development-Production Parity**: Same environment in dev shell and Docker

## Benefits for Your Codebase

- **OCR Dependencies**: Tesseract and all image processing libs are version-locked
- **GTK4 Runtime**: Exact same GTK version in Docker as development
- **AI/ML Stack**: LangChain, Ollama, and all Python packages perfectly matched
- **CI/CD Ready**: GitHub Actions can build the Nix Docker image without Docker installed
