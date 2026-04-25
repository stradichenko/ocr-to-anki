"""
GPU backend auto-detection for llama.cpp.

Probes the system for available GPU backends (Vulkan, CUDA, SYCL, Metal)
and selects the best llama-mtmd-cli binary accordingly.
"""

import functools
import os
import re
import shutil
import subprocess
import platform
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


class Backend(str, Enum):
    """Supported GPU compute backends, ordered by preference."""
    CUDA   = "cuda"
    METAL  = "metal"
    VULKAN = "vulkan"
    OPENCL = "opencl"
    SYCL   = "sycl"
    CPU    = "cpu"


@dataclass
class GPUDevice:
    """Describes a single GPU device."""
    name: str
    backend: Backend
    vram_mb: int = 0
    compute_capability: str = ""
    driver_version: str = ""

    @property
    def is_discrete(self) -> bool:
        """Heuristic: return True if the device appears to be a discrete GPU.

        Integrated GPUs typically contain keywords like UHD, Iris, HD Graphics,
        Intel(R) Graphics, etc.  Discrete GPUs from NVIDIA, AMD, and Intel Arc
        are identified by brand-specific keywords or by exclusion of known
        integrated patterns.
        """
        n = self.name.lower()
        # Known integrated GPU patterns
        _INTEGRATED = (
            "uhd", "iris", "hd graphics", "intel(r) graphics",
            "vega 3", "vega 5", "vega 6", "vega 7", "vega 8",
            "radeon(tm) graphics",  # AMD APU
            "apple m",              # Apple Silicon (unified, but powerful)
        )
        if any(kw in n for kw in _INTEGRATED):
            return False
        # Known discrete GPU patterns
        _DISCRETE = (
            "geforce", "rtx", "gtx", "quadro", "tesla",  # NVIDIA
            "radeon rx", "radeon pro",                     # AMD discrete
            "arc a", "arc b",                              # Intel Arc
        )
        if any(kw in n for kw in _DISCRETE):
            return True
        # If VRAM > 2 GB, very likely discrete
        if self.vram_mb > 2048:
            return True
        # Default: unknown — treat as integrated to be conservative
        return False


@dataclass
class DetectionResult:
    """Full hardware probe result."""
    os_name: str
    arch: str
    devices: list[GPUDevice] = field(default_factory=list)
    recommended_backend: Backend = Backend.CPU
    binary_path: Optional[str] = None
    details: dict = field(default_factory=dict)


# -------------------------------------------------------------------
# Binary resolution
# -------------------------------------------------------------------


def llama_bin_cache() -> Path:
    """Directory for auto-downloaded llama.cpp binaries."""
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))
        return Path(base) / "llama.cpp" / "bin"
    return Path.home() / ".cache" / "llama.cpp" / "bin"


# Backward-compatible alias
_llama_bin_cache = llama_bin_cache


def _search_paths() -> list[Path]:
    """Return the list of directories to search for backend-specific binaries."""
    return [
        llama_bin_cache(),                # Auto-downloaded binaries
        Path.home() / ".local" / "bin",    # User-installed (Linux/macOS)
        Path("/usr/local/bin"),
        Path("/usr/bin"),
    ]

# Mapping of backend → possible binary names (most specific first)
_BINARY_NAMES: dict[Backend, list[str]] = {
    Backend.CUDA:   ["llama-mtmd-cli-cuda", "llama-mtmd-cli"],
    Backend.METAL:  ["llama-mtmd-cli-metal", "llama-mtmd-cli"],
    Backend.VULKAN: ["llama-mtmd-cli-vulkan", "llama-mtmd-cli"],
    Backend.OPENCL: ["llama-mtmd-cli-opencl"],
    Backend.SYCL:   ["llama-mtmd-cli-sycl", "llama-mtmd-cli"],
    Backend.CPU:    ["llama-mtmd-cli-cpu", "llama-mtmd-cli"],
}


def _find_binary_impl(
    backend: Backend,
    names: Optional[dict[Backend, list[str]]] = None,
) -> Optional[Path]:
    """Core binary resolution (uncached so custom name maps work)."""
    is_win = platform.system() == "Windows"
    cache = llama_bin_cache()
    name_list = (names or _BINARY_NAMES).get(backend, [])

    for name in name_list:
        ext_names = [f"{name}.exe", name] if is_win else [name]

        for n in ext_names:
            # 1. Flat check in explicit search dirs
            for d in _search_paths():
                candidate = d / n
                if candidate.is_file() and (is_win or os.access(candidate, os.X_OK)):
                    return candidate

            # 2. Recursive search inside cache dir (archives may nest)
            if cache.exists():
                for f in cache.rglob(n):
                    if f.is_file() and (is_win or os.access(f, os.X_OK)):
                        return f

        # 3. PATH lookup (shutil.which handles .exe on Windows)
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


@functools.lru_cache(maxsize=None)
def find_binary(backend: Backend) -> Optional[Path]:
    """Locate a working llama-mtmd-cli binary for *backend*.

    Search order: explicit search paths (~/.cache/llama.cpp/bin,
    ~/.local/bin, /usr/local/bin) first — flat check then recursive
    search in the cache dir (handles archives with subdirectories) —
    then PATH.

    On Windows ``.exe`` extensions are tried automatically.
    """
    return _find_binary_impl(backend)


# Backward-compatible alias
_find_binary = find_binary


def _binary_supports_vulkan(binary: Path) -> bool:
    """Quick check: does this binary actually have Vulkan compiled in?"""
    try:
        out = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=5,
        )
        combined = out.stdout + out.stderr
        return "vulkan" in combined.lower() or "ggml_vulkan" in combined
    except Exception:
        return False


def opencl_env() -> dict[str, str]:
    """Return environment dict with OCL_ICD_VENDORS set for Intel NEO runtime."""
    env = dict(os.environ)
    if env.get("OCL_ICD_VENDORS"):
        return env  # already set by user

    # Fast path: system-wide vendor directory (works on most distros)
    system_icd = Path("/etc/OpenCL/vendors")
    if system_icd.exists() and list(system_icd.glob("*.icd")):
        env["OCL_ICD_VENDORS"] = str(system_icd)
        log.debug("OpenCL ICD (system): %s", system_icd)
        return env

    # NixOS: search the Nix store for intel-compute-runtime ICD files.
    # Prefer "legacy1" builds -- Gen9 / Gen9.5 iGPUs (e.g. UHD 620/630)
    # are NOT supported by the newer intel-compute-runtime (>=25.x).
    nix_store = Path("/nix/store")
    if nix_store.exists():
        try:
            candidates = []
            for d in nix_store.iterdir():
                if "intel-compute-runtime" in d.name and not d.name.endswith(".drv"):
                    icd_dir = d / "etc" / "OpenCL" / "vendors"
                    if icd_dir.exists() and list(icd_dir.glob("*.icd")):
                        is_legacy = "legacy" in d.name
                        candidates.append((is_legacy, d.name, icd_dir))
            # Sort: legacy first (True > False), then by name descending
            candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
            if candidates:
                icd_dir = candidates[0][2]
                env["OCL_ICD_VENDORS"] = str(icd_dir)
                log.debug("OpenCL ICD (nix): %s (from %s)", icd_dir, candidates[0][1])
                return env
        except Exception as exc:
            log.warning("Nix store scan for OpenCL ICD failed: %s", exc)

    log.warning("Could not locate OpenCL ICD vendors directory")
    return env


# Backward-compatible alias
_opencl_env = opencl_env


# -------------------------------------------------------------------
# Per-backend probes
# -------------------------------------------------------------------

def _probe_vulkan() -> list[GPUDevice]:
    """Detect Vulkan-capable GPUs."""
    devices = []

    # First try: the binary itself reports Vulkan devices
    binary = find_binary(Backend.VULKAN)
    if binary:
        try:
            out = subprocess.run(
                [str(binary), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            combined = out.stdout + out.stderr
            # Parse lines like: "ggml_vulkan: 0 = Intel(R) UHD Graphics ..."
            for line in combined.splitlines():
                if "ggml_vulkan:" in line and "=" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        name = parts[1].split("|")[0].strip()
                        devices.append(GPUDevice(
                            name=name,
                            backend=Backend.VULKAN,
                        ))
        except Exception as e:
            log.debug("Vulkan binary probe failed: %s", e)

    if devices:
        return devices

    # Fallback: vulkaninfo
    vulkaninfo = shutil.which("vulkaninfo")
    if vulkaninfo:
        try:
            out = subprocess.run(
                [vulkaninfo, "--summary"],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                if "deviceName" in line:
                    name = line.split("=")[-1].strip()
                    devices.append(GPUDevice(name=name, backend=Backend.VULKAN))
        except Exception as e:
            log.debug("vulkaninfo probe failed: %s", e)

    return devices


def _probe_cuda() -> list[GPUDevice]:
    """Detect NVIDIA CUDA GPUs."""
    devices = []
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return devices

    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                devices.append(GPUDevice(
                    name=parts[0],
                    backend=Backend.CUDA,
                    vram_mb=int(float(parts[1])),
                    compute_capability=parts[2],
                    driver_version=parts[3],
                ))
    except Exception as e:
        log.debug("CUDA probe failed: %s", e)

    return devices


def _probe_metal() -> list[GPUDevice]:
    """Detect Apple Metal GPUs (macOS only)."""
    if platform.system() != "Darwin":
        return []

    devices = []
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(out.stdout)
        for display in data.get("SPDisplaysDataType", []):
            name = display.get("sppci_model", "Apple GPU")
            vram_str = display.get("spdisplays_vram", "0")
            vram_mb = int("".join(c for c in vram_str if c.isdigit()) or 0)
            devices.append(GPUDevice(
                name=name,
                backend=Backend.METAL,
                vram_mb=vram_mb,
            ))
    except Exception as e:
        log.debug("Metal probe failed: %s", e)

    return devices


def _probe_opencl() -> list[GPUDevice]:
    """Detect OpenCL-capable GPUs."""
    devices = []

    # Try the OpenCL binary's --list-devices
    binary = find_binary(Backend.OPENCL)
    if binary:
        try:
            env = opencl_env()
            out = subprocess.run(
                [str(binary), "--list-devices"],
                capture_output=True, text=True, timeout=10, env=env,
            )
            combined = out.stdout + out.stderr
            for line in combined.splitlines():
                if "GPUOpenCL:" in line:
                    # e.g. "  GPUOpenCL: Intel(R) UHD Graphics (0 MiB, 0 MiB free)"
                    raw = line.split("GPUOpenCL:")[1].strip()
                    # Remove trailing "(0 MiB, ...)" but keep "(R)" etc.
                    name = re.sub(r"\(\d+\s*MiB.*$", "", raw).strip()
                    devices.append(GPUDevice(name=name, backend=Backend.OPENCL))
        except Exception as e:
            log.debug("OpenCL binary probe failed: %s", e)

    if devices:
        return devices

    # Fallback: clinfo
    clinfo = shutil.which("clinfo")
    if clinfo:
        try:
            env = opencl_env()
            out = subprocess.run(
                [clinfo, "-l"],
                capture_output=True, text=True, timeout=10, env=env,
            )
            for line in out.stdout.splitlines():
                line_stripped = line.strip()
                if line_stripped.startswith("`--"):
                    name = line_stripped.lstrip("`-").strip()
                    devices.append(GPUDevice(name=name, backend=Backend.OPENCL))
        except Exception as e:
            log.debug("clinfo probe failed: %s", e)

    return devices


def _probe_sycl() -> list[GPUDevice]:
    """Detect Intel SYCL devices."""
    devices = []

    # Check for sycl-ls (Intel OneAPI)
    sycl_ls = shutil.which("sycl-ls")
    if not sycl_ls:
        # Try standard OneAPI path
        oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
        candidate = Path(oneapi_root) / "compiler" / "latest" / "bin" / "sycl-ls"
        if candidate.exists():
            sycl_ls = str(candidate)

    if sycl_ls:
        try:
            out = subprocess.run(
                [sycl_ls],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                # e.g. "[opencl:gpu]   Intel(R) ..."
                if "gpu" in line.lower():
                    name = line.split("]")[-1].strip() if "]" in line else line.strip()
                    devices.append(GPUDevice(name=name, backend=Backend.SYCL))
        except Exception as e:
            log.debug("SYCL probe failed: %s", e)

    return devices


# -------------------------------------------------------------------
# Main detection
# -------------------------------------------------------------------

# Backend selection priority (higher = preferred)
#
# Vulkan is preferred over OpenCL for Intel iGPUs because:
#   - Vulkan supports flash attention (FA), cutting prompt eval from
#     ~1536 s to ~40 s on a Gemma-3 4B vision workload.
#   - OpenCL does NOT support FA (crashes during warmup) and also
#     lacks GGML_OP_POOL_2D needed by Gemma-3's SigLIP encoder,
#     so mmproj must stay on CPU regardless.
#   - Both backends still require --no-mmproj-offload on Intel iGPU
#     (Vulkan produces corrupted CLIP embeddings when offloading to
#     the Intel UHD GPU; OpenCL crashes on POOL_2D).
_BACKEND_PRIORITY = {
    Backend.CUDA:   100,
    Backend.METAL:  90,
    Backend.VULKAN: 80,   # Preferred: has flash attention
    Backend.OPENCL: 65,   # Fallback: no FA, no POOL_2D
    Backend.SYCL:   60,
    Backend.CPU:    0,
}


def detect(*, prefer: Optional[Backend] = None, prefer_discrete: bool = True) -> DetectionResult:
    """
    Probe the system and return a :class:`DetectionResult`.

    Parameters
    ----------
    prefer : Backend, optional
        Force a specific backend.  If the backend has no binary, falls back.
    prefer_discrete : bool
        When True (default), favour discrete GPUs over integrated ones
        within the same backend priority tier.  A discrete GPU on a
        lower-priority backend can also win if no discrete GPU exists on
        the higher-priority backend.
    """
    result = DetectionResult(
        os_name=platform.system(),
        arch=platform.machine(),
    )

    # Run all probes
    probes = {
        Backend.CUDA:   _probe_cuda,
        Backend.METAL:  _probe_metal,
        Backend.OPENCL: _probe_opencl,
        Backend.VULKAN: _probe_vulkan,
        Backend.SYCL:   _probe_sycl,
    }

    for backend, probe_fn in probes.items():
        try:
            devs = probe_fn()
            result.devices.extend(devs)
            result.details[backend.value] = [d.name for d in devs]
        except Exception as e:
            log.warning("Probe %s failed: %s", backend.value, e)

    # Determine which backends have both devices AND binaries.
    # Track whether a backend has a discrete GPU available.
    viable: list[tuple[Backend, Path, bool]] = []     # (backend, binary, has_discrete)
    backend_has_discrete: dict[Backend, bool] = {}
    for dev in result.devices:
        binary = find_binary(dev.backend)
        if binary:
            is_disc = dev.is_discrete
            if dev.backend not in backend_has_discrete:
                backend_has_discrete[dev.backend] = is_disc
            else:
                backend_has_discrete[dev.backend] = backend_has_discrete[dev.backend] or is_disc
            viable.append((dev.backend, binary, is_disc))

    # Deduplicate by backend (prefer discrete device when deduplicating)
    seen = set()
    unique_viable = []
    for b, p, disc in viable:
        if b not in seen:
            seen.add(b)
            unique_viable.append((b, p, backend_has_discrete.get(b, False)))

    # CPU always viable
    cpu_binary = find_binary(Backend.CPU)
    if cpu_binary and Backend.CPU not in seen:
        unique_viable.append((Backend.CPU, cpu_binary, False))

    if prefer and prefer != Backend.CPU:
        # User override
        for b, p, _ in unique_viable:
            if b == prefer:
                result.recommended_backend = b
                result.binary_path = str(p)
                return result
        log.warning("Preferred backend %s not available, auto-selecting", prefer.value)

    if not unique_viable:
        result.recommended_backend = Backend.CPU
        result.binary_path = str(cpu_binary) if cpu_binary else None
        return result

    # Sort by (backend_priority, has_discrete_gpu).  When prefer_discrete
    # is True a discrete GPU on a lower-priority backend can beat an
    # integrated-only higher-priority backend (bonus +50 keeps it within
    # the same ballpark).
    def _sort_key(entry: tuple[Backend, Path, bool]) -> int:
        b, _, has_disc = entry
        base = _BACKEND_PRIORITY.get(b, 0)
        if prefer_discrete and has_disc:
            base += 50
        return base

    unique_viable.sort(key=_sort_key, reverse=True)
    best_backend, best_binary, best_disc = unique_viable[0]
    result.recommended_backend = best_backend
    result.binary_path = str(best_binary)

    if best_disc:
        discrete_names = [d.name for d in result.devices
                          if d.backend == best_backend and d.is_discrete]
        log.info("Preferring discrete GPU: %s (%s)", discrete_names, best_backend.value)

    return result


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def print_report(result: DetectionResult) -> None:
    """Pretty-print a detection result."""
    print("╔══════════════════════════════════════════════════╗")
    print("║          Hardware & Backend Detection            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  OS:   {result.os_name} ({result.arch})")
    print()

    if result.devices:
        print("  Detected GPUs:")
        for i, d in enumerate(result.devices):
            vram = f"  ({d.vram_mb} MB)" if d.vram_mb else ""
            tag = " [discrete]" if d.is_discrete else " [integrated]"
            print(f"    [{d.backend.value:>6}] {d.name}{vram}{tag}")
    else:
        print("  No GPUs detected -- CPU-only mode")

    print()
    print(f"  Recommended backend:  {result.recommended_backend.value}")
    print(f"  Binary:               {result.binary_path or '[ERR] not found'}")

    # Show all available binaries
    print()
    print("  Binary availability:")
    for backend in Backend:
        binary = find_binary(backend)
        status = f"[OK] {binary}" if binary else "-"
        print(f"    {backend.value:>6}: {status}")
    print()


def main():
    logging.basicConfig(level=logging.DEBUG)
    result = detect()
    print_report(result)


if __name__ == "__main__":
    main()
