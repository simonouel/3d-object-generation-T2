#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Automatic dependency installer based on config.py settings.

This script reads the configuration and installs the appropriate dependencies
using the corresponding requirements files:

- requirements.txt           : Core dependencies (always installed via -r)
- requirements-nim.txt       : NIM backend (when USE_NATIVE_LLM = False)
- requirements-native.txt    : Native LLM (when USE_NATIVE_LLM = True)
- requirements-trellis.txt   : Native TRELLIS (when USE_NATIVE_TRELLIS = True)

Usage:
    python install_dependencies.py
"""

import importlib.util
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path


def is_package_installed(import_name: str) -> bool:
    """Return True if the package is already importable in the current environment."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def run_pip_install(requirements_file: str, description: str) -> bool:
    """Run pip install for a requirements file."""
    print(f"\n{'='*60}")
    print(f"Installing {description}...")
    print(f"  Using: {requirements_file}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file],
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"✓ {description} installed successfully")
        return True
    else:
        print(f"✗ Failed to install {description}")
        return False


def verify_pytorch_cuda() -> bool:
    """Verify that PyTorch has CUDA support.
    
    Returns True if PyTorch has CUDA, False otherwise.
    """
    print(f"\n{'='*60}")
    print("Verifying PyTorch CUDA support...")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-c", 
         "import torch; "
         "cuda_ok = '+cu' in torch.__version__; "
         "print(f'  PyTorch version: {torch.__version__}'); "
         "print(f'  CUDA available: {torch.cuda.is_available()}'); "
         "print(f'  CUDA version: {torch.version.cuda}'); "
         "exit(0 if cuda_ok else 1)"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("✓ PyTorch with CUDA verified")
        return True
    else:
        print("✗ PyTorch does NOT have CUDA support!")
        print("  To fix, run:")
        print("    pip uninstall torch torchvision torchaudio -y")
        print("    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
        return False


def parse_config(config_path: Path) -> dict:
    """Parse config.py to extract relevant settings."""
    settings = {
        "USE_NATIVE_LLM": True,
        "USE_NATIVE_TRELLIS": True,
    }
    
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Extract USE_NATIVE_LLM
                if line.startswith("USE_NATIVE_LLM") and "=" in line:
                    value = line.split("=")[1].strip()
                    # Remove inline comments (e.g., "True  # comment" -> "True")
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    settings["USE_NATIVE_LLM"] = value.lower() == "true"
                
                # Extract USE_NATIVE_TRELLIS
                if line.startswith("USE_NATIVE_TRELLIS") and "=" in line:
                    value = line.split("=")[1].strip()
                    # Remove inline comments (e.g., "True  # comment" -> "True")
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    settings["USE_NATIVE_TRELLIS"] = value.lower() == "true"
                    
    except Exception as e:
        print(f"Warning: Could not parse config.py: {e}")
        print("Using default settings...")
    
    return settings


def setup_cuda_env() -> bool:
    """Set up CUDA environment variables for Windows and Linux."""
    if os.environ.get("CUDA_HOME"):
        print(f"  CUDA_HOME already set: {os.environ['CUDA_HOME']}")
        return True

    if os.name == "nt":
        # Windows: check CUDA 12.8 default path
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
        if Path(cuda_path).exists():
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path
            print(f"  Auto-detected CUDA 12.8 (Windows): {cuda_path}")
            return True
        print("  WARNING: CUDA 12.8 not found!")
        print("  Set CUDA_HOME manually: set CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8")
        return False
    else:
        # Linux: check standard CUDA paths in order of preference
        cuda_candidates = [
            "/usr/local/cuda-12.8",
            "/usr/local/cuda",
            "/usr/cuda",
        ]
        for cuda_path in cuda_candidates:
            if Path(cuda_path).exists() and Path(f"{cuda_path}/bin/nvcc").exists():
                os.environ["CUDA_HOME"] = cuda_path
                os.environ["CUDA_PATH"] = cuda_path
                print(f"  Auto-detected CUDA: {cuda_path}")
                return True
        # nvcc may be on PATH even without CUDA_HOME
        import shutil
        if shutil.which("nvcc"):
            nvcc_path = shutil.which("nvcc")
            cuda_path = str(Path(nvcc_path).parent.parent)
            os.environ["CUDA_HOME"] = cuda_path
            print(f"  Detected CUDA via nvcc: {cuda_path}")
            return True
        print("  WARNING: CUDA not found! Install CUDA 12.8 and set CUDA_HOME.")
        print("  Example: export CUDA_HOME=/usr/local/cuda")
        return False


def install_cuda_extension_from_git(name: str, git_url: str, import_name: str = None,
                                    branch: str = None, recursive: bool = False) -> bool:
    """Clone a git repo and install it, skipping if already installed.

    Args:
        name: Display name of the extension
        git_url: The git repository URL
        import_name: Python import name used to detect if already installed
        branch: Optional branch or tag to check out
        recursive: Whether to clone submodules recursively

    Returns:
        True if installation succeeded (or already installed), False otherwise
    """
    if import_name and is_package_installed(import_name):
        print(f"\n  Skipping {name} (already installed)")
        return True

    print(f"\n  Installing {name}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_cmd = ["git", "clone"]
        if branch:
            clone_cmd += ["-b", branch]
        if recursive:
            clone_cmd += ["--recursive"]
        clone_cmd += [git_url, tmpdir + "/" + name]
        result = subprocess.run(clone_cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  Failed to clone {name}")
            return False
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", tmpdir + "/" + name],
            capture_output=False
        )
        if result.returncode == 0:
            print(f"  {name} installed successfully")
            return True
        else:
            print(f"  {name} installation failed")
            return False


def install_trellis_extensions() -> bool:
    """Install TRELLIS 2 CUDA extensions from source.

    These extensions must be built from source to support the user's
    specific CUDA version and GPU architecture.

    Extensions:
        - nvdiffrast v0.4.0: NVIDIA differentiable rasterizer
        - nvdiffrec (renderutils branch): Differentiable mesh reconstruction utilities
        - CuMesh: CUDA mesh processing library
        - FlexGEMM: Flexible GEMM CUDA kernels
        - o-voxel: Occupancy voxel library (local package in TRELLIS.2 repo)
    """
    print(f"\n{'='*60}")
    print("Installing TRELLIS 2 CUDA Extensions")
    print("(Building from source - this may take several minutes)")
    print(f"{'='*60}")

    # Ensure CUDA_HOME is set
    if not setup_cuda_env():
        print("\n  ERROR: Cannot install CUDA extensions without CUDA_HOME!")
        return False

    # Initialize TRELLIS.2 git submodules (e.g. o-voxel/third_party/eigen)
    script_dir = Path(__file__).parent
    trellis2_path = script_dir / "TRELLIS.2"
    if (trellis2_path / ".git").exists() or (trellis2_path / ".gitmodules").exists():
        print("\n  Initializing TRELLIS.2 submodules (third_party/eigen, ...)...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(trellis2_path),
            capture_output=False,
        )
        if result.returncode != 0:
            print("  Warning: git submodule update failed in TRELLIS.2 — o-voxel build may fail")

    all_success = True

    # nvdiffrast v0.4.0
    if not install_cuda_extension_from_git(
        "nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git",
        import_name="nvdiffrast", branch="v0.4.0",
    ):
        all_success = False

    # nvdiffrec (renderutils branch)
    if not install_cuda_extension_from_git(
        "nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git",
        import_name="renderutils", branch="renderutils",
    ):
        all_success = False

    # CuMesh
    if not install_cuda_extension_from_git(
        "CuMesh", "https://github.com/JeffreyXiang/CuMesh.git",
        import_name="cumesh", recursive=True,
    ):
        all_success = False

    # FlexGEMM
    if not install_cuda_extension_from_git(
        "FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git",
        import_name="flexgemm", recursive=True,
    ):
        all_success = False

    # o-voxel (local package in TRELLIS.2 repo)
    ovoxel_path = script_dir / "TRELLIS.2" / "o-voxel"
    if is_package_installed("o_voxel"):
        print("\n  Skipping o-voxel (already installed)")
    else:
        print(f"\n  Installing o-voxel from {ovoxel_path}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(ovoxel_path)],
            capture_output=False
        )
        if result.returncode == 0:
            print("  o-voxel installed successfully")
        else:
            print("  o-voxel installation failed")
            all_success = False

    print(f"\n{'='*60}")
    if all_success:
        print("✓ All CUDA extensions installed successfully!")
    else:
        print("⚠ Some CUDA extensions failed to install.")
        print("  You may need to install them manually.")
    print(f"{'='*60}")

    return all_success


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    print("=" * 60)
    print("3D Object Generation - Dependency Installer")
    print("=" * 60)
    
    # Parse config.py
    config_path = script_dir / "config.py"
    settings = parse_config(config_path)
    
    use_native_llm = settings["USE_NATIVE_LLM"]
    use_native_trellis = settings["USE_NATIVE_TRELLIS"]
    
    print(f"\nDetected configuration:")
    print(f"  USE_NATIVE_LLM: {use_native_llm}")
    print(f"  USE_NATIVE_TRELLIS: {use_native_trellis}")
    
    all_success = True
    
    # =========================================================================
    # Step 1: Install LLM dependencies
    # =========================================================================
    if use_native_llm:
        requirements_file = script_dir / "requirements-native.txt"
        description = "Native LLM dependencies"
    else:
        requirements_file = script_dir / "requirements-nim.txt"
        description = "NIM backend with Griptape"
    
    # Check if the requirements file exists
    if not requirements_file.exists():
        print(f"\n✗ Error: Requirements file not found: {requirements_file}")
        print(f"  Falling back to core requirements.txt")
        requirements_file = script_dir / "requirements.txt"
        description = "Core dependencies"
    
    # Install LLM dependencies
    if not run_pip_install(str(requirements_file), description):
        all_success = False
    
    # =========================================================================
    # Step 2: Install TRELLIS dependencies (if enabled)
    # =========================================================================
    if use_native_trellis:
        trellis_requirements = script_dir / "requirements-trellis.txt"
        
        if trellis_requirements.exists():
            if not run_pip_install(str(trellis_requirements), "Native TRELLIS dependencies"):
                all_success = False
        else:
            print(f"\nWarning: requirements-trellis.txt not found")
            print("  Skipping TRELLIS dependencies")
    
    # =========================================================================
    # Step 3: Verify PyTorch has CUDA support
    # =========================================================================
    # requirements-trellis.txt includes --extra-index-url for PyTorch CUDA
    # This check ensures the correct version is installed
    if not verify_pytorch_cuda():
        print("\n⚠ Warning: PyTorch CUDA verification failed!")
        all_success = False
    
    # =========================================================================
    # Step 4: Install TRELLIS CUDA extensions (requires PyTorch with CUDA)
    # =========================================================================
    if use_native_trellis:
        if not install_trellis_extensions():
            all_success = False
    
    # =========================================================================
    # Summary
    # =========================================================================
    if not all_success:
        print("\n" + "=" * 60)
        print("⚠ Installation completed with warnings!")
        print("  Some packages may need manual installation.")
        print("=" * 60)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All dependencies installed successfully!")
    print("=" * 60)
    print("\nYou can now run the application with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
