# TRELLIS Patches for Commercial License Compliance

This directory contains replacement files for the TRELLIS submodule that:

1. Use **gsplat** (Apache-2.0) instead of **diff-gaussian-rasterization** (Inria Non-Commercial)
2. Remove **kaolin** dependency (replaced with local implementation)
3. Remove **igraph** (GPL) dependency (replaced with **scipy** max-flow, BSD licensed)

## Source

These patch files are from **NVIDIA's Visual GenAI NIM** which provides 
inference-optimized, commercially-licensed versions of TRELLIS components.

## Pinned Versions

These patches are designed for specific commits to ensure reproducibility:

### TRELLIS
| Field | Value |
|-------|-------|
| **Repository** | https://github.com/microsoft/TRELLIS |
| **Commit** | `442aa1e1afb9014e80681d3bf604e8d728a86ee7` |
| **Description** | Merge pull request #335 from ForeverFancy/main |

### FlexiCubes (nested submodule)
| Field | Value |
|-------|-------|
| **Repository** | https://github.com/MaxtirError/FlexiCubes |
| **Commit** | `815e075a2a400d06c48d94c347674344ed6ae5c5` |
| **Description** | Resolve merge conflict in flexicubes.py |
| **Path** | `trellis/trellis/representations/mesh/flexicubes/` |

The `install.bat` script pins these exact commits for reproducibility.

### gsplat Version
| Field | Value |
|-------|-------|
| **Package** | gsplat==1.4.0 |
| **License** | Apache-2.0 |
| **Note** | Latest version compatible with Windows + PyTorch 2.7.0 (v1.5.x has MSVC/PyTorch API issues)

## Why These Patches?

The original TRELLIS code uses:
1. **Inria/GRAPHDECO's 3D Gaussian Splatting** - Non-commercial research license
2. **NVIDIA kaolin** - Apache-2.0 but heavy dependency (~1GB install)
3. **igraph** - GPL v2+ copyleft license (may require project to be GPL)

These patches replace that code with:

1. **gsplat renderer** - Apache-2.0 licensed from [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
2. **Inference-optimized Gaussian model** - MIT licensed from Microsoft (via NVIDIA NIM)
3. **Local `check_tensor` implementation** - Replaces kaolin dependency in FlexiCubes
4. **scipy max-flow** - BSD licensed, replaces igraph for mincut algorithm

## Files Patched

| Original Location | Patch File | License | Purpose |
|-------------------|------------|---------|---------|
| `trellis/trellis/renderers/gaussian_render.py` | `renderers/gaussian_render.py` | MIT (Microsoft) | Gaussian splatting renderer using gsplat |
| `trellis/trellis/representations/gaussian/gaussian_model.py` | `representations/gaussian/gaussian_model.py` | MIT (Microsoft) | Inference-only Gaussian model |
| `trellis/trellis/representations/gaussian/general_utils.py` | *(excluded, not replaced)* | - | Training utilities (not needed for inference) |
| `trellis/trellis/representations/mesh/flexicubes/` | `representations/mesh/flexicubes/flexicubes.py` | Apache-2.0 (NVIDIA NIM) | FlexiCubes with local `check_tensor` (no kaolin) |
| `trellis/trellis/representations/mesh/flexicubes/` | `representations/mesh/flexicubes/tables.py` | Apache-2.0 (NVIDIA) | Lookup tables for FlexiCubes |
| `trellis/trellis/utils/postprocessing_utils.py` | `utils/postprocessing_utils.py` | MIT (Microsoft) | Mesh postprocessing with scipy max-flow (no igraph) |

## How These Files Are Installed

The `setup_trellis.bat` script uses **sparse checkout** to completely exclude the 
Inria-licensed and kaolin-dependent files during clone, then installs our patched versions:

```batch
REM 1. Clone TRELLIS with sparse checkout (excludes Inria, kaolin, and igraph-dependent files)
git clone --no-checkout https://github.com/microsoft/TRELLIS.git trellis
git sparse-checkout init
echo /* > .git/info/sparse-checkout
echo !trellis/representations/gaussian/general_utils.py >> .git/info/sparse-checkout
echo !trellis/representations/gaussian/gaussian_model.py >> .git/info/sparse-checkout
echo !trellis/renderers/gaussian_render.py >> .git/info/sparse-checkout
echo !trellis/representations/mesh/flexicubes/ >> .git/info/sparse-checkout
echo !trellis/utils/postprocessing_utils.py >> .git/info/sparse-checkout
git checkout <commit>

REM 2. Install patched files
copy /Y "trellis_patch\renderers\gaussian_render.py" "trellis\trellis\renderers\"
copy /Y "trellis_patch\representations\gaussian\gaussian_model.py" "trellis\trellis\representations\gaussian\"
copy /Y "trellis_patch\utils\postprocessing_utils.py" "trellis\trellis\utils\"
mkdir "trellis\trellis\representations\mesh\flexicubes"
copy /Y "trellis_patch\representations\mesh\flexicubes\flexicubes.py" "trellis\trellis\representations\mesh\flexicubes\"
copy /Y "trellis_patch\representations\mesh\flexicubes\tables.py" "trellis\trellis\representations\mesh\flexicubes\"
```

**Result**: 
- Inria-licensed code is **never downloaded** to your system
- Kaolin dependency is **completely removed** (~1GB saved)
- FlexiCubes submodule is **completely skipped** (we use only `flexicubes.py` + `tables.py`)
- igraph (GPL) is **completely removed** (replaced with scipy BSD-licensed max-flow)

## NIM vs Original Gaussian Model

| Feature | Original | NIM (Inference) |
|---------|----------|-----------------|
| Lines of code | ~210 | 82 |
| `save_ply()` | ✓ | ✗ (not needed) |
| `load_ply()` | ✓ | ✗ (not needed) |
| `get_covariance()` | ✓ | ✗ (training only) |
| `general_utils.py` dependency | ✓ | ✗ (uses torch.logit) |
| Memory efficiency | Standard | Optimized |

## FlexiCubes Patch (kaolin removal)

The original `flexicubes.py` imports `kaolin.utils.testing.check_tensor` for input validation.
This is the **only** kaolin usage in the inference path.

Our patch replaces this with a lightweight local implementation:

```python
def check_tensor(tensor, expected_shape, throw=True):
    """Validates tensor has expected shape. None = any value (wildcard)."""
    if len(tensor.shape) != len(expected_shape):
        if throw: raise ValueError(...)
        return False
    for actual, expected in zip(tensor.shape, expected_shape):
        if expected is not None and actual != expected:
            if throw: raise ValueError(...)
            return False
    return True
```

| Comparison | With kaolin | Patched |
|------------|------------|---------|
| Install size | ~1GB | 0 (local implementation) |
| Dependencies | kaolin, many CUDA extensions | None |
| Functionality | Full kaolin toolkit | Just shape validation |

## postprocessing_utils Patch (igraph optional)

The original `postprocessing_utils.py` uses **igraph** (GPL v2+) for the minimum s-t cut algorithm
in mesh hole filling. GPL is a copyleft license that may require your project to be GPL-licensed.

Our patch supports **both** igraph and scipy:
- If **igraph is installed**: Uses igraph (faster, ~6-10x)
- If **igraph is NOT installed**: Falls back to scipy (BSD licensed)

```python
# Automatic backend selection
try:
    import igraph
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

def _compute_mincut(...):
    if IGRAPH_AVAILABLE:
        return _igraph_mincut(...)  # Faster, GPL
    else:
        return _scipy_mincut(...)   # BSD licensed fallback
```

| Comparison | igraph (optional) | scipy (default fallback) |
|------------|-------------------|--------------------------|
| License | GPL v2+ (copyleft) | BSD (permissive) |
| Performance | C-based, fastest | C-based (Cython), ~6-10x slower |
| Install | `pip install igraph` | Already included |
| Commercial use | ⚠️ May require GPL | ✅ Safe |

**To use igraph for better performance** (if GPL is acceptable):
```bash
pip install igraph
```

## License

Files in this directory are licensed under MIT License from Microsoft Corporation:

```
MIT License
Copyright (c) Microsoft Corporation.
```

## References

- gsplat: https://github.com/nerfstudio-project/gsplat
- gsplat Documentation: https://docs.gsplat.studio/
- NVIDIA Visual GenAI NIM: Source of inference-optimized TRELLIS code
- Original Issue: Inria/GRAPHDECO code requires commercial license for commercial use
