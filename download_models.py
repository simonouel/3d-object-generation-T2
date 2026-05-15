#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Download all required models for the 3D Object Generation application.

This script downloads:
- NSFW Prompt Detector (guardrail)
- Native LLM model (if USE_NATIVE_LLM = True)
- Native TRELLIS model (if USE_NATIVE_TRELLIS = True)

Note: Image generation uses a local safetensors file (RealVisXL Lightning) —
no download needed. See config.IMAGE_MODEL_PATH.
"""

import torch
import gc
import sys
from pathlib import Path

from huggingface_hub import scan_cache_dir
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config settings directly
import config


def is_model_cached(repo_id: str) -> bool:
    """Return True if the model is already fully downloaded in the HF cache."""
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception as e:
        logger.debug(f"Cache scan failed for {repo_id}: {e}")
        return False


def download_sana_model():
    """Verify the local image generation model file exists (no download needed)."""
    model_path = Path(config.IMAGE_MODEL_PATH)
    if model_path.exists():
        logger.info(f"✓ Image generation model found: {model_path}")
        return True
    logger.error(f"✗ Image generation model not found: {model_path}")
    logger.error("  Set IMAGE_MODEL_PATH env var to a valid .safetensors file.")
    return False


def download_guardrail_model():
    """Download the NSFW Prompt Detector guardrail model."""
    repo_id = "ezb/NSFW-Prompt-Detector"
    if is_model_cached(repo_id):
        logger.info(f"✓ NSFW Prompt Detector already downloaded — skipping")
        return True
    logger.info("Downloading NSFW Prompt Detector model...")
    try:
        guardrail_pipe = pipeline("text-classification", model=repo_id)
        del guardrail_pipe
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("✓ NSFW Prompt Detector model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading NSFW Prompt Detector model: {e}")
        return False


def download_native_llm_model():
    """Download the native LLM model for text generation."""
    if getattr(config, 'USE_OPENAI_COMPATIBLE_LLM', False):
        logger.info("Skipping native LLM download (USE_OPENAI_COMPATIBLE_LLM = True, using remote endpoint)")
        return True
    if not config.USE_NATIVE_LLM:
        logger.info("Skipping native LLM download (USE_NATIVE_LLM = False)")
        return True

    if is_model_cached(config.NATIVE_LLM_MODEL):
        logger.info(f"✓ LLM model {config.NATIVE_LLM_MODEL} already downloaded — skipping")
        return True

    logger.info(f"Downloading native LLM model: {config.NATIVE_LLM_MODEL}...")
    try:
        # Download tokenizer
        logger.info("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.NATIVE_LLM_MODEL)
        del tokenizer

        # Download model weights (don't load to GPU, just cache)
        logger.info("  Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.NATIVE_LLM_MODEL,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"✓ Native LLM model downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ Error downloading native LLM model: {e}")
        logger.error("  Make sure you have accepted the model license on HuggingFace")
        logger.error("  and set HF_TOKEN environment variable if required.")
        return False


TRELLIS_GATED_DEPS = [
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
]


def check_trellis_gated_deps() -> bool:
    """Verify access to gated HuggingFace models required by TRELLIS 2."""
    from huggingface_hub import model_info
    all_ok = True
    for repo_id in TRELLIS_GATED_DEPS:
        if is_model_cached(repo_id):
            logger.info(f"✓ {repo_id} already downloaded — skipping")
            continue
        try:
            model_info(repo_id)  # raises if no access
            logger.info(f"✓ Access confirmed: {repo_id}")
        except Exception as e:
            err = str(e)
            if "403" in err or "gated" in err or "restricted" in err:
                logger.error(f"✗ Access denied: {repo_id}")
                logger.error(f"  TRELLIS 2 requires this gated model.")
                logger.error(f"  1. Request access at: https://huggingface.co/{repo_id}")
                logger.error(f"  2. Set HF_TOKEN: export HF_TOKEN=hf_xxx")
                all_ok = False
            else:
                logger.warning(f"  Could not verify {repo_id}: {e}")
    return all_ok


def download_native_trellis_model():
    """Download the native TRELLIS 3D generation model."""
    if not config.USE_NATIVE_TRELLIS:
        logger.info("Skipping native TRELLIS download (USE_NATIVE_TRELLIS = False)")
        return True

    if not check_trellis_gated_deps():
        logger.error("✗ Cannot download TRELLIS: missing access to required gated models (see above).")
        return False

    if is_model_cached(config.NATIVE_TRELLIS_MODEL):
        logger.info(f"✓ TRELLIS model {config.NATIVE_TRELLIS_MODEL} already downloaded — skipping")
        return True

    # Need to add TRELLIS.2 to sys.path so trellis2 module is importable
    trellis2_path = str(Path(__file__).parent / "TRELLIS.2")
    if trellis2_path not in sys.path:
        sys.path.insert(0, trellis2_path)

    logger.info(f"Downloading native TRELLIS model: {config.NATIVE_TRELLIS_MODEL}...")
    try:
        # Import TRELLIS 2 pipeline
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Download by loading the pipeline (will cache to HuggingFace cache)
        logger.info("  Downloading TRELLIS pipeline (this may take a while)...")
        pipeline_trellis = Trellis2ImageTo3DPipeline.from_pretrained(config.NATIVE_TRELLIS_MODEL)
        del pipeline_trellis
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("✓ Native TRELLIS model downloaded successfully!")
        return True
    except ImportError as e:
        logger.warning(f"Could not import TRELLIS pipeline: {e}")
        logger.warning("  TRELLIS dependencies may not be installed yet.")
        logger.warning("  The model will be downloaded when first used.")
        return True  # Not a fatal error
    except Exception as e:
        err = str(e)
        if "403" in err or "gated" in err or "restricted" in err:
            logger.error(f"✗ Access denied downloading TRELLIS model.")
            logger.error(f"  Request access and set HF_TOKEN (see above).")
        else:
            logger.error(f"✗ Error downloading TRELLIS model: {e}")
        return False


def download_models():
    """Download all required models."""
    logger.info("=" * 60)
    logger.info("Starting model downloads...")
    logger.info("=" * 60)
    use_openai = getattr(config, 'USE_OPENAI_COMPATIBLE_LLM', False)
    logger.info(f"Configuration:")
    logger.info(f"  USE_OPENAI_COMPATIBLE_LLM: {use_openai}")
    logger.info(f"  USE_NATIVE_LLM: {config.USE_NATIVE_LLM}")
    logger.info(f"  USE_NATIVE_TRELLIS: {config.USE_NATIVE_TRELLIS}")
    if use_openai:
        logger.info(f"  OPENAI_COMPATIBLE_BASE_URL: {getattr(config, 'OPENAI_COMPATIBLE_BASE_URL', '?')}")
        logger.info(f"  OPENAI_COMPATIBLE_MODEL: {getattr(config, 'OPENAI_COMPATIBLE_MODEL', 'default')} (auto-detected at startup)")
    elif config.USE_NATIVE_LLM:
        logger.info(f"  NATIVE_LLM_MODEL: {config.NATIVE_LLM_MODEL}")
    if config.USE_NATIVE_TRELLIS:
        logger.info(f"  NATIVE_TRELLIS_MODEL: {config.NATIVE_TRELLIS_MODEL}")
    logger.info("")
    
    all_success = True
    
    # Core models (always needed)
    if not download_sana_model():
        all_success = False
    
    if not download_guardrail_model():
        all_success = False
    
    # Native LLM model (if enabled)
    if not download_native_llm_model():
        all_success = False
    
    # Native TRELLIS model (if enabled)
    if not download_native_trellis_model():
        all_success = False
    
    logger.info("")
    logger.info("=" * 60)
    if all_success:
        logger.info("✓ All models downloaded successfully!")
    else:
        logger.warning("⚠ Some models failed to download. Check the logs above.")
    logger.info("=" * 60)
    
    return all_success


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
