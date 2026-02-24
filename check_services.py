#!/usr/bin/env python3

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
Service health checker for LLM and Trellis services.
This script checks if the services are running and ready.

Supports both NIM mode (external containers) and native mode (in-process models).
In native mode, services run inside the Gradio app, so we use Gradio's health
as a proxy for service readiness.
"""

import requests
import time
import sys
import logging
from pathlib import Path

# Import config to detect native vs NIM mode
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_service_health(url, service_name, timeout=5):
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"{service_name} is ready")
            return True
        else:
            logger.debug(f"{service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"{service_name} not ready: {e}")
        return False

def main():
    """Main function to check services based on configuration mode."""
    gradio_url = "http://localhost:7860"
    llm_url = "http://localhost:19002/v1/health/ready"
    trellis_url = "http://localhost:8000/v1/health/ready"
    
    # Check Gradio first - needed for native mode checks
    gradio_ready = check_service_health(gradio_url, "Gradio")
    
    # Determine service readiness based on mode
    if config.USE_NATIVE_LLM and config.USE_NATIVE_TRELLIS:
        # Full native mode: both services run in-process with Gradio
        # If Gradio is up, all services are ready
        if gradio_ready:
            logger.info("Native mode: All services running in-process with Gradio")
            print("ALL_READY")
            sys.exit(0)
        else:
            print("NONE_READY")
            sys.exit(3)
    
    elif config.USE_NATIVE_LLM:
        # Native LLM + NIM Trellis
        llm_ready = gradio_ready  # LLM runs in Gradio process
        trellis_ready = check_service_health(trellis_url, "Trellis Service")
        
    elif config.USE_NATIVE_TRELLIS:
        # NIM LLM + Native Trellis
        llm_ready = check_service_health(llm_url, "LLM Service")
        trellis_ready = gradio_ready  # Trellis runs in Gradio process
        
    else:
        # Full NIM mode: check external endpoints
        llm_ready = check_service_health(llm_url, "LLM Service")
        trellis_ready = check_service_health(trellis_url, "Trellis Service")
    
    # Return exit code based on service status
    if llm_ready and trellis_ready:
        print("ALL_READY")
        sys.exit(0)
    elif llm_ready:
        print("LLM_READY")
        sys.exit(1)
    elif trellis_ready:
        print("TRELLIS_READY")
        sys.exit(2)
    else:
        print("NONE_READY")
        sys.exit(3)

if __name__ == "__main__":
    main() 