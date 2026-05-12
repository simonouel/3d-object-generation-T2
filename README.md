# 3D Object Generation Blueprint

## Description

The 3D Object Generation Blueprint is an end-to-end generative AI workflow that allows users to prototype 3D scenes quickly by simply describing the scene. The Blueprint takes a user's 3D scene idea, generates object recommendations, associated prompts and previews using a Llama 3.1 8B LLM and NVIDIA SANA, and ready-to-use 3D objects with Microsoft TRELLIS 2.  

> This blueprint supports the following NVIDIA GPUs: RTX 5090, RTX 5080, RTX 4090, RTX 4080, RTX 6000 Ada. We're planning to add wider GPU support in the near future. We recommend at least 48 GB of system RAM. 

## Features

- Chat interface for scene planning
- AI-assisted object and prompt generation
- Automatic 3D asset generation from text prompts
- Blender import functionality for generated assets
- **GPU memory management** - Intelligent model loading/unloading
- VRAM management with model termination

## Installation 

### Prerequisites

Before you begin, ensure you have:

- **Windows 10/11**
- **NVIDIA GPU** (RTX 4080 or higher recommended)
- **CUDA Toolkit 12.8** - Install from [NVIDIA CUDA 12.8 Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64)
- **~50GB disk space** for AI models
- **HuggingFace Account** (free) - Required for downloading some models. Create an account at [huggingface.co](https://huggingface.co/join) and generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Note:** TRELLIS 2 CUDA extensions (nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel) are built from source during installation. Ensure CUDA 12.8 and Visual Studio Build Tools are installed before running the installer.

---

### Installation

**Step 1:** Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/NVIDIA-AI-Blueprints/3d-object-generation.git
```

**Step 2:** Set the required environment variables:

```powershell
# PowerShell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:HF_TOKEN = "your_huggingface_token_here"
```
```cmd
# Command Prompt
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set HF_TOKEN=your_huggingface_token_here
```

> **Note:** Adjust the `CUDA_HOME` path if your CUDA 12.8 is installed in a different location.

**Step 3:** Open PowerShell **as Administrator** and navigate to this repository. Run the automated installation script. Select **(n)** if prompted to re-clone the repo:

```powershell
.\setup_environment.ps1
```

The setup script will automatically:
- Install Git and Git LFS (if not present)
- Install Miniconda (if not present)
- Create and configure the Conda environment
- Install all dependencies and CUDA extensions
- Download required AI models (~50GB)

**Optional Parameters:**

```powershell
.\setup_environment.ps1 -InstallPath "D:\my-custom-path" -CondaEnvName "myenv"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-InstallPath` | `C:\3d-object-generation` | Where to install the project |
| `-CondaEnvName` | `3dwithtrellis` | Name of the Conda environment |

> **Note:** The installation process may take 30-60 minutes depending on your internet connection and hardware.

---

### Install Blender

This blueprint requires Blender 4.2+ for the add-on integration. You can download and install manually from:
- [Blender 4.2.7 LTS](https://www.blender.org/download/release/Blender4.2/blender-4.2.7-windows-x64.msi)

Or via winget:
```
winget install --id 9NW1B444LDLW
```

> **Note:** Blender can be installed before or after the main setup. The installation script will automatically copy the add-ons to your Blender installation.

---

## Configuration

You can customize the following settings in `config.py`:

### LLM Model Options

```python
# Model name from HuggingFace
# Qwen3-4B: "Qwen/Qwen3-4B" (4B params)
# Llama-3.1-8B: "meta-llama/Llama-3.1-8B-Instruct" (8B params)
NATIVE_LLM_MODEL = "Qwen/Qwen3-4B"

# Precision options: "float16", "bfloat16", "float32", "int4" (for GPTQ)
NATIVE_LLM_PRECISION = "bfloat16"
```

### GPU Memory Management

The application automatically manages GPU memory across three models:
- **LLM** (Qwen3-4B or Llama 3.1 8B)
- **SANA** (Image generation)
- **TRELLIS 2** (3D generation — 4B parameters, O-Voxel architecture, PBR materials)

**Memory Management Strategy:**
- All models are pre-loaded at startup
- Models are moved to CPU when not actively in use
- Only one model runs on GPU at a time
- GPU cache is cleared between model switches

---

## Usage

There are two ways to run the application:

| Method | Best For | Description |
|--------|----------|-------------|
| **Blender Add-on** (Recommended) | 3D artists using Blender | Start services from within Blender, with integrated asset import |
| **Standalone** | Testing or non-Blender workflows | Run `python app.py` manually from command line |

Both methods launch the same Gradio web interface. **If you're working in Blender, use the add-on — there's no need to run `python app.py` separately.**

---

## Usage - Blender Add-on (Recommended)

The **3D Object Generation** add-on launches and manages all services directly from Blender.

### Initial Setup

1. Open Blender
2. Go to **Edit → Preferences → Add-ons**
3. <img width="996" height="384" alt="image" src="https://github.com/user-attachments/assets/a858877d-d182-44f2-bcc9-72f47358070c" />
4. Enable **3D Object Generation** and **Asset Importer** by checking the boxes
5. Expand the 3D Object Generation add-on and set **Blueprint Base Path** to your installation directory (e.g., `E:\3d-object-generation`)
6. <img width="983" height="537" alt="image" src="https://github.com/user-attachments/assets/55cb9cc8-3493-4b19-a139-14572e254c9a" />

### Starting Services

1. In the 3D Viewport, press **N** to open the sidebar
2. Click the **3D Object Generation** tab
3. <img width="590" height="476" alt="image" src="https://github.com/user-attachments/assets/51a7b2dd-be42-44f4-8572-d35a3c3967ad" />
4. (Optional) Open the system console for monitoring: **Window → Toggle System Console**
5. Click **Start Services** — this launches the LLM, SANA, and TRELLIS services (may take up to 3 minutes)
6. Once all services show **READY**, click **Open 3D Object Generation UI**
7. <img width="547" height="625" alt="image" src="https://github.com/user-attachments/assets/e4a62fd1-8948-4750-bba6-59438febc7a0" />

To stop services and free GPU memory, click **Services Started .. Click to Terminate**.

---

## Usage - Standalone (Alternative)

If you prefer to run the application outside of Blender:

### Starting the Application

1. Open a new terminal and run:
```
conda activate 3dwithtrellis
cd C:\3d-object-generation

python app.py
```

2. Open your browser to the URL shown in the terminal (typically http://127.0.0.1:7860/)

**💡 Recommended**: For the best experience, use the light theme by accessing the application with: `http://127.0.0.1:7860/?__theme=light`

**⚠️ Important Note**: Browser refresh is not supported and may cause the application to crash. In that case, please restart the application instead.

### Stopping the Application

To stop the application and free VRAM, simply press `Ctrl+C` in the terminal where the application is running.

---

### Using the Interface

Once the application is running, you can:

1. **Scene Planning**:
   - Describe your desired scene in natural language
   <kbd>
   <img width="1508" height="999" alt="image" src="https://github.com/user-attachments/assets/a65600ee-3507-4bb1-86d9-66d734134f63" />
   </kbd>

2. **Asset Generation**:
   - The LLM will automatically create prompts for suggested items which will be sent to the 2D image generator
     <kbd>
     <img width="1606" height="1352" alt="image" src="https://github.com/user-attachments/assets/55b11095-1b0a-42c5-95e6-7d32f56ed72f" />
     </kbd>
     
   - Each image contains additional controls
     
     <kbd>
     <img width="640" height="370" alt="image" src="https://github.com/user-attachments/assets/6a03b8ab-ee65-40ee-a458-ef64475d7a50" />
     </kbd>
     
     - <img width="86" height="74" alt="image" src="https://github.com/user-attachments/assets/4dd79a46-bb97-46f9-bcba-905606c168bf" /> **Refresh** - Generate a new image based on the existing prompt.
     - <img width="87" height="69" alt="image" src="https://github.com/user-attachments/assets/577d88ee-fc54-47f2-be77-922d0df2fba0" /> **Edit** - Edit the prompt an generate a new image.
     - <img width="77" height="65" alt="image" src="https://github.com/user-attachments/assets/7e740931-29e0-4ff4-af0b-cc080e130c2a" /> **Delete** - Remove the image from the gallery display.
     - <img width="114" height="63" alt="image" src="https://github.com/user-attachments/assets/e125e259-bf10-4e28-97f1-b416c08a168e" /> **Generate 3D** - Generate a 3D object from the image.
        - The color of the Generate 3D button indicates the status of 3D generation for that object
        -  <img width="122" height="72" alt="Screenshot 2025-08-22 141820" src="https://github.com/user-attachments/assets/f35824da-8311-4af6-ada0-a34c2795d8be" /> Object has not been queued for 3D generation.
        -  <img width="130" height="68" alt="Screenshot 2025-08-22 141836" src="https://github.com/user-attachments/assets/b58d7341-68dd-4bdb-ae75-281f4491327d" /> Object has been queued for 3D generation, but object generation has not completed.
        -  <img width="117" height="64" alt="Screenshot 2025-08-22 141810" src="https://github.com/user-attachments/assets/fd1119a5-b51b-4013-9c16-6cca27ed7834" /> 3D model has been generated for this object.
        -  <img width="120" height="66" alt="Screenshot 2025-08-22 141708" src="https://github.com/user-attachments/assets/66cd9956-d622-4059-b132-2470eb2ba42f" /> Object has been flagged by guardrails as potentially inappropriate, 3D object will not be generated.

<img width="2313" height="125" alt="image" src="https://github.com/user-attachments/assets/8ca38e36-c245-4e06-93e9-5bec518025c9" />

   - Convert all images to 3D Objects (Delete unwanted images before converting to 3D)
     
   - **NOTE**: Image to 3D Object processing takes 3–60 seconds *per object* on a RTX 5090 (TRELLIS 2 at 1024³ resolution), when using the Convert All image option this time will be a multiple of the number of objects being converted, using the Convert All option may take a significant amount of time. The UI will not be updated until all objects have been converted. 

3. **Save Objects**:
   - The Export Objects to File allows saving the generated objects to a folder.
   <kbd>
   <img width="2384" height="613" alt="image" src="https://github.com/user-attachments/assets/21a5b43f-7dda-42d9-bbf3-40518a3d3754" />
   </kbd>
   <img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/b302f89a-e282-4a22-ba8a-607cc2a40c82" />

4. **Blender Integration**:
   - Import generated assets directly into Blender
   - <img width="455" height="222" alt="image" src="https://github.com/user-attachments/assets/11b4f471-3fb1-4980-bba6-338886219202" />
   - Use the Asset Importer add-on and select the desired scene folder, and click Import assets
   - <img width="498" height="133" alt="image" src="https://github.com/user-attachments/assets/da88971b-ce42-454c-b3db-3ec8f32d0f68" />
   - Assets are imported and the asset tag is applied, saving the scene to the %userprofile%\Documents\Blender\assets folder will add the imported objects to the Blender asset browser.
   - <img width="1933" height="1234" alt="image" src="https://github.com/user-attachments/assets/f148936c-27da-428c-9d92-5603446deb37" />
   - Continue working with the assets in your 3D workflow
   - Can be used with [3D Guided Gen AI BP](https://github.com/NVIDIA-AI-Blueprints/3d-guided-genai-rtx)

---

## Configuration Reference

### `config.py` Key Settings

```python
# =============================================================================
# LLM Settings
# =============================================================================
NATIVE_LLM_MODEL = "Qwen/Qwen3-4B"    # HuggingFace model ID
NATIVE_LLM_PRECISION = "bfloat16"      # float16, bfloat16, or int4 (for GPTQ)

# =============================================================================
# Logging
# =============================================================================
VERBOSE = False                        # Detailed timing/memory logs
```

---

## Troubleshooting

### Common Issues

1. **CUDA not found during installation**
   - Ensure CUDA 12.8 is installed from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64)
   - If the installer can't auto-detect CUDA, set `CUDA_HOME` before running `install.bat`:
     ```powershell
     # PowerShell
     $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
     ```
     ```cmd
     # Command Prompt
     set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
     ```
   - Verify your CUDA installation path:
     ```powershell
     Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\"
     ```

2. **Out of VRAM**
   - Use a smaller LLM model: Qwen3-4B instead of Llama-3.1-8B
   - Close other GPU-using applications
   - The application automatically moves inactive models to CPU

3. **Slow LLM inference**
   - Ensure using Gradio 5.x (not 6.x)
   - Check `requirements.txt` has `gradio==5.50.0`

4. **Model download fails**
   - Set HuggingFace token: `set HF_TOKEN=your_token`
   - Check internet connection

5. **TRELLIS 2 import errors**
   - Ensure the `TRELLIS.2` directory is present at the root of the project (cloned from `https://github.com/microsoft/TRELLIS.2`)
   - Ensure TRELLIS 2 dependencies are installed: `python install_dependencies.py`
   - Verify the `trellis2` module is importable: `python -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline"`

6. **Installation Issues**:
   - Run PowerShell as Administrator
   - Check if Python is in your system PATH
   - Verify Visual Studio Build Tools installation

### Logs

- Application logs: Console output
- Verbose logging: Set `VERBOSE = True` in `config.py`

## Acknowledgments

- [TRELLIS 2](https://github.com/microsoft/TRELLIS.2) - Microsoft's 3D generation model (4B params, O-Voxel, PBR materials)
- [Qwen3](https://github.com/QwenLM/Qwen3) - Alibaba's LLM
- [SANA](https://github.com/NVlabs/Sana) - NVIDIA's image generation model
- [Gradio](https://github.com/gradio-app/gradio) - Web interface framework

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
