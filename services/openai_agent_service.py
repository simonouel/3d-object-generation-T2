"""OpenAI-compatible LLM agent service — drop-in replacement for NativeAgentService.

Works with any OpenAI API-compatible server: vLLM, llama.cpp server,
Ollama (with OpenAI adapter), LM Studio, etc.
"""

import re
import logging
from typing import Optional, Tuple, List

import config

logger = logging.getLogger(__name__)


class OpenAIAgentService:
    """Agent service backed by an OpenAI-compatible REST API (e.g. vLLM).

    No GPU is used locally. All inference is delegated to the remote endpoint.
    The interface is identical to NativeAgentService.
    """

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai>=1.0")

        base_url = getattr(config, 'OPENAI_COMPATIBLE_BASE_URL', 'http://localhost:8000/v1')
        model    = getattr(config, 'OPENAI_COMPATIBLE_MODEL', 'default')

        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        self._model  = model
        self._base_url = base_url
        logger.info(f"OpenAIAgentService: endpoint={base_url}, model={model}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str, temperature: float = None) -> str:
        temp = temperature if temperature is not None else getattr(config, 'LLM_TEMPERATURE', 0.4)
        max_tokens = getattr(config, 'NATIVE_LLM_MAX_NEW_TOKENS', 1024)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _planning_system_prompt(self) -> str:
        n = getattr(config, 'NUM_OF_OBJECTS', 20)
        return (
            "You are a helpful scene planning assistant for 3D content creation.\n"
            "Your primary task is to suggest objects for the 3D scene based on user requests.\n"
            "Always suggest objects in singular form (e.g., 'Palm Tree' instead of 'Palm Trees').\n"
            "Always format object names with proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table').\n"
            f"When suggesting objects for a general scene request: Suggest exactly {n} objects that would be appropriate for this scene.\n"
            f"Always format your object suggestions in this exact format: Suggested objects: 1. object_name 2. object_name ... {n}. object_name\n"
            "Focus only on suggesting appropriate objects for the scene based on the user's requests."
        )

    def _prompt_gen_system_prompt(self) -> str:
        length = getattr(config, 'TWO_D_PROMPT_LENGTH', 30)
        return (
            "You are now in 2D prompt generation mode. Your task is to create detailed prompts for each object in the scene.\n"
            "The descriptions should be highly detailed and visually rich, suitable for a text-to-image generation model.\n"
            "The prompt must specify a plain or empty background (e.g., 'on a white background', 'isolated', 'no background').\n"
            "Focus ONLY on the physical and visual characteristics of each object itself.\n"
            f"Keep each object's prompt to exactly {length} words or less for optimal generation quality.\n"
            "Format each object's prompt with 'Object:' and 'Prompt:' labels.\n"
            "Do not add any explanatory notes or comments after the prompt.\n"
            "Do not use asterisks or any special formatting characters.\n"
            "Output only the Object and Prompt labels with clean text - no additional formatting or notes.\n\n"
            "The prompt text should describe the object's visual characteristics in detail.\n"
            "Example:\nObject: Beach Chair\nPrompt: A comfortable beach chair with ergonomic design and colorful fabric, on a white background\n"
            "Example:\nObject: Beach Umbrella\nPrompt: A vibrant beach umbrella with colorful stripes and sturdy metal frame, on a white background"
        )

    @staticmethod
    def _classification_system_prompt() -> str:
        return (
            "You are an input classifier for a 3D scene creation application.\n"
            "Your task is to determine if the user's input is a scene description or something else.\n"
            "A scene description should describe a physical environment, location, or setting that can contain objects.\n"
            "Scene descriptions typically include: locations (beach, kitchen, garden), environments (outdoor, indoor), settings (modern, rustic, tropical), or specific places.\n"
            "Non-scene inputs include: greetings (hello, hi), questions (what can you do, how does this work), general chat (thanks, nice app), or requests for help.\n\n"
            "You must respond with exactly one of these classifications:\n"
            "1. 'SCENE' - if the input describes a scene, environment, or location that could contain objects\n"
            "2. 'GREETING' - if the input is a greeting or salutation\n"
            "3. 'QUESTION' - if the input is asking about capabilities, how to use the app, or seeking information\n"
            "4. 'GENERAL_CHAT' - if the input is general conversation, thanks, or other non-scene content\n\n"
            "Respond with only the classification word (SCENE, GREETING, QUESTION, or GENERAL_CHAT) - no additional text or explanation."
        )

    # ------------------------------------------------------------------
    # Public interface (matches NativeAgentService exactly)
    # ------------------------------------------------------------------

    def _initialize_agent(self, load_model: bool = True):
        pass  # no local model to load

    def _ensure_agent_loaded(self, load_model: bool = True):
        pass

    def is_healthy(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI endpoint health check failed: {e}")
            return False

    def chat(self, message: str, current_objects=None) -> str:
        try:
            return self._chat(self._planning_system_prompt(), message)
        except Exception as e:
            return f"Error communicating with LLM endpoint: {e}"

    def classify_input(self, user_input: str) -> Tuple[str, Optional[str]]:
        if not user_input.strip():
            return "EMPTY", "Please enter a scene description."
        try:
            classification = self._chat(
                self._classification_system_prompt(), user_input, temperature=0.1
            ).upper()
            valid = {"SCENE", "GREETING", "QUESTION", "GENERAL_CHAT"}
            if classification not in valid:
                classification = "SCENE"
            messages = {
                "GREETING":     "Hello! To get started, please describe a scene you'd like to create. For example: 'A cozy living room with a fireplace' or 'A tropical beach with palm trees'",
                "QUESTION":     "I can help you create 3D scenes! Describe what you want to build, like 'A modern kitchen' or 'A garden with flowers'.",
                "GENERAL_CHAT": "Thanks! To create a 3D scene, please describe what you'd like to build. Try something like 'A beach scene' or 'A modern office'.",
            }
            return classification, messages.get(classification)
        except Exception as e:
            logger.error(f"classify_input error: {e}")
            return "SCENE", None

    def generate_objects_for_scene(self, scene_description: str) -> List[str]:
        n = getattr(config, 'NUM_OF_OBJECTS', 20)
        try:
            response = self._chat(
                self._planning_system_prompt(),
                f'Based on this scene description: "{scene_description}"\nPlease suggest exactly {n} objects.',
            )
            matches = re.findall(r'\d+\.\s+(.+?)(?=\s*\d+\.|$)', response, re.DOTALL)
            objects = [m.strip() for m in matches if m.strip()][:n]
            return objects if objects else []
        except Exception as e:
            logger.error(f"generate_objects_for_scene error: {e}")
            return []

    def generate_objects_and_prompts_with_progress(self, description: str):
        try:
            yield (0, 0, "Generating object list...", False, None)
            objects = self.generate_objects_for_scene(description)
            if not objects:
                yield (0, 0, "No objects generated", True, (False, None, "No objects generated"))
                return

            total = len(objects)
            prompts = {}
            system = self._prompt_gen_system_prompt()

            for idx, obj in enumerate(objects, 1):
                yield (idx, total, f"Generating prompt {idx}/{total}: {obj}", False, None)
                user_msg = f"Generate a visual prompt for this object:\nObject: {obj}"
                response = self._chat(system, user_msg)

                if "Prompt:" in response:
                    prompt_text = response.split("Prompt:")[-1].strip()
                else:
                    prompt_text = response.strip()
                prompts[obj] = prompt_text or f"{obj}, detailed 2D image on white background"

            yield (total, total, "Generation complete!", True, (True, prompts, "2D prompts generated successfully"))

        except Exception as e:
            yield (0, 1, f"Error: {e}", True, (False, None, f"Error generating prompts: {e}"))

    def clear_memory(self) -> bool:
        return True  # stateless API, nothing to clear

    def move_agent_to_cpu(self):
        pass  # no local model

    def move_agent_to_gpu(self):
        pass

    def unload_agent(self):
        pass

    def cleanup(self):
        pass
