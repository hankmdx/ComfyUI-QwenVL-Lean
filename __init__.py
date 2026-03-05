"""ComfyUI-QwenVL-Lean — GGUF-only auto-prompting nodes.

Standalone replacement for ComfyUI-QwenVL-Mod's GGUF nodes.
Zero dependency on bitsandbytes or transformers.
"""

from .qwenvl_gguf import QwenVL_Lean, QwenVL_Lean_Advanced
from .prompt_enhancer import QwenVL_Lean_PromptEnhancer

NODE_CLASS_MAPPINGS = {
    "QwenVL_Lean": QwenVL_Lean,
    "QwenVL_Lean_Advanced": QwenVL_Lean_Advanced,
    "QwenVL_Lean_PromptEnhancer": QwenVL_Lean_PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVL_Lean": "QwenVL Lean (GGUF)",
    "QwenVL_Lean_Advanced": "QwenVL Lean Advanced (GGUF)",
    "QwenVL_Lean_PromptEnhancer": "QwenVL Lean Prompt Enhancer (GGUF)",
}
