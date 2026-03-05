"""Self-contained prompt cache for ComfyUI-QwenVL-Lean.

MD5 key from (model, preset, custom_prompt, image_hash, seed).
Persistent via prompt_cache.json. No external dependencies.
"""

import hashlib
import json
from pathlib import Path

CACHE_FILE = Path(__file__).parent / "prompt_cache.json"
PROMPT_CACHE: dict = {}


def _load_cache():
    global PROMPT_CACHE
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                PROMPT_CACHE = json.load(f)
                print(f"[QwenVL-Lean] Loaded {len(PROMPT_CACHE)} cached prompts")
    except Exception as e:
        print(f"[QwenVL-Lean] Failed to load prompt cache: {e}")
        PROMPT_CACHE = {}


def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(PROMPT_CACHE, f, indent=2)
    except Exception as e:
        print(f"[QwenVL-Lean] Failed to save prompt cache: {e}")


def get_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None):
    key_data = {
        "model": model_name,
        "preset": preset_prompt,
        "custom": custom_prompt.strip() if custom_prompt else "",
        "image": image_hash,
        "video": video_hash,
        "seed": seed,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_image_hash(image):
    if image is None:
        return None
    try:
        shape = str(image.shape)
        dtype = str(image.dtype)
        if len(image.shape) >= 3:
            sample_pixels = image.flatten()[:100].tolist() if image.numel() > 0 else []
        else:
            sample_pixels = image.flatten().tolist() if image.numel() > 0 else []
        content = f"{shape}_{dtype}_{sample_pixels[:10]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return None


def get_video_hash(video):
    return get_image_hash(video)


# Load cache on import
_load_cache()
