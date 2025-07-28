import re
import time

from comfy.comfy_types.node_typing import IO

from .ollama_client import (
    DEFAULT_MODEL,
    MAX_GENERATE_ATTEMPTS,
    OllamaRequestError,
    build_generate_payload,
    build_options,
    generate_text_streaming,
    get_default_host,
    get_model_names,
    image_tensor_to_base64_png,
    normalize_host,
    require_vision_support,
    resolve_model,
    strip_think_tags,
)


IMAGE_TO_PROMPT_SYSTEM = """You are an expert visual analyst and prompt extractor. Convert the input image into a dense, precise, hierarchical description that can be reused as a high-quality image generation prompt. Describe only visually supported information, separate observation from interpretation, and prioritize concrete attributes over vague adjectives. At the end, synthesize the details into a generation-ready prompt."""

ENHANCE_PROMPT_POLISH_SYSTEM = """You are an expert prompt editor for image generation. Rewrite the user's prompt with better clarity, structure, specificity, and visual usefulness while keeping it grounded and controlled. Preserve the original subject, intent, style constraints, and requested details. Improve weak wording, remove ambiguity, and add only modest, plausible visual details that support the original idea. Return only the polished prompt."""

ENHANCE_PROMPT_CREATIVE_SYSTEM = """You are an imaginative visual prompt designer for image generation. Transform the user's prompt into a richer, more evocative, more artistic version while preserving the core subject and intent. Add creative flair through composition, lighting, atmosphere, texture, style direction, color relationships, and tasteful embellishments. Do not add unrelated subjects or contradict explicit constraints. Return only the creatively enhanced prompt."""

IMAGE_TO_PROMPT_DEFAULT_PROMPT = "Analyze the input image and produce a detailed, generation-ready prompt."
DEFAULT_PASSTHROUGH_DELIMITER = "Prompt: "

TASK_PRESETS = ["custom", "image_to_prompt", "enhance_prompt_polish", "enhance_prompt_creative"]
LEGACY_TASK_PRESETS = {
    "enhance_prompt": "enhance_prompt_polish",
}
SAMPLING_MODES = ["preset_optimized", "model_defaults", "custom"]

UNUSABLE_RESPONSE_MARKERS = (
    "as an ai language model",
    "i am unable to",
    "i'm unable to",
    "i cannot",
    "i can't",
    "cannot assist",
    "can't assist",
    "unable to analyze",
    "unable to process",
    "unable to view",
    "unable to see",
    "i do not have the ability",
    "i don't have the ability",
    "no image was provided",
    "no input image",
)

_THINK_CONTENT_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_CONTENT_RE = re.compile(r"<think\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
PROGRESS_TEXT_INTERVAL_SECONDS = 0.5

PRESET_PARAMETER_PROFILES = {
    "custom": {
        "num_predict": 4096,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 1.5,
        "seed": -1,
        "thinking": True,
        "strip_thinking": True,
        "keep_alive": "5m",
        "timeout_seconds": 120,
    },
    "image_to_prompt": {
        "num_predict": 4096,
        "temperature": 0.25,
        "top_k": 20,
        "top_p": 0.9,
        "min_p": 0.0,
        "repeat_penalty": 1.05,
        "presence_penalty": 0.2,
        "seed": -1,
        "thinking": True,
        "strip_thinking": True,
        "keep_alive": "15m",
        "timeout_seconds": 300,
    },
    "enhance_prompt_polish": {
        "num_predict": 4096,
        "temperature": 0.35,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.0,
        "repeat_penalty": 1.05,
        "presence_penalty": 0.25,
        "seed": -1,
        "thinking": True,
        "strip_thinking": True,
        "keep_alive": "10m",
        "timeout_seconds": 300,
    },
    "enhance_prompt_creative": {
        "num_predict": 4096,
        "temperature": 0.85,
        "top_k": 80,
        "top_p": 0.95,
        "min_p": 0.03,
        "repeat_penalty": 1.08,
        "presence_penalty": 0.6,
        "seed": -1,
        "thinking": True,
        "strip_thinking": True,
        "keep_alive": "10m",
        "timeout_seconds": 300,
    },
}


def _normalize_task_preset(task_preset):
    task_preset = (task_preset or "custom").strip()
    return LEGACY_TASK_PRESETS.get(task_preset, task_preset)


def _resolve_system_prompt(task_preset, system_prompt):
    task_preset = _normalize_task_preset(task_preset)
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        return system_prompt
    if task_preset == "image_to_prompt":
        return IMAGE_TO_PROMPT_SYSTEM
    if task_preset == "enhance_prompt_polish":
        return ENHANCE_PROMPT_POLISH_SYSTEM
    if task_preset == "enhance_prompt_creative":
        return ENHANCE_PROMPT_CREATIVE_SYSTEM
    return ""


def _resolve_prompt(task_preset, prompt):
    task_preset = _normalize_task_preset(task_preset)
    prompt = prompt or ""
    if task_preset == "image_to_prompt" and not prompt.strip():
        return IMAGE_TO_PROMPT_DEFAULT_PROMPT
    if task_preset in ("enhance_prompt_polish", "enhance_prompt_creative") and not prompt.strip():
        raise ValueError(f"The {task_preset} preset requires a non-empty prompt.")
    return prompt


def _extract_passthrough_prompt(prompt, delimiter):
    value = prompt or ""
    delimiter = delimiter if delimiter is not None else DEFAULT_PASSTHROUGH_DELIMITER
    if delimiter:
        _, found, remainder = value.partition(delimiter)
        if found:
            value = remainder.lstrip()
    return value


def _passthrough_result(prompt, delimiter=DEFAULT_PASSTHROUGH_DELIMITER, thinking_stream=""):
    value = _extract_passthrough_prompt(prompt, delimiter)
    return {"result": (value, thinking_stream or "")}


def _extract_thinking_stream(value):
    value = value or ""
    thinking_blocks = [match.group(1).strip() for match in _THINK_CONTENT_RE.finditer(value)]
    without_closed_blocks = _THINK_CONTENT_RE.sub(" ", value)
    unclosed_match = _UNCLOSED_THINK_CONTENT_RE.search(without_closed_blocks)
    if unclosed_match:
        thinking_blocks.append(unclosed_match.group(1).strip())
    return "\n\n".join(block for block in thinking_blocks if block).strip()


def _is_unusable_model_result(value):
    text = (value or "").strip()
    if not text:
        return True

    normalized = " ".join(text.casefold().split())
    if normalized in ("none", "null", "n/a"):
        return True
    return any(marker in normalized for marker in UNUSABLE_RESPONSE_MARKERS)


def _compact_progress_reason(reason, max_length=220):
    reason = " ".join((reason or "").split())
    if len(reason) <= max_length:
        return reason
    return reason[: max_length - 3].rstrip() + "..."


def _send_node_progress_text(unique_id, text):
    if not unique_id:
        return
    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_progress_text(text, unique_id)
    except Exception:
        pass


class _NoOpProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total


def _make_progress_bar(total, node_id=None):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total, node_id=node_id)
    except Exception:
        return _NoOpProgressBar(total, node_id=node_id)


class _OllamaProgressReporter:
    def __init__(self, unique_id, num_predict, max_attempts=MAX_GENERATE_ATTEMPTS):
        self.unique_id = unique_id
        self.units_per_attempt = max(1, int(num_predict or 1))
        self.max_attempts = max(1, int(max_attempts or 1))
        self.total = self.units_per_attempt * self.max_attempts
        self.progress_bar = _make_progress_bar(self.total, node_id=unique_id)
        self._last_text_time = 0.0

    def phase(self, text, value=None, force=True):
        if value is not None:
            self.progress_bar.update_absolute(max(0, min(int(value), self.total)), self.total)
        self._send_text(text, force=force)

    def finish(self, text):
        self.progress_bar.update_absolute(self.total, self.total)
        self._send_text(text, force=True)

    def __call__(self, event):
        event_type = event.get("type")
        attempt = max(1, int(event.get("attempt") or 1))
        max_attempts = max(1, int(event.get("max_attempts") or self.max_attempts))
        if max_attempts != self.max_attempts:
            self.max_attempts = max_attempts
            self.total = self.units_per_attempt * self.max_attempts
        offset = (attempt - 1) * self.units_per_attempt

        if event_type == "attempt_start":
            self.progress_bar.update_absolute(min(offset, self.total), self.total)
            self._send_text(f"Ollama: generating text (attempt {attempt}/{self.max_attempts})", force=True)
            return

        if event_type == "token":
            generated_units = max(1, int(event.get("generated_chunks") or 1))
            progress_value = min(offset + generated_units, max(0, self.total - 1))
            self.progress_bar.update_absolute(progress_value, self.total)
            generated_characters = int(event.get("generated_characters") or 0)
            self._send_text(
                f"Ollama: generating text (attempt {attempt}/{self.max_attempts})\n"
                f"chunks: {generated_units}, chars: {generated_characters}",
                force=False,
            )
            return

        if event_type == "stream_done":
            progress_value = min(offset + self.units_per_attempt, max(0, self.total - 1))
            self.progress_bar.update_absolute(progress_value, self.total)
            self._send_text(f"Ollama: validating output (attempt {attempt}/{self.max_attempts})", force=True)
            return

        if event_type == "attempt_retry":
            reason = _compact_progress_reason(event.get("reason"))
            progress_value = min(offset + self.units_per_attempt, max(0, self.total - 1))
            self.progress_bar.update_absolute(progress_value, self.total)
            self._send_text(
                f"Ollama: retrying generation (attempt {attempt}/{self.max_attempts})\n{reason}",
                force=True,
            )
            return

        if event_type == "attempt_error":
            reason = _compact_progress_reason(event.get("reason"))
            progress_value = min(offset + self.units_per_attempt, max(0, self.total - 1))
            self.progress_bar.update_absolute(progress_value, self.total)
            self._send_text(
                f"Ollama: generation error (attempt {attempt}/{self.max_attempts})\n{reason}",
                force=True,
            )
            return

        if event_type == "attempt_success":
            generated_characters = int(event.get("generated_characters") or 0)
            self.finish(f"Ollama: generation complete\nchars: {generated_characters}")
            return

        if event_type == "failed":
            reason = _compact_progress_reason(event.get("reason"))
            self.finish(f"Ollama: generation failed, passing through prompt\n{reason}")

    def _send_text(self, text, force=False):
        now = time.monotonic()
        if not force and now - self._last_text_time < PROGRESS_TEXT_INTERVAL_SECONDS:
            return
        self._last_text_time = now
        _send_node_progress_text(self.unique_id, text)


def _get_preset_parameter_profile(task_preset):
    task_preset = _normalize_task_preset(task_preset)
    return dict(PRESET_PARAMETER_PROFILES.get(task_preset, PRESET_PARAMETER_PROFILES["custom"]))


def _resolve_effective_generation_settings(
    task_preset,
    sampling_mode,
    num_predict,
    temperature,
    top_k,
    top_p,
    min_p,
    repeat_penalty,
    presence_penalty,
    seed,
    thinking,
    strip_thinking,
    keep_alive,
    timeout_seconds,
):
    settings = {
        "sampling_mode": sampling_mode,
        "num_predict": num_predict,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "seed": seed,
        "thinking": thinking,
        "strip_thinking": strip_thinking,
        "keep_alive": keep_alive,
        "timeout_seconds": timeout_seconds,
    }
    if sampling_mode == "preset_optimized":
        settings.update(_get_preset_parameter_profile(task_preset))
        settings["sampling_mode"] = "custom"
    return settings


class OllamaGenerateText:
    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names(timeout_seconds=1)
        default_model = DEFAULT_MODEL if DEFAULT_MODEL in model_names else model_names[0]
        defaults = _get_preset_parameter_profile("custom")
        return {
            "required": {
                "prompt": (IO.STRING, {"default": "", "multiline": True, "dynamicPrompts": True}),
                "model": (model_names, {"default": default_model}),
                "task_preset": (TASK_PRESETS, {"default": "custom"}),
                "num_predict": ("INT", {"default": defaults["num_predict"], "min": 1, "max": 262144, "step": 1}),
                "sampling_mode": (SAMPLING_MODES, {"default": "preset_optimized"}),
                "temperature": ("FLOAT", {"default": defaults["temperature"], "min": 0.0, "max": 2.0, "step": 0.000001}),
                "top_k": ("INT", {"default": defaults["top_k"], "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": defaults["top_p"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": defaults["min_p"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": defaults["repeat_penalty"], "min": 0.0, "max": 5.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": defaults["presence_penalty"], "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": defaults["seed"], "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "thinking": ("BOOLEAN", {"default": defaults["thinking"]}),
                "strip_thinking": ("BOOLEAN", {"default": defaults["strip_thinking"]}),
                "keep_alive": ("STRING", {"default": defaults["keep_alive"], "multiline": False, "advanced": True}),
                "timeout_seconds": ("INT", {"default": defaults["timeout_seconds"], "min": 1, "max": 3600, "step": 1, "advanced": True}),
                "model_override": ("STRING", {"default": "", "multiline": False, "advanced": True}),
                "ollama_host": ("STRING", {"default": get_default_host(), "multiline": False, "advanced": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "advanced": True}),
            },
            "optional": {
                "delimiter": (IO.STRING, {"default": DEFAULT_PASSTHROUGH_DELIMITER, "multiline": False}),
                "image": (IO.IMAGE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("generated_text", "thinking_stream")
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "text/ollama"
    SEARCH_ALIASES = ["ollama", "prompt enhance", "image to prompt", "vision llm", "vision language"]

    def main(
        self,
        prompt,
        model,
        task_preset,
        num_predict,
        sampling_mode,
        temperature,
        top_k,
        top_p,
        min_p,
        repeat_penalty,
        presence_penalty,
        seed,
        thinking,
        strip_thinking,
        keep_alive,
        timeout_seconds,
        model_override,
        ollama_host,
        system_prompt,
        image=None,
        delimiter=DEFAULT_PASSTHROUGH_DELIMITER,
        unique_id=None,
    ):
        task_preset = _normalize_task_preset(task_preset)
        settings = _resolve_effective_generation_settings(
            task_preset,
            sampling_mode,
            num_predict,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            presence_penalty,
            seed,
            thinking,
            strip_thinking,
            keep_alive,
            timeout_seconds,
        )
        selected_model = resolve_model(model, model_override)
        host = normalize_host(ollama_host)
        final_prompt = _resolve_prompt(task_preset, prompt)
        resolved_system_prompt = _resolve_system_prompt(task_preset, system_prompt)
        progress = _OllamaProgressReporter(unique_id, settings["num_predict"])
        progress.phase("Ollama: preparing request", value=0)

        try:
            images = []
            if image is not None:
                progress.phase(f"Ollama: checking vision support for {selected_model}")
                require_vision_support(selected_model, host=host, timeout_seconds=settings["timeout_seconds"])
                progress.phase("Ollama: encoding image")
                images.append(image_tensor_to_base64_png(image))
            elif task_preset == "image_to_prompt":
                raise ValueError("The image_to_prompt preset requires an IMAGE input.")

            progress.phase("Ollama: building generation payload")
            options = build_options(
                settings["sampling_mode"],
                settings["num_predict"],
                settings["temperature"],
                settings["top_k"],
                settings["top_p"],
                settings["min_p"],
                settings["repeat_penalty"],
                settings["presence_penalty"],
                settings["seed"],
            )
            payload = build_generate_payload(
                selected_model,
                final_prompt,
                resolved_system_prompt,
                images,
                settings["thinking"],
                settings["keep_alive"].strip(),
                options,
                stream=True,
            )
            progress.phase(f"Ollama: connecting to {host}")
            value = generate_text_streaming(
                host,
                payload,
                timeout_seconds=settings["timeout_seconds"],
                progress_callback=progress,
            )
        except OllamaRequestError as exc:
            progress.finish(f"Ollama: generation failed, passing through prompt\n{_compact_progress_reason(str(exc))}")
            return _passthrough_result(prompt, delimiter)

        thinking_stream = _extract_thinking_stream(value)
        generated_text = strip_think_tags(value)
        if _is_unusable_model_result(generated_text):
            progress.finish("Ollama: unusable output, passing through prompt")
            return _passthrough_result(prompt, delimiter, thinking_stream)
        progress.finish("Ollama: complete")
        return {"result": (generated_text, thinking_stream)}
