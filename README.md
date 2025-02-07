# ComfyUI Ollama Prompt Tools

ComfyUI Ollama Prompt Tools is a custom ComfyUI node package for local prompt generation and prompt refinement through Ollama. The main node, `Ollama Generate Text`, can act as a general text generator, an image-to-prompt extractor for vision models, or a prompt enhancer with either a controlled or creative preset.

## What This Repository Contains

- A ComfyUI custom node that connects to a local Ollama server.

## Main Features

- Local Ollama text generation from inside ComfyUI.
- Task presets for image-to-prompt, polished prompt enhancement, and creative prompt enhancement.
- Optional image input with explicit vision-model validation before request submission.
- Streaming generation with ComfyUI progress text updates.
- Automatic retry when Ollama returns unusably short or long content.
- Safe fallback to the original prompt when Ollama errors, refuses, or returns unusable output.
- Separate `thinking_stream` output extracted from `<think>...</think>` content.

## Repository Layout

```text
.
├── .env.example
├── __init__.py
├── nodes.py
├── ollama_client.py
```

## ComfyUI Node

### Registered Node

- Display name: `Ollama Generate Text`
- Internal name: `OllamaGenerateText`
- Category: `text/ollama`
- Outputs:
  - `generated_text`
  - `thinking_stream`

### Task Presets

The node exposes four task presets:

- `custom`: plain text generation with manually controllable behavior.
- `image_to_prompt`: converts an input image into a dense generation-ready prompt.
- `enhance_prompt_polish`: rewrites a prompt for better clarity and grounded usefulness.
- `enhance_prompt_creative`: rewrites a prompt with more artistic flair and richer visual direction.

There is also legacy handling for older workflows that still use `enhance_prompt`; it is normalized internally to `enhance_prompt_polish`.

### Sampling Modes

- `preset_optimized`: replaces the visible sampling fields with the preset profile defined in the code.
- `model_defaults`: only sends `num_predict`, letting the Ollama model keep its own defaults for the remaining sampling parameters.
- `custom`: sends the manual values from the node widgets.

### Inputs

Required inputs:

- `prompt`
- `model`
- `task_preset`
- `num_predict`
- `sampling_mode`
- `temperature`
- `top_k`
- `top_p`
- `min_p`
- `repeat_penalty`
- `presence_penalty`
- `seed`
- `keep_alive`
- `timeout_seconds`
- `model_override`
- `ollama_host`
- `system_prompt`

Optional inputs:

- `delimiter`
- `image`

Hidden input:

- `unique_id` for ComfyUI progress reporting.

### Runtime Behavior

The node builds an Ollama `/api/generate` request and streams the response back through `ollama_client.generate_text_streaming`. During execution it sends status updates such as request preparation, image encoding, connection state, token progress, retry notices, and completion state to ComfyUI.

If `image` is connected, the node first validates that the chosen model advertises `vision` capability through Ollama `/api/show`. If the `image_to_prompt` preset is selected without an image input, the node raises a validation error before contacting Ollama.

If the Ollama request fails, if the model produces an empty or refusal-style answer, or if the output is otherwise judged unusable, the node does not hard-fail the workflow. Instead, it falls back to the original prompt and returns that prompt as `generated_text`.

### Fallback and Delimiter Semantics

The passthrough fallback supports prompt strings that contain metadata before the actual prompt. By default the node looks for the delimiter `Prompt:` followed by a space and, on fallback, returns only the text after that delimiter.

Examples:

- `metadata Prompt: final prompt` becomes `final prompt` on fallback.
- If a custom delimiter such as `###PROMPT###` is supplied, the fallback uses that delimiter instead.
- If the delimiter is empty, the entire original prompt is returned.

### Thinking Stream Behavior

The node extracts content inside `<think>...</think>` blocks into the second output, `thinking_stream`. Closed and unclosed `<think>` sections are both handled.

The code cleans `<think>` tags out of `generated_text` before returning it. In practice, treat `thinking_stream` as the place where reasoning-like content is preserved and `generated_text` as the cleaned text output.

### Output Quality Guardrails

The Ollama client enforces a response-length window before accepting generated content:

- Minimum visible length after cleanup: 250 words and 1600 characters.
- Maximum visible length after cleanup: 800 words and 6000 characters.
- Maximum attempts: 3.

Outputs outside that range trigger retries. If all attempts fail or the request errors, the node falls back to the original prompt.

## Ollama Client Details

The Python client in `ollama_client.py` is responsible for:

- Host normalization through `OLLAMA_HOST` or the default `http://127.0.0.1:11434`.
- Listing models from `/api/tags`.
- Prioritizing the configured default model in the dropdown when available.
- Request/response JSON handling for standard and NDJSON streaming generation.
- Building generation payloads and sampling options.
- Extracting and cleaning `<think>` blocks.
- Logging success and failure messages with timing information.
- PNG encoding for ComfyUI image tensors before submission to vision-capable models.

The default model can be set locally through `OLLAMA_MODEL` in a repo-local `.env` file. The checked-in example configuration uses `llava:13b`.

## Installation

### Prerequisites

- A working ComfyUI installation.
- A running Ollama server.
- At least one Ollama model pulled locally.
- For image-to-prompt usage, a model that advertises vision capability.

### Python Dependencies

The runtime code imports:

- `numpy`
- `Pillow`

ComfyUI itself provides the Comfy-specific runtime imports such as `comfy`, `server`, and `comfy.comfy_types.node_typing`.

### Install Steps

1. Place this repository inside your ComfyUI `custom_nodes` directory.
2. Copy `.env.example` to `.env` and set `OLLAMA_MODEL` to the local Ollama model you want to use.
3. Install Python dependencies into the same environment ComfyUI uses.
4. Start or verify the Ollama service.
5. Pull the models you intend to use.
6. Restart ComfyUI.

Example dependency install:

```bash
pip install numpy pillow
```

Example local model configuration:

```bash
cp .env.example .env
```

Example Ollama setup:

```bash
ollama serve
ollama pull llava:13b
```

For image-to-prompt workflows, also pull a vision-capable model that Ollama reports with `vision` support.

## Usage

### Basic Text Generation

1. Add the `Ollama Generate Text` node to your graph.
2. Enter a prompt.
3. Choose a model from the dropdown or supply `model_override`.
4. Pick a task preset.
5. Use `preset_optimized` for the built-in profile or `custom` to control sampling parameters manually.
6. Read the generated result from `generated_text`.

### Image To Prompt

1. Connect an `IMAGE` output to the node.
2. Set `task_preset` to `image_to_prompt`.
3. Use a model that supports vision.
4. Leave the prompt blank to use the built-in default analysis instruction, or provide your own task wording.

### Prompt Enhancement

- Use `enhance_prompt_polish` for cleaner, more controlled rewrites.
- Use `enhance_prompt_creative` for more stylized and embellished rewrites.

## Known Behavior Notes

- If Ollama is unreachable, the model list falls back to the configured default model instead of crashing node registration.
- The node normalizes host strings so bare `127.0.0.1:11434` becomes `http://127.0.0.1:11434`.
- A repo-local `.env` can set `OLLAMA_MODEL` and other environment variables without committing local model choices.
- `model_override` takes precedence over the dropdown-selected model.
- `sampling_mode=model_defaults` only sends `num_predict` in the Ollama options payload.
- Fallback output is intentionally non-fatal so downstream ComfyUI text workflows can continue.

## License

This repository is source-available, not open source in the usual permissive or copyleft sense. The included license is `LicenseRef-flickleafy-personal-noncommercial-source-available-1.0`.

Important restrictions in the included license text include:

- personal, non-commercial use only
- no organizational use without separate permission
- no network service deployment
- no AI/ML use
- no distribution of modified versions without written permission

Read `LICENSE` in full before using or redistributing any part of this repository.
