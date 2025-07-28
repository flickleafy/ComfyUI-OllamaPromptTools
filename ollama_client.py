import base64
import json
import os
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request
from io import BytesIO
from urllib.parse import urljoin

import numpy as np
from PIL import Image


def _load_local_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return

    try:
        env_lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in env_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_local_env()

DEFAULT_MODEL = (os.environ.get("OLLAMA_MODEL") or "llava:13b").strip() or "llava:13b"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_GENERATE_ATTEMPTS = 3
MIN_GENERATED_WORDS = 250
MIN_GENERATED_CHARACTERS = 1600
MAX_GENERATED_WORDS = 800
MAX_GENERATED_CHARACTERS = 6000
LOG_PREFIX = "[ComfyUI-OllamaPromptTools]"

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_STRAY_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_WORD_RE = re.compile(r"\b\w+\b")


class OllamaRequestError(RuntimeError):
    pass


def _log_ollama_failure(message):
    full_message = f"{LOG_PREFIX} {message}"
    print(full_message, flush=True)


def _log_ollama_message(message):
    full_message = f"{LOG_PREFIX} {message}"
    print(full_message, flush=True)


def _format_elapsed_time(start_time):
    return f"{time.perf_counter() - start_time:.2f}s"


def _payload_model(payload):
    if isinstance(payload, dict):
        return payload.get("model") or "<unknown model>"
    return "<unknown model>"


def _log_attempt_failure(action, attempt, attempts, payload, reason):
    retry_text = "; retrying" if attempt < attempts else ""
    _log_ollama_failure(
        f"{action} attempt {attempt}/{attempts} failed for model '{_payload_model(payload)}'{retry_text}. "
        f"Reason: {reason}"
    )


def get_default_host():
    return normalize_host(os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)


def normalize_host(host):
    host = (host or DEFAULT_HOST).strip()
    if not host:
        host = DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def build_url(host, path):
    return urljoin(f"{normalize_host(host)}/", path.lstrip("/"))


def _request_json(method, url, payload=None, timeout_seconds=30):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            message = parsed.get("error") or body
        except json.JSONDecodeError:
            message = body
        raise OllamaRequestError(f"Ollama HTTP {exc.code} for {url}: {message}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise OllamaRequestError(
            f"Could not reach Ollama at {url}. Check ollama_host and make sure the Ollama service is running."
        ) from exc

    if not body:
        return {}

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaRequestError(f"Ollama returned invalid JSON from {url}: {body[:200]}") from exc


def _request_json_stream(method, url, payload=None, timeout_seconds=30):
    data = None
    headers = {"Accept": "application/x-ndjson, application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        response = urllib.request.urlopen(request, timeout=float(timeout_seconds))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            message = parsed.get("error") or body
        except json.JSONDecodeError:
            message = body
        raise OllamaRequestError(f"Ollama HTTP {exc.code} for {url}: {message}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise OllamaRequestError(
            f"Could not reach Ollama at {url}. Check ollama_host and make sure the Ollama service is running."
        ) from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise OllamaRequestError(f"Ollama returned invalid streaming JSON from {url}: {line[:200]}") from exc


def list_models(host=None, timeout_seconds=2):
    response = _request_json("GET", build_url(host or get_default_host(), "/api/tags"), timeout_seconds=timeout_seconds)
    names = []
    for model in response.get("models", []):
        name = model.get("name") or model.get("model")
        if name and name not in names:
            names.append(name)
    return names


def get_model_names(host=None, timeout_seconds=2):
    start_time = time.perf_counter()
    try:
        names = list_models(host=host, timeout_seconds=timeout_seconds)
    except OllamaRequestError as exc:
        _log_ollama_failure(
            f"Could not list Ollama models from {normalize_host(host or get_default_host())}; "
            f"using fallback model '{DEFAULT_MODEL}'. Processing time: {_format_elapsed_time(start_time)}. Reason: {exc}"
        )
        return [DEFAULT_MODEL]

    if not names:
        _log_ollama_message(
            f"Listed 0 Ollama models from {normalize_host(host or get_default_host())}; "
            f"using fallback model '{DEFAULT_MODEL}'. Processing time: {_format_elapsed_time(start_time)}."
        )
        return [DEFAULT_MODEL]

    _log_ollama_message(
        f"Listed {len(names)} Ollama model(s) from {normalize_host(host or get_default_host())}. "
        f"Processing time: {_format_elapsed_time(start_time)}."
    )
    if DEFAULT_MODEL in names:
        return [DEFAULT_MODEL] + [name for name in names if name != DEFAULT_MODEL]
    return names


def resolve_model(model, model_override=None):
    override = (model_override or "").strip()
    if override:
        return override
    model = (model or "").strip()
    return model or DEFAULT_MODEL


def strip_think_tags(text):
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    text = _STRAY_THINK_CLOSE_RE.sub("", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generated_text_quality_error(text):
    cleaned = strip_think_tags(text)
    word_count = len(_WORD_RE.findall(cleaned))
    character_count = len(cleaned)
    if word_count < MIN_GENERATED_WORDS or character_count < MIN_GENERATED_CHARACTERS:
        return OllamaRequestError(
            "Ollama generated text was too short after removing thinking tags "
            f"({word_count} words, {character_count} characters; "
            f"minimum {MIN_GENERATED_WORDS} words and {MIN_GENERATED_CHARACTERS} characters)."
        )
    if word_count > MAX_GENERATED_WORDS or character_count > MAX_GENERATED_CHARACTERS:
        return OllamaRequestError(
            "Ollama generated text was too long after removing thinking tags "
            f"({word_count} words, {character_count} characters; "
            f"maximum {MAX_GENERATED_WORDS} words and {MAX_GENERATED_CHARACTERS} characters)."
        )
    return None


def image_tensor_to_base64_png(image):
    if image is None:
        return None

    if hasattr(image, "detach"):
        image = image.detach().cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()

    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {array.shape}.")

    channels = array.shape[-1]
    if channels == 1:
        array = np.repeat(array, 3, axis=-1)
    elif channels >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Expected at least 1 image channel, got {channels}.")

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_options(
    sampling_mode,
    num_predict,
    temperature,
    top_k,
    top_p,
    min_p,
    repeat_penalty,
    presence_penalty,
    seed,
):
    options = {"num_predict": int(num_predict)}
    if sampling_mode != "custom":
        return options

    options.update(
        {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repeat_penalty": float(repeat_penalty),
            "presence_penalty": float(presence_penalty),
        }
    )
    if int(seed) >= 0:
        options["seed"] = int(seed)
    return options


def build_generate_payload(model, prompt, system_prompt, images, thinking, keep_alive, options, stream=False):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "think": bool(thinking),
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if images:
        payload["images"] = images
    if keep_alive:
        payload["keep_alive"] = keep_alive
    return payload


def _emit_progress(progress_callback, event_type, **event):
    if progress_callback is not None:
        try:
            progress_callback({"type": event_type, **event})
        except Exception:
            pass


def show_model(model, host=None, timeout_seconds=30):
    payload = {"model": model}
    return _request_json(
        "POST",
        build_url(host or get_default_host(), "/api/show"),
        payload=payload,
        timeout_seconds=timeout_seconds,
    )


def require_vision_support(model, host=None, timeout_seconds=30):
    start_time = time.perf_counter()
    try:
        response = show_model(model, host=host, timeout_seconds=timeout_seconds)
    except OllamaRequestError as exc:
        _log_ollama_failure(
            f"Vision capability check failed for model '{model}'. "
            f"Processing time: {_format_elapsed_time(start_time)}. Reason: {exc}"
        )
        raise

    capabilities = response.get("capabilities")
    if isinstance(capabilities, list) and "vision" not in capabilities:
        message = f"Ollama model '{model}' does not advertise vision capability."
        _log_ollama_failure(f"{message} Processing time: {_format_elapsed_time(start_time)}.")
        raise OllamaRequestError(message)

    _log_ollama_message(
        f"Vision capability check completed for model '{model}'. "
        f"Processing time: {_format_elapsed_time(start_time)}."
    )


def generate_text(host, payload, timeout_seconds=120, max_attempts=MAX_GENERATE_ATTEMPTS):
    start_time = time.perf_counter()
    attempts = max(1, int(max_attempts))
    url = build_url(host, "/api/generate")
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = _request_json(
                "POST",
                url,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            if "error" in response:
                raise OllamaRequestError(f"Ollama generation failed: {response['error']}")
            if "response" not in response:
                raise OllamaRequestError(f"Ollama response did not include generated text: {response}")

            generated_text = response["response"]
            quality_error = generated_text_quality_error(generated_text)
            if quality_error is None:
                _log_ollama_message(
                    f"Ollama generation completed for model '{_payload_model(payload)}'. "
                    f"Processing time: {_format_elapsed_time(start_time)}."
                )
                return generated_text
            last_error = quality_error
            _log_attempt_failure("Ollama generation", attempt, attempts, payload, quality_error)
        except OllamaRequestError as exc:
            last_error = exc
            _log_attempt_failure("Ollama generation", attempt, attempts, payload, exc)

    _log_ollama_failure(
        f"Ollama generation failed after {attempts} attempts for model '{_payload_model(payload)}'. "
        f"Processing time: {_format_elapsed_time(start_time)}. Reason: {last_error}"
    )
    raise last_error


def generate_text_streaming(
    host,
    payload,
    timeout_seconds=120,
    max_attempts=MAX_GENERATE_ATTEMPTS,
    progress_callback=None,
):
    start_time = time.perf_counter()
    attempts = max(1, int(max_attempts))
    url = build_url(host, "/api/generate")
    last_error = None

    for attempt in range(1, attempts + 1):
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        generated_parts = []
        generated_chunks = 0
        generated_characters = 0

        _emit_progress(
            progress_callback,
            "attempt_start",
            attempt=attempt,
            max_attempts=attempts,
        )

        try:
            done_seen = False
            for chunk in _request_json_stream(
                "POST",
                url,
                payload=stream_payload,
                timeout_seconds=timeout_seconds,
            ):
                if "error" in chunk:
                    raise OllamaRequestError(f"Ollama generation failed: {chunk['error']}")

                text_piece = chunk.get("response")
                if text_piece:
                    generated_parts.append(text_piece)
                    generated_chunks += 1
                    generated_characters += len(text_piece)
                    _emit_progress(
                        progress_callback,
                        "token",
                        attempt=attempt,
                        max_attempts=attempts,
                        generated_chunks=generated_chunks,
                        generated_characters=generated_characters,
                        text_piece=text_piece,
                    )

                if chunk.get("done"):
                    done_seen = True
                    eval_count = chunk.get("eval_count")
                    if isinstance(eval_count, int) and eval_count > generated_chunks:
                        generated_chunks = eval_count
                    _emit_progress(
                        progress_callback,
                        "stream_done",
                        attempt=attempt,
                        max_attempts=attempts,
                        generated_chunks=generated_chunks,
                        generated_characters=generated_characters,
                    )
                    break

            if not done_seen:
                raise OllamaRequestError("Ollama streaming response ended before a done event.")

            generated_text = "".join(generated_parts)
            quality_error = generated_text_quality_error(generated_text)
            if quality_error is None:
                _emit_progress(
                    progress_callback,
                    "attempt_success",
                    attempt=attempt,
                    max_attempts=attempts,
                    generated_chunks=generated_chunks,
                    generated_characters=generated_characters,
                )
                _log_ollama_message(
                    f"Ollama streaming generation completed for model '{_payload_model(payload)}'. "
                    f"Processing time: {_format_elapsed_time(start_time)}."
                )
                return generated_text

            last_error = quality_error
            _log_attempt_failure("Ollama streaming generation", attempt, attempts, stream_payload, quality_error)
            _emit_progress(
                progress_callback,
                "attempt_retry" if attempt < attempts else "attempt_error",
                attempt=attempt,
                max_attempts=attempts,
                reason=str(quality_error),
            )
        except OllamaRequestError as exc:
            last_error = exc
            _log_attempt_failure("Ollama streaming generation", attempt, attempts, stream_payload, exc)
            _emit_progress(
                progress_callback,
                "attempt_error",
                attempt=attempt,
                max_attempts=attempts,
                reason=str(exc),
            )

    _emit_progress(
        progress_callback,
        "failed",
        attempt=attempts,
        max_attempts=attempts,
        reason=str(last_error),
    )
    _log_ollama_failure(
        f"Ollama streaming generation failed after {attempts} attempts for model '{_payload_model(payload)}'. "
        f"Processing time: {_format_elapsed_time(start_time)}. Reason: {last_error}"
    )
    raise last_error
