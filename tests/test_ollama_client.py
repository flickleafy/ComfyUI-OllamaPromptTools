import base64
import contextlib
import io
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_DIR / "ollama_client.py"
SPEC = importlib.util.spec_from_file_location("ollama_prompt_tools_client", MODULE_PATH)
ollama_client = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ollama_client
SPEC.loader.exec_module(ollama_client)


def long_generated_text():
    return "descriptive " * 260


def too_long_generated_text():
    return "descriptive " * 510


class OllamaClientTests(unittest.TestCase):
    def test_get_model_names_moves_default_first_and_falls_back(self):
        original_request_json = ollama_client._request_json

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            return {
                "models": [
                    {"name": "other:model"},
                    {"model": ollama_client.DEFAULT_MODEL},
                ]
            }

        try:
            ollama_client._request_json = fake_request_json
            self.assertEqual(
                ollama_client.get_model_names("http://127.0.0.1:11434"),
                [ollama_client.DEFAULT_MODEL, "other:model"],
            )

            def failing_request_json(method, url, payload=None, timeout_seconds=30):
                raise ollama_client.OllamaRequestError("offline")

            ollama_client._request_json = failing_request_json
            self.assertEqual(ollama_client.get_model_names("http://127.0.0.1:11434"), [ollama_client.DEFAULT_MODEL])
        finally:
            ollama_client._request_json = original_request_json

    def test_resolve_model_prefers_override(self):
        self.assertEqual(ollama_client.resolve_model("dropdown:model", " override:model "), "override:model")
        self.assertEqual(ollama_client.resolve_model("dropdown:model", ""), "dropdown:model")
        self.assertEqual(ollama_client.resolve_model("", ""), ollama_client.DEFAULT_MODEL)

    def test_build_options_model_defaults_only_sends_num_predict(self):
        options = ollama_client.build_options(
            "model_defaults",
            128,
            0.7,
            20,
            0.95,
            0.0,
            1.0,
            1.5,
            42,
        )
        self.assertEqual(options, {"num_predict": 128})

    def test_build_options_custom_includes_sampling_and_optional_seed(self):
        options = ollama_client.build_options(
            "custom",
            256,
            0.5,
            80,
            0.9,
            0.1,
            1.2,
            0.3,
            123,
        )
        self.assertEqual(
            options,
            {
                "num_predict": 256,
                "temperature": 0.5,
                "top_k": 80,
                "top_p": 0.9,
                "min_p": 0.1,
                "repeat_penalty": 1.2,
                "presence_penalty": 0.3,
                "seed": 123,
            },
        )

        no_seed = ollama_client.build_options("custom", 64, 0.7, 20, 0.95, 0.0, 1.0, 1.5, -1)
        self.assertNotIn("seed", no_seed)

    def test_strip_think_tags_removes_closed_unclosed_and_stray_tags(self):
        text = "A <think>hidden reasoning</think>visible</think>\nB <think>unfinished"
        self.assertEqual(ollama_client.strip_think_tags(text), "A visible\nB")

    def test_image_tensor_to_base64_png_encodes_first_batch_item(self):
        image = torch.zeros((2, 2, 2, 3), dtype=torch.float32)
        image[0, 0, 0, 0] = 1.0
        encoded = ollama_client.image_tensor_to_base64_png(image)
        raw = base64.b64decode(encoded)
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_generate_text_posts_generate_payload(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            return {"response": long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            value = ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(value, long_generated_text())
        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/api/generate"))
        self.assertEqual(calls[0][2], {"model": "m"})
        self.assertEqual(calls[0][3], 7)
        self.assertEqual(len(calls), 1)

    def test_generate_text_retries_request_errors_then_returns_success(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            if len(calls) < 3:
                raise ollama_client.OllamaRequestError("temporary failure")
            return {"response": long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            value = ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(value, long_generated_text())
        self.assertEqual(len(calls), 3)

    def test_generate_text_retries_short_text_after_removing_thinking_tags(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            if len(calls) < 3:
                return {"response": "<think>" + long_generated_text() + "</think>short visible text"}
            return {"response": long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            value = ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(value, long_generated_text())
        self.assertEqual(len(calls), 3)

    def test_generate_text_raises_after_three_short_outputs(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            return {"response": "short visible text"}

        try:
            ollama_client._request_json = fake_request_json
            with self.assertRaises(ollama_client.OllamaRequestError) as ctx:
                ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(len(calls), 3)
        self.assertIn("too short", str(ctx.exception))

    def test_generate_text_failure_prints_terminal_message(self):
        original_request_json = ollama_client._request_json

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            raise ollama_client.OllamaRequestError("temporary failure")

        try:
            ollama_client._request_json = fake_request_json
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with self.assertRaises(ollama_client.OllamaRequestError):
                    ollama_client.generate_text(
                        "http://localhost:11434",
                        {"model": "m"},
                        timeout_seconds=7,
                        max_attempts=1,
                    )
        finally:
            ollama_client._request_json = original_request_json

        output = stdout.getvalue()
        self.assertIn("[ComfyUI-OllamaPromptTools]", output)
        self.assertIn("temporary failure", output)
        self.assertIn("failed after 1 attempts", output)
        self.assertIn("Processing time:", output)

    def test_generate_text_success_prints_processing_time(self):
        original_request_json = ollama_client._request_json

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            return {"response": long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                value = ollama_client.generate_text(
                    "http://localhost:11434",
                    {"model": "m"},
                    timeout_seconds=7,
                    max_attempts=1,
                )
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(value, long_generated_text())
        output = stdout.getvalue()
        self.assertIn("Ollama generation completed", output)
        self.assertIn("Processing time:", output)

    def test_generate_text_retries_long_text_after_removing_thinking_tags(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            if len(calls) < 3:
                return {"response": "<think>hidden reasoning</think>" + too_long_generated_text()}
            return {"response": long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            value = ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(value, long_generated_text())
        self.assertEqual(len(calls), 3)

    def test_generate_text_streaming_collects_chunks_and_reports_progress(self):
        original_request_json_stream = ollama_client._request_json_stream
        calls = []
        progress_events = []

        def fake_request_json_stream(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            yield {"response": "descriptive " * 130}
            yield {"response": "descriptive " * 130}
            yield {"done": True, "eval_count": 260}

        try:
            ollama_client._request_json_stream = fake_request_json_stream
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                value = ollama_client.generate_text_streaming(
                    "http://localhost:11434",
                    {"model": "m", "stream": False},
                    timeout_seconds=7,
                    progress_callback=progress_events.append,
                )
        finally:
            ollama_client._request_json_stream = original_request_json_stream

        self.assertEqual(value, long_generated_text())
        self.assertIn("Ollama streaming generation completed", stdout.getvalue())
        self.assertIn("Processing time:", stdout.getvalue())
        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/api/generate"))
        self.assertTrue(calls[0][2]["stream"])
        self.assertEqual(calls[0][3], 7)
        self.assertEqual([event["type"] for event in progress_events], ["attempt_start", "token", "token", "stream_done", "attempt_success"])

    def test_generate_text_streaming_retries_short_output(self):
        original_request_json_stream = ollama_client._request_json_stream
        calls = []
        progress_events = []

        def fake_request_json_stream(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            if len(calls) < 3:
                yield {"response": "short visible text"}
                yield {"done": True}
                return
            yield {"response": long_generated_text()}
            yield {"done": True}

        try:
            ollama_client._request_json_stream = fake_request_json_stream
            value = ollama_client.generate_text_streaming(
                "http://localhost:11434",
                {"model": "m"},
                timeout_seconds=7,
                progress_callback=progress_events.append,
            )
        finally:
            ollama_client._request_json_stream = original_request_json_stream

        self.assertEqual(value, long_generated_text())
        self.assertEqual(len(calls), 3)
        self.assertEqual([event["type"] for event in progress_events].count("attempt_retry"), 2)
        self.assertEqual(progress_events[-1]["type"], "attempt_success")

    def test_generate_text_raises_after_three_long_outputs(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            return {"response": too_long_generated_text()}

        try:
            ollama_client._request_json = fake_request_json
            with self.assertRaises(ollama_client.OllamaRequestError) as ctx:
                ollama_client.generate_text("http://localhost:11434", {"model": "m"}, timeout_seconds=7)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(len(calls), 3)
        self.assertIn("too long", str(ctx.exception))

    def test_show_model_and_vision_validation_use_api_show(self):
        original_request_json = ollama_client._request_json
        calls = []

        def fake_request_json(method, url, payload=None, timeout_seconds=30):
            calls.append((method, url, payload, timeout_seconds))
            return {"capabilities": ["completion"]}

        try:
            ollama_client._request_json = fake_request_json
            with self.assertRaises(ollama_client.OllamaRequestError):
                ollama_client.require_vision_support("text-only", "http://localhost:11434", timeout_seconds=5)
        finally:
            ollama_client._request_json = original_request_json

        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/api/show"))
        self.assertEqual(calls[0][2], {"model": "text-only"})
        self.assertEqual(calls[0][3], 5)

    def test_build_generate_payload_omits_empty_optional_fields(self):
        payload = ollama_client.build_generate_payload(
            "model",
            "prompt",
            "",
            [],
            True,
            "",
            {"num_predict": 8},
        )
        self.assertEqual(
            payload,
            {
                "model": "model",
                "prompt": "prompt",
                "stream": False,
                "think": True,
                "options": {"num_predict": 8},
            },
        )

        payload = ollama_client.build_generate_payload(
            "model",
            "prompt",
            "system",
            ["image"],
            False,
            "10m",
            {"num_predict": 8},
        )
        self.assertEqual(payload["system"], "system")
        self.assertEqual(payload["images"], ["image"])
        self.assertEqual(payload["keep_alive"], "10m")
        self.assertFalse(payload["think"])


if __name__ == "__main__":
    unittest.main()
