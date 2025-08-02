import importlib.util
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parent.parent
sys.path.insert(0, str(REPO_DIR))
MODULE_PATH = PACKAGE_DIR / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "ComfyUI_OllamaPromptTools",
    MODULE_PATH,
    submodule_search_locations=[str(PACKAGE_DIR)],
)
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)
nodes = sys.modules[f"{SPEC.name}.nodes"]

TEST_MODEL = "vision-test-model"


class OllamaNodePresetTests(unittest.TestCase):
    def test_task_presets_are_split_and_legacy_enhance_maps_to_polish(self):
        self.assertIn("enhance_prompt_polish", nodes.TASK_PRESETS)
        self.assertIn("enhance_prompt_creative", nodes.TASK_PRESETS)
        self.assertNotIn("enhance_prompt", nodes.TASK_PRESETS)
        self.assertEqual(nodes._normalize_task_preset("enhance_prompt"), "enhance_prompt_polish")

    def test_preset_optimized_overrides_stale_widget_parameters(self):
        settings = nodes._resolve_effective_generation_settings(
            "image_to_prompt",
            "preset_optimized",
            num_predict=12,
            temperature=2.0,
            top_k=999,
            top_p=1.0,
            min_p=1.0,
            repeat_penalty=5.0,
            presence_penalty=5.0,
            seed=123,
            thinking=False,
            strip_thinking=False,
            keep_alive="1s",
            timeout_seconds=1,
        )
        self.assertEqual(settings["sampling_mode"], "custom")
        self.assertEqual(settings["num_predict"], 4096)
        self.assertEqual(settings["temperature"], 0.25)
        self.assertEqual(settings["presence_penalty"], 0.2)
        self.assertTrue(settings["thinking"])
        self.assertEqual(settings["keep_alive"], "15m")
        self.assertEqual(settings["timeout_seconds"], 300)

    def test_custom_sampling_mode_preserves_manual_parameters(self):
        settings = nodes._resolve_effective_generation_settings(
            "enhance_prompt_creative",
            "custom",
            num_predict=12,
            temperature=2.0,
            top_k=999,
            top_p=1.0,
            min_p=1.0,
            repeat_penalty=5.0,
            presence_penalty=5.0,
            seed=123,
            thinking=True,
            strip_thinking=False,
            keep_alive="1s",
            timeout_seconds=1,
        )
        self.assertEqual(settings["sampling_mode"], "custom")
        self.assertEqual(settings["num_predict"], 12)
        self.assertEqual(settings["temperature"], 2.0)
        self.assertEqual(settings["presence_penalty"], 5.0)
        self.assertTrue(settings["thinking"])
        self.assertFalse(settings["strip_thinking"])
        self.assertEqual(settings["keep_alive"], "1s")
        self.assertEqual(settings["timeout_seconds"], 1)

    def test_prompt_presets_use_distinct_system_prompts(self):
        polish = nodes._resolve_system_prompt("enhance_prompt_polish", "")
        creative = nodes._resolve_system_prompt("enhance_prompt_creative", "")
        self.assertIn("grounded and controlled", polish)
        self.assertIn("creative flair", creative)
        self.assertNotEqual(polish, creative)

    def test_passthrough_prompt_uses_default_or_custom_delimiter(self):
        self.assertEqual(
            nodes._extract_passthrough_prompt("metadata Prompt: final prompt", "Prompt: "),
            "final prompt",
        )
        self.assertEqual(
            nodes._extract_passthrough_prompt("metadata ###PROMPT### final prompt", "###PROMPT###"),
            "final prompt",
        )
        self.assertEqual(
            nodes._extract_passthrough_prompt("final prompt", "Prompt: "),
            "final prompt",
        )
        self.assertEqual(
            nodes._extract_passthrough_prompt("metadata Prompt: final prompt", ""),
            "metadata Prompt: final prompt",
        )

    def test_thinking_stream_extracts_closed_and_unclosed_think_blocks(self):
        self.assertEqual(
            nodes._extract_thinking_stream("<think>first step</think>usable text<think>second step"),
            "first step\n\nsecond step",
        )

    def test_node_main_uses_preset_profile_for_generate_payload(self):
        original_generate_text_streaming = nodes.generate_text_streaming
        captured = {}

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            captured["host"] = host
            captured["payload"] = payload
            captured["timeout_seconds"] = timeout_seconds
            if progress_callback is not None:
                progress_callback({"type": "attempt_success", "attempt": 1, "max_attempts": 3, "generated_characters": 30})
            return "<think>hidden</think>visible"

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="a city at night",
                delimiter="Prompt: ",
                model=TEST_MODEL,
                task_preset="enhance_prompt_creative",
                num_predict=1,
                sampling_mode="preset_optimized",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=42,
                thinking=True,
                strip_thinking=True,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
                unique_id="123",
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertNotIn("ui", result)
        self.assertEqual(result["result"][0], "visible")
        self.assertEqual(result["result"][1], "hidden")
        self.assertEqual(captured["host"], "http://127.0.0.1:11434")
        self.assertEqual(captured["timeout_seconds"], 300)
        self.assertEqual(captured["payload"]["keep_alive"], "10m")
        self.assertEqual(captured["payload"]["options"]["num_predict"], 4096)
        self.assertEqual(captured["payload"]["options"]["temperature"], 0.85)
        self.assertEqual(captured["payload"]["options"]["presence_penalty"], 0.6)
        self.assertTrue(captured["payload"]["think"])
        self.assertTrue(captured["payload"]["stream"])
        self.assertIn("creative flair", captured["payload"]["system"])

    def test_node_main_resolves_placeholders_before_generate_payload(self):
        original_generate_text_streaming = nodes.generate_text_streaming
        original_materialize_prompt_placeholders = nodes.materialize_prompt_placeholders
        captured = {}

        def fake_materialize_prompt_placeholders(
            prompt,
            placeholder_templates,
            delimiter,
            embedding_model,
            similarity_threshold,
        ):
            captured["placeholder_prompt"] = prompt
            captured["placeholder_templates"] = placeholder_templates
            captured["placeholder_delimiter"] = delimiter
            captured["placeholder_embedding_model"] = embedding_model
            captured["placeholder_similarity_threshold"] = similarity_threshold
            return prompt.replace("[PLACEHOLDER_OBJECTIVE]", "resolved boat objective")

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            captured["payload"] = payload
            return "generated prompt"

        try:
            nodes.materialize_prompt_placeholders = fake_materialize_prompt_placeholders
            nodes.generate_text_streaming = fake_generate_text_streaming
            package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="Base [PLACEHOLDER_OBJECTIVE]\nPrompt: image has boats",
                delimiter="Prompt: ",
                placeholder_templates="[PLACEHOLDER_OBJECTIVE_DEFAULT]\nfallback objective",
                placeholder_embedding_model="test-embedding-model",
                placeholder_similarity_threshold=0.72,
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=False,
                strip_thinking=True,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming
            nodes.materialize_prompt_placeholders = original_materialize_prompt_placeholders

        self.assertEqual(captured["placeholder_delimiter"], "Prompt: ")
        self.assertEqual(captured["placeholder_embedding_model"], "test-embedding-model")
        self.assertEqual(captured["placeholder_similarity_threshold"], 0.72)
        self.assertIn("[PLACEHOLDER_OBJECTIVE_DEFAULT]", captured["placeholder_templates"])
        self.assertEqual(
            captured["payload"]["prompt"],
            "Base resolved boat objective\nPrompt: image has boats",
        )
        self.assertNotIn("[PLACEHOLDER_OBJECTIVE]", captured["payload"]["prompt"])

    def test_node_main_falls_back_to_prompt_when_model_errors(self):
        original_generate_text_streaming = nodes.generate_text_streaming

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            raise nodes.OllamaRequestError("model failed")

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="metadata Prompt: original prompt",
                delimiter="Prompt: ",
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=False,
                strip_thinking=True,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertEqual(result["result"][0], "original prompt")
        self.assertEqual(result["result"][1], "")
        self.assertNotIn("ui", result)

    def test_node_main_uses_default_delimiter_when_missing_from_old_workflow(self):
        original_generate_text_streaming = nodes.generate_text_streaming

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            raise nodes.OllamaRequestError("model failed")

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="metadata Prompt: original prompt",
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=False,
                strip_thinking=True,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertEqual(result["result"][0], "original prompt")
        self.assertEqual(result["result"][1], "")

    def test_node_main_falls_back_to_prompt_when_output_is_blank_after_cleanup(self):
        original_generate_text_streaming = nodes.generate_text_streaming

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            return "<think>only hidden reasoning</think>"

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="metadata ### fallback prompt",
                delimiter="###",
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=True,
                strip_thinking=True,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertEqual(result["result"][0], "fallback prompt")
        self.assertEqual(result["result"][1], "only hidden reasoning")

    def test_node_main_falls_back_to_prompt_when_output_is_refusal(self):
        original_generate_text_streaming = nodes.generate_text_streaming

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            return "I cannot analyze this image because no input image was provided."

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="fallback prompt",
                delimiter="Prompt: ",
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=False,
                strip_thinking=False,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertEqual(result["result"][0], "fallback prompt")
        self.assertEqual(result["result"][1], "")

    def test_node_main_keeps_generated_text_usable_when_strip_thinking_is_false(self):
        original_generate_text_streaming = nodes.generate_text_streaming

        def fake_generate_text_streaming(host, payload, timeout_seconds=120, progress_callback=None):
            return "<think>debug reasoning</think>usable prompt"

        try:
            nodes.generate_text_streaming = fake_generate_text_streaming
            result = package.NODE_CLASS_MAPPINGS["OllamaGenerateText"]().main(
                prompt="fallback prompt",
                delimiter="Prompt: ",
                model=TEST_MODEL,
                task_preset="custom",
                num_predict=1,
                sampling_mode="custom",
                temperature=0.0,
                top_k=0,
                top_p=0.0,
                min_p=0.0,
                repeat_penalty=0.0,
                presence_penalty=0.0,
                seed=-1,
                thinking=True,
                strip_thinking=False,
                keep_alive="1s",
                timeout_seconds=1,
                model_override="",
                ollama_host="127.0.0.1:11434",
                system_prompt="",
                image=None,
            )
        finally:
            nodes.generate_text_streaming = original_generate_text_streaming

        self.assertEqual(result["result"][0], "usable prompt")
        self.assertEqual(result["result"][1], "debug reasoning")
        self.assertNotIn("ui", result)


if __name__ == "__main__":
    unittest.main()
