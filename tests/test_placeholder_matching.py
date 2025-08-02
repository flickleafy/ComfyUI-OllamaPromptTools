import importlib.util
import contextlib
import io
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_DIR / "placeholder_matching.py"
SPEC = importlib.util.spec_from_file_location("ollama_prompt_tools_placeholder_matching", MODULE_PATH)
placeholder_matching = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = placeholder_matching
SPEC.loader.exec_module(placeholder_matching)


def keyword_embedder(texts):
    vectors = []
    for text in texts:
        normalized = text.casefold()
        if (
            "boat" in normalized
            or "ship" in normalized
            or "harbor" in normalized
            or "rigging" in normalized
            or "maritime" in normalized
        ):
            vectors.append([1.0, 0.0])
        elif "portrait" in normalized or "person" in normalized:
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.0, 0.0])
    return vectors


def zero_embedder(texts):
    return [[0.0, 0.0] for _text in texts]


class PlaceholderTemplateParserTests(unittest.TestCase):
    def test_parser_collects_candidates_defaults_and_repeated_labels(self):
        catalog = placeholder_matching.parse_placeholder_templates(
            """
[PLACEHOLDER_OBJECTIVE]
boat-specific objective
---
[placeholder_objective]
ignored lowercase block

[PLACEHOLDER_OBJECTIVE_DEFAULT]
generic objective fallback

[PLACEHOLDER_CORE_RULES]
first rules block

[PLACEHOLDER_CORE_RULES]
second rules block
"""
        )

        self.assertEqual(catalog.candidates["PLACEHOLDER_OBJECTIVE"], ["boat-specific objective"])
        self.assertEqual(catalog.defaults["PLACEHOLDER_OBJECTIVE"], "generic objective fallback")
        self.assertEqual(catalog.candidates["PLACEHOLDER_CORE_RULES"], ["first rules block", "second rules block"])
        self.assertNotIn("placeholder_objective", catalog.candidates)

    def test_parser_ignores_empty_and_malformed_blocks(self):
        catalog = placeholder_matching.parse_placeholder_templates(
            """
[PLACEHOLDER_EMPTY]
---
[INVALID-LABEL]
ignored
[VALID_LABEL]
kept
"""
        )

        self.assertNotIn("PLACEHOLDER_EMPTY", catalog.candidates)
        self.assertEqual(catalog.candidates["VALID_LABEL"], ["kept"])

    def test_parser_extracts_private_matching_metadata_from_candidate_blocks(self):
        catalog = placeholder_matching.parse_placeholder_templates(
            """
[PLACEHOLDER_CORE_RULES]
@match: wooden boats, ships, harbor rigging, maritime scene
@match: preserve historical rigging, weathered wood, dock details
@weight: boat=4, boats=4, ship=3, harbor=2
@require_any: boat, boats, ship
1. Keep the original maritime scene content intact.
2. Preserve historical rigging details.
"""
        )

        candidate = catalog.candidates["PLACEHOLDER_CORE_RULES"][0]
        self.assertEqual(
            candidate.text,
            "1. Keep the original maritime scene content intact.\n2. Preserve historical rigging details.",
        )
        self.assertEqual(
            candidate.match_texts,
            [
                "wooden boats, ships, harbor rigging, maritime scene",
                "preserve historical rigging, weathered wood, dock details",
            ],
        )
        self.assertEqual(candidate.weights["boat"], 4.0)
        self.assertEqual(candidate.weights["harbor"], 2.0)
        self.assertEqual(candidate.require_any, ["boat", "boats", "ship"])


class PlaceholderResolverTests(unittest.TestCase):
    def test_semantic_match_replaces_repeated_base_markers(self):
        prompt = "Primary:\n[PLACEHOLDER_OBJECTIVE]\nAgain:\n[PLACEHOLDER_OBJECTIVE]"
        templates = """
[PLACEHOLDER_OBJECTIVE]
Enhance wooden boats and ship rigging with historical detail.

[PLACEHOLDER_OBJECTIVE]
Describe studio portrait lighting.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
Preserve the visible subject faithfully.
"""

        result = placeholder_matching.resolve_placeholders(
            prompt,
            "The image shows boats in a harbor.",
            templates,
            similarity_threshold=0.5,
            embedder=keyword_embedder,
        )

        self.assertEqual(result.count("Enhance wooden boats and ship rigging with historical detail."), 2)
        self.assertNotIn("[PLACEHOLDER_OBJECTIVE]", result)

    def test_match_metadata_drives_matching_without_leaking_into_prompt(self):
        result = placeholder_matching.resolve_placeholders(
            "Primary:\n[PLACEHOLDER_OBJECTIVE]",
            "A cinematic maritime scene with wooden boats in a harbor.",
            """
[PLACEHOLDER_OBJECTIVE]
@match: wooden boats, ships, harbor rigging, maritime scene
@match: preserve historical rigging, weathered wood, dock details
@weight: boat=4, boats=4, ship=3, harbor=2
@require_any: boat, boats, ship
1. Keep the original maritime scene content intact.
2. Preserve historical rigging and dock details.

[PLACEHOLDER_OBJECTIVE]
@match: car, vehicle, street racing scene
@weight: car=4, vehicle=3
@require_any: car, vehicle
Use vehicle-specific handling.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
@match: generic fallback metadata that must not be injected
Preserve the visible subject faithfully.
""",
            similarity_threshold=0.5,
            embedder=zero_embedder,
        )

        self.assertIn("Preserve historical rigging and dock details.", result)
        self.assertNotIn("@match:", result)
        self.assertNotIn("@weight:", result)
        self.assertNotIn("@require_any:", result)

    def test_match_metadata_is_embedded_instead_of_visible_body(self):
        result = placeholder_matching.resolve_placeholders(
            "Primary:\n[PLACEHOLDER_OBJECTIVE]",
            "The image shows boats in a harbor.",
            """
[PLACEHOLDER_OBJECTIVE]
@match: wooden boats, ships, harbor rigging
Use the selected specialized rule block.

[PLACEHOLDER_OBJECTIVE]
@match: portrait, person, studio lighting
Use portrait-specific handling.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
Preserve the visible subject faithfully.
""",
            similarity_threshold=0.5,
            embedder=keyword_embedder,
        )

        self.assertEqual(result, "Primary:\nUse the selected specialized rule block.")

    def test_semantic_match_logs_selected_block_text(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            placeholder_matching.resolve_placeholders(
                "Primary:\n[PLACEHOLDER_OBJECTIVE]",
                "The image shows boats in a harbor.",
                """
[PLACEHOLDER_OBJECTIVE]
Enhance wooden boats and ship rigging with historical detail.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
Preserve the visible subject faithfully.
""",
                similarity_threshold=0.5,
                embedder=keyword_embedder,
            )

        output = stdout.getvalue()
        self.assertIn("[ComfyUI-OllamaPromptTools]", output)
        self.assertIn("Placeholder [PLACEHOLDER_OBJECTIVE] selected block", output)
        self.assertIn("source=semantic_match", output)
        self.assertIn("Enhance wooden boats and ship rigging with historical detail.", output)

    def test_unmatched_candidate_uses_default_block(self):
        result = placeholder_matching.resolve_placeholders(
            "Primary:\n[PLACEHOLDER_OBJECTIVE]",
            "The image shows a portrait of a person.",
            """
[PLACEHOLDER_OBJECTIVE]
Enhance wooden boats and ship rigging with historical detail.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
@match: private fallback metadata
Preserve the visible subject faithfully.
""",
            similarity_threshold=0.5,
            embedder=keyword_embedder,
        )

        self.assertEqual(result, "Primary:\nPreserve the visible subject faithfully.")
        self.assertNotIn("@match:", result)

    def test_require_any_gate_blocks_semantic_match_without_required_terms(self):
        result = placeholder_matching.resolve_placeholders(
            "Primary:\n[PLACEHOLDER_OBJECTIVE]",
            "The image shows harbor rigging in a maritime scene.",
            """
[PLACEHOLDER_OBJECTIVE]
@match: wooden boats, ships, harbor rigging, maritime scene
@require_any: boat, boats, ship
Use maritime-subject-specific handling.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
Preserve the visible subject faithfully.
""",
            similarity_threshold=0.5,
            embedder=keyword_embedder,
        )

        self.assertEqual(result, "Primary:\nPreserve the visible subject faithfully.")

    def test_weight_metadata_can_select_when_embeddings_are_neutral(self):
        result = placeholder_matching.resolve_placeholders(
            "Primary:\n[PLACEHOLDER_OBJECTIVE]",
            "Wooden boats near a harbor dock.",
            """
[PLACEHOLDER_OBJECTIVE]
@match: wooden boats, ships, harbor rigging, maritime scene
@weight: boat=4, boats=4, ship=3, harbor=2
@require_any: boat, boats, ship
Use maritime-subject-specific handling.

[PLACEHOLDER_OBJECTIVE]
@weight: car=4, vehicle=3
@require_any: car, vehicle
Use vehicle-specific handling.

[PLACEHOLDER_OBJECTIVE_DEFAULT]
Preserve the visible subject faithfully.
""",
            similarity_threshold=0.5,
            embedder=zero_embedder,
        )

        self.assertEqual(result, "Primary:\nUse maritime-subject-specific handling.")

    def test_fallback_match_text_removes_shared_boilerplate_and_low_priority_lines(self):
        candidates = [
            placeholder_matching.PlaceholderCandidate(
                "1. Keep the original scene content intact.\n"
                "2. Emphasize jewelry and crown details.\n"
                "3. Avoid mirror effects."
            ),
            placeholder_matching.PlaceholderCandidate(
                "1. Keep the original scene content intact.\n"
                "2. Emphasize wooden boat rigging.\n"
                "3. Avoid mirror effects."
            ),
        ]

        match_texts = placeholder_matching._build_candidate_match_texts(
            candidates,
            "1. Keep the original scene content intact.",
        )

        self.assertEqual(match_texts[0], "Emphasize jewelry and crown details.")
        self.assertEqual(match_texts[1], "Emphasize wooden boat rigging.")

    def test_unmatched_candidate_without_default_raises(self):
        with self.assertRaises(placeholder_matching.PlaceholderResolutionError) as ctx:
            placeholder_matching.resolve_placeholders(
                "Primary:\n[PLACEHOLDER_OBJECTIVE]",
                "The image shows a portrait of a person.",
                """
[PLACEHOLDER_OBJECTIVE]
Enhance wooden boats and ship rigging with historical detail.
""",
                similarity_threshold=0.5,
                embedder=keyword_embedder,
            )

        self.assertIn("[PLACEHOLDER_OBJECTIVE]", str(ctx.exception))
        self.assertIn("[PLACEHOLDER_OBJECTIVE_DEFAULT]", str(ctx.exception))

    def test_materialize_only_replaces_base_side_when_delimiter_is_present(self):
        result = placeholder_matching.materialize_prompt_placeholders(
            "Base [PLACEHOLDER_OBJECTIVE]\nPrompt: image has boats and [PLACEHOLDER_OBJECTIVE]",
            """
[PLACEHOLDER_OBJECTIVE]
Enhance wooden boats and ship rigging with historical detail.
""",
            "Prompt: ",
            similarity_threshold=0.5,
            embedder=keyword_embedder,
        )

        self.assertEqual(
            result,
            "Base Enhance wooden boats and ship rigging with historical detail.\n"
            "Prompt: image has boats and [PLACEHOLDER_OBJECTIVE]",
        )

    def test_empty_templates_leave_prompt_unchanged(self):
        prompt = "Base [PLACEHOLDER_OBJECTIVE]\nPrompt: image has boats"
        self.assertEqual(
            placeholder_matching.materialize_prompt_placeholders(
                prompt,
                "",
                "Prompt: ",
                embedder=keyword_embedder,
            ),
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
