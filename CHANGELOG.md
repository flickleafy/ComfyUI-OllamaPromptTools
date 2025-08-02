# Changelog

## 2025-08-01

Latest updates:

- Added semantic placeholder injection for prompt templates, including placeholder parsing, embedding-based matching, weighted lexical boosts, and default fallback blocks before Ollama generation.
- Extended the Ollama Generate Text node with placeholder template, embedding model, and similarity threshold inputs to resolve prompt placeholders during request preparation.
- Increased the maximum generated word limit to better accommodate the expanded prompt composition flow.
- Expanded README documentation for placeholder-driven prompt enhancement, dependency requirements, and matcher behavior.
- Added regression coverage for placeholder parsing and resolution along with node-level placeholder workflow behavior.

## 2025-07-27

Latest updates:

- Enabled Ollama thinking by default for every task preset and now always sends the `think` flag in generation requests.
- Restored strict vision-model validation through Ollama `/api/show` capability metadata before image workflows run.
- Updated the frontend preset sync defaults so the `thinking` widget matches the backend preset profiles.
- Expanded README coverage for the frontend extension, test suite, and test dependency notes.
- Updated regression tests for thinking-enabled payloads and the stricter vision capability behavior.

## 2025-02-06

Initial development:

- Added the Ollama Generate Text custom node for ComfyUI prompt generation and prompt refinement workflows.
- Added task presets for image-to-prompt analysis, polished prompt enhancement, and creative prompt enhancement.
- Added streaming generation, passthrough fallback behavior, and local environment-based default model configuration.
- Added frontend preset synchronization for node widget defaults.
- Added regression tests for node behavior and Ollama client request handling.
