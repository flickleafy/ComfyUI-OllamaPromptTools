from .nodes import OllamaGenerateText

NODE_CLASS_MAPPINGS = {
    "OllamaGenerateText": OllamaGenerateText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaGenerateText": "Ollama Generate Text",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
