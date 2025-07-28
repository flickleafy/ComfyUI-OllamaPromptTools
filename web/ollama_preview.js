import { app } from "../../scripts/app.js";

const PRESET_PARAMETERS = {
    custom: {
        sampling_mode: "preset_optimized",
        num_predict: 4096,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.0,
        repeat_penalty: 1.0,
        presence_penalty: 1.5,
        seed: -1,
        thinking: true,
        strip_thinking: true,
        keep_alive: "5m",
        timeout_seconds: 120,
    },
    image_to_prompt: {
        sampling_mode: "preset_optimized",
        num_predict: 4096,
        temperature: 0.25,
        top_k: 20,
        top_p: 0.9,
        min_p: 0.0,
        repeat_penalty: 1.05,
        presence_penalty: 0.2,
        seed: -1,
        thinking: true,
        strip_thinking: true,
        keep_alive: "15m",
        timeout_seconds: 300,
    },
    enhance_prompt_polish: {
        sampling_mode: "preset_optimized",
        num_predict: 4096,
        temperature: 0.35,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.0,
        repeat_penalty: 1.05,
        presence_penalty: 0.25,
        seed: -1,
        thinking: true,
        strip_thinking: true,
        keep_alive: "10m",
        timeout_seconds: 300,
    },
    enhance_prompt_creative: {
        sampling_mode: "preset_optimized",
        num_predict: 4096,
        temperature: 0.85,
        top_k: 80,
        top_p: 0.95,
        min_p: 0.03,
        repeat_penalty: 1.08,
        presence_penalty: 0.6,
        seed: -1,
        thinking: true,
        strip_thinking: true,
        keep_alive: "10m",
        timeout_seconds: 300,
    },
};

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = widgetByName(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
}

function applyPresetParameters(node, preset) {
    const profile = PRESET_PARAMETERS[preset] ?? PRESET_PARAMETERS.custom;
    for (const [name, value] of Object.entries(profile)) {
        setWidgetValue(node, name, value);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function chainWidgetCallback(widget, callback) {
    if (!widget) {
        return;
    }
    const originalCallback = widget.callback;
    widget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        callback.apply(this, arguments);
        return result;
    };
}

app.registerExtension({
    name: "ComfyUI.OllamaPromptTools",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "OllamaGenerateText") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            chainWidgetCallback(widgetByName(this, "task_preset"), (preset) => {
                applyPresetParameters(this, preset);
            });
        };
    },
});
