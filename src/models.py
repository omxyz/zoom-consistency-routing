"""Model loading and VLM inference."""

import json
import os
import re
import torch
from PIL import Image

DEFAULT_KV_PATH = os.environ.get("KV_GROUND_PATH", "models/kv-ground-8b")
DEFAULT_QWEN_PATH = os.environ.get("QWEN_PATH", "models/qwen3.5-27b-awq")
DEFAULT_PHI4_PATH = os.environ.get("PHI4_PATH", "models/phi4-vision-15b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user will give you an instruction, "
    "and you MUST left click on the corresponding UI element via tool call. "
    "If you are not sure about where to click, guess a most likely one.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name": "computer_use", '
    '"description": "Use a mouse to interact with a computer.\\n'
    "* The screen's resolution is 1000x1000.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip "
    "in the center of the element. \\n"
    '* You can only use the left_click action to interact with the computer.", '
    '"parameters": {"properties": {"action": {"description": '
    '"The action to perform. The available actions are:\\n'
    '* `left_click`: Click the left mouse button with coordinate (x, y).", '
    '"enum": ["left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) '
    "and y (pixels from the top edge) coordinates to move the mouse to. "
    'Required only by `action=left_click`.", "type": "array"}, '
    '"required": ["action"], "type": "object"}}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

_models = {}


def load_model(name, path=None):
    """Load a model by name ('kv', 'qwen', or 'phi4'). Cached after first load."""
    if name in _models:
        return _models[name]

    if name == "phi4":
        return _load_phi4(path)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    if path is None:
        path = DEFAULT_KV_PATH if name == "kv" else DEFAULT_QWEN_PATH

    print(f"Loading {name} from {path}...", flush=True)
    processor = AutoProcessor.from_pretrained(
        path, min_pixels=65536, max_pixels=99_999_999,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        path, device_map=DEVICE, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    _models[name] = (model, processor)
    print(f"{name} loaded.", flush=True)
    return model, processor


def _load_phi4(path=None):
    """Load Phi-4-Reasoning-Vision-15B."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    if path is None:
        path = DEFAULT_PHI4_PATH

    print(f"Loading phi4 from {path}...", flush=True)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.eval()
    _models["phi4"] = (model, processor)
    print("phi4 loaded.", flush=True)
    return model, processor


def run_vlm(name, image, instruction):
    """Run a named model on an image+instruction.
    Returns (x, y) in 1000x1000 space, or None on parse failure."""
    if name == "phi4":
        return _run_phi4(image, instruction)

    model, processor = load_model(name)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_tensors="pt",
        return_dict=True, add_generation_prompt=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return parse_tool_call(response)


def _run_phi4(image, instruction):
    """Run Phi-4-Reasoning-Vision on an image+instruction.
    Returns (x, y) in 1000x1000 space, or None on parse failure."""
    model, processor = load_model("phi4")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{instruction}"},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return parse_tool_call(response)


def parse_tool_call(response):
    """Parse (x, y) from tool call output."""
    if not response:
        return None

    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
    if tc_match:
        try:
            data = json.loads(tc_match.group(1))
            coord = data.get("arguments", {}).get("coordinate", [])
            if len(coord) >= 2:
                return (float(coord[0]), float(coord[1]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    coord_match = re.search(
        r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response)
    if coord_match:
        return (float(coord_match.group(1)), float(coord_match.group(2)))

    return None
