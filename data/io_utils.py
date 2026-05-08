import os
import json
import hashlib
from PIL import Image
import uuid
import re

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

def save_image(image, prefix, image_root_dir):
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(image_root_dir, filename)

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image.save(filepath, "JPEG", quality=90)
    return os.path.join("images", filename)

def get_image_filename_from_url(url):
    hash_object = hashlib.md5(url.encode())
    extend = url.split(".")[-1].split("?")[0]
    if len(extend) > 4 or len(extend) < 2:
        extend = "jpg"
    return f"{hash_object.hexdigest()}.{extend}"

def append_metadata(meta_filepath, entry):
    with open(meta_filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def sample_jsonl(input_filepath, output_filepath, num_samples, seed=None):
    import random

    if seed is not None:
        random.seed(seed)

    data = load_jsonl(input_filepath)
    sampled_data = random.sample(data, num_samples)
    save_jsonl(sampled_data, output_filepath)
    print(f"[Info] Saved {num_samples} samples to {output_filepath}")

def repair_think_answer_tags(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    has_open_think = "<think>" in text
    has_close_think = "</think>" in text
    has_open_answer = "<answer>" in text
    has_close_answer = "</answer>" in text

    if not has_open_think and has_close_think:
        think_end = text.find("</think>")
        prefix = text[:think_end].strip()
        suffix = text[think_end:]
        if prefix:
            text = f"<think>\n{prefix}\n{suffix}"

    # recompute
    has_open_think = "<think>" in text
    has_close_think = "</think>" in text
    has_open_answer = "<answer>" in text
    has_close_answer = "</answer>" in text

    if has_open_think and not has_close_think:
        if has_open_answer:
            text = text.replace("<answer>", "</think>\n<answer>", 1)
        else:
            text = f"{text}\n</think>"

    if "<answer>" not in text:
        boxed_matches = list(re.finditer(r"\\boxed\{.*?\}", text, flags=re.DOTALL))
        if boxed_matches:
            last_boxed = boxed_matches[-1]
            boxed_text = last_boxed.group(0)
            body = (text[:last_boxed.start()] + text[last_boxed.end():]).strip()

            if "<think>" in body and "</think>" not in body:
                body = f"{body}\n</think>"

            text = f"{body}\n<answer>{boxed_text}</answer>".strip()

    if "<answer>" in text and "</answer>" not in text:
        text = f"{text}</answer>"

    return text.strip()
