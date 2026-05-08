import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from tqdm import tqdm

from .io_utils import load_jsonl


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _strip_vision_tokens(text: str) -> str:
    for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        text = text.replace(token, "")
    return text.strip()


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if os.path.isabs(path) or os.path.isfile(path):
        return path
    if image_root:
        return os.path.join(image_root, path)
    return path


def _encode_image_to_data_url(path: str, max_side: int = 1024) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_side:
        scale = max_side / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"


def _extract_user_message(item: Dict[str, Any], image_root: Optional[str]) -> Dict[str, Any]:
    messages = item.get("messages", item.get("message", []))
    for msg in messages:
        if msg.get("role", "").lower() != "user":
            continue

        text = _strip_vision_tokens(msg.get("dialog", "") or "")
        image_filenames = msg.get("image_filenames", []) or []
        resolved_images = [_resolve_image_path(p, image_root) for p in image_filenames]

        return {
            "text": text,
            "image_paths": resolved_images,
        }

    return {
        "text": "",
        "image_paths": [],
    }


def _build_user_content(text: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _encode_image_to_data_url(image_path),
                },
            }
        )

    if text:
        content.append(
            {
                "type": "text",
                "text": text,
            }
        )

    return content


def _wait_for_server(api_base: str, poll_interval: int = 10, max_wait: int = 3600) -> None:
    health_url = api_base.rstrip("/") + "/health"
    elapsed = 0

    while elapsed < max_wait:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"[Info] vLLM server is ready: {health_url}")
                return
        except requests.RequestException:
            pass

        print(
            f"[Info] Waiting for vLLM server at {health_url} ({elapsed}s / {max_wait}s)...",
            file=sys.stderr,
        )
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"vLLM server at {api_base} did not become ready within {max_wait}s")


def _post_chat_completion(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = api_base.rstrip("/") + "/v1/chat/completions"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Request failed after {max_retries} retries: {exc}") from exc
            wait_time = min(2 ** attempt, 8)
            print(f"[Warn] Request failed: {exc}. Retry in {wait_time}s...", file=sys.stderr)
            time.sleep(wait_time)

    return ""


def _sample_n_responses(
    model: str,
    api_base: str,
    api_key: str,
    system_prompt: str,
    user_content: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    timeout: int,
    max_retries: int,
    dry_run: bool,
) -> List[str]:
    responses: List[str] = []

    for i in range(n):
        if dry_run:
            responses.append(f"[DRY_RUN_SAMPLE_{i}]")
            continue

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response_text = _post_chat_completion(
            api_base=api_base,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
            max_retries=max_retries,
        )
        responses.append(response_text)

    return responses


def reinfer_passk(
    input_path: str,
    output_path: str,
    model: str,
    api_base: str,
    api_key: str,
    image_root: Optional[str],
    system_prompt_path: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    timeout: int,
    max_retries: int,
    dry_run: bool,
    max_samples: Optional[int],
) -> None:
    if not dry_run:
        _wait_for_server(api_base)

    data = load_jsonl(input_path)
    if max_samples is not None:
        data = data[:max_samples]

    system_prompt = ""
    if system_prompt_path:
        system_prompt = _read_text_file(system_prompt_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Qwen2.5-VL pass@{n}", unit="sample"):
            item_id = item.get("id")
            try:
                user_msg = _extract_user_message(item, image_root=image_root)
                user_content = _build_user_content(
                    text=user_msg["text"],
                    image_paths=user_msg["image_paths"],
                )

                responses = _sample_n_responses(
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )

                new_item = dict(item)
                new_item["reinfer_model"] = model
                new_item["reinfer_n"] = n
                new_item["reinfer_temperature"] = temperature
                new_item["reinfer_top_p"] = top_p
                new_item["reinfer_max_tokens"] = max_tokens
                new_item["reinfer_system_prompt_path"] = system_prompt_path
                new_item["reinfer_responses"] = responses

            except Exception as exc:
                print(f"[Error] Failed on sample {item_id}: {exc}", file=sys.stderr)
                new_item = dict(item)
                new_item["reinfer_model"] = model
                new_item["reinfer_n"] = n
                new_item["reinfer_error"] = str(exc)
                new_item["reinfer_responses"] = []

            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[Info] Saved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-Instruct pass@k re-inference")
    parser.add_argument("--input_path", required=True, help="Input JSONL path")
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument("--model", default="qwen2p5-vl-7b", help="Served model name")
    parser.add_argument("--api_base", default="http://localhost:8000", help="vLLM OpenAI API base")
    parser.add_argument("--api_key", default="", help="API key if needed")
    parser.add_argument("--image_root", default=None, help="Root directory for relative image paths")
    parser.add_argument("--system_prompt_path", default=None, help="Optional system prompt txt path")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=8, help="Number of samples per item")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reinfer_passk(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        image_root=args.image_root,
        system_prompt_path=args.system_prompt_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        timeout=args.timeout,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()