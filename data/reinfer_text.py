import argparse
import base64
import json
import io
import os
import re
import sys
import time
from string import Template
from typing import Dict, List, Optional

from tqdm import tqdm

from .io_utils import load_jsonl, repair_think_answer_tags


def _read_text_file(path: str) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read().strip()


def _strip_vision_tokens(text: str) -> str:
	"""Remove VL special tokens so the text prompt stays clean."""
	for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
		text = text.replace(token, "")
	return text


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
	if os.path.isabs(path) or os.path.isfile(path):
		return path
	if image_root:
		return os.path.join(image_root, path)
	return path


def _encode_image_to_data_url(path: str, max_side: int = 1024) -> str:
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Image not found: {path}")

	from PIL import Image as PILImage

	image = PILImage.open(path).convert("RGB")
	width, height = image.size
	if width > max_side or height > max_side:
		scale = max_side / max(width, height)
		image = image.resize(
			(int(width * scale), int(height * scale)),
			PILImage.LANCZOS,
		)

	buffer = io.BytesIO()
	image.save(buffer, format="JPEG", quality=85)
	img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
	return f"data:image/jpeg;base64,{img_base64}"


def _build_user_content(text: str, image_paths: List[str], image_root: Optional[str]) -> List[Dict[str, object]]:
	content: List[Dict[str, object]] = []
	for image_path in image_paths:
		resolved_path = _resolve_image_path(image_path, image_root)
		data_url = _encode_image_to_data_url(resolved_path)
		content.append(
			{
				"type": "image_url",
				"image_url": {"url": data_url},
			}
		)

	if text.strip():
		content.append({"type": "text", "text": _strip_vision_tokens(text)})

	return content


def _load_few_shots(path: Optional[str]) -> str:
	if not path:
		return ""
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Few-shot file not found: {path}")

	if path.endswith(".jsonl"):
		lines = load_jsonl(path)
		rendered = []
		for idx, item in enumerate(lines, start=1):
			prompt = (
				item.get("prompt")
				or item.get("input")
				or item.get("question")
				or item.get("user")
				or ""
			)
			response = (
				item.get("response")
				or item.get("output")
				or item.get("answer")
				or item.get("assistant")
				or ""
			)
			if not prompt and not response:
				continue
			rendered.append(
				"\n".join(
					[
						f"Example {idx}:",
						"User:",
						prompt.strip(),
						"Assistant:",
						response.strip(),
					]
				)
			)
		return "\n\n".join(rendered)

	return _read_text_file(path)


def _extract_question(item: Dict, image_root: Optional[str]) -> Dict[str, object]:
	messages = item.get("messages", item.get("message", []))
	for msg in messages:
		if msg.get("role", "").lower() == "user":
			image_filenames = msg.get("image_filenames", [])
			resolved_images = [_resolve_image_path(path, image_root) for path in image_filenames]
			return {
				"dialog": _strip_vision_tokens(msg.get("dialog", "").strip()),
				"image_filenames": resolved_images,
			}
	return {"dialog": "", "image_filenames": []}


def _extract_reference_answer(item: Dict) -> str:
	messages = item.get("messages", item.get("message", []))
	for msg in reversed(messages):
		if msg.get("role", "").lower() == "assistant":
			return msg.get("dialog", "").strip()
	return ""


def _build_system_prompt(template_text: str, few_shots: str) -> str:
	return Template(template_text).safe_substitute(FEW_SHOTS=few_shots.strip())


def _wait_for_server(api_base: str, poll_interval: int = 10, max_wait: int = 3600) -> None:
	import requests

	health_url = api_base.rstrip("/") + "/health"
	elapsed = 0
	while elapsed < max_wait:
		try:
			response = requests.get(health_url, timeout=5)
			if response.status_code == 200:
				print(f"[Info] vLLM server is ready ({health_url})")
				return
		except requests.ConnectionError:
			pass
		print(
			f"[Info] Waiting for vLLM server at {health_url} "
			f"({elapsed}s / {max_wait}s) ...",
			file=sys.stderr,
		)
		time.sleep(poll_interval)
		elapsed += poll_interval

	raise TimeoutError(
		f"vLLM server at {api_base} did not become ready within {max_wait}s"
	)


def _post_chat_completion(
	api_base: str,
	api_key: str,
	payload: Dict,
	timeout: int,
	max_retries: int,
) -> str:
	import requests

	headers = {"Content-Type": "application/json"}
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"

	url = api_base.rstrip("/") + "/v1/chat/completions"

	for attempt in range(1, max_retries + 1):
		try:
			response = requests.post(url, headers=headers, json=payload, timeout=timeout)
			if response.status_code == 500:
				raise RuntimeError(f"500 Server Error from {url}")
			response.raise_for_status()
			data = response.json()
			return data["choices"][0]["message"]["content"].strip()
		except (requests.ConnectionError, ConnectionRefusedError) as exc:
			print(
				f"[Warn] vLLM server unreachable ({exc}), waiting for recovery...",
				file=sys.stderr,
			)
			_wait_for_server(api_base, poll_interval=15, max_wait=3600)
			continue
		except Exception as exc:  # noqa: BLE001
			if attempt == max_retries:
				raise
			wait = min(2 ** attempt, 8)
			print(f"[Warn] Request failed ({exc}), retrying in {wait}s...", file=sys.stderr)
			time.sleep(wait)
	return ""


def _load_jsonl_from_oss(filename: str):
	file_path = os.path.join(f"${{PRIMUS_INPUT_DIR}}/{filename}")
	if os.path.exists(file_path):
		os.system(f"cp {file_path} /root/data/{filename}")
		return load_jsonl(f"/root/data/{filename}")

	print(f"[Warning] File not found in OSS: {file_path}")
	return []


def _json_file_to_id_set(json_file) -> set:
	return {item.get("id") for item in json_file if item.get("id") is not None}


def _extract_text_reasonings(response: str) -> str:
	res = re.sub(r"</?text_reasoning>", "", response, flags=re.IGNORECASE)
	return res.strip()


def reinfer_text(
	input_path: str,
	output_path: str,
	prompt_template_path: str,
	model: str,
	api_base: str,
	api_key: str,
	image_root: Optional[str],
	few_shot_path: Optional[str],
	max_tokens: int,
	temperature: float,
	top_p: float,
	timeout: int,
	max_retries: int,
	dry_run: bool,
	replace_assistant: bool,
	max_samples: Optional[int],
	save_per_steps: int = 1000,
):
	if not dry_run:
		_wait_for_server(api_base)

	template_text = _read_text_file(prompt_template_path)
	few_shots = _load_few_shots(few_shot_path)
	data = load_jsonl(input_path)

	if max_samples is not None:
		data = data[:max_samples]

	total_count = 0
	failed_count = 0

	existing_jsonl = _load_jsonl_from_oss(os.path.basename(output_path))
	existing_ids = _json_file_to_id_set(existing_jsonl)

	with open(output_path, "w", encoding="utf-8") as out_fh:
		for item in tqdm(data, desc="Re-inferring text", unit="sample", total=len(data)):
			item_id = item.get("id")
			if item_id in existing_ids:
				print(f"[Info] Skipping already processed sample {item_id}")
				out_fh.write(json.dumps(item, ensure_ascii=False) + "\n")
				continue

			try:
				question_info = _extract_question(item, image_root)
				reference_answer = _extract_reference_answer(item)
				system_prompt = _build_system_prompt(template_text, few_shots)
				user_content = _build_user_content(
					question_info["dialog"],
					question_info["image_filenames"],
					image_root,
				)

				messages = [
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_content},
				]

				if dry_run:
					generated = "<think>\n[DRY_RUN]\n</think>\n<answer>[DRY_RUN]</answer>"
				else:
					payload = {
						"model": model,
						"messages": messages,
						"max_tokens": max_tokens,
						"temperature": temperature,
						"top_p": top_p,
					}
					generated = _post_chat_completion(api_base, api_key, payload, timeout, max_retries)

				generated = _extract_text_reasonings(generated)
				generated = repair_think_answer_tags(generated)

			except Exception as exc:
				failed_count += 1
				print(
					f"[Error] Failed on sample {item.get('id', 'unknown')} ({exc}), skipping.",
					file=sys.stderr,
				)
				generated = ""
				reference_answer = ""

			new_item = dict(item)
			if generated:
				new_item["reinfer_response"] = generated
				new_item["reinfer_model"] = model
				new_item["reinfer_prompt_template"] = os.path.basename(prompt_template_path)
			new_item["reference_answer"] = reference_answer

			if replace_assistant:
				messages_list = new_item.get("messages", new_item.get("message", []))
				replaced = False
				for msg in reversed(messages_list):
					if msg.get("role", "").lower() == "assistant":
						msg["dialog"] = generated
						msg["image_filenames"] = []
						msg["need_mask"] = False
						replaced = True
						break
				if not replaced:
					messages_list.append(
						{
							"role": "assistant",
							"dialog": generated,
							"image_filenames": [],
							"need_mask": False,
						}
					)

			out_fh.write(json.dumps(new_item, ensure_ascii=False) + "\n")
			out_fh.flush()
			total_count += 1

			if total_count % save_per_steps == 0:
				output_dir = os.environ.get("PRIMUS_OUTPUT_DIR")
				if output_dir:
					os.system(f"cp {output_path} {output_dir}")
				print(
					f"[Info] Intermediate save to {output_path} "
					f"({total_count} samples, {failed_count} failed)."
				)

	output_dir = os.environ.get("PRIMUS_OUTPUT_DIR")
	if output_dir:
		os.system(f"cp {output_path} {output_dir}")
	print(f"[Info] Saved to {output_path} ({total_count} samples, {failed_count} failed).")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Re-infer text-only reasoning samples via an OpenAI-compatible vLLM endpoint"
	)
	parser.add_argument("--input_path", required=True, help="Input JSONL file")
	parser.add_argument("--output_path", required=True, help="Output JSONL file")
	parser.add_argument(
		"--prompt_template",
		required=True,
		help="Prompt template file with $FEW_SHOTS placeholders",
	)
	parser.add_argument("--model", required=True, help="Model name served by vLLM")
	parser.add_argument("--api_base", default="http://localhost:8000", help="vLLM OpenAI API base")
	parser.add_argument("--api_key", default="", help="API key if required")
	parser.add_argument("--image_root", default=None, help="Optional image root dir for relative paths")
	parser.add_argument("--few_shots", default=None, help="Optional few-shot text/jsonl file")
	parser.add_argument("--max_tokens", type=int, default=2048)
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top_p", type=float, default=0.9)
	parser.add_argument("--timeout", type=int, default=120)
	parser.add_argument("--max_retries", type=int, default=3)
	parser.add_argument("--dry_run", action="store_true", help="Skip API calls and write placeholders")
	parser.add_argument(
		"--replace_assistant",
		action="store_true",
		help="Replace the last assistant message dialog with re-infer output",
	)
	parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick runs")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	reinfer_text(
		input_path=args.input_path,
		output_path=args.output_path,
		prompt_template_path=args.prompt_template,
		model=args.model,
		api_base=args.api_base,
		api_key=args.api_key,
		image_root=args.image_root,
		few_shot_path=args.few_shots,
		max_tokens=args.max_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		timeout=args.timeout,
		max_retries=args.max_retries,
		dry_run=args.dry_run,
		replace_assistant=args.replace_assistant,
		max_samples=args.max_samples,
	)


if __name__ == "__main__":
	main()
