# Re-infer (Qwen3-VL + vLLM)

Brief guide for `data/re_infer.py`, which re-infers raw samples into higher-quality reasoning text plus auxiliary image descriptions.

## Main Features
- Read JSONL samples and extract questions and images.
- Call Qwen3-VL through an OpenAI-compatible vLLM endpoint.
- Write a `reinfer_response` field, optionally replacing the final assistant reply.
- Support few-shot files and custom prompt templates.

## Key Arguments
- `--input`: Input JSONL.
- `--output`: Output JSONL.
- `--prompt-template`: Template file supporting `$QUESTION`, `$REFERENCE_ANSWER`, and `$FEW_SHOTS`.
- `--model`: Model name served by vLLM.
- `--api-base`: OpenAI-compatible vLLM API base.
- `--image-root`: Root directory for relative image paths.
- `--few-shot`: Few-shot text or JSONL file.
- `--replace-assistant`: Replace the final assistant reply. By default, only a new field is added.
- `--dry-run`: Skip API calls for quick pipeline testing.

## Output Fields
- `reinfer_response`: Re-inferred response matching the `<text reasoning> + <aux image> + <parser>...</parser>` structure.
- `reinfer_model`: Model name used for re-inference.
- `reinfer_prompt_template`: Prompt template filename.

## Notes
- Use `--image-root` when sample image paths are relative.
- Few-shot files can be plain text or JSONL. JSONL fields such as `prompt`, `input`, `question`, `response`, `output`, and `answer` are auto-detected.
