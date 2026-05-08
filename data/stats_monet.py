import sys
import os
import json
import argparse
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Optional
from .io_utils import IMAGE_PLACEHOLDER

from transformers import AutoTokenizer

# Try to import tokenizer if provided
def load_tokenizer(tokenizer_path="Qwen/Qwen2.5-VL-7B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer

def extract_categories_from_filenames(filenames):
    cats = set()
    if not filenames:
        return cats
    for fn in filenames:
        if not isinstance(fn, str):
            continue
        m = re.search(r"monet/([^/\\\s]+)", fn)
        if m:
            cats.add(m.group(1))
    return cats

def extract_filenames_from_text(text):
    return re.findall(r"monet/[^ \t\r\n\"'()<>]+", text or "")

def normalize_msg_content(msg):
    # support different schemas: 'dialog' or 'content' or 'answer' etc.
    if isinstance(msg, dict):
        for key in ("dialog", "content", "answer", "text"):
            if key in msg and isinstance(msg[key], str):
                return msg[key]
    if isinstance(msg, str):
        return msg
    return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="input JSONL (monet_metadata.jsonl)")
    parser.add_argument("--tokenizer", "-t", default=None, help="optional tokenizer path (HF) for token counts")
    parser.add_argument("--max-responses", type=int, default=0, help="process at most N responses (0 = all)")
    parser.add_argument("--output_path", "-o", default="/primus_oss/output/monet_stats.json", help="optional output JSON path to save the stats")
    args = parser.parse_args()

    if args.tokenizer:
        print(f"[Info] loading tokenizer from {args.tokenizer}")

    tokenizer_path = args.tokenizer if args.tokenizer else "Qwen/Qwen2.5-VL-7B-Instruct"
    tok = load_tokenizer(tokenizer_path)

    image_counts = []
    token_counts = []
    total_responses = 0
    category_counter = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[Warning] skipping invalid JSON line {line_no}: {e}")
                continue

            # messages may be under "messages" or "message" or other structures
            messages = obj.get("messages", obj.get("message", None))
            if messages is None:
                # maybe single-turn format
                # try to treat obj as one message
                messages = []
                if "role" in obj and ("dialog" in obj or "content" in obj or "answer" in obj):
                    messages.append(obj)

            # iterate messages
            for msg in messages:
                role = ""
                if isinstance(msg, dict):
                    role = (msg.get("role", "") or "").lower()
                # only count assistant responses
                if role != "assistant":
                    continue

                content = normalize_msg_content(msg)
                total_responses += 1

                # count placeholder occurrences
                img_calls = content.count(IMAGE_PLACEHOLDER)
                # also try to include explicit filenames in image_filenames field or in text
                fname_list = msg.get("image_filenames", []) if isinstance(msg, dict) else []
                # extract filenames mentioned in text
                fname_list += extract_filenames_from_text(content)
                # categories from filenames
                cats = extract_categories_from_filenames(fname_list)
                for c in cats:
                    category_counter[c] += 1

                # if image_filenames exist, also count them (prefer filenames count if > placeholder count)
                if fname_list:
                    # count only monet/... occurrences among fname_list
                    monet_count = sum(1 for fn in fname_list if isinstance(fn, str) and "monet/" in fn)
                    # use max of detected filename count and placeholder count to be robust
                    img_calls = max(img_calls, monet_count)

                image_counts.append(img_calls)

                # tokens
                if tok:
                    try:
                        enc = tok.encode(content, add_special_tokens=False)
                        token_counts.append(len(enc))
                    except Exception:
                        token_counts.append(len(content.split()))
                else:
                    token_counts.append(len(content.split()))

                if args.max_responses and total_responses >= args.max_responses:
                    break

            if args.max_responses and total_responses >= args.max_responses:
                break

    # compute stats
    def stats_from_list(lst):
        if not lst:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0}
        return {
            "count": len(lst),
            "mean": mean(lst),
            "std": pstdev(lst) if len(lst) > 1 else 0.0,
            "min": min(lst),
            "max": max(lst),
        }

    img_stats = stats_from_list(image_counts)
    token_stats = stats_from_list(token_counts)

    # top categories
    top_cats = category_counter.most_common(50)

    out = {
        "input": os.path.abspath(args.input),
        "total_assistant_responses": total_responses,
        "image_calls_stats": img_stats,
        "token_counts_stats": token_stats,
        "top_categories": top_cats,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))

    with open(args.output_path, "w", encoding="utf-8") as outf:
        json.dump(out, outf, ensure_ascii=False, indent=2)
    print(f"[Info] saved stats to {args.output_path}")


if __name__ == "__main__":
    main()
