"""
Data filtering and sampling utility.

Keep all rows whose data_source does not match the selected value.
Randomly sample rows whose data_source matches the selected value.
Merge both groups into the output JSONL.

Usage:
    python3 -m Gap.data.filter_and_sample \
        --input_path /root/data/train.jsonl \
        --output_path /root/data/train_filtered.jsonl \
        --filter_source monet_Visual_CoT \
        --sample_size 20000 \
        --seed 42
"""

import argparse
import json
import random
import sys


def filter_and_sample(
    input_path: str,
    output_path: str,
    filter_source: str,
    sample_size: int,
    seed: int,
) -> None:
    kept_data = []
    filtered_data = []

    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("data_source") == filter_source:
                filtered_data.append(item)
            else:
                kept_data.append(item)

    print(f"[Info] data_source != '{filter_source}': {len(kept_data)} samples (all kept)")
    print(f"[Info] data_source == '{filter_source}': {len(filtered_data)} samples")

    if sample_size >= len(filtered_data):
        print(f"[Info] sample_size ({sample_size}) >= available ({len(filtered_data)}), keeping all")
        sampled = filtered_data
    else:
        random.seed(seed)
        sampled = random.sample(filtered_data, sample_size)
        print(f"[Info] Sampled {len(sampled)} from {len(filtered_data)} (seed={seed})")

    merged = kept_data + sampled
    print(f"[Info] Final output: {len(merged)} samples")

    with open(output_path, "w", encoding="utf-8") as fh:
        for item in merged:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Info] Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter by data_source and sample")
    parser.add_argument("--input_path", required=True, help="Input JSONL file")
    parser.add_argument("--output_path", required=True, help="Output JSONL file")
    parser.add_argument("--filter_source", required=True, help="data_source value to filter and sample")
    parser.add_argument("--sample_size", type=int, required=True, help="Number of samples to keep from filtered source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    filter_and_sample(
        input_path=args.input_path,
        output_path=args.output_path,
        filter_source=args.filter_source,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
