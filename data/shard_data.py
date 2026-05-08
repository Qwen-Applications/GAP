"""
Data sharding and merging utility.

Split: divide one JSONL file evenly into shard files.
Merge: merge shard output files into one JSONL file.

Usage:
    # Split into 8 shards.
    python3 -m Gap.data.shard_data split \
        --input_path /root/data/train.jsonl \
        --num_shards 8 \
        --output_dir /root/data/shards

    # Merge 8 shard outputs.
    python3 -m Gap.data.shard_data merge \
        --input_dir /root/data/shards \
        --output_path /root/data/train_reinfer.jsonl \
        --num_shards 8
"""

import argparse
import json
import math
import os
import sys


def split_data(input_path: str, num_shards: int, output_dir: str, rank: int = -1) -> None:
    """Split a JSONL file into shard files.

    If *rank* >= 0, only write the shard for that rank (useful in multi-node
    scenarios where each node only needs its own shard).  If *rank* < 0, write
    all shards.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data.append(line)

    total = len(data)
    shard_size = math.ceil(total / num_shards)
    print(f"[Split] Total samples: {total}, Shards: {num_shards}, ~{shard_size} per shard")

    indices = [rank] if rank >= 0 else range(num_shards)
    for shard_idx in indices:
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        shard = data[start:end]

        shard_path = os.path.join(output_dir, f"shard_{shard_idx}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as fh:
            for line in shard:
                fh.write(line if line.endswith("\n") else line + "\n")

        print(f"  Shard {shard_idx}: [{start}:{end}] = {len(shard)} samples -> {shard_path}")

    print(f"[Split] Done. Output dir: {output_dir}")


def merge_data(input_dir: str, output_path: str, num_shards: int, max_wait: int = 0) -> None:
    """Merge shard output files back into a single JSONL file.

    If *max_wait* > 0, poll for missing shard files up to *max_wait* seconds
    before giving up (useful when other nodes may still be running).
    """
    import time

    shard_paths = [os.path.join(input_dir, f"output_{i}.jsonl") for i in range(num_shards)]

    # Wait for all shards if requested
    if max_wait > 0:
        elapsed = 0
        poll_interval = 30
        while elapsed < max_wait:
            missing = [p for p in shard_paths if not os.path.isfile(p)]
            if not missing:
                break
            if elapsed % 300 == 0:
                ready = num_shards - len(missing)
                print(
                    f"[Merge] Waiting for shards: {ready}/{num_shards} ready "
                    f"({elapsed}s / {max_wait}s)",
                    file=sys.stderr,
                )
            time.sleep(poll_interval)
            elapsed += poll_interval

    all_lines = []
    for shard_idx, shard_path in enumerate(shard_paths):
        if not os.path.isfile(shard_path):
            print(f"  [Warn] Missing shard: {shard_path}, skipping", file=sys.stderr)
            continue

        count = 0
        with open(shard_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_lines.append(line if line.endswith("\n") else line + "\n")
                    count += 1
        print(f"  Shard {shard_idx}: {count} samples")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for line in all_lines:
            fh.write(line)

    print(f"[Merge] Total: {len(all_lines)} samples -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL data split / merge utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # split
    split_parser = subparsers.add_parser("split", help="Split JSONL into shards")
    split_parser.add_argument("--input_path", required=True, help="Input JSONL file")
    split_parser.add_argument("--num_shards", type=int, required=True, help="Number of shards")
    split_parser.add_argument("--output_dir", required=True, help="Output directory for shard files")
    split_parser.add_argument("--rank", type=int, default=-1, help="Only write this rank's shard (-1 = all)")

    # merge
    merge_parser = subparsers.add_parser("merge", help="Merge shard outputs into one JSONL")
    merge_parser.add_argument("--input_dir", required=True, help="Directory containing output_*.jsonl files")
    merge_parser.add_argument("--output_path", required=True, help="Final merged output JSONL file")
    merge_parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to merge")
    merge_parser.add_argument("--max_wait", type=int, default=0, help="Max seconds to wait for missing shards (0 = no wait)")

    args = parser.parse_args()

    if args.command == "split":
        split_data(args.input_path, args.num_shards, args.output_dir, rank=args.rank)
    elif args.command == "merge":
        merge_data(args.input_dir, args.output_path, args.num_shards, max_wait=args.max_wait)


if __name__ == "__main__":
    main()
