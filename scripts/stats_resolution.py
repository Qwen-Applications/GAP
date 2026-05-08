#!/usr/bin/env python3
"""
Collect original resolutions for user-provided images in a GAP training dataset.
Only user-role images are counted; assistant target images are ignored.

Single-process usage:
    python -m scripts.stats_resolution \
        --model_name_or_path /path/to/model \
        --train_meta_filepath /path/to/train.jsonl \
        --image_root_dirs '{"source1": "/path/to/root1"}' \
        --latent_seq_len 36 \
        --max_seq_len 4096 \
        --num_samples 5000

Distributed usage:
    torchrun --nproc_per_node=8 -m scripts.stats_resolution \
        --model_name_or_path /path/to/model \
        --train_meta_filepath /path/to/train.jsonl \
        --image_root_dirs '{"source1": "/path/to/root1"}' \
        --latent_seq_len 36 \
        --num_samples 50000 \
        --plot --save_csv user_resolution_stats.csv
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image as PILImage

# Add the GAP project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_utils import InterleaveDataset
from transformers import AutoTokenizer, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Collect user-provided image resolution statistics from a GAP dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_meta_filepath", type=str, required=True)
    parser.add_argument("--image_root_dirs", type=str, required=True)
    parser.add_argument("--latent_seq_len", type=int, default=36)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to inspect")
    parser.add_argument("--plot", action="store_true", help="Generate a resolution scatter plot")
    parser.add_argument("--save_csv", type=str, default=None, help="Save results to CSV")
    parser.add_argument("--parser_start_token", type=str, default="<parser>")
    parser.add_argument("--parser_end_token", type=str, default="</parser>")
    parser.add_argument("--parse_prompt", type=str, default="\nPlease parse latent information from the above conversation and provide it in a structured format between <parser> and </parser> tags. The latent information should be concise and directly related to the visual content of the conversation.")
    return parser.parse_args()


def get_user_image_resolutions(dataset, num_samples, local_rank=0, world_size=1):
    """Collect original resolutions from user-role image messages only."""
    sizes = []
    failed = 0

    n = min(num_samples, len(dataset))
    indices = list(range(local_rank, n, world_size))

    for idx in indices:
        item = dataset.data[idx]
        data_source = item.get("data_source", "default")
        image_root_dir = dataset.image_root_dirs.get(data_source, "")
        messages = item.get("original_msgs", item.get("messages", item.get("message", [])))

        for msg in messages:
            # Count user-role images only.
            if msg["role"].lower() != "user":
                continue

            image_paths = msg.get("image_filenames", [])
            image_types = msg.get("image_types", None)

            for i, img_rel_path in enumerate(image_paths):
                img_type = image_types[i] if image_types and i < len(image_types) else None

                if os.path.isfile(img_rel_path):
                    img_path = img_rel_path
                else:
                    img_path = os.path.join(image_root_dir, img_rel_path)

                try:
                    with PILImage.open(img_path) as img:
                        w, h = img.size
                        sizes.append((w, h, img_type))
                except Exception:
                    failed += 1

    # Gather results across workers.
    if world_size > 1:
        from torch.distributed import gather_object
        local_data = [{"sizes": sizes, "failed": failed}]
        if local_rank == 0:
            gathered = [None for _ in range(world_size)]
            gather_object(local_data, gathered)
        else:
            gather_object(local_data, None)

        if local_rank == 0:
            all_sizes = []
            all_failed = 0
            for d in gathered:
                if d is not None:
                    all_sizes.extend(d[0]["sizes"])
                    all_failed += d[0]["failed"]
            return all_sizes, all_failed
        return None, None

    return sizes, failed


def print_stats(sizes, failed):
    if not sizes:
        print("No user-provided images found.")
        return

    widths = [w for w, h, _ in sizes]
    heights = [h for w, h, _ in sizes]
    aspect_ratios = [w / h if h > 0 else 0 for w, h, _ in sizes]

    # Group by image type.
    by_type = {}
    for w, h, t in sizes:
        key = t if t else "unspecified"
        by_type.setdefault(key, []).append((w, h))

    total = len(sizes)

    print(f"\n{'='*60}")
    print("User-provided image resolution statistics")
    print(f"{'='*60}")
    print(f"  Total images: {total}")
    print(f"  Failed files: {failed}")
    print()
    print(f"  Width   min: {min(widths):>6}  max: {max(widths):>6}  mean: {sum(widths)/total:.1f}  median: {sorted(widths)[total//2]}")
    print(f"  Height  min: {min(heights):>6}  max: {max(heights):>6}  mean: {sum(heights)/total:.1f}  median: {sorted(heights)[total//2]}")
    print(f"  Aspect  min: {min(aspect_ratios):.3f}  max: {max(aspect_ratios):.3f}  mean: {sum(aspect_ratios)/total:.3f}")

    # Print per-type statistics.
    for t, pts in sorted(by_type.items()):
        tw = [w for w, h in pts]
        th = [h for w, h in pts]
        print(f"\n  [{t}] count: {len(pts)}")
        print(f"    Width   min: {min(tw):>6}  max: {max(tw):>6}  mean: {sum(tw)/len(tw):.1f}")
        print(f"    Height  min: {min(th):>6}  max: {max(th):>6}  mean: {sum(th)/len(th):.1f}")

    # Top 15 common resolutions.
    counter = Counter((w, h) for w, h, _ in sizes)
    print("\nTop 15 common resolutions:")
    print(f"  {'Resolution':>20}  {'Count':>6}  {'Share':>7}  Distribution")
    print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*30}")
    for (w, h), count in counter.most_common(15):
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f"  {w:>6} x {h:<10}  {count:>6}  {pct:5.1f}%  {bar}")

    # Aspect-ratio buckets.
    print("\nAspect-ratio distribution:")
    buckets = [(0, 0.5, "very narrow (<0.5)"), (0.5, 0.75, "portrait (0.5-0.75)"),
               (0.75, 1.0, "near-square (0.75-1.0)"), (1.0, 1.33, "near-square (1.0-1.33)"),
               (1.33, 1.78, "landscape (1.33-1.78)"), (1.78, float('inf'), "very wide (>1.78)")]
    for lo, hi, label in buckets:
        cnt = sum(1 for ar in aspect_ratios if lo <= ar < hi)
        print(f"  {label:>25}  {cnt:>6}  ({cnt/total*100:5.1f}%)")


def plot_stats(sizes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    widths = [w for w, h, _ in sizes]
    heights = [h for w, h, _ in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot.
    ax = axes[0]
    ax.scatter(widths, heights, alpha=0.3, s=15, color='steelblue', edgecolors='none')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title(f'User Image Resolution ({len(sizes)})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Aspect-ratio histogram.
    ax = axes[1]
    aspect_ratios = [w / h if h > 0 else 0 for w, h, _ in sizes]
    ax.hist(aspect_ratios, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1:1')
    ax.axvline(x=1.78, color='green', linestyle='--', alpha=0.5, label='16:9')
    ax.axvline(x=1.33, color='orange', linestyle='--', alpha=0.5, label='4:3')
    ax.set_xlabel('Aspect Ratio (W/H)')
    ax.set_ylabel('Count')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = 'user_resolution_stats.png'
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nScatter plot saved to: {output}")


def save_csv(sizes, path):
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['width', 'height', 'type'])
        for w, h, t in sizes:
            writer.writerow([w, h, t if t else 'unspecified'])
    print(f"CSV saved to: {path}")


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group(backend="gloo")

    if local_rank == 0:
        print(f"Loading tokenizer and processor from {args.model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    image_root_dirs = json.loads(args.image_root_dirs)

    if local_rank == 0:
        print("Building dataset...")

    dataset = InterleaveDataset(
        data_paths=[args.train_meta_filepath],
        tokenizer=tokenizer,
        processor=processor,
        image_root_dirs=image_root_dirs,
        latent_seq_len=args.latent_seq_len,
        max_seq_len=args.max_seq_len,
        parser_start_token=args.parser_start_token,
        parser_end_token=args.parser_end_token,
        parse_prompt=args.parse_prompt,
        is_train=False,
    )

    if local_rank == 0:
        print(f"Dataset size: {len(dataset)}. Collecting user-provided image resolutions...")

    sizes, failed = get_user_image_resolutions(
        dataset, args.num_samples, local_rank, world_size
    )

    if local_rank == 0:
        print_stats(sizes, failed)

        if args.plot:
            plot_stats(sizes)

        if args.save_csv:
            save_csv(sizes, args.save_csv)

        print("\nDone!")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
