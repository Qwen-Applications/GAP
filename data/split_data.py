import os
import argparse
import random

from .io_utils import load_jsonl, save_jsonl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, default="data/meta.jsonl")
    parser.add_argument("--train_path", type=str, default="data/meta_split.jsonl")
    parser.add_argument("--val_path", type=str, default="data/meta_split.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)

    return parser.parse_args()

def main():
    args = get_args()
    print(f"[Info] {args=}")

    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(42)

    all_data = load_jsonl(args.meta_path)
    random.shuffle(all_data)
    if args.num_samples is not None:
        all_data = all_data[:args.num_samples]
    val_size = int(len(all_data) * args.val_ratio)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    directory = os.path.dirname(args.train_path)
    os.makedirs(directory, exist_ok=True)
    directory = os.path.dirname(args.val_path)
    os.makedirs(directory, exist_ok=True)
    
    save_jsonl(train_data, args.train_path)
    save_jsonl(val_data, args.val_path)
    print(f"[Info] Saved {len(train_data)} train data and {len(val_data)} val data.")

if __name__ == "__main__":
    main()
