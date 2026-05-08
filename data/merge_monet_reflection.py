import argparse
import os
import random
import json
from tqdm import tqdm

from .io_utils import *

RANDOM_SUBSET = "Visual_CoT"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--random_samples", type=int, default=20000)
    parser.add_argument("--monet_jsonl_path", type=str, default=None)
    parser.add_argument("--reflection_jsonl_path", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="/root/data")
    args = parser.parse_args()
    return args

def process_monet_data(args):
    print("Merging Monet and reflection data ...")
    all_data = []

    # Load Monet data
    catgories = {}
    with open(args.monet_jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Monet data"):
            item = json.loads(line)
            catgory = item["data_source"]
            if catgory not in catgories:
                catgories[catgory] = []
            catgories[catgory].append(item)

    # Sample random subset
    random.seed(args.random_seed)
    for category in catgories.keys():
        if RANDOM_SUBSET in category:
            catgories[category] = random.sample(catgories[category], args.random_samples)
            print(f"[Info] Sampled {len(catgories[category])} random samples from {category}.")
        
        all_data.extend(catgories[category])

    # Load reflection data
    with open(args.reflection_jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reflection data"):
            item = json.loads(line)
            all_data.append(item)
    
    print(f"[Info] Total loaded data: {len(all_data)}")
    # Save
    output_path = os.path.join(args.output_dir, "monet_reflection_merged.jsonl")
    save_jsonl(all_data, output_path)
    print(f"[Info] Saved all {len(all_data)} data to {output_path}.")
    print("[Info] Done.")

def main():
    args = get_args()
    process_monet_data(args)

if __name__ == "__main__":
    main()
