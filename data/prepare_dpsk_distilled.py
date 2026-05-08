import argparse
import os
import json
import uuid
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from .io_utils import *

ALL_CONFIGS = [
    'code_1.5b_1pass', 'code_1.5b_2pass', 'code_1.5b_3pass', 'code_1.5b_4pass', 
    'code_7b_1pass', 'code_7b_2pass', 'code_7b_3pass', 'code_7b_4pass', 
    'code_r1_1pass', 'code_r1_2pass', 'code_r1_3pass', 'code_r1_4pass', 
    'if_1.5b_1pass', 'if_1.5b_2pass', 'if_1.5b_3pass', 'if_1.5b_4pass', 
    'if_7b_1pass', 'if_7b_2pass', 'if_7b_3pass', 'if_7b_4pass', 
    'if_r1_1pass', 'if_r1_2pass', 'if_r1_3pass', 'if_r1_4pass', 
    'math_1.5b_1pass', 'math_1.5b_2pass', 'math_1.5b_3pass', 'math_1.5b_4pass', 
    'math_7b_1pass', 'math_7b_2pass', 'math_7b_3pass', 'math_7b_4pass', 
    'math_r1_1pass', 'math_r1_2pass', 'math_r1_3pass', 'math_r1_4pass'
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="/root/data")
    parser.add_argument("--tokenizer_path", type=str, default="/root/model/Qwen2.5-VL-7B-Instruct") 
    parser.add_argument("--max_tokens", type=int, default=32000, help="Max token length for assistant response")
    parser.add_argument("--sample_size", type=int, default=25000, help="Number of samples to collect")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size for shuffling streaming dataset")
    args = parser.parse_args()
    return args


def process_am_deepseek_data(args):
    print(f"Processing AM-DeepSeek-Distilled-40M data ...")
    
    # 1. Prepare tokenizer.
    print(f"[Info] Loading Tokenizer from {args.tokenizer_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        return

    # 2. Create output directory.
    output_root = os.path.join(args.output_dir, "am_deepseek")
    os.makedirs(output_root, exist_ok=True)

    collected_data = []
    pbar = tqdm(total=args.sample_size, desc="Collecting Samples")
    
    # Compute the per-config sample quota with an approximately uniform distribution.
    # There are about 36 configs; 700 samples per config gives about 25k samples.
    samples_per_config = (args.sample_size // len(ALL_CONFIGS)) + 1
    
    # Shuffle configs to improve randomness.
    random.seed(42)
    random.shuffle(ALL_CONFIGS)

    # 3. Iterate over each config.
    for config_name in ALL_CONFIGS:
        if len(collected_data) >= args.sample_size:
            break

        print(f"\n[Info] Streaming config: {config_name} ...")
        
        try:
            ds_stream = load_dataset(
                "a-m-team/AM-DeepSeek-Distilled-40M", 
                config_name,  # Config name is required.
                split="train", 
                streaming=True
            )
            ds_stream = ds_stream.shuffle(buffer_size=args.buffer_size, seed=42)
        except Exception as e:
            print(f"[Warning] Failed to load config {config_name}: {e}. Skipping.")
            continue

        current_config_count = 0
        
        # 4. Iterate over streaming data.
        for item in ds_stream:
            # Stop when the per-config or global sample limit is reached.
            if current_config_count >= samples_per_config:
                break
            if len(collected_data) >= args.sample_size:
                break

            question, response_text = item["question"], item["answer"]
            
            if not question or not response_text:
                continue

            # Coarse length filter for speed.
            if len(response_text) > 200000: 
                continue

            try:
                token_ids = tokenizer.encode(response_text, add_special_tokens=False)
                token_count = len(token_ids)
            except Exception:
                continue

            if token_count <= args.max_tokens:
                message_list = []
                message_list.append(
                    {
                        "role": "user",
                        "dialog": question,
                        "image_filenames": [], 
                        "need_mask": False
                    }
                )
                message_list.append(
                    {
                        "role": "assistant",
                        "dialog": response_text,
                        "image_filenames": [], 
                        "need_mask": True
                    }
                )

                collected_data.append({
                    "id": str(uuid.uuid4()),
                    "data_source": f"am_deepseek",
                    "data_type": "text",
                    "message": message_list,
                })
                
                current_config_count += 1
                pbar.update(1)

    pbar.close()

    if len(collected_data) == 0:
        print("[Warning] No valid data found!")
        return

    output_path = os.path.join(args.output_dir, "am_deepseek_25k.jsonl")
    save_jsonl(collected_data, output_path)
    
    print(f"[Info] Saved {len(collected_data)} items to {output_path}.")
    print("[Info] Done.")

def main():
    args = get_args()
    process_am_deepseek_data(args)

if __name__ == "__main__":
    main()
