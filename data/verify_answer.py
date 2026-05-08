import json
import os
import re
from openai import OpenAI
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "/root/model/Qwen3-235B-A22B-Instruct"

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_API_BASE,
)

def extract_ground_truth(messages):
    """Extract ground truth from the original messages."""
    for msg in messages:
        if msg["role"] == "assistant":
            # Extract the answer inside \boxed{}.
            match = re.search(r'\\boxed\{([^}]*)\}', msg["dialog"])
            if match:
                return match.group(1).strip()
    return None

def qwen_vllm_judge(ground_truth, model_response, model_name):
    """Use a local Qwen model served by vLLM as the judge."""
    
    prompt = f"""You are an intelligent evaluator for visual question answering.

Target Answer (Ground Truth): {ground_truth}

Model Output:
{model_response}

Does the final conclusion in the Model Output conceptually match the Target Answer? 
Consider it a match ("True") if:
1. It is exactly the same mathematically or logically.
2. It is a valid synonym or expresses the same semantic meaning.
3. The model output is a more general but correct category of the ground truth (e.g., Target: "a sun hat", Output: "a hat").
4. It correctly answers the core question despite extra conversational text.

Respond exactly with ONLY "True" if it matches, or "False" if it does not. Do not include any other words or punctuation.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().lower()
        # Treat any response containing "true" as correct.
        return 'true' in result
    except Exception as e:
        print(f"vLLM API call failed: {e}")
        return False

def evaluate_record(record, model_name):
    # Avoid passing a raw string into extract_ground_truth.
    ground_truth = extract_ground_truth(record["message"])
    if not ground_truth:
        ground_truth = record["message"][-1]['dialog']

    responses = record.get("reinfer_responses", [])
    if len(responses) == 0:
        return record
    
    correct_count = 0
    record["reinfer_correct"] = []
    
    for resp in responses:
        is_correct = qwen_vllm_judge(ground_truth, resp, model_name)
        record["reinfer_correct"].append(is_correct)
        if is_correct:
            correct_count += 1
        
    record["accuracy"] = correct_count / len(responses)
    return record

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="vLLM model name or path")
    parser.add_argument("--input_file", type=str, default="/root/data/qwen2p5_reinfer.jsonl")
    parser.add_argument("--output_file", type=str, default="/root/data/qwen2p5_reinfer_w_acc.jsonl")
    # Sharding arguments for distributed evaluation.
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards, default 1")
    parser.add_argument("--shard_rank", type=int, default=0, help="Current shard rank, default 0")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print(f"[Info] {args=}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"[Info] Loaded {len(records)} records from {args.input_file}")

    # Split records by shard if requested.
    if args.total_shards > 1:
        chunk_size = (len(records) + args.total_shards - 1) // args.total_shards
        start_idx = args.shard_rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(records))
        records = records[start_idx:end_idx]
        print(f"[Info] Shard {args.shard_rank}/{args.total_shards}: Assigned {len(records)} records (index {start_idx} to {end_idx-1})")

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_record = json.loads(line)
                        if "id" in processed_record and "accuracy" in processed_record:
                            processed_ids.add(processed_record["id"])
                    except:
                        pass
        print(f"[Info] Found {len(processed_ids)} already processed records in {args.output_file}. Skipping them.")
    else:
        # Create an empty output file.
        open(args.output_file, 'w', encoding='utf-8').close()

    records = [r for r in records if r.get("id") not in processed_ids]
    print(f"[Info] {len(records)} records remaining to process.")
    
    if len(records) == 0:
        print("[Info] All records have been processed!")
        return

    # Use a lock because multiple worker threads write to the same file.
    file_lock = threading.Lock()

    def process_and_save(record):
        res_record = evaluate_record(record, args.model_name)
        if res_record:
            # Write after acquiring the lock.
            with file_lock:
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res_record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=32) as executor:
        # Wrap executor.map with tqdm for progress display.
        list(tqdm(executor.map(process_and_save, records), total=len(records), desc="Evaluating Records"))

    print(f"[Info] Finished Evaluation! Output saved to: {args.output_file}")
    
if __name__ == "__main__":
    main()
    
