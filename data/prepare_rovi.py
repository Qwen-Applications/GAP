import os
import argparse
import random
import json
import ijson
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download 
from .io_utils import *

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/root/data")
    return parser.parse_args()

def process_rovi_data(args):
    print("Processing VL data (Chang/ROVI)...")
    # dataset = load_dataset("CHang/ROVI", split="train")
    data_file = hf_hub_download(
        repo_id="CHang/ROVI", 
        filename="ROVI_train.json", 
        repo_type="dataset", 
        local_dir="/root/data/rovi",
        local_dir_use_symlinks=False
    )
    processed_data = []
    download_list = []

    with open(data_file, "rb") as f:
        parser = ijson.kvitems(f, "")
        for item in tqdm(parser):
            (_, row) = item
            image_url = row["url"]
            caption = row["vlm_description"]
            if image_url is None or caption is None:
                continue

            file_name = get_image_filename_from_url(image_url)
            local_rel_path = f"rovi/{file_name}"
            download_list.append({"url": image_url, "save_name": file_name})

            message_list = [
                {
                    "role": "user",
                    "dialog": f"{IMAGE_PLACEHOLDER}\nDescribe this image in detail.",
                    "image_filenames": [local_rel_path],
                    "need_mask": False,
                },
                {
                    "role": "assistant",
                    "dialog": caption,
                    "image_filenames": [],
                    "need_mask": True,
                }
            ]     

            processed_data.append({
                "id": str(uuid.uuid4()),
                "data_source": "rovi",
                "data_type": "image2text",
                "message": message_list,
                "original_url": image_url
            })
    
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(args.output_dir, "rovi_metadata.jsonl")
    save_jsonl(processed_data, file_path)
    with open(f"{args.output_dir}/rovi_download_list.json", "w") as f:
        json.dump(download_list, f, indent=4)
        
    print(f"[Info] Saved {len(processed_data)} rovi data.")
    print(f"[Info] Saved {len(download_list)} rovi download list.")

if __name__ == "__main__":
    args = get_args()
    process_rovi_data(args)
