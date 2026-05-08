import argparse
import os
import json
import uuid
import zipfile
import shutil
import re
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .io_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="/root/data")
    args = parser.parse_args()
    return args

def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def clean_prompt(text):
    if "<image>" in text:
        text = re.sub(r'\s*<image>\s*', IMAGE_PLACEHOLDER + '\n', text)
        text = re.sub(r'\s*</image>\s*', '', text)
    else:
        text = f"{IMAGE_PLACEHOLDER}\n{text}"
    return text.strip()

def process_geo170k_data(args):
    print("Processing Geo170K (qa_tuning) data ...")
    all_processed_data = []
    
    dataset_root = os.path.join(args.output_dir, "geo170k")
    os.makedirs(dataset_root, exist_ok=True)
    
    images_dir = os.path.join(dataset_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Download json metadata
    repo_id = "Luckyjhg/Geo170K"
    json_filename = "qa_tuning.json"
    try:
        json_path = hf_hub_download(
            repo_id=repo_id,
            filename=json_filename,
            repo_type="dataset",
            local_dir=dataset_root,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[Error] Could not download {json_filename} from HF: {e}")
        return

    # Download images
    zip_filename = "images.zip"
    try:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
            local_dir=dataset_root,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[Warning] Could not download {zip_filename} from HF: {e}")
        zip_path = None

    if zip_path and os.path.exists(zip_path):
        if len(os.listdir(images_dir)) > 0:
            print(f"[Warning] {images_dir} is not empty, skipping unzip.")
        else:
            print(f"[Info] Unzipping {zip_filename}...")
            unzip_file(zip_path, images_dir)

    print(f"[Info] Start parsing metadata from {json_filename}...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"[Debug] Found image subdirectories: {subdirs}")
    if "images" in subdirs:
        image_subdir = "images"

    elif len(subdirs) == 1:
        image_subdir = subdirs[0]
        
    if image_subdir:
        print(f"[Info] Detected nested image subdirectory: '{image_subdir}'")

    for item in tqdm(raw_data):
        image_name = item.get("image", None)
        if image_name is None:
            continue

        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        user_turn = conversations[0]
        assistant_turn = conversations[1]

        if user_turn['from'] != 'human' or assistant_turn['from'] != 'gpt':
            continue

        problem_text = clean_prompt(user_turn["value"])
        solution_text = assistant_turn["value"]
        
        current_images = []
        
        if image_subdir:
            local_rel_path = os.path.join("geo170k", "images", image_subdir, image_name)
        else:
            local_rel_path = os.path.join("geo170k", "images", image_name)
            
        current_images.append(local_rel_path)
        
        message_list = [
            {
                "role": "user",
                "dialog": problem_text,
                "image_filenames": current_images,
                "need_mask": False,
            },
            {
                "role": "assistant",
                "dialog": solution_text,
                "image_filenames": [],
                "need_mask": True,
            }
        ]

        all_processed_data.append(
            {
                "id": str(uuid.uuid4()),
                "data_source": "geo170k_qa_tuning",
                "data_type": "geometry",
                "message": message_list,
            }
        )

    # Save
    output_path = os.path.join(args.output_dir, "geo170k_metadata.jsonl")
    save_jsonl(all_processed_data, output_path)
    print(f"[Info] Saved all {len(all_processed_data)} data to {output_path}.")
    print("[Info] Done.")

def main():
    args = get_args()
    process_geo170k_data(args)

if __name__ == "__main__":
    main()