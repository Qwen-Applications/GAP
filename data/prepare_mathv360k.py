import argparse
import os
import json
import uuid
import zipfile
import shutil
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import re
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
    return text

def process_mathv360k_data(args):
    print("Processing MathV360K data ...")
    all_processed_data = []
    
    # Define local storage paths.
    dataset_root = os.path.join(args.output_dir, "mathv360k")
    os.makedirs(dataset_root, exist_ok=True)
    
    images_dir = os.path.join(dataset_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    repo_id = "Zhiqiang007/MathV360K"
    
    json_filename = "train_samples_all_tuning.json"
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

    # Download the image archive.
    zip_filename = "data_images.zip"
    try:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
            local_dir=dataset_root,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"[Warning] Could not download {zip_filename} from HF (Images might need manual handling): {e}")
        zip_path = None

    # Extract images.
    if zip_path and os.path.exists(zip_path):
        if len(os.listdir(images_dir)) > 0:
            print(f"[Warning] {images_dir} is not empty, skipping unzip.")
        else:
            print(f"[Info] Unzipping {zip_filename}...")
            unzip_file(zip_path, images_dir)

    print(f"[Info] Start parsing metadata for MathV360K...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    subdir_check = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    image_subdir = ""
    if len(subdir_check) == 1 and len(os.listdir(images_dir)) == 1:
        image_subdir = subdir_check[0]
        print(f"[Info] Detected image subdirectory: {image_subdir}")

    for item in tqdm(raw_data):
        image_name = item.get("image", None)
        if image_name is None:
            print(f"[Warning] Missing image for item: {item}, skipping...")
            continue

        conversations = item.get("conversations", [])
        if len(conversations) != 2:
            print(f"[Warning] Unexpected conversation format for item: {item}, skipping...")
            continue

        problem_text = clean_prompt(conversations[0]["value"])
        solution_text = conversations[1]["value"]
        
        # Build the user message.
        user_content_list = []
        current_images = []

        if image_subdir:
            local_rel_path = os.path.join("mathv360k", "images", image_subdir, image_name)
        else:
            local_rel_path = os.path.join("mathv360k", "images", image_name)
        
        current_images.append(local_rel_path)
        
        user_content_list.append(problem_text)
        
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
                "data_source": "mathv360k",
                "data_type": "image2text",
                "message": message_list,
            }
        )

    # Save
    output_path = os.path.join(args.output_dir, "mathv360k_metadata.jsonl")
    save_jsonl(all_processed_data, output_path)
    print(f"[Info] Saved all {len(all_processed_data)} data to {output_path}.")
    print("[Info] Done.")

def main():
    args = get_args()
    process_mathv360k_data(args)

if __name__ == "__main__":
    main()
