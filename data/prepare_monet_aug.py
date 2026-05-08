import argparse
import os
import json
import uuid
import zipfile
import re
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .io_utils import *

# MONET_SUBSETS = ["CogCoM", "ReFocus", "Visual_CoT", "Zebra_CoT_count", "Zebra_CoT_geometry", "Zebra_CoT_visual_search"]
MONET_SUBSETS = ["ReFocus", "Visual_CoT", "Zebra_CoT_geometry", "Zebra_CoT_visual_search"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monet_root", type=str, default="/root/data/monet")
    parser.add_argument("--output_dir", "-o", type=str, default="/root/data")
    args = parser.parse_args()
    return args

def clean_monet_text(text):
    text = text.replace("<abs_vis_token></abs_vis_token>", "")
    text = text.replace("<abs_vis_token>", "")
    text = text.replace("</abs_vis_token>", "")

    text = text.replace("<observation>", "")
    text = text.replace("</observation>", "")

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def process_monet_data(args):
    print("Processing Monet-Augment data ...")
    all_processed_data = []
    
    monet_root = args.monet_root

    for subset in MONET_SUBSETS:
        print("=" * 32)
        print(f"\nProcessing {subset}...")
        print("=" * 32)

        # Obtain subset path
        subset_dir = os.path.join(monet_root, subset)

        images_dir = os.path.join(subset_dir, "images")
        json_path = os.path.join(subset_dir, "train.json")

        print(f"[Info] Start parsing metadata for {subset}...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for item in tqdm(raw_data):
            meta = item["metadata"]
            source = meta["dataset_name"]
            conversations = item["data"]
            message_list = []
            for turn in conversations:
                role = turn["role"]
                content_list = turn["content"]

                if role == "system":
                    continue

                dialog_segements = []
                current_images = []
                current_image_types = []

                for part in content_list:
                    content_type = part["type"]
                    if content_type == "text":
                        text_content = part["text"]
                        text_content = clean_monet_text(text_content)
                        if text_content:
                            dialog_segements.append(text_content)
                    elif content_type == "image":
                        raw_image_path = part["image"]
                        if raw_image_path:
                            file_name = os.path.basename(raw_image_path)
                            local_rel_path = os.path.join(images_dir, file_name)

                            current_images.append(local_rel_path)
                            dialog_segements.append(IMAGE_PLACEHOLDER)
                            if role == "assistant":
                                if "wrong" in file_name.lower():
                                    current_image_types.append("input")
                                else:
                                    current_image_types.append("target")
                            else:
                                current_image_types.append("input")

                full_dialog = "\n".join(dialog_segements)

                if not full_dialog.strip() and not current_images:
                    continue

                message_list.append(
                    {
                        "role": role,
                        "dialog": full_dialog,
                        "image_filenames": current_images,
                        "need_mask": role == "assistant",
                        "image_types": current_image_types
                    }
                )
            
            all_processed_data.append(
                {
                    "id": str(uuid.uuid4()),
                    "data_source": f"monet_{source}",
                    "data_type": "interleave",
                    "message": message_list,
                }
            )

    # Save
    output_path = os.path.join(args.output_dir, "monet_aug_metadata.jsonl")
    save_jsonl(all_processed_data, output_path)
    print(f"[Info] Saved all {len(all_processed_data)} data to {output_path}.")
    print("[Info] Done.")

def main():
    args = get_args()
    process_monet_data(args)

if __name__ == "__main__":
    main()
