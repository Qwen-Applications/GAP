import os
import json
import random
import argparse
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Stage 2 Data for Training-Free GRPO")
    parser.add_argument("--output_dir", type=str, default="../data/stage2_grpo", help="Base directory for outputs")
    parser.add_argument("--image_dir", type=str, default="images", help="Sub-directory name for images")
    parser.add_argument("--output_jsonl", type=str, default="stage2_dataset.jsonl", help="Filename for the output JSONL file")
    parser.add_argument("--total_target_samples", type=int, default=10000, help="Total number of samples to keep")
    return parser.parse_args()

# Tune each dataset quota as needed. If a dataset has fewer samples, all available samples are kept.
DATASET_MAX_QUOTAS = {
    "ScienceQA": 5000,
    "DynaMath": 5000,
    "RealWorldQA": 5000,
    "LogicVista": 5000
}

def save_image_and_get_path(img: Image.Image, dataset_name: str, item_id: str, image_dir: str) -> str:
    """Save a PIL image and return its path, converting non-RGB images first."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    filename = f"{dataset_name}_{item_id}.jpg"
    filepath = os.path.join(image_dir, filename)
    img.save(filepath)
    return filepath

def process_science_qa(max_quota, image_dir):
    """
    ScienceQA includes text-only items, so keep only examples with images.
    """
    print("Loading ScienceQA...")
    dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    processed = []
    
    # Keep only image-bearing questions.
    with_image = [item for item in dataset if item.get('image') is not None]
    random.shuffle(with_image)
    
    for idx, item in enumerate(with_image):
        if idx >= max_quota:
            break
        # Format answer choices.
        choices = item.get("choices", [])
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        query = f"{item['question']}\nOptions:\n{choices_text}"

        answer_idx = item.get("answer")
        # Use the actual answer text as ground_truth.
        ground_truth = choices[answer_idx] if isinstance(answer_idx, int) and answer_idx < len(choices) else str(answer_idx)
        
        img_path = save_image_and_get_path(item['image'], "ScienceQA", str(idx), image_dir)
        
        processed.append({
            "id": f"ScienceQA_{idx}",
            "dataset_source": "ScienceQA",
            "image_path": img_path,
            "query": query,
            "ground_truth": ground_truth
        })
    return processed

def process_dynamath(max_quota, image_dir):
    print("Loading DynaMath...")
    processed = []
    
    # DynaMath splits are like sample_variant0, sample_variant1..., sample_variant10
    total_items = []
    for split_id in range(11):
        split_name = f"sample_variant{split_id}"
        try:
            dataset = load_dataset("DynaMath/DynaMath_Sample", split=split_name)
            total_items.extend([item for item in dataset if item.get('decoded_image') is not None])
        except Exception as e:
            print(f"Skipping DynaMath split {split_name} (Not found or error).")
            
    if not total_items:
        print("Warning: No items loaded for DynaMath.")
        return processed
        
    random.shuffle(total_items)
    
    for idx, item in enumerate(total_items):
        if idx >= max_quota:
            break
        query = str(item.get('question', ''))
        ground_truth = str(item.get('ground_truth', ''))
        item_id = str(item.get('id', f"DynaMath_{idx}"))
        
        # Save image using the decoded_image which is the PIL Image
        img_path = save_image_and_get_path(item['decoded_image'], "DynaMath", item_id.replace("/", "_"), image_dir)
        
        processed.append({
            "id": item_id,
            "dataset_source": "DynaMath",
            "image_path": img_path,
            "query": query,
            "ground_truth": ground_truth
        })
    return processed

def process_realworld_qa(max_quota, image_dir):
    print("Loading RealWorldQA...")
    dataset = load_dataset("visheratin/realworldqa", split="test")
    processed = []
    
    valid_items = [item for item in dataset if item.get('image') is not None]
    random.shuffle(valid_items)
    
    for idx, item in enumerate(valid_items):
        if idx >= max_quota:
            break
        query = str(item.get('question', ''))
        ground_truth = str(item.get('answer', ''))
        item_id = f"RealWorldQA_{idx}"
        
        img_path = save_image_and_get_path(item['image'], "RealWorldQA", str(idx), image_dir)
        
        processed.append({
            "id": item_id,
            "dataset_source": "RealWorldQA",
            "image_path": img_path,
            "query": query,
            "ground_truth": ground_truth
        })
    return processed

def process_logic_vista(max_quota, image_dir):
    print("Loading LogicVista...")
    dataset = load_dataset("lscpku/LogicVista", split="test")
    processed = []
    
    valid_items = [item for item in dataset if item.get('image') is not None]
    random.shuffle(valid_items)
    
    for idx, item in enumerate(valid_items):
        if idx >= max_quota:
            break
        query = str(item.get('question', ''))
        # Adjust depends on original keys, typical is 'answer' or 'ground_truth'
        ground_truth = str(item.get('answer', item.get('ground_truth', '')))
        item_id = f"LogicVista_{idx}"
        
        img_path = save_image_and_get_path(item['image'], "LogicVista", str(idx), image_dir)
        
        processed.append({
            "id": item_id,
            "dataset_source": "LogicVista",
            "image_path": img_path,
            "query": query,
            "ground_truth": ground_truth
        })
    return processed

def main():
    args = get_args()
    
    # Build absolute output paths.
    output_dir = os.path.abspath(args.output_dir)
    image_dir = os.path.join(output_dir, args.image_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, args.output_jsonl)
    
    os.makedirs(image_dir, exist_ok=True)

    all_data = []
    
    # 1. Process ScienceQA.
    try:
        data = process_science_qa(DATASET_MAX_QUOTAS["ScienceQA"], image_dir)
        print(f"✅ Picked {len(data)} from ScienceQA")
        all_data.extend(data)
    except Exception as e:
        print(f"⚠️ Skipped ScienceQA: {e}")
    
    # 2. Process DynaMath.
    try:
        data = process_dynamath(DATASET_MAX_QUOTAS["DynaMath"], image_dir)
        print(f"✅ Picked {len(data)} from DynaMath")
        all_data.extend(data)
    except Exception as e:
        print(f"⚠️ Skipped DynaMath: {e}")
        
    # 3. Process RealWorldQA.
    try:
        data = process_realworld_qa(DATASET_MAX_QUOTAS["RealWorldQA"], image_dir)
        print(f"✅ Picked {len(data)} from RealWorldQA")
        all_data.extend(data)
    except Exception as e:
        print(f"⚠️ Skipped RealWorldQA: {e}")

    # 4. Process LogicVista.
    try:
        data = process_logic_vista(DATASET_MAX_QUOTAS["LogicVista"], image_dir)
        print(f"✅ Picked {len(data)} from LogicVista")
        all_data.extend(data)
    except Exception as e:
        print(f"⚠️ Skipped LogicVista: {e}")

    # ===== Shuffle and Save =====
    print(f"\nCollected {len(all_data)} samples. Shuffling...")
    random.shuffle(all_data)
    
    # Keep the requested number of samples, or all available samples if fewer exist.
    final_data = all_data[:args.total_target_samples]
    
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    print(f"🎉 Successfully prepared {len(final_data)} multimodal GRPO samples!")
    print(f"Dataset JSONL saved to: {output_jsonl}")
    print(f"Images saved to: {image_dir}")

if __name__ == "__main__":
    main()
