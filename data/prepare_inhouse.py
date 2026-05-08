import os
import argparse
import json
import uuid
import glob
from tqdm import tqdm
import random
from .io_utils import *

RELPACE_FROM = "primus/datasets/"
RELPACE_TO = "/primus_datasets/"

GENERIC_PROMPTS = [
    # --- Chinese: general generation prompts ---
    "请生成一篇图文并茂的文章。",
    "写一篇包含图片的内容。",
    "请创作一段图文混排的内容。",
    "生成带有多张插图的详细描述。",
    "请根据你的知识，写一篇配图文章。",
    "帮我生成一份图文内容。",
    "创建一个包含图片和文字的展示。",
    "请输出一段包含视觉信息的文本。",
    "写一段内容，并在适当的位置插入图片。",
    "生成一篇图文结合的经验分享。",

    # --- Chinese: tutorial, guide, and explainer prompts ---
    "请写一篇带图片的详细教程。",
    "制作一份图文步骤指南。",
    "请一步步图解说明这个过程。",
    "写一篇科普文章，并配上相应的图片。",
    "如何做？请用图文形式解答。",
    "请解释一下这个话题，并提供辅助图片。",
    "写一个生活小技巧的图文介绍。",
    "请生成一份包含步骤图的说明书。",
    "介绍一下相关知识，最好图文并茂。",
    "写一篇带有示例图片的指南。",

    # --- Chinese: blog, story, and lifestyle prompts ---
    "写一篇图文博客。",
    "分享一段经历，并配上图片。",
    "写一个带图的故事。",
    "记录一下生活点滴，图文结合。",
    "生成一篇类似于小红书或知乎的图文笔记。",
    "写一篇游记或生活分享，包含照片。",
    "描述一个场景，并给出相应的画面。",
    "请以博主的口吻写一篇图文推送。",
    "创作一篇生动的图文日记。",
    "写一篇关于日常生活的图文分享。",

    # --- English: general/article prompts ---
    "Generate an article with images.",
    "Create an illustrated blog post.",
    "Write a text response interleaved with images.",
    "Produce a piece of content containing both text and visuals.",
    "Please generate a detailed description with images.",
    "Write a post that includes relevant pictures.",
    "Create an interleaved image-text narrative.",
    "Generate a visual guide with text descriptions.",
    "Write an article and insert images where appropriate.",
    "I need a blog post with images. Please generate one.",
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, default="/root/data/inhouse_raw")
    parser.add_argument("--output_dir", "-o", type=str, default="/root/data/inhouse_research")
    args = parser.parse_args()

    return args

def fix_image_path(path):
    if not path:
        return path
    if path.startswith(RELPACE_FROM):
        path = path.replace(RELPACE_FROM, RELPACE_TO, 1)
    
    return path

def process_inhouse_interleave(args):
    print(f"[Info] {args=}")
    print(f"Processing inhouse interleave data from {args.input_dir}")

    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))

    all_processed_data = []

    for fpath in jsonl_files:
        print(f"[Info] Reading {os.path.basename(fpath)}")

        with open(fpath, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                item = json.loads(line)
                content_str = item["content"]
                if not content_str:
                    print(f"[Warning] Empty content: {content_str}")
                    continue
                try:
                    inner_data = json.loads(content_str)
                except json.JSONDecodeError:
                    print(f"[Warning] Invalid content: {content_str}")
                    continue

                messages = inner_data["message"]
                if not messages:
                    print(f"[Warning] Empty messages: {messages}")
                    continue

                messages_list = []
                for msg in messages:
                    raw_dialog = msg["dialog"]
                    raw_images = msg["image_filenames"]
                    fixed_images = [fix_image_path(img) for img in raw_images]

                    user_prompt = random.choice(GENERIC_PROMPTS)
                    messages_list.append(
                        {
                            "role": "user",
                            "dialog": user_prompt,
                            "image_filenames": [],
                            "need_mask": False
                        }
                    )
                    messages_list.append(
                        {
                            "role": "assistant",
                            "dialog": raw_dialog,
                            "image_filenames": fixed_images,
                            "need_mask": True
                        }
                    )
                
                all_processed_data.append({
                    "id": str(uuid.uuid4()),
                    "messages": messages_list,
                    "data_source": "inhouse_interleave",
                    "data_type": "interleave"
                })

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "inhouse_interleave.jsonl")
    save_jsonl(all_processed_data, output_path)
    print(f"[Info] Saved {len(all_processed_data)} samples to {output_path}")

def main():
    args = get_args()
    process_inhouse_interleave(args)

if __name__ == "__main__":
    main()
