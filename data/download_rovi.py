import os
import json
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/root/data/rovi_download_list.json")
    parser.add_argument("--output_dir", type=str, default="/root/data/images/rovi")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    return args

def download_one(item, output_dir):
    url = item["url"]
    save_path = os.path.join(output_dir, item["save_name"])
    if os.path.exists(save_path):
        return
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(resp.content)
    
    except Exception as e:
        print(f"Error {e} downloading {url}")
    
def main():
    args = get_args()
    print(f"[Info] {args=}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading list from {args.input_file}...")

    with open(args.input_file, "r", encoding="utf-8") as f:
        download_list = json.load(f)

    download_function = partial(download_one, output_dir=args.output_dir)
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(download_function, download_list), total=len(download_list)))

    print("Download finished!")

if __name__ == "__main__":
    main()
