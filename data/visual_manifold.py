import os
import json
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_world_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def is_distributed():
    _, _, world_size = get_world_info()
    return world_size > 1


def setup_distributed():
    local_rank, rank, world_size = get_world_info()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return get_world_info()[1] == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def load_model_and_processor(device):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def mean_pool(hidden_states, attention_mask=None):
    # hidden_states: [B, T, D]
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def extract_representations(model, processor, text, image_path, device):
    image = load_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    # 1) input text embeddings
    input_embeddings = model.get_input_embeddings()(input_ids)  # [B, T, D]

    # 2) last_hidden_state
    last_hidden_state = outputs.hidden_states[-1]  # [B, T, D]

    # 3) image embeddings
    # Vision token positions can vary across implementations. The
    # multimodal-input-embedding minus text-embedding approximation is not exact.
    # A more robust approach is to extract features directly from the vision tower or projector.
    image_embeddings = None

    if hasattr(model, "visual"):
        try:
            pixel_values = inputs.get("pixel_values", None)
            if pixel_values is not None:
                vision_out = model.visual(pixel_values)
                if isinstance(vision_out, torch.Tensor):
                    image_embeddings = vision_out
                elif hasattr(vision_out, "last_hidden_state"):
                    image_embeddings = vision_out.last_hidden_state
        except Exception as e:
            print(f"[Warn] failed to extract image embeddings from model.visual: {e}")

    # Pool into sentence-level or image-level vectors.
    text_vec = mean_pool(input_embeddings, attention_mask).detach().float().cpu().numpy()
    last_vec = mean_pool(last_hidden_state, attention_mask).detach().float().cpu().numpy()

    if image_embeddings is not None:
        if image_embeddings.dim() == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
        image_vec = image_embeddings.mean(dim=1).detach().float().cpu().numpy()
    else:
        image_vec = None

    return {
        "text_token_embeddings": input_embeddings.detach().float().cpu().numpy(),
        "last_hidden_state": last_hidden_state.detach().float().cpu().numpy(),
        "text_pooled": text_vec,
        "last_pooled": last_vec,
        "image_pooled": image_vec,
    }


def reduce_dim(X, method="pca", random_state=42):
    if len(X) < 2:
        raise ValueError("At least 2 vectors are required for dimensionality reduction.")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        perplexity = min(30, max(1, len(X) - 1))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("Please install umap-learn first.")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)


def plot_points(X_2d, labels, title, save_path=None):
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        idx = [i for i, x in enumerate(labels) if x == label]
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, alpha=0.8)

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize multimodal manifolds with single-node multi-GPU execution.")
    parser.add_argument("--data-file", type=str, required=True, help="JSON/JSONL data file")
    parser.add_argument("--image-root", type=str, default=None, help="Image root directory")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for plots")
    return parser.parse_args()


def load_samples(data_file):
    if data_file.endswith(".jsonl"):
        records = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_samples(samples, image_root=None, default_text="Please describe this image."):
    normalized = []
    for item in samples:
        text = item.get("text") or item.get("query") or default_text
        image_path = item.get("image") or item.get("image_path") or item.get("img")
        if image_path is None:
            continue
        if image_root is not None and not os.path.isabs(image_path):
            image_path = os.path.join(image_root, image_path)
        normalized.append({"text": text, "image": image_path})
    return normalized


def sample_dataset(samples, sample_size, seed):
    if sample_size is None or sample_size <= 0 or sample_size >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, sample_size)


def shard_samples_for_rank(samples, rank, world_size):
    if world_size <= 1:
        return samples
    return samples[rank::world_size]


def gather_results(local_payload):
    if not dist.is_initialized():
        return [local_payload]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_payload)
    return gathered


def main():
    args = parse_args()
    device, local_rank, rank, world_size = setup_distributed()

    try:
        model, processor = load_model_and_processor(device)

        raw_samples = load_samples(args.data_file)
        sampled = sample_dataset(raw_samples, args.sample_size, args.seed)
        samples = normalize_samples(sampled, image_root=args.image_root)
        local_samples = shard_samples_for_rank(samples, rank, world_size)

        rank0_print(f"[Info] total sampled records: {len(samples)}")
        rank0_print(f"[Info] world_size={world_size}, using torch device sharding by local_rank")
        print(f"[Rank {rank}] processing {len(local_samples)} samples on {device}")

        text_vecs = []
        image_vecs = []
        last_vecs = []

        for idx, sample in enumerate(local_samples):
            try:
                feats = extract_representations(
                    model=model,
                    processor=processor,
                    text=sample["text"],
                    image_path=sample["image"],
                    device=device,
                )

                text_vecs.append(feats["text_pooled"][0])
                last_vecs.append(feats["last_pooled"][0])

                if feats["image_pooled"] is not None:
                    image_vecs.append(feats["image_pooled"][0])

                if (idx + 1) % 10 == 0:
                    print(f"[Rank {rank}] processed {idx + 1}/{len(local_samples)} local samples")
            except Exception as e:
                print(f"[Warn][Rank {rank}] failed on sample {idx}: {e}")

        gathered = gather_results(
            {
                "text_vecs": text_vecs,
                "image_vecs": image_vecs,
                "last_vecs": last_vecs,
            }
        )

        if is_main_process():
            merged_text_vecs = []
            merged_image_vecs = []
            merged_last_vecs = []

            for payload in gathered:
                merged_text_vecs.extend(payload["text_vecs"])
                merged_image_vecs.extend(payload["image_vecs"])
                merged_last_vecs.extend(payload["last_vecs"])

            all_vecs = []
            labels = []

            for v in merged_text_vecs:
                all_vecs.append(v)
                labels.append("text_embedding")

            for v in merged_image_vecs:
                all_vecs.append(v)
                labels.append("image_embedding")

            for v in merged_last_vecs:
                all_vecs.append(v)
                labels.append("last_hidden_state")

            if len(all_vecs) < 2:
                raise ValueError("Not enough valid vectors to visualize.")

            os.makedirs(args.output_dir, exist_ok=True)
            all_vecs = np.stack(all_vecs, axis=0)

            for method in ["pca", "tsne"] + (["umap"] if HAS_UMAP else []):
                X_2d = reduce_dim(all_vecs, method=method, random_state=args.seed)
                plot_points(
                    X_2d,
                    labels,
                    title=f"Qwen2.5-VL-7B manifold visualization ({method.upper()})",
                    save_path=os.path.join(args.output_dir, f"qwen25_vl_manifold_{method}.png"),
                )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
