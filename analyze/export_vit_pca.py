import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from data.io_utils import IMAGE_PLACEHOLDER
from gap import GapForConditionalGeneration


@dataclass
class PCAResult:
    x_path: str
    mean_path: str
    components_path: str
    metadata_path: str
    reconstruction_path: str
    hidden_size: int
    num_components: int
    explained_variance_ratio: float
    total_samples: int


class RunningMeanAccumulator:
    def __init__(self, hidden_size: int, dtype: np.dtype = np.float64):
        self.hidden_size = int(hidden_size)
        self.dtype = np.dtype(dtype)
        self.count = 0
        self.sum_x = np.zeros((self.hidden_size,), dtype=self.dtype)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        if x.ndim != 2 or x.shape[1] != self.hidden_size:
            raise ValueError(f"Expected x with shape (N, {self.hidden_size}), got {x.shape}")
        self.sum_x += x.sum(axis=0, dtype=self.dtype)
        self.count += int(x.shape[0])

    def finalize(self) -> np.ndarray:
        if self.count < 1:
            raise ValueError("Need at least 1 sample to compute mean.")
        return self.sum_x / self.count


class RunningCovarianceAccumulator:
    def __init__(self, hidden_size: int, mean: np.ndarray, dtype: np.dtype = np.float64):
        self.hidden_size = int(hidden_size)
        self.dtype = np.dtype(dtype)
        self.mean = np.asarray(mean, dtype=self.dtype)
        self.count = 0
        self.sum_centered_xx = np.zeros((self.hidden_size, self.hidden_size), dtype=self.dtype)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        centered = x.astype(self.dtype, copy=False) - self.mean[None, :]
        self.sum_centered_xx += centered.T @ centered
        self.count += int(centered.shape[0])

    def finalize(self) -> np.ndarray:
        if self.count < 2:
            raise ValueError(f"Need at least 2 samples for covariance, got {self.count}")
        covariance = self.sum_centered_xx / (self.count - 1)
        covariance = (covariance + covariance.T) * 0.5
        return covariance


class PCAVisionDataset(Dataset):
    def __init__(
        self,
        data_paths: Sequence[str],
        processor: AutoProcessor,
        image_root_dirs: Dict[str, str],
        source: str = "input",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.processor = processor
        self.image_root_dirs = image_root_dirs
        self.source = source
        self.data: List[dict] = []

        for path in data_paths:
            print(f"[PCAVisionDataset] Loading metadata from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        if max_samples is not None:
            self.data = self.data[:max_samples]

        print(f"[PCAVisionDataset] Loaded {len(self.data)} samples (source={self.source}).")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_image_path(self, image_path: str, image_root_dir: str) -> str:
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        candidate = os.path.join(image_root_dir, image_path)
        if os.path.exists(candidate):
            return candidate
        return image_path

    def _load_image(self, image_path: str) -> Image.Image:
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {image_path}: {e}") from e

    def _process_image(self, image_path: str, is_target: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(image_path)
        out = self.processor.image_processor(images=image, return_tensors="pt")

        pixel_values = out.get("pixel_values")
        image_grid_thw = out.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError(f"Processor did not return pixel_values/image_grid_thw for {image_path}")
        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.tensor(image_grid_thw)
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)
        return pixel_values, image_grid_thw

    def __getitem__(self, idx: int) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        item = self.data[idx]
        data_source = item.get("data_source", "default")
        image_root_dir = self.image_root_dirs.get(data_source, "")
        messages = copy.deepcopy(item.get("messages", item.get("message", [])))

        input_pixel_values_list: List[torch.Tensor] = []
        input_image_grid_thw_list: List[torch.Tensor] = []
        target_pixel_values_list: List[torch.Tensor] = []
        target_image_grid_thw_list: List[torch.Tensor] = []

        try:
            for msg in messages:
                role = str(msg.get("role", "")).lower()
                dialog = msg.get("dialog", "")
                image_paths = msg.get("image_filenames", []) or []
                image_types = msg.get("image_types", None)

                if isinstance(image_paths, str):
                    image_paths = [image_paths]

                parts = dialog.split(IMAGE_PLACEHOLDER)
                num_image_slots = max(len(parts) - 1, len(image_paths))

                for i in range(num_image_slots):
                    if i >= len(image_paths):
                        continue

                    img_path = self._resolve_image_path(image_paths[i], image_root_dir)
                    img_type = image_types[i] if image_types and i < len(image_types) else None

                    is_target = False
                    if img_type == "input":
                        is_target = False
                    elif img_type is not None:
                        is_target = (role == "assistant")
                    else:
                        is_target = (role == "assistant")

                    include_input = self.source in ("input", "both") and not is_target
                    include_target = self.source in ("target", "both") and is_target
                    if not include_input and not include_target:
                        continue

                    pixel_values, image_grid_thw = self._process_image(img_path, is_target=is_target)
                    if is_target:
                        target_pixel_values_list.append(pixel_values)
                        target_image_grid_thw_list.append(image_grid_thw)
                    else:
                        input_pixel_values_list.append(pixel_values)
                        input_image_grid_thw_list.append(image_grid_thw)
        except Exception as e:
            print(f"[PCAVisionDataset] skip sample idx={idx}: {e}")
            return None

        def safe_concat(tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
            return torch.cat(tensor_list, dim=0) if tensor_list else None

        result = {
            "pixel_values": safe_concat(input_pixel_values_list),
            "image_grid_thw": safe_concat(input_image_grid_thw_list),
            "target_pixel_values": safe_concat(target_pixel_values_list),
            "target_image_grid_thw": safe_concat(target_image_grid_thw_list),
        }

        if all(value is None for value in result.values()):
            return None
        return result


@dataclass
class PCAVisionCollator:
    def __call__(self, examples):
        examples = [example for example in examples if example is not None]
        if not examples:
            return {}

        batch = {}
        for key in ["pixel_values", "image_grid_thw", "target_pixel_values", "target_image_grid_thw"]:
            tensors = [example[key] for example in examples if example.get(key) is not None]
            batch[key] = torch.cat(tensors, dim=0) if tensors else None
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export X, mu, W from ViT embeddings in GAP training data.")
    parser.add_argument("--model-name-or-path", required=True, help="HF model path or local checkpoint.")
    parser.add_argument(
        "--train-meta-filepaths",
        required=True,
        nargs="+",
        help="One or more JSONL metadata files used for training.",
    )
    parser.add_argument(
        "--image-root-dirs",
        type=str,
        default="{}",
        help="JSON dict: data_source -> image root directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save memmap and PCA outputs.")
    parser.add_argument("--variance-threshold", type=float, default=0.95, help="Target explained variance ratio.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for data loading.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on dataset samples.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Inference dtype for the VL model.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=3584,
        help="Hidden size of the vision embeddings; default is 3584 for Qwen2.5-7B ViT"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="Dataset max sequence length; only affects text packing, not image features.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16384,
        help="Rows per chunk when reading/writing memmap and computing PCA statistics.",
    )
    parser.add_argument(
        "--memmap-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Disk dtype for X memmap. float32 is safer; float16 saves space.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="target",
        choices=["input", "target", "both"],
        help="Which vision embeddings to dump into X. GAP paper defaults to target auxiliary-image embeddings.",
    )
    parser.add_argument(
        "--fixed-vision-tokens",
        type=int,
        default=None,
        help="If set, adaptively pool each image's vision embeddings to a fixed token count (e.g. 64).",
    )
    parser.add_argument(
        "--compute-reconstruction-loss",
        action="store_true",
        help="If set, compute reconstruction MSE/L2 metrics using mu and W over the memmap X.",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Progress log frequency in steps.")
    return parser.parse_args()


def to_torch_dtype(name: str) -> Optional[torch.dtype]:
    if name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def to_numpy_dtype(name: str) -> np.dtype:
    return np.dtype({"float16": np.float16, "float32": np.float32}[name])


def build_dataset_and_loader(
    train_meta_filepaths: Sequence[str],
    model_name_or_path: str,
    image_root_dirs: Dict[str, str],
    batch_size: int,
    num_workers: int,
    max_seq_len: int,
    max_samples: Optional[int],
    source: str,
):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    dataset = PCAVisionDataset(
        data_paths=list(train_meta_filepaths),
        processor=processor,
        image_root_dirs=image_root_dirs,
        source=source,
        max_samples=max_samples,
    )

    collator = PCAVisionCollator()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def build_model(model_name_or_path: str, device: torch.device, torch_dtype: Optional[torch.dtype]):
    model_kwargs = {"trust_remote_code": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if "gap" in model_name_or_path.lower():
        model = GapForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    elif "qwen" in model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def pool_vision_tokens_per_image(
    embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    target_tokens: Optional[int],
) -> torch.Tensor:
    if target_tokens is None:
        return embeds
    if target_tokens <= 0:
        raise ValueError(f"fixed_vision_tokens must be positive, got {target_tokens}")
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required when fixed_vision_tokens is set")

    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.tensor(image_grid_thw, device=embeds.device)
    image_grid_thw = image_grid_thw.to(device=embeds.device)
    if image_grid_thw.dim() == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    pooled_chunks = []
    start = 0
    for grid in image_grid_thw:
        t = int(grid[0].item())
        h = int(grid[1].item())
        w = int(grid[2].item())
        seq_len = t * max(h // 2, 1) * max(w // 2, 1)
        if seq_len <= 0:
            continue

        chunk = embeds[start : start + seq_len]
        start += seq_len
        if chunk.shape[0] != seq_len:
            raise ValueError(f"Vision token count mismatch: expected {seq_len}, got {chunk.shape[0]}")

        if seq_len == target_tokens:
            pooled_chunks.append(chunk)
            continue

        chunk_t = chunk.transpose(0, 1).unsqueeze(0)
        pooled = torch.nn.functional.adaptive_avg_pool1d(chunk_t, target_tokens)
        pooled = pooled.squeeze(0).transpose(0, 1).contiguous()
        pooled_chunks.append(pooled)

    if start != embeds.shape[0]:
        raise ValueError(f"Unused vision tokens after pooling: consumed {start}, total {embeds.shape[0]}")
    if not pooled_chunks:
        return embeds.new_zeros((0, embeds.shape[-1]))
    return torch.cat(pooled_chunks, dim=0)


def extract_batch_embeddings(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    source: str,
    fixed_vision_tokens: Optional[int] = None,
) -> Iterator[torch.Tensor]:
    with torch.no_grad():
        if source in ("input", "both"):
            pixel_values = batch.get("pixel_values")
            image_grid_thw = batch.get("image_grid_thw")
            if pixel_values is not None and image_grid_thw is not None:
                embeds = model.model.visual(pixel_values.to(device), grid_thw=image_grid_thw.to(device)).detach()
                embeds = pool_vision_tokens_per_image(embeds, image_grid_thw, fixed_vision_tokens)
                yield embeds

        if source in ("target", "both"):
            target_pixel_values = batch.get("target_pixel_values")
            target_image_grid_thw = batch.get("target_image_grid_thw")
            if target_pixel_values is not None and target_image_grid_thw is not None:
                embeds = model.model.visual(
                    target_pixel_values.to(device),
                    grid_thw=target_image_grid_thw.to(device),
                ).detach()
                embeds = pool_vision_tokens_per_image(embeds, target_image_grid_thw, fixed_vision_tokens)
                yield embeds


def iter_vision_embeddings(
    model,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    source: str,
    fixed_vision_tokens: Optional[int] = None,
) -> Iterator[np.ndarray]:
    for batch in loader:
        if not batch:
            continue
        for embeds in extract_batch_embeddings(model, batch, device, source, fixed_vision_tokens=fixed_vision_tokens):
            yield embeds.float().cpu().numpy()


def count_total_tokens(
    model,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    source: str,
    log_every: int,
    fixed_vision_tokens: Optional[int] = None,
) -> Tuple[int, int]:
    total_tokens = 0
    steps = 0
    progress = tqdm(
        iter_vision_embeddings(model, loader, device, source, fixed_vision_tokens=fixed_vision_tokens),
        desc="count_tokens",
    )
    for chunk in progress:
        total_tokens += int(chunk.shape[0])
        steps += 1
        progress.set_postfix(total_tokens=total_tokens)
        if log_every > 0 and steps % log_every == 0:
            print(f"[export_vit_pca] counting steps={steps} total_tokens={total_tokens}")
    return total_tokens, steps


def write_memmap_matrix(
    x_path: str,
    shape: Tuple[int, int],
    data_iter: Iterable[np.ndarray],
    dtype: np.dtype,
    log_every: int,
) -> int:
    x_memmap = np.memmap(x_path, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    steps = 0
    progress = tqdm(data_iter, desc="write_memmap")
    for chunk in progress:
        rows = int(chunk.shape[0])
        x_memmap[offset : offset + rows] = chunk.astype(dtype, copy=False)
        offset += rows
        steps += 1
        progress.set_postfix(rows_written=offset)
        if log_every > 0 and steps % log_every == 0:
            print(f"[export_vit_pca] writing steps={steps} rows_written={offset}")
    x_memmap.flush()
    del x_memmap
    return offset


def iter_memmap_chunks(x_memmap: np.memmap, chunk_size: int) -> Iterator[np.ndarray]:
    total_rows = x_memmap.shape[0]
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        yield np.asarray(x_memmap[start:end])


def iter_memmap_chunks_with_progress(
    x_memmap: np.memmap,
    chunk_size: int,
    desc: str,
) -> Iterator[np.ndarray]:
    total_rows = x_memmap.shape[0]
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    for chunk in tqdm(iter_memmap_chunks(x_memmap, chunk_size), total=total_chunks, desc=desc):
        yield chunk


def compute_mean_from_memmap(x_memmap: np.memmap, chunk_size: int) -> Tuple[np.ndarray, int]:
    hidden_size = x_memmap.shape[1]
    accumulator = RunningMeanAccumulator(hidden_size=hidden_size)
    for chunk in iter_memmap_chunks_with_progress(x_memmap, chunk_size, desc="mean"):
        accumulator.update(chunk)
    return accumulator.finalize(), accumulator.count


def compute_covariance_from_memmap(x_memmap: np.memmap, mean: np.ndarray, chunk_size: int) -> np.ndarray:
    hidden_size = x_memmap.shape[1]
    accumulator = RunningCovarianceAccumulator(hidden_size=hidden_size, mean=mean)
    for chunk in iter_memmap_chunks_with_progress(x_memmap, chunk_size, desc="covariance"):
        accumulator.update(chunk)
    return accumulator.finalize()


def compute_reconstruction_metrics_from_memmap(
    x_memmap: np.memmap,
    mean: np.ndarray,
    components: np.ndarray,
    chunk_size: int,
) -> Dict[str, float]:
    if x_memmap.shape[0] < 1:
        raise ValueError("Need at least 1 sample to compute reconstruction metrics.")
    if components.ndim != 2:
        raise ValueError(f"Expected components to be 2D, got shape {components.shape}")
    if mean.ndim != 1:
        raise ValueError(f"Expected mean to be 1D, got shape {mean.shape}")

    total_squared_error = 0.0
    total_squared_norm = 0.0
    total_rows = 0
    total_dims = int(x_memmap.shape[1])
    mean64 = mean.astype(np.float64, copy=False)
    components64 = components.astype(np.float64, copy=False)

    for chunk in iter_memmap_chunks_with_progress(x_memmap, chunk_size, desc="reconstruction"):
        x = chunk.astype(np.float64, copy=False)
        centered = x - mean64[None, :]
        coeff = centered @ components64
        reconstructed = coeff @ components64.T + mean64[None, :]
        residual = x - reconstructed

        total_squared_error += float(np.sum(residual * residual))
        total_squared_norm += float(np.sum(centered * centered))
        total_rows += int(x.shape[0])

    mse = total_squared_error / (total_rows * total_dims)
    rmse = float(np.sqrt(mse))
    relative_mse = total_squared_error / max(total_squared_norm, 1e-12)

    return {
        "mse": float(mse),
        "rmse": rmse,
        "relative_mse": float(relative_mse),
        "total_squared_error": float(total_squared_error),
        "total_squared_centered_norm": float(total_squared_norm),
        "num_rows": int(total_rows),
        "num_dims": int(total_dims),
    }


def compute_components_from_covariance(
    covariance: np.ndarray,
    variance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], a_min=0.0, a_max=None)
    evecs = evecs[:, order]

    total = float(evals.sum())
    if total <= 0:
        raise ValueError("Covariance matrix has non-positive total variance; cannot compute PCA basis.")

    ratios = evals / total
    cumulative = np.cumsum(ratios)
    k = int(np.searchsorted(cumulative, variance_threshold, side="left")) + 1
    k = min(k, covariance.shape[0])
    explained = float(cumulative[k - 1])
    components = evecs[:, :k]
    return components, evals, k, explained


def save_outputs(
    output_dir: str,
    x_path: str,
    mean: np.ndarray,
    components: np.ndarray,
    eigenvalues: np.ndarray,
    explained_variance_ratio: float,
    total_samples: int,
    memmap_dtype: np.dtype,
    source: str,
    fixed_vision_tokens: Optional[int],
    reconstruction_metrics: Optional[Dict[str, float]] = None,
) -> PCAResult:
    os.makedirs(output_dir, exist_ok=True)

    mean_path = os.path.join(output_dir, "vit_pca_mu.npy")
    components_path = os.path.join(output_dir, "vit_pca_W.npy")
    metadata_path = os.path.join(output_dir, "vit_pca_metadata.json")
    reconstruction_path = os.path.join(output_dir, "vit_pca_reconstruction.json")

    np.save(mean_path, mean.astype(np.float32, copy=False))
    np.save(components_path, components.astype(np.float32, copy=False))

    metadata = {
        "x_file": os.path.basename(x_path),
        "x_dtype": str(np.dtype(memmap_dtype)),
        "hidden_size": int(components.shape[0]),
        "num_components": int(components.shape[1]),
        "explained_variance_ratio": explained_variance_ratio,
        "total_samples": int(total_samples),
        "source": source,
        "fixed_vision_tokens": fixed_vision_tokens,
        "top_eigenvalues": eigenvalues[: min(32, eigenvalues.shape[0])].tolist(),
        "mu_file": os.path.basename(mean_path),
        "components_file": os.path.basename(components_path),
        "reconstruction_file": os.path.basename(reconstruction_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if reconstruction_metrics is None:
        reconstruction_metrics = {}
    with open(reconstruction_path, "w", encoding="utf-8") as f:
        json.dump(reconstruction_metrics, f, ensure_ascii=False, indent=2)

    return PCAResult(
        x_path=x_path,
        mean_path=mean_path,
        components_path=components_path,
        metadata_path=metadata_path,
        reconstruction_path=reconstruction_path,
        hidden_size=int(components.shape[0]),
        num_components=int(components.shape[1]),
        explained_variance_ratio=explained_variance_ratio,
        total_samples=int(total_samples),
    )


def run_export(args: argparse.Namespace) -> PCAResult:
    if not (0.0 < args.variance_threshold <= 1.0):
        raise ValueError(f"variance_threshold must be in (0, 1], got {args.variance_threshold}")
    if args.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {args.chunk_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    image_root_dirs = json.loads(args.image_root_dirs) if args.image_root_dirs else {}
    infer_device = torch.device(args.device)
    torch_dtype = to_torch_dtype(args.torch_dtype)
    memmap_dtype = to_numpy_dtype(args.memmap_dtype)

    _, loader = build_dataset_and_loader(
        train_meta_filepaths=args.train_meta_filepaths,
        model_name_or_path=args.model_name_or_path,
        image_root_dirs=image_root_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        source=args.source,
    )
    model = build_model(args.model_name_or_path, infer_device, torch_dtype)

    hidden_size = args.hidden_size
    total_tokens, _ = count_total_tokens(
        model=model,
        loader=loader,
        device=infer_device,
        source=args.source,
        log_every=args.log_every,
        fixed_vision_tokens=args.fixed_vision_tokens,
    )
    if total_tokens < 2:
        raise ValueError(f"Need at least 2 vision tokens for PCA, got {total_tokens}")

    x_path = os.path.join(args.output_dir, "vit_embeddings_X.memmap")
    _, loader = build_dataset_and_loader(
        train_meta_filepaths=args.train_meta_filepaths,
        model_name_or_path=args.model_name_or_path,
        image_root_dirs=image_root_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        source=args.source,
    )
    rows_written = write_memmap_matrix(
        x_path=x_path,
        shape=(total_tokens, hidden_size),
        data_iter=iter_vision_embeddings(
            model,
            loader,
            infer_device,
            args.source,
            fixed_vision_tokens=args.fixed_vision_tokens,
        ),
        dtype=memmap_dtype,
        log_every=args.log_every,
    )
    if rows_written != total_tokens:
        raise RuntimeError(f"Rows written to memmap ({rows_written}) != counted total tokens ({total_tokens})")

    x_memmap = np.memmap(x_path, mode="r", dtype=memmap_dtype, shape=(total_tokens, hidden_size))
    mean, counted_rows = compute_mean_from_memmap(x_memmap, args.chunk_size)
    if counted_rows != total_tokens:
        raise RuntimeError(f"Rows used for mean ({counted_rows}) != total tokens ({total_tokens})")

    covariance = compute_covariance_from_memmap(x_memmap, mean, args.chunk_size)
    components, eigenvalues, _, explained = compute_components_from_covariance(
        covariance=covariance,
        variance_threshold=args.variance_threshold,
    )
    reconstruction_metrics = None
    if args.compute_reconstruction_loss:
        reconstruction_metrics = compute_reconstruction_metrics_from_memmap(
            x_memmap=x_memmap,
            mean=mean,
            components=components,
            chunk_size=args.chunk_size,
        )
    result = save_outputs(
        output_dir=args.output_dir,
        x_path=x_path,
        mean=mean,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained,
        total_samples=total_tokens,
        memmap_dtype=memmap_dtype,
        source=args.source,
        fixed_vision_tokens=args.fixed_vision_tokens,
        reconstruction_metrics=reconstruction_metrics,
    )
    print(
        "[export_vit_pca] done: "
        f"X={result.x_path}, mu={result.mean_path}, W={result.components_path}, "
        f"hidden_size={result.hidden_size}, num_components={result.num_components}, "
        f"explained_variance_ratio={result.explained_variance_ratio:.6f}, total_samples={result.total_samples}"
    )
    if reconstruction_metrics is not None:
        print(
            "[export_vit_pca] reconstruction: "
            f"mse={reconstruction_metrics['mse']:.8f}, "
            f"rmse={reconstruction_metrics['rmse']:.8f}, "
            f"relative_mse={reconstruction_metrics['relative_mse']:.8f}"
        )
    return result


def main() -> None:
    args = parse_args()
    run_export(args)


if __name__ == "__main__":
    main()
