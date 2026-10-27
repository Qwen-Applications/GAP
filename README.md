# GAP

GAP is the codebase for **Fill the GAP: A Granular Alignment Paradigm for Visual Reasoning in Multimodal Large Language Models**.

The implementation builds on Qwen2.5-VL and trains a visual latent reasoning path with:

- context-grounded latent supervision using auxiliary visual targets,
- a PCA-aligned latent head for input-compatible latent feedback,
- difficulty-aware latent assignment through per-sample accuracy metadata,
- a default 36-token latent budget.

This repository is GAP-only. Deprecated method variants have been removed.

Large checkpoints and datasets are not bundled.

## Repository Layout

```text
gap/                          GAP model and config
training/train.py             Main GAP training entry point
data/data_utils.py            Interleaved text/image/latent dataset and collator
analyze/export_vit_pca.py     PCA export for auxiliary-image vision embeddings
scripts/run_train.sh          Main distributed training launcher
scripts/debug_stage1_train.sh Small debug launcher for the same GAP training path
scripts/run_export_vit_pca.sh PCA export launcher
deepspeed/                    DeepSpeed configs
tests/                        Lightweight data/collator diagnostics
```

## Setup

```bash
cd Gap
pip install -r requirements.txt
pip install -e .
```

Expected base model:

```bash
export MODEL_PATH=/root/model/Qwen2.5-VL-7B-Instruct
```

The package exposes the model as:

```python
from gap import GapConfig, GapForConditionalGeneration
```

New GAP checkpoints use:

- `model_type`: `gap`
- architecture: `GapForConditionalGeneration`
- example checkpoint path: `/root/model/Gap-VL-7B`

## Data

The training data used in the paper is released on Hugging Face as [AlibabaQwenintern/gap-data](https://huggingface.co/datasets/AlibabaQwenintern/gap-data). It contains 36,985 examples (33,279 in `train.jsonl`, 3,706 in `val.jsonl`), the image archives, and the PCA basis used in the paper. See the [dataset card](https://huggingface.co/datasets/AlibabaQwenintern/gap-data) for sources, construction, field descriptions, and licenses.

Access is gated: log in to Hugging Face and accept the conditions on the dataset page, then download and extract:

```bash
hf auth login
hf download AlibabaQwenintern/gap-data --repo-type dataset --local-dir /root/data
cd /root/data
for f in images/*.tar; do tar -xf "$f" -C /root/data; done
```

With older versions of `huggingface_hub`, run `huggingface-cli download` with the same arguments. The archives take about 50 GB, and extraction needs roughly the same again; you can delete `images/*.tar` afterwards.

After extraction, `/root/data` matches the launcher defaults:

```text
/root/data/
├── train.jsonl        # TRAIN_DATA_PATH, TRAIN_META_FILEPATHS
├── val.jsonl          # VAL_DATA_PATH
├── vit_pca_mu.npy     # VISION_PCA_MU_PATH
├── vit_pca_W.npy      # VISION_PCA_W_PATH
├── monet/
├── chart/
├── GEOQA/
├── MM_MATH/
├── MATHVISION/
└── geo3k_train/
```

Image paths in the JSONL files are either absolute (`/root/data/...`) or relative to `/root/data`, which is the image root for every `data_source` in the default `IMAGE_ROOTS` of `scripts/run_train.sh`. To keep the data elsewhere, rewrite the `/root/data/` prefix in the JSONL files and set `IMAGE_ROOTS` to your root.

The released PCA basis (629 components, 95% retained variance) was fit on the training split, so you can train directly without the PCA export step below:

```bash
export TRAIN_DATA_PATH=/root/data/train.jsonl
export VAL_DATA_PATH=/root/data/val.jsonl
export VISION_PCA_MU_PATH=/root/data/vit_pca_mu.npy
export VISION_PCA_W_PATH=/root/data/vit_pca_W.npy
```

## Data Format

Training data is JSONL. Each line should contain `messages` or `message` with role/dialog fields. Images are referenced by `image_filenames`, and `image_types` should distinguish query images from auxiliary target images:

```json
{
  "data_source": "monet_Visual_CoT",
  "accuracy": 0.0,
  "messages": [
    {
      "role": "user",
      "dialog": "Question text <|vision_start|><|image_pad|><|vision_end|>",
      "image_filenames": ["query.jpg"],
      "image_types": ["input"]
    },
    {
      "role": "assistant",
      "dialog": "Reasoning <|vision_start|><|image_pad|><|vision_end|> final answer",
      "image_filenames": ["auxiliary.jpg"],
      "image_types": ["target"]
    }
  ]
}
```

Difficulty-aware assignment uses `accuracy`: samples with `accuracy > ACC_THRESHOLD` are treated as easy and have auxiliary target latent supervision removed.

## Export PCA Features

GAP’s paper setting uses target auxiliary-image embeddings, 95% retained variance, and a fixed 36-token target grid.

```bash
MODEL_NAME_OR_PATH=/root/model/Qwen2.5-VL-7B-Instruct \
TRAIN_META_FILEPATHS=/root/data/train.jsonl \
OUTPUT_DIR=/root/output/gap_pca \
bash scripts/run_export_vit_pca.sh
```

This writes PCA artifacts such as the mean vector and projection matrix. Pass them into training with:

```bash
export VISION_PCA_MU_PATH=/root/output/gap_pca/vit_pca_mu.npy
export VISION_PCA_W_PATH=/root/output/gap_pca/vit_pca_W.npy
```

Adjust filenames if your export directory uses different names.

## Train GAP

The main launcher is:

```bash
cd Gap

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NPROC_PER_NODE=8

export MODEL_PATH=/root/model/Qwen2.5-VL-7B-Instruct
export TRAIN_DATA_PATH=/root/data/train.jsonl
export VAL_DATA_PATH=/root/data/val.jsonl
export PRIMUS_SAVE_CHECKPOINT_DIR=/root/output/gap_stage1_train
export PRIMUS_TENSORBOARD_LOG_DIR=/root/output/gap_stage1_logs

bash scripts/run_train.sh
```

Important defaults in `scripts/run_train.sh`:

```bash
NPROC_PER_NODE=8
LATENT_SEQ_LEN=36
VISION_HEAD_MODE=linear
GLOBAL_BATCH_SIZE=128
LR=1e-5
VISION_LR=1e-5
EPOCHS=2
ACC_THRESHOLD=0.0
SCHEDULE_SAMPLING_ENABLED=false
```

For a smaller smoke run, use:

```bash
export NPROC_PER_NODE=1
bash scripts/debug_stage1_train.sh
```

The launchers use environment variables for all machine-specific paths. The
default `/root/...` values are placeholders for a container or cluster setup;
override them for local paths.

## Useful Ablations

Pure-text / no-latent mode:

```bash
NO_LATENT=true bash scripts/run_train.sh
```

Disable latent loss while keeping token serialization:

```bash
NO_LATENT_LOSS=true bash scripts/run_train.sh
```

Change the difficulty-aware threshold:

```bash
ACC_THRESHOLD=0.5 bash scripts/run_train.sh
```

Change latent budget:

```bash
LATENT_SEQ_LEN=16 bash scripts/run_train.sh
```

`LATENT_SEQ_LEN` must be a perfect square, such as `4`, `16`, `36`, `64`, or `144`.

## Diagnostics

Run lightweight data/collator checks:

```bash
python tests/check_data_collator.py
```

Run syntax and import checks:

```bash
python -c "from gap import GapConfig, GapForConditionalGeneration; print(GapConfig.model_type)"
python -c "from data.data_utils import InterleaveDataset, InterleaveDataCollator; print('ok')"
```

Inspect user-image resolutions in a training JSONL:

```bash
python -m scripts.stats_resolution \
  --model_name_or_path /root/model/Qwen2.5-VL-7B-Instruct \
  --train_meta_filepath /root/data/train.jsonl \
  --image_root_dirs '{"monet_Visual_CoT": "/root/data"}' \
  --latent_seq_len 36 \
  --num_samples 5000
```

## Notes

- The default code path is the GAP paper path: PCA-aligned latent head, 36 latent tokens, and difficulty-aware target removal.
- Auxiliary target images are used only for latent supervision during training. They are not query inputs.
- Deprecated checkpoints from earlier method variants are not supported by this cleaned codebase.
