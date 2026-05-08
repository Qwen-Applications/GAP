import os
import sys
from dataclasses import dataclass, field
import json
from typing import Optional

import torch
from torch.utils.data import Subset
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
)

from gap import GapForConditionalGeneration, GapConfig
from data.data_utils import InterleaveDataset, InterleaveDataCollator

@dataclass
class GapTrainingArguments(TrainingArguments):
    use_sinkhorn_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use Sinkhorn loss for vision prediction."}
    )
    vision_head_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the vision head."}
    )
    no_latent_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use vision loss for vision prediction."}
    )
    schedule_sampling_enabled: bool = field(
        default=False,
        metadata={"help": "Enable two-pass scheduled sampling for latent vision training."}
    )
    schedule_sampling_ratio: float = field(
        default=0.0,
        metadata={"help": "Probability of running the recon-features pass instead of real features. 0 = always real features (teacher forcing), 1 = always recon features (inference-aligned)."}
    )
    schedule_sampling_max_ratio: float = field(
        default=0.0,
        metadata={"help": "If > 0, linearly ramp the probability up to this max value over schedule_sampling_warmup_steps."}
    )
    schedule_sampling_warmup_steps: int = field(
        default=0,
        metadata={"help": "Warmup steps for linearly increasing scheduled sampling ratio."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path to the pretrained VL model."}
    )
    vision_feature_size: int = field(
        default=3584,
        metadata={"help": "Vision feature size."}
    )
    vision_pca_mu_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the vision PCA mean vector. Leave empty to train without PCA / vision latent head."}
    )
    vision_pca_W_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the vision PCA projection matrix. Leave empty to train without PCA / vision latent head."}
    )
    latent_start_token: str = field(
        default="<|latent_start|>",
        metadata={"help": "Latent start token."}
    )
    latent_end_token: str = field(
        default="<|latent_end|>",
        metadata={"help": "Latent end token."}
    )
    latent_pad_token: str = field(
        default="<|latent_pad|>",
        metadata={"help": "Latent pad token."}
    )
    latent_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Vision loss weight."}
    )
    is_transformer_block: bool = field(
        default=False,
        metadata={"help": "Whether to use transformer blocks in the vision head."}
    )
    num_transformer_layers: int = field(
        default=1,
        metadata={"help": "Number of transformer layers in the vision head if using transformer blocks."}
    )
    detach_vision_hidden: bool = field(
        default=False,
        metadata={"help": "Detach vision hidden states before the vision head (preserve base model math ability)."}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM during training."}
    )
    freeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LM head during training."}
    )
    freeze_embed: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the embedding layer during training."}
    )
    vision_head_mode: str = field(
        default="linear",
        metadata={"help": "Vision prediction head mode: 'linear' or 'none'."}
    )
    no_latent: bool = field(
        default=False,
        metadata={"help": "Pure-text ablation mode: skip latent tokens, vision loss, and schedule sampling entirely."}
    )


@dataclass
class DataArguments:
    train_meta_filepath: str = field(
        default="data/train_metadata.jsonl",
        metadata={"help": "Path to the training metadata."}
    )
    val_meta_filepath: str = field(
        default="data/val_metadata.jsonl",
        metadata={"help": "Path to the validation metadata."}
    )
    image_root_dirs: str = field(
        default=None,
        metadata={"help": "Path to the image root directories."}
    )
    latent_seq_len: int = field(
        default=36,
        metadata={"help": "Latent sequence length."}
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Whether to run in debug mode."}
    )
    debug_evalsubset_size: int = field(
        default=128,
        metadata={"help": "Number of samples to evaluate."}
    )
    data_max_seq_len: int = field(
        default=16384,
        metadata={"help": "Maximum sequence length for data samples."}
    )
    parser_start_token: str = field(
        default="<parser>",
        metadata={"help": "Parser start token."}
    )
    parser_end_token: str = field(
        default="</parser>",
        metadata={"help": "Parser end token."}
    )
    parse_prompt: str = field(
        default="\nPlease parse latent information from the above conversation and provide it in a structured format between <parser> and </parser> tags. The latent information should be concise and directly related to the visual content of the conversation.",
        metadata={"help": "Prompt to guide the model in parsing latent information."}
    )
    no_split: bool = field(
        default=False,
        metadata={"help": "If True, do not split each sample into parser/non-parser variants; keep original single sample."}
    )
    case_probs: str = field(
        default="1,0,0",
        metadata={"help": "Comma-separated probabilities for the 3 cases (both/latent/parser) when generating training samples. Should sum to 1."}
    )
    acc_threshold: float = field(
        default=0.0,
        metadata={"help": "Accuracy threshold: samples with accuracy > acc_threshold skip latent (pure LM loss); others use latent."}
    )

class GapTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clear_eval_buffer()
        self._micro_step = 0

    def _clear_eval_buffer(self):
        self.eval_metrics_buffer = {
            "lm_loss_sum": 0.0,
            "lm_loss_count": 0,
            "vis_loss_sum": 0.0,
            "vis_loss_count": 0,
        }

    def _get_schedule_sampling_ratio(self):
        if not getattr(self.args, "schedule_sampling_enabled", False):
            return 0.0

        base_ratio = float(getattr(self.args, "schedule_sampling_ratio", 0.0))
        max_ratio = float(getattr(self.args, "schedule_sampling_max_ratio", 0.0))
        warmup_steps = int(getattr(self.args, "schedule_sampling_warmup_steps", 0))

        if max_ratio <= 0.0 or warmup_steps <= 0:
            return base_ratio

        progress = min(1.0, max(float(self.state.global_step), 0.0) / float(max(warmup_steps, 1)))
        return min(max_ratio, progress * max_ratio)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"NaN detected in input {k}")
                
        target_pixel_values = inputs.pop("target_pixel_values", None)
        target_image_grid_thw = inputs.pop("target_image_grid_thw", None)
        target_image_seq_lens = inputs.pop("target_image_seq_lens", None)
        target_image_counts_input = inputs.pop("target_image_counts", None)
        need_latent = inputs.pop("need_latent", None)  # (B,) bool tensor
        
        vision_labels = None
        vision_mask = None
        target_image_seq_lens_tensor = None
        target_image_counts_tensor = target_image_counts_input
        if target_pixel_values is not None and target_image_grid_thw is not None:
            base_model = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                vision_features = base_model.model.visual(
                    target_pixel_values, grid_thw=target_image_grid_thw
                )

            if target_image_seq_lens is not None:
                seq_lens = target_image_seq_lens.tolist()
            else:
                seq_lens = []
                for grid in target_image_grid_thw:
                    h = int(grid[1].item())
                    w = int(grid[2].item())
                    seq_lens.append(int((h // 2) * (w // 2)))

            per_image = []
            pos = 0
            for seq_len in seq_lens:
                per_image.append(vision_features[pos : pos + seq_len])
                pos += seq_len
            
            seq_lens_for_model = [int(s) for s in seq_lens]
            target_image_seq_lens_tensor = torch.tensor(seq_lens_for_model, dtype=torch.long, device=vision_features.device)

            if target_image_counts_input is None:
                latent_start_id = getattr(base_model, "latent_start_token_id", None)
                if latent_start_id is None:
                    latent_start_id = getattr(base_model.config, "latent_start_token_id", None)
                if latent_start_id is None:
                    raise ValueError("latent_start_token_id not found for building vision labels")
                target_image_counts_list = (inputs["input_ids"] == latent_start_id).sum(dim=1).tolist()
            else:
                if isinstance(target_image_counts_input, torch.Tensor):
                    target_image_counts_list = target_image_counts_input.tolist()
                else:
                    target_image_counts_list = list(target_image_counts_input)

            target_image_counts_tensor = torch.tensor(
                target_image_counts_list,
                dtype=torch.long,
                device=vision_features.device,
            )

            per_sample = []
            img_idx = 0
            for count in target_image_counts_list:
                count = int(count)
                if count == 0:
                    per_sample.append(vision_features.new_zeros((0, vision_features.shape[-1])))
                    continue
                sample_imgs = per_image[img_idx : img_idx + count]
                img_idx += count

                per_sample.append(torch.cat(list(sample_imgs), dim=0))

            max_len = max((t.shape[0] for t in per_sample), default=0)
            if max_len > 0:
                batch_size = len(per_sample)
                feat_dim = vision_features.shape[-1]
                vision_labels = vision_features.new_zeros((batch_size, max_len, feat_dim))
                vision_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=vision_features.device)
                for b, tgt in enumerate(per_sample):
                    if tgt.numel() == 0:
                        continue
                    vision_labels[b, : tgt.shape[0]] = tgt
                    vision_mask[b, : tgt.shape[0]] = 1

                # Zero out vision supervision for samples where need_latent=False
                if need_latent is not None:
                    need_latent_dev = need_latent.to(vision_mask.device)
                    vision_mask[~need_latent_dev] = 0
                    target_image_counts_tensor[~need_latent_dev] = 0


        packed_vision_features = vision_labels[vision_mask.bool()] if vision_labels is not None else None
        need_vision_loss = not self.args.no_latent_loss
        schedule_sampling_ratio = self._get_schedule_sampling_ratio()

        outputs = model(
            **inputs,
            vision_labels=vision_labels if need_vision_loss else None,
            vision_mask=vision_mask,
            target_image_seq_lens=target_image_seq_lens_tensor,
            target_image_counts=target_image_counts_tensor,
            vision_features=packed_vision_features,
            scheduled_sampling_enabled=self.args.schedule_sampling_enabled and need_vision_loss,
            scheduled_sampling_ratio=schedule_sampling_ratio,
        )
        
        loss = outputs.loss

        self._micro_step += 1
        is_last_microbatch = self._micro_step % self.args.gradient_accumulation_steps == 0
        if model.training and is_last_microbatch and self.state.global_step % self.args.logging_steps == 0:
            custom_logs = {}
            if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                custom_logs["lm_loss"] = outputs.lm_loss.item()
            if hasattr(outputs, "vision_loss") and outputs.vision_loss is not None:
                v_loss = outputs.vision_loss
                if v_loss is not None:
                    custom_logs["vision_loss"] = v_loss.item()
            if self.args.schedule_sampling_enabled:
                custom_logs["schedule_sampling_ratio"] = schedule_sampling_ratio
            if len(custom_logs) > 0:
                self.log(custom_logs)

        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._clear_eval_buffer()
        
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        global_metrics = self._sync_custom_metrics()
        
        if global_metrics["lm_loss_count"] > 0:
            output[f"{metric_key_prefix}_lm_loss"] = global_metrics["lm_loss_sum"] / global_metrics["lm_loss_count"]

        if global_metrics["vis_loss_count"] > 0:
            output[f"{metric_key_prefix}_vision_loss"] = global_metrics["vis_loss_sum"] / global_metrics["vis_loss_count"]

        if self.is_world_process_zero():
            print(f"\n[Eval Metrics] "
                  f"Loss: {output.get(f'{metric_key_prefix}_loss', 'N/A'):.4f} | "
                  f"LM: {output.get(f'{metric_key_prefix}_lm_loss', 0.0):.4f} | "
                  f"Vis: {output.get(f'{metric_key_prefix}_vision_loss', 0.0):.4f}\n")
            
        return output

    def _sync_custom_metrics(self):
        if not dist.is_initialized():
            return self.eval_metrics_buffer
        
        device = self.args.device
        metrics_tensor = torch.tensor([
            self.eval_metrics_buffer["lm_loss_sum"],
            self.eval_metrics_buffer["lm_loss_count"],
            self.eval_metrics_buffer["vis_loss_sum"],
            self.eval_metrics_buffer["vis_loss_count"],
        ], dtype=torch.float64, device=device)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        return {
            "lm_loss_sum": metrics_tensor[0].item(),
            "lm_loss_count": metrics_tensor[1].item(),
            "vis_loss_sum": metrics_tensor[2].item(),
            "vis_loss_count": metrics_tensor[3].item(),
        }
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = True,
        ignore_keys = None,
    ):
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                batch_size = inputs["input_ids"].shape[0]

                if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                    # outputs.lm_loss is usually a mean scalar, so multiply by batch size to recover a sum.
                    self.eval_metrics_buffer["lm_loss_sum"] += outputs.lm_loss.mean().item() * batch_size
                    self.eval_metrics_buffer["lm_loss_count"] += batch_size
                
                # Accumulate local vision loss.
                if hasattr(outputs, "vision_loss") and outputs.vision_loss is not None:
                    self.eval_metrics_buffer["vis_loss_sum"] += outputs.vision_loss.mean().item() * batch_size
                    self.eval_metrics_buffer["vis_loss_count"] += batch_size

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels")
        
        return (loss, logits, labels)

    
class EvalCallback(TrainerCallback):
    def __init__(
        self, 
        eval_dataset, 
        data_collator,
        model, 
        tokenizer, 
        processor, 
        num_samples=8, 
        latent_pad_token="<|latent_pad|>", 
        latent_seq_len=36,
    ):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_samples = num_samples
        self.sample_indices = torch.randperm(len(self.eval_dataset))[:self.num_samples].tolist()
        self.latent_pad_token = latent_pad_token
        self.stop_token_id = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None:
            self.stop_token_id.append(im_end_id)
        self.latent_seq_len = latent_seq_len

    def on_evaluate(self, args, state, control, **kwargs):
        # Ensure all ranks reach the evaluation entry point to avoid collective mismatches.
        if dist.is_initialized():
            dist.barrier()

        if not state.is_world_process_zero:
            if dist.is_initialized():
                dist.barrier()
            return
        
        print(f"\n{'='*20} [Native Generate Sanity Check] {'='*20}", flush=True)
        
        inference_model = self.model.module if hasattr(self.model, "module") else self.model
        inference_model.eval()
        device = args.device

        # Sanity check
        test_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(test_prompt, return_tensors="pt").to(device)

        print(f">>> [Sanity Prompt]: {test_prompt.strip()}")

        try:
            with torch.no_grad():
                response = inference_model.generate_interleaved(
                    input_ids=inputs.input_ids,
                    # attention_mask=inputs.attention_mask,
                    tokenizer=self.tokenizer,
                    max_new_tokens=128,
                    max_latent_tokens=self.latent_seq_len,
                    do_sample=True, 
                    temperature=0.6,
                )

            print(f">>> [Sanity Response]: {response}")

        except Exception as e:
            print(f"❌ [Error in Native Generate]: {e}")
            import traceback
            traceback.print_exc()

        print("="*60 + "\n")

        print(f"\n[Info] Evaluate callback triggered.", flush=True)
        inference_model = self.model.module if hasattr(self.model, "module") else self.model
        inference_model.eval()

        device = args.device

        for i, idx in enumerate(self.sample_indices):
            raw_sample = self.eval_dataset[int(idx)]
            if "prompt_ids" in raw_sample:
                input_ids = raw_sample["prompt_ids"].unsqueeze(0).to(device)
            else:
                print(f"[Warning] No prompt_ids in sample, use input_ids instead")
                input_ids = raw_sample["input_ids"].unsqueeze(0).to(device)

            prompt_str = self.tokenizer.decode(input_ids[0])
            if not prompt_str.endswith("\n"):
                print(f"[Debug] no new_line")
                nl_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
                input_ids = torch.cat([input_ids, torch.tensor([[nl_id]], device=device)], dim=1)

            pixel_values = raw_sample.get("input_pixel_values")
            image_grid_thw = raw_sample.get("input_image_grid_thw")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
                image_grid_thw = image_grid_thw.to(device)

            gt_text = "N/A"
            if "ground_truth_ids" in raw_sample:
                gt_text = self.tokenizer.decode(raw_sample["ground_truth_ids"], skip_special_tokens=False)
                # Simplify display by compacting repeated pad tokens.
                pad_str = self.latent_pad_token * 2
                if pad_str in gt_text:
                    gt_text = gt_text.replace(self.latent_pad_token, "")

            prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if "<|im_start|>user" in prompt_text:
                display_prompt = prompt_text.split("<|im_start|>user")[-1].strip()[-200:]
            else:
                display_prompt = prompt_text[-200:]

            print(f"\n>>> [Sample {i} | ID {idx}]")
            print(f">>> [Prompt]: ...{display_prompt}")
            print(f">>> [Has Image Input]: {'YES' if pixel_values is not None else 'NO'}")

            if "<|im_start|>assistant" not in prompt_text.split("<|im_start|>")[-1]:
                im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                assistant_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
                nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
                trigger_ids = torch.tensor(
                    [[im_start_id] + assistant_ids + nl_ids],
                    device=device
                )
                input_ids = torch.cat([input_ids, trigger_ids], dim=1)


            with torch.no_grad():
                generated_output = inference_model.generate_interleaved(
                    input_ids=input_ids,
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=1024,
                    max_latent_tokens=self.latent_seq_len,
                    do_sample=True,
                    temperature=0.6,
                )

            print(f">>> [Generated]:\n{generated_output}")
            print(f">>> [Ground Truth]:\n{gt_text}...")
        
        inference_model.train()
        # Let other ranks proceed after generation completes
        if dist.is_initialized():
            dist.barrier()

def get_gap_optim_groups(model, base_lr, weight_decay, vision_head_lr=None):
    if vision_head_lr is None:
        vision_head_lr = base_lr

    base_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision_latent_head" in name:
            head_params.append(param)
        else:
            base_params.append(param)

    optim_groups = [
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay}
    ]
    if len(head_params) > 0:
        optim_groups.append({"params": head_params, "lr": vision_head_lr, "weight_decay": weight_decay})

    return optim_groups

def get_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, GapTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    train_args.remove_unused_columns = False
    train_args.ddp_broadcast_buffers = False
    train_args.ddp_find_unused_parameters = False

    return model_args, data_args, train_args


class SaveTokenizerProcessorCallback(TrainerCallback):
    """Save tokenizer and processor into each checkpoint directory when Trainer saves a checkpoint.

    This ensures tokenizer and processor are paired with model checkpoints for later reload.
    """
    def __init__(self, tokenizer, processor):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor

    def _save_to_dir(self, dir_path: str):
        try:
            os.makedirs(dir_path, exist_ok=True)
            self.tokenizer.save_pretrained(dir_path)
        except Exception as e:
            print(f"[SaveCallback] Failed to save tokenizer to {dir_path}: {e}")
        try:
            if hasattr(self.processor, 'save_pretrained'):
                self.processor.save_pretrained(dir_path)
        except Exception as e:
            print(f"[SaveCallback] Failed to save processor to {dir_path}: {e}")

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._save_to_dir(ckpt_dir)

    def on_train_end(self, args, state, control, **kwargs):
        self._save_to_dir(args.output_dir)

def main():
    model_args, data_args, train_args = get_args()
    if train_args.process_index == 0:
        print(f"[Info] Model args: {model_args}", flush=True)
        print(f"[Info] Data args: {data_args}", flush=True)
        print(f"[Info] Train args: {train_args}", flush=True)
    if train_args.seed:
        set_seed(train_args.seed)
    else:
        set_seed(42)

    os.makedirs(train_args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    config = GapConfig.from_pretrained(model_args.model_name_or_path)

    if not model_args.no_latent:
        new_tokens = [model_args.latent_start_token, model_args.latent_end_token, model_args.latent_pad_token]
        tokenizer.add_tokens(new_tokens)
        _ = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        processor.tokenizer.add_tokens(new_tokens)
        _ = processor.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        if train_args.process_index == 0:
            print(f"[Info] Added new tokens: {new_tokens}")
        config.latent_start_token_id = tokenizer.convert_tokens_to_ids(model_args.latent_start_token)
        config.latent_end_token_id = tokenizer.convert_tokens_to_ids(model_args.latent_end_token)
        config.latent_pad_token_id = tokenizer.convert_tokens_to_ids(model_args.latent_pad_token)
    else:
        if train_args.process_index == 0:
            print(f"[Info] Pure-text mode: skipping latent token addition.")
        config.latent_start_token_id = None
        config.latent_end_token_id = None
        config.latent_pad_token_id = None

    config.vision_pca_mu_path = model_args.vision_pca_mu_path
    config.vision_pca_W_path = model_args.vision_pca_W_path
    config.vision_head_mode = model_args.vision_head_mode
    config.vision_head_detach = model_args.detach_vision_hidden
    config.schedule_sampling_enabled = train_args.schedule_sampling_enabled
    config.schedule_sampling_ratio = train_args.schedule_sampling_ratio
    config.model_type = "gap"
    config.architectures = ["GapForConditionalGeneration"]

    # Decouple vision token length from image resolution.
    # In Qwen2.5-VL, image_grid_thw uses a 2x2 merge, so:
    #   latent_seq_len == (target_h // 2) * (target_w // 2)
    latent_seq_len = data_args.latent_seq_len
    side = int(latent_seq_len ** 0.5)
    if side * side != latent_seq_len:
        raise ValueError(
            f"latent_seq_len must be a perfect square (e.g., 4/16/36/64/144), got {latent_seq_len}."
        )
    merge_factor = 2
    config.target_h = side * merge_factor
    config.target_w = side * merge_factor

    no_latent_loss_effective = train_args.no_latent_loss or model_args.no_latent
    config.vision_loss_weight = model_args.latent_loss_weight if not no_latent_loss_effective else 0.0
    config.is_transformer_block = model_args.is_transformer_block
    config.num_transformer_layers = model_args.num_transformer_layers
    config.use_sinkhorn_loss = train_args.use_sinkhorn_loss

    if train_args.process_index == 0:
        print(f"[Info] Initializing GAP Model...", flush=True)

    model = GapForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True
    )

    if not model_args.no_latent:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    model.freeze_vision_model(
        freeze_llm=model_args.freeze_llm,
        freeze_lm_head=model_args.freeze_lm_head,
        freeze_embed=model_args.freeze_embed,
    )
    model.gradient_checkpointing_enable()

    if not model_args.no_latent:
        if train_args.process_index == 0:
            print(f"[Info] Initializing new vision token embeddings...", flush=True)
        with torch.no_grad():
            input_emb_layer = model.get_input_embeddings()
            output_emb_layer = model.get_output_embeddings()

            in_mean, in_std = input_emb_layer.weight.mean(), input_emb_layer.weight.std()
            out_mean, out_std = output_emb_layer.weight.mean(), output_emb_layer.weight.std()

            latent_tokens = [
                model.config.latent_start_token_id,
                model.config.latent_pad_token_id,
                model.config.latent_end_token_id
            ]

            for tid in latent_tokens:
                if tid is not None:
                    input_emb_layer.weight[tid].normal_(mean=in_mean, std=in_std)
                    output_emb_layer.weight[tid].normal_(mean=out_mean, std=out_std)

        if train_args.process_index == 0:
            print(f"[Info] Initialized {len(latent_tokens)} tokens with noise.")

        latent_start_id = model.config.latent_start_token_id
        emb_weight = model.get_input_embeddings().weight[latent_start_id]
        print(f"[Debug] Gen Start Embedding - Mean: {emb_weight.mean()}, Std: {emb_weight.std()}, Max: {emb_weight.max()}")
        if torch.isnan(emb_weight).any() or emb_weight.abs().max() > 100:
            print("[ALARM] Embedding initialization is BROKEN!")

    case_probs = [float(p) for p in data_args.case_probs.split(",")]
    train_dataset = InterleaveDataset(
        data_paths=[data_args.train_meta_filepath],
        tokenizer=tokenizer,
        processor=processor,
        max_seq_len=data_args.data_max_seq_len,
        image_root_dirs=json.loads(data_args.image_root_dirs),
        latent_seq_len=data_args.latent_seq_len,
        latent_start_token=model_args.latent_start_token,
        latent_end_token=model_args.latent_end_token,
        latent_pad_token=model_args.latent_pad_token,
        parser_start_token=data_args.parser_start_token,
        parser_end_token=data_args.parser_end_token,
        parse_prompt=data_args.parse_prompt,
        is_train=True,
        learn_footer_tokens=True,
        no_latent_loss=no_latent_loss_effective,
        no_split=data_args.no_split,
        case_probs=case_probs,
        pure_text_mode=model_args.no_latent,
        acc_threshold=data_args.acc_threshold,
    )

    eval_dataset = InterleaveDataset(
        data_paths=[data_args.val_meta_filepath],
        tokenizer=tokenizer,
        processor=processor,
        max_seq_len=data_args.data_max_seq_len,
        image_root_dirs=json.loads(data_args.image_root_dirs),
        latent_seq_len=data_args.latent_seq_len,
        latent_start_token=model_args.latent_start_token,
        latent_end_token=model_args.latent_end_token,
        latent_pad_token=model_args.latent_pad_token,
        parser_start_token=data_args.parser_start_token,
        parser_end_token=data_args.parser_end_token,
        parse_prompt=data_args.parse_prompt,
        is_train=False,
        no_latent_loss=no_latent_loss_effective,
        no_split=data_args.no_split,
        case_probs=case_probs,
        pure_text_mode=model_args.no_latent,
        acc_threshold=data_args.acc_threshold,
    )

    if data_args.debug_mode:
        from torch.utils.data import Subset
        eval_dataset = Subset(eval_dataset, range(data_args.debug_evalsubset_size))

    if train_args.process_index == 0:
        print(f"[Info] Initializing data collator, eval_callback, and trainer...")
    data_collator = InterleaveDataCollator(tokenizer=tokenizer)

    eval_callback = EvalCallback(
        eval_dataset, 
        data_collator, 
        model, 
        tokenizer, 
        processor, 
        num_samples=8, 
        latent_pad_token=model_args.latent_pad_token,
        latent_seq_len=data_args.latent_seq_len,
    )

    save_tp_callback = SaveTokenizerProcessorCallback(tokenizer, processor)

    optim_groups = get_gap_optim_groups(
        model, 
        train_args.learning_rate, 
        train_args.weight_decay,
        vision_head_lr=train_args.vision_head_lr,
    )
    optimizer = torch.optim.AdamW(optim_groups)

    trainer = GapTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[eval_callback, save_tp_callback],
        optimizers=(optimizer, None)
    )

    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint

    if train_args.process_index == 0:
        print("[Info] Start training...", flush=True)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if train_args.process_index == 0:
        print("[Info] Training completed.", flush=True)

if __name__ == "__main__":
    main()
