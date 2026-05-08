import os
import re
import types
from dataclasses import dataclass
from typing import Optional, Union, List, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from transformers import Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel, Qwen2_5_VLForConditionalGeneration

from .configuration_gap import GapConfig
logger = logging.get_logger(__name__)


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

def _is_valid_local_file(path: Optional[str]) -> bool:
    return isinstance(path, str) and len(path.strip()) > 0 and os.path.isfile(path)

@dataclass
class GapCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    # record loss
    lm_loss: Optional[torch.FloatTensor] = None
    vision_loss: Optional[torch.FloatTensor] = None

def sinkhorn_loss(pred, target, epsilon=0.2, num_iters=5):
    pred = pred.float()
    target = target.float()
    B, L_p, _ = pred.shape
    _, L_t, _ = target.shape
    device = pred.device

    cost = torch.cdist(pred, target, p=2) ** 2

    log_a = -torch.log(torch.tensor(L_p, device=device, dtype=torch.float32))
    log_b = -torch.log(torch.tensor(L_t, device=device, dtype=torch.float32))

    f = torch.zeros((B, L_p), device=device)
    g = torch.zeros((B, L_t), device=device)

    for _ in range(num_iters):
        # f_n+1 = epsilon * [log_a - logsumexp((g - C)/epsilon)]
        f = epsilon * (log_a - torch.logsumexp((g.unsqueeze(1) - cost) / epsilon, dim=2))
        g = epsilon * (log_b - torch.logsumexp((f.unsqueeze(2) - cost) / epsilon, dim=1))

    log_P = (f.unsqueeze(2) + g.unsqueeze(1) - cost) / epsilon
    loss = torch.sum(torch.exp(log_P) * cost, dim=(1, 2))
    
    return loss.mean()


class LatentMLP(nn.Module):
    def __init__(self, in_dim: int, intermediate_size: int, out_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, out_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LatentProjectorHead(nn.Module):
    """Linear projector to reduce dimension, followed by LatentMLP for non-linearity."""
    def __init__(self, in_dim: int, proj_dim: int, out_dim: int):
        super().__init__()
        self.linear_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.latent_mlp = LatentMLP(proj_dim, proj_dim * 4, out_dim)

    def forward(self, x):
        return self.latent_mlp(self.linear_proj(x))


class GapForConditionalGeneration(Qwen2_5_VLForConditionalGeneration, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # self.model = Qwen2_5_VLModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        self.config = config
        # Fixed vision target size
        self.target_h = config.target_h
        self.target_w = config.target_w
        self.target_vision_tokens = int(self.target_h * self.target_w)
        # Special tokens
        self.latent_start_token_id = config.latent_start_token_id
        self.latent_end_token_id = config.latent_end_token_id
        self.latent_pad_token_id = config.latent_pad_token_id
        # Cache rope_deltas as a plain Python object to avoid Long tensors
        # being attached to the model state/parameters.
        self._rope_deltas_cache = None
        self.schedule_sampling_enabled = bool(getattr(config, "schedule_sampling_enabled", False))
        self.schedule_sampling_ratio = float(getattr(config, "schedule_sampling_ratio", 0.0))
        self.schedule_sampling_detach_features = bool(getattr(config, "schedule_sampling_detach_features", True))
        env_vision_head_mode = os.environ.get("VISION_HEAD_MODE")
        configured_vision_head_mode = getattr(config, "vision_head_mode", None)
        self.vision_head_mode = str(env_vision_head_mode or configured_vision_head_mode or "none").lower()

        mu_path = getattr(config, "vision_pca_mu_path", None)
        W_path = getattr(config, "vision_pca_W_path", None)
        self.use_vision_pca = _is_valid_local_file(mu_path) and _is_valid_local_file(W_path)
        self.vision_feature_size = getattr(config, "vision_feature_size", config.hidden_size)

        if self.use_vision_pca:
            mu = torch.from_numpy(np.load(mu_path)).float()
            self.register_buffer("vision_pca_mu", mu)

            W = torch.from_numpy(np.load(W_path)).float()
            self.register_buffer("vision_pca_W", W)

            self.vision_output_dim = W.shape[1]
            if self.vision_head_mode is None or str(self.vision_head_mode).lower() == "none":
                self.vision_latent_head = None
            else:
                in_dim, out_dim = W.shape[0], W.shape[1]
                self.vision_latent_head = LatentProjectorHead(in_dim, out_dim, out_dim)
                # Initialize the first linear layer with PCA basis transpose (W^T shape: k x d)
                # so that at init the MLP approximates the PCA projection of the hidden state.
                self.vision_latent_head.linear_proj.weight.data.copy_(W.transpose(0, 1))
            logger.info(
                "GAP vision PCA mode enabled. mu=%s, W=%s, latent_dim=%s, vision_head_mode=%s",
                mu_path,
                W_path,
                self.vision_output_dim,
                self.vision_head_mode,
            )
        else:
            self.vision_pca_mu = None
            self.vision_pca_W = None
            self.vision_output_dim = config.hidden_size
            if self.vision_head_mode is None or str(self.vision_head_mode).lower() == "none":
                self.vision_latent_head = None
            else:
                self.vision_latent_head = LatentProjectorHead(config.hidden_size, self.vision_feature_size, config.hidden_size)
            logger.warning(
                "GAP vision PCA files are missing or not provided; falling back to direct vision hidden prediction. "
                "mu_path=%s, W_path=%s, vision_head_mode=%s",
                mu_path,
                W_path,
                self.vision_head_mode,
            )

        self.post_init()

    # --- Cross-version compatibility shims ---
    # transformers 4.52 nested visual/language_model under self.model and moved
    # get_rope_index / rope_deltas there too. 4.49 keeps visual at top level,
    # treats self.model as the LM, and exposes get_rope_index on self.
    def _qwen_visual(self):
        if hasattr(self.model, "visual"):
            return self.model.visual
        return self.visual

    def _qwen_language_model(self):
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        return self.model

    def _qwen_get_rope_index(self, *args, **kwargs):
        if hasattr(self.model, "get_rope_index"):
            return self.model.get_rope_index(*args, **kwargs)
        return super().get_rope_index(*args, **kwargs)

    def _qwen_clear_rope_deltas(self):
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        if hasattr(self, "rope_deltas") and not isinstance(getattr(type(self), "rope_deltas", None), property):
            try:
                self.rope_deltas = None
            except Exception:
                pass

    def _qwen_filter_model_kwargs(self, **kwargs):
        """Drop kwargs not accepted by self.model.forward (vision-related kwargs were
        added to Qwen2_5_VLModel.forward in transformers 4.52; on 4.49 they belong to
        the wrapper class instead and must not be forwarded to the LM)."""
        if not hasattr(self, "_qwen_supported_model_kwargs"):
            import inspect
            try:
                params = inspect.signature(self.model.forward).parameters
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                self._qwen_supported_model_kwargs = None if has_var_kw else set(params.keys())
            except (TypeError, ValueError):
                self._qwen_supported_model_kwargs = None
        supported = self._qwen_supported_model_kwargs
        if supported is None:
            return kwargs
        return {k: v for k, v in kwargs.items() if k in supported}

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values, images_crop, images_spatial_crop):
        return self.model.get_image_features(pixel_values, images_crop, images_spatial_crop)

    def freeze_vision_model(self, freeze_llm=True, freeze_lm_head=True, freeze_embed=True):
        for param in self._qwen_visual().parameters():
            param.requires_grad = False

        for param in self._qwen_language_model().parameters():
            param.requires_grad = not freeze_llm

        self._qwen_language_model().embed_tokens.weight.requires_grad = not freeze_embed
        self.lm_head.weight.requires_grad = not freeze_lm_head

        print(f"[Info] Vision model is frozen.")

    def extract_vision_gt(self, pixel_values, grid_thw):
        vision_features = self._qwen_visual()(pixel_values, grid_thw=grid_thw)
        batch_size = grid_thw.shape[0]
        targets = []
        pos = 0
        # Split fixed-length vision features per sample.
        for i in range(batch_size):
            h, w = grid_thw[i, 1] // 2, grid_thw[i, 2] // 2
            seq_len = h * w

            features = vision_features[pos: pos + seq_len]
            pos += seq_len
            targets.append(features)

        return torch.stack(targets)

    def _encode_vision_target_with_pca(self, target):
        if not self.use_vision_pca:
            return target
        mu = self.vision_pca_mu.to(device=target.device, dtype=target.dtype)
        W = self.vision_pca_W.to(device=target.device, dtype=target.dtype)
        return (target - mu.unsqueeze(0)) @ W

    def _decode_vision_latent_with_pca(self, latent):
        if not self.use_vision_pca:
            return latent
        mu = self.vision_pca_mu.to(device=latent.device, dtype=latent.dtype)
        W = self.vision_pca_W.to(device=latent.device, dtype=latent.dtype)
        return (latent @ W.transpose(0, 1)) + mu.unsqueeze(0)

    def _project_vision_prediction(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.vision_latent_head is None:
            return hidden_states
        return self.vision_latent_head(hidden_states)

    def _collect_predicted_vision_segments(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
    ):
        # Shifted-MSE alignment: collect exactly N prediction positions per block,
        # where N = number of <latent_pad> tokens. Positions kept are
        # [<latent_start>, <latent_pad>_1, ..., <latent_pad>_{N-1}]. These are the
        # positions whose projections are consumed at inference (projection at k feeds
        # the embedding at position k+1). <latent_pad>_N and <latent_end> are excluded
        # because their projections are never used during generate_interleaved.
        latent_start_mask = (input_ids == self.latent_start_token_id) if self.latent_start_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        latent_pad_mask = (input_ids == self.latent_pad_token_id) if self.latent_pad_token_id is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        union_mask = latent_start_mask | latent_pad_mask

        batch_size = input_ids.shape[0]
        pred_per_sample = []

        for b in range(batch_size):
            union_positions = torch.nonzero(union_mask[b], as_tuple=False).squeeze(1)
            if union_positions.numel() == 0:
                pred_per_sample.append([])
                continue

            sample_hidden = hidden_states[b, union_positions]
            start_positions = torch.nonzero(latent_start_mask[b], as_tuple=False).squeeze(1)
            if start_positions.numel() == 0:
                pred_per_sample.append([])
                continue

            start_indices = torch.searchsorted(union_positions, start_positions)
            per_img = []
            for i in range(start_indices.numel()):
                s_idx = int(start_indices[i].item())
                e_idx = int(start_indices[i + 1].item()) if i + 1 < start_indices.numel() else sample_hidden.shape[0]
                block = sample_hidden[s_idx:e_idx]
                if block.shape[0] >= 1:
                    block = block[:-1]
                per_img.append(block)
            pred_per_sample.append(per_img)

        return pred_per_sample

    def _build_target_vision_segments(
        self,
        input_ids: torch.LongTensor,
        vision_labels: torch.FloatTensor,
        vision_mask: Optional[torch.LongTensor] = None,
        target_image_seq_lens: Optional[torch.LongTensor] = None,
        target_image_counts: Optional[torch.LongTensor] = None,
        pred_per_sample: Optional[list] = None,
    ):
        latent_start_mask = (input_ids == self.latent_start_token_id)
        batch_size = input_ids.shape[0]

        if target_image_counts is None:
            target_image_counts = latent_start_mask.sum(dim=1)

        seq_lens_flat = target_image_seq_lens.tolist() if target_image_seq_lens is not None else None
        lens_offset = 0
        flat_ptr = 0
        target_per_sample = []


        for b in range(batch_size):
            count = int(target_image_counts[b].item()) if isinstance(target_image_counts, torch.Tensor) else int(target_image_counts[b])
            if count == 0:
                target_per_sample.append([])
                continue
            
            if vision_labels.dim() == 3:
                target_flat = vision_labels[b]
                if vision_mask is not None:
                    sample_mask = vision_mask[b].bool()
                    if sample_mask.dim() > 1:
                        sample_mask = sample_mask.reshape(-1)
                    sample_mask = sample_mask.to(device=target_flat.device).bool()
                
                    target_flat = target_flat[sample_mask]
            elif vision_labels.dim() == 2:
                if seq_lens_flat is not None:
                    sample_lens = seq_lens_flat[lens_offset: lens_offset + count]
                    sample_total_len = sum(int(x) for x in sample_lens)
                else:
                    pred_imgs = pred_per_sample[b] if pred_per_sample is not None else []
                    sample_lens = [int(pred_img.shape[0]) for pred_img in pred_imgs[:count]]
                    sample_total_len = sum(sample_lens)
                
                target_flat = vision_labels[flat_ptr: flat_ptr + sample_total_len]
                flat_ptr += sample_total_len
            else:
                raise ValueError(f"Unexpected vision_labels dimension: {vision_labels.dim()}")
            
            per_img = []
            if seq_lens_flat is not None:
                sample_lens = seq_lens_flat[lens_offset:lens_offset + count]
                lens_offset += count
                ptr = 0
                for seq_len in sample_lens:
                    seq_len = int(seq_len)
                    per_img.append(target_flat[ptr:ptr + seq_len])
                    ptr += seq_len
            else:
                ptr = 0
                pred_imgs = pred_per_sample[b] if pred_per_sample is not None else []
                for pred_img in pred_imgs[:count]:
                    seq_len = int(pred_img.shape[0])
                    per_img.append(target_flat[ptr:ptr + seq_len])
                    ptr += seq_len

            target_per_sample.append(per_img)

        return target_per_sample
            

    def _compute_vision_loss_from_segments(self, pred_per_sample, target_per_sample, device):
        vision_loss_terms = []

        for b, (pred_imgs, tgt_imgs) in enumerate(zip(pred_per_sample, target_per_sample)):
            if len(pred_imgs) == 0 or len(tgt_imgs) == 0:
                continue

            num_pairs = min(len(pred_imgs), len(tgt_imgs))
            for i in range(num_pairs):
                pred_vision = pred_imgs[i]
                target = tgt_imgs[i]

                if pred_vision.numel() == 0 or target.numel() == 0:
                    continue

                if pred_vision.shape[0] != target.shape[0]:
                    raise ValueError(f"Length mismatch between predicted vision features and target for sample {b}, pair {i}: {pred_vision.shape[0]} vs {target.shape[0]}. Consider enabling automatic alignment or check the data preprocessing.")

                projected_pred = self._project_vision_prediction(pred_vision)
                recon_pred = self._decode_vision_latent_with_pca(
                    projected_pred.view(-1, projected_pred.shape[-1])
                )
                vision_loss_terms.append(F.mse_loss(recon_pred, target))

        if len(vision_loss_terms) > 0:
            return torch.stack(vision_loss_terms).mean()
        return torch.tensor(0.0, device=device)

    def _flatten_vision_segments(self, segments, reference_tensor: torch.Tensor):
        per_sample_flat = []
        max_len = 0

        for per_img in segments:
            if len(per_img) == 0:
                sample = reference_tensor.new_zeros((0, reference_tensor.shape[-1]))
            else:
                sample = torch.cat(per_img, dim=0)
            per_sample_flat.append(sample)
            max_len = max(max_len, sample.shape[0])

        if max_len == 0:
            return None, None

        batch_size = len(per_sample_flat)
        hidden_size = reference_tensor.shape[-1]
        flat = reference_tensor.new_zeros((batch_size, max_len, hidden_size))
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=reference_tensor.device)

        for b, sample in enumerate(per_sample_flat):
            if sample.shape[0] == 0:
                continue
            flat[b, :sample.shape[0]] = sample
            mask[b, :sample.shape[0]] = True

        return flat, mask

    def _mix_schedule_sampling_segments(
        self,
        teacher_segments,
        student_segments,
        ss_ratio,
    ):
        if teacher_segments is None:
            return student_segments
        if student_segments is None or ss_ratio <= 0.0:
            return teacher_segments
        if ss_ratio >= 1.0:
            return student_segments

        mixed_segments = []
        batch_size = len(teacher_segments)
        

        for b in range(batch_size):
            teacher_imgs = teacher_segments[b]
            student_imgs = student_segments[b]

            if len(teacher_imgs) == 0:
                mixed_segments.append(student_imgs)
                continue
            if len(student_imgs) == 0:
                mixed_segments.append(teacher_imgs)
                continue

            num_pairs = min(len(teacher_imgs), len(student_imgs))
            sample_segments = []

            for i in range(num_pairs):
                teacher_seg = teacher_imgs[i]
                student_seg = student_imgs[i]

                if teacher_seg.shape != student_seg.shape:
                    raise ValueError(
                        f"Teacher/student segment shape mismatch at sample {b}, image {i}: "
                        f"{teacher_seg.shape} vs {student_seg.shape}"
                    )

                use_student = bool(
                    (torch.rand((), device=teacher_seg.device) < ss_ratio).item()
                )
                sample_segments.append(student_seg if use_student else teacher_seg)

            if len(teacher_imgs) > num_pairs:
                sample_segments.extend(teacher_imgs[num_pairs:])
            elif len(student_imgs) > num_pairs:
                sample_segments.extend(student_imgs[num_pairs:])

            mixed_segments.append(sample_segments)

        return mixed_segments

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # vision inputs
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.FloatTensor] = None,
        # other
        rope_deltas: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Optional[int] = 0,
        # labels
        labels: Optional[torch.LongTensor] = None,
        # vision labels and mask
        vision_labels: Optional[torch.FloatTensor] = None,
        vision_mask: Optional[torch.LongTensor] = None,
        target_image_seq_lens: Optional[torch.LongTensor] = None,
        target_image_counts: Optional[torch.LongTensor] = None,
        scheduled_sampling_ratio: Optional[float] = None,
        scheduled_sampling_enabled: Optional[bool] = None,
        schedule_sampling_teacher_force: bool = False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        effective_scheduled_sampling_enabled = self.schedule_sampling_enabled if scheduled_sampling_enabled is None else bool(scheduled_sampling_enabled)
        effective_scheduled_sampling_ratio = self.schedule_sampling_ratio if scheduled_sampling_ratio is None else float(scheduled_sampling_ratio)

        should_run_schedule_sampling = (
            effective_scheduled_sampling_enabled
            and not schedule_sampling_teacher_force
            and vision_labels is not None
            and input_ids is not None
            and self.latent_pad_token_id is not None
            and (input_ids == self.latent_pad_token_id).any()
            and torch.rand(1).item() < effective_scheduled_sampling_ratio
            and labels is not None
        )

        if should_run_schedule_sampling:
            teacher_outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                vision_features=vision_features if vision_features is not None else vision_labels,
                rope_deltas=rope_deltas,
                second_per_grid_ts=second_per_grid_ts,
                logits_to_keep=logits_to_keep,
                labels=None,
                vision_labels=None,
                vision_mask=vision_mask,
                target_image_seq_lens=target_image_seq_lens,
                target_image_counts=target_image_counts,
                scheduled_sampling_ratio=0.0,
                scheduled_sampling_enabled=False,
                schedule_sampling_teacher_force=True,
                **kwargs,
            )

            teacher_hidden_states = teacher_outputs.hidden_states[-1] if teacher_outputs.hidden_states is not None else teacher_outputs.last_hidden_state

            # Collect hidden states at latent positions from the teacher pass,
            # then apply the same vision head + inverse PCA transformation used at inference.
            # This makes the injected features at training time match the distribution seen
            # at inference (decode(project(hidden))), closing the real-features vs.
            # recon-features gap.
            pred_segments = self._collect_predicted_vision_segments(teacher_hidden_states, input_ids)
            recon_segments = []
            for sample_segs in pred_segments:
                sample_recon = []
                for seg in sample_segs:
                    if seg.numel() == 0:
                        sample_recon.append(seg)
                        continue
                    projected = self._project_vision_prediction(seg.detach())
                    recon = self._decode_vision_latent_with_pca(
                        projected.view(-1, projected.shape[-1])
                    ).view_as(seg).detach()
                    sample_recon.append(recon)
                recon_segments.append(sample_recon)

            recon_features, _ = self._flatten_vision_segments(recon_segments, teacher_hidden_states)

            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                vision_features=recon_features,
                rope_deltas=rope_deltas,
                second_per_grid_ts=second_per_grid_ts,
                logits_to_keep=logits_to_keep,
                labels=labels,
                vision_labels=vision_labels,
                vision_mask=vision_mask,
                target_image_seq_lens=target_image_seq_lens,
                target_image_counts=target_image_counts,
                scheduled_sampling_ratio=0.0,
                scheduled_sampling_enabled=False,
                schedule_sampling_teacher_force=True,
                **kwargs,
            )
        
        if position_ids is None and input_ids is not None:
            is_training_latent_vis = (self.latent_pad_token_id is not None) and (input_ids == self.latent_pad_token_id).any()
            if is_training_latent_vis:
                rope_input_ids = input_ids.clone()
                latent_mask = (rope_input_ids == self.latent_pad_token_id)
                rope_input_ids[latent_mask] = self.config.image_token_id
                
                img_start_mask = (input_ids == self.config.image_token_id)
                img_block_starts = img_start_mask & (~torch.roll(img_start_mask, 1, dims=1))
                img_block_starts[:, 0] = img_start_mask[:, 0]
                
                latent_start_mask = (input_ids == self.latent_pad_token_id)
                latent_block_starts = latent_start_mask & (~torch.roll(latent_start_mask, 1, dims=1))
                latent_block_starts[:, 0] = latent_start_mask[:, 0]
                
                input_indices = torch.nonzero(img_block_starts)
                latent_indices = torch.nonzero(latent_block_starts)
                
                input_indices = torch.cat([input_indices, torch.zeros((input_indices.shape[0], 1), device=input_ids.device, dtype=torch.long)], dim=1)
                latent_indices = torch.cat([latent_indices, torch.ones((latent_indices.shape[0], 1), device=input_ids.device, dtype=torch.long)], dim=1)
                
                all_indices = torch.cat([input_indices, latent_indices], dim=0)
                sort_keys = all_indices[:, 0] * input_ids.shape[1] + all_indices[:, 1]
                sorted_order = torch.argsort(sort_keys)
                sorted_records = all_indices[sorted_order]
                
                total_images = sorted_records.shape[0]
                if total_images > 0:
                    device = input_ids.device
                    merged_grid_thw = torch.zeros((total_images, 3), dtype=torch.long, device=device)
                    
                    if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                        is_input = (sorted_records[:, 2] == 0)
                        merged_grid_thw[is_input] = image_grid_thw
                    
                    is_gen = (sorted_records[:, 2] == 1)
                    if is_gen.any():
                        target_grid = torch.tensor([1, self.target_h, self.target_w], dtype=torch.long, device=device)
                        merged_grid_thw[is_gen] = target_grid
                        
                    position_ids, rope_deltas = self._qwen_get_rope_index(
                        input_ids=rope_input_ids,
                        image_grid_thw=merged_grid_thw,
                        video_grid_thw=None,
                        attention_mask=attention_mask
                    )
                    self._qwen_clear_rope_deltas()

        raw_token_embeds = None
        query_text_embeds = None
        query_text_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if input_ids is not None:
            raw_token_embeds = inputs_embeds.clone()
            special_token_ids = {
                self.config.image_token_id,
                getattr(self.config, "vision_start_token_id", None),
                getattr(self.config, "vision_pad_token_id", None),
                getattr(self.config, "video_token_id", None),
                getattr(self.config, "vision_end_token_id", None),
                self.latent_start_token_id,
                self.latent_end_token_id,
                self.latent_pad_token_id,
            }

            query_text_mask = torch.ones_like(input_ids, dtype=torch.bool)
            for token_id in special_token_ids:
                if token_id is not None:
                    query_text_mask &= (input_ids != token_id)
            
            query_text_embeds = inputs_embeds[query_text_mask]
            
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.clone()

        if pixel_values is not None and image_grid_thw is not None:
            with torch.no_grad():
                image_embeds = self._qwen_visual()(
                    pixel_values, grid_thw=image_grid_thw
                )

            image_mask = (input_ids == self.config.image_token_id)
            # Use only the first image_embeds.shape[0] image tokens.
            flat_image_mask = image_mask.view(-1)
            if flat_image_mask.sum() > 0:
                idx = torch.nonzero(flat_image_mask, as_tuple=False)[:image_embeds.shape[0]]
                flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
                flat_inputs_embeds[idx.squeeze()] = image_embeds[:idx.shape[0]].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = flat_inputs_embeds.view_as(inputs_embeds)
            elif image_mask.sum() == image_embeds.shape[0]:
                inputs_embeds[image_mask] = image_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            else:
                raise ValueError(f"Image token number and image embeds number do not match: {image_mask.sum()} vs {image_embeds.shape[0]}")
        
        if vision_features is not None:
            latent_mask = (input_ids == self.latent_pad_token_id)
            if latent_mask.any():
                if vision_mask is not None:
                    if vision_features.dim() == 2:
                        valid_features = vision_features
                    else:
                        valid_features = vision_features[vision_mask.bool()]
                else:
                    hidden_size = vision_features.shape[-1]
                    valid_features = vision_features.view(-1, hidden_size)

                num_slots = latent_mask.sum()
                if valid_features.shape[0] >= num_slots:
                    inputs_embeds[latent_mask] = valid_features[:num_slots].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        pixel_values = None
        pixel_values_videos = None

        outputs = self.model(**self._qwen_filter_model_kwargs(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            second_per_grid_ts=second_per_grid_ts,
        ))

        hidden_states = outputs[0]
        # save query text embeddings and image embeddings 
        if kwargs.get("save_embeddings", False):
            save_dir = kwargs.get("save_embeddings_dir", "./saved_embeddings")
            save_idx = getattr(self, "_save_embeddings_idx", 0)
            self._save_embeddings_idx = save_idx + 1

            embeds_to_save = {}
            if query_text_embeds is not None:
                embeds_to_save["query_text_embeds"] = query_text_embeds.cpu().detach()
            if "image_embeds" in locals() and image_embeds is not None:
                embeds_to_save["image_embeds"] = image_embeds.cpu().detach()
            
            self._save_tensors_dict(
                embeds_to_save,
                dir_path=save_dir,
                file_name=f"embeddings_{self._save_embeddings_idx}.pt"
            )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        lm_loss, vision_loss = None, None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.ignore_index
            )
            loss = lm_loss

        if vision_labels is not None:
            pred_per_sample = self._collect_predicted_vision_segments(hidden_states, input_ids)
            target_per_sample = self._build_target_vision_segments(
                input_ids=input_ids,
                vision_labels=vision_labels,
                vision_mask=vision_mask,
                target_image_seq_lens=target_image_seq_lens,
                target_image_counts=target_image_counts,
                pred_per_sample=pred_per_sample,
            )
            vision_loss = self._compute_vision_loss_from_segments(
                pred_per_sample=pred_per_sample,
                target_per_sample=target_per_sample,
                device=hidden_states.device,
            )

            # Total loss
            vision_loss_weight = getattr(self.config, "vision_loss_weight", 1.0)
            if loss is None:
                loss = vision_loss_weight * vision_loss
            else:
                loss = loss + vision_loss_weight * vision_loss

        # Ensure vision_latent_head always participates in the backward pass
        # so DDP allreduce doesn't hang when some ranks have no vision loss.
        if self.vision_latent_head is not None:
            dummy = sum(p.sum() * 0.0 for p in self.vision_latent_head.parameters())
            loss = loss + dummy if loss is not None else dummy

        if not return_dict:
            output = (hidden_states,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GapCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=getattr(outputs, "rope_deltas", rope_deltas),
            lm_loss=lm_loss.detach() if lm_loss is not None else None,
            vision_loss=vision_loss.detach() if vision_loss is not None else None,
        )
    
    def _save_tensors_dict(self, tensors_dict, dir_path, file_name):
        if dir_path is None:
            dir_path = "./saved_tensors"
        os.makedirs(dir_path, exist_ok=True)
        
        cpu_dict = {}
        for k, v in tensors_dict.items():
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.float().detach().cpu()
            else:
                cpu_dict[k] = v
        
        torch.save(cpu_dict, os.path.join(dir_path, file_name))
    
    def _sample_token(
        self, 
        logits, 
        history_ids=None,
        temperature=0.7, 
        top_p=0.9, 
        repetition_penalty=1.0,
        do_sample=True
    ):

        logits = torch.nan_to_num(logits, nan=float("-inf"), posinf=1e5, neginf=-1e5)

        batch_size = logits.shape[0]
        vocab_size = logits.shape[1]

        # Normalize history to per-sample lists
        per_sample_hist = [None] * batch_size
        if history_ids is None:
            per_sample_hist = [None] * batch_size
        elif isinstance(history_ids, list) and len(history_ids) == batch_size and all(isinstance(h, list) for h in history_ids):
            per_sample_hist = history_ids
        else:
            # treat as shared history for all samples (history_ids may be list/tensor of tokens)
            if isinstance(history_ids, torch.Tensor):
                try:
                    shared = history_ids.flatten().tolist()
                except Exception:
                    shared = []
            elif isinstance(history_ids, list):
                shared = history_ids
            else:
                shared = []
            per_sample_hist = [shared] * batch_size

        # We'll sample per-row to honor per-sample histories and top-p/temperature correctly.
        next_tokens = []
        for b in range(batch_size):
            row = logits[b].clone()
            hist = per_sample_hist[b] or []
            if repetition_penalty > 1.0 and len(hist) > 0:
                try:
                    hist_tensor = torch.tensor(hist[-200:], dtype=torch.long, device=row.device)
                except Exception:
                    hist_tensor = torch.tensor([], dtype=torch.long, device=row.device)
                if hist_tensor.numel() > 0:
                    for token_id in torch.unique(hist_tensor).tolist():
                        if token_id < 0 or token_id >= vocab_size:
                            continue
                        val = row[token_id]
                        if val < 0:
                            row[token_id] = val * repetition_penalty
                        else:
                            row[token_id] = val / repetition_penalty

            # If greedy
            if not do_sample:
                next_tokens.append(torch.argmax(row).unsqueeze(0))
                continue

            # temperature
            proc = row / temperature if (temperature > 0 and temperature != 1.0) else row

            # top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(proc, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...] = sorted_indices_to_remove.clone()
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                proc[indices_to_remove] = -float('inf')

            probs = torch.softmax(proc, dim=-1)
            # fallback if numerical issues
            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
                next_tokens.append(torch.argmax(row).unsqueeze(0))
            else:
                tok = torch.multinomial(probs, num_samples=1)
                next_tokens.append(tok)

        return torch.stack(next_tokens, dim=0)
    
    def _build_judge_prompt_text(self, few_shot_examples=None, include_image=True):
        parts = [
            "You are a judge deciding whether latent visual reasoning should be kept.",
            "For each example, compare the plain text-only answer and the interleaved answer.",
            "Output exactly one label: KEEP_LATENT or DROP_LATENT.",
            "",
        ]

        if few_shot_examples is not None:
            for i, example in enumerate(few_shot_examples, 1):
                parts.append(f"Example {i}:")
                parts.append(f"Question: {example.get('question', '')}")
                if include_image and example.get("image", None):
                    parts.append("Query Image: <IMAGE>")
                if example.get("text_response", None) is not None:
                    parts.append(f"Text-only Answer: {example.get('text_response', '')}")
                if example.get("interleave_response", None) is not None:
                    parts.append(f"Interleaved Answer: {example.get('interleave_response', '')}")
                parts.append(f"Decision: {example.get('decision', 'KEEP_LATENT')}")
                parts.append("")

        parts.append("Now judge the current case.")
        parts.append("Question: {QUESTION}")
        if include_image:
            parts.append("Query Image: <IMAGE>")
        parts.append("Latent Features: <LATENT>")
        parts.append("Decision:")
        return "\n".join(parts)

    def _build_judge_input_ids(
        self,
        tokenizer,
        question,
        num_latent_tokens,
        few_shot_examples=None,
        image_grid_thw=None,
        device=None,
    ):
        num_images = image_grid_thw.shape[0] if image_grid_thw is not None else 0
        num_fs_images = sum(1 for ex in (few_shot_examples or []) if ex.get("image", None)) if few_shot_examples is not None else 0
        include_image = num_images > num_fs_images

        prompt_text = self._build_judge_prompt_text(few_shot_examples, include_image)
        prompt_text = prompt_text.replace("{QUESTION}", question)

        if "<LATENT>" not in prompt_text:
            raise ValueError("Prompt template must contain <LATENT> placeholder")
        
        if include_image and "<IMAGE>" not in prompt_text:
            raise ValueError("Prompt template must contain <IMAGE> placeholder when num_image_tokens > 0")
        
        image_placeholder = "IMAGE_BLOCK"
        latent_placeholder = "LATENT_BLOCK"

        if include_image:
            prompt_text = prompt_text.replace("<IMAGE>", image_placeholder)
        prompt_text = prompt_text.replace("<LATENT>", latent_placeholder)
        
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        image_token_id = self.config.image_token_id

        latent_start_token_id = self.config.latent_start_token_id
        latent_end_token_id = self.config.latent_end_token_id
        latent_pad_token_id = self.config.latent_pad_token_id

        token_ids = []
        ptr = 0
        img_idx = 0

        while ptr < len(prompt_text):
            next_image = prompt_text.find(image_placeholder, ptr)
            next_latent = prompt_text.find(latent_placeholder, ptr)

            candidates = []
            if next_image != -1:
                candidates.append((next_image, "image"))
            
            if next_latent != -1:
                candidates.append((next_latent, "latent"))

            if len(candidates) == 0:
                tail_text = prompt_text[ptr:]
                if tail_text:
                    token_ids.extend(
                        tokenizer.encode(tail_text, add_special_tokens=False)
                    )
                break
            
            next_pos, next_type = min(candidates, key=lambda x: x[0])

            # tokenize the plain text before the next placeholder
            if next_pos > ptr:
                text_chunk = prompt_text[ptr:next_pos]
                token_ids.extend(tokenizer.encode(text_chunk, add_special_tokens=False))
            
            if next_type == "image":
                num_img_toks = 0
                if image_grid_thw is not None and img_idx < num_images:
                    num_img_toks = int(image_grid_thw[img_idx].prod().item())

                if vision_start_token_id is not None:
                    token_ids.append(vision_start_token_id)
                token_ids.extend([image_token_id] * num_img_toks)
                if vision_end_token_id is not None:
                    token_ids.append(vision_end_token_id)
                ptr = next_pos + len(image_placeholder)
                img_idx += 1
            else:
                token_ids.append(latent_start_token_id)
                token_ids.extend([latent_pad_token_id] * num_latent_tokens)
                token_ids.append(latent_end_token_id)
                ptr = next_pos + len(latent_placeholder)

        return torch.tensor([token_ids], dtype=torch.long, device=device)
    
    @torch.no_grad()
    def _reprefill_from_history(
        self,
        history_ids,
        device,
        pixel_values=None,
        image_grid_thw=None,
    ):
        # Rebuild KV cache from the kept history only.
        # Original image inputs are preserved; latent features are intentionally not reused.
        self._rope_deltas_cache = None

        seq_len = history_ids.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(history_ids.shape[0], -1)
        attention_mask = torch.ones_like(history_ids, dtype=torch.long, device=device)

        model_inputs = self.prepare_inputs_for_generation(
            input_ids=history_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            is_first_iteration=True,
            cache_position=cache_position,
        )

        return self.forward(**model_inputs)
    
    @torch.no_grad()
    def _judge_latent_with_image(
        self,
        tokenizer,
        question,
        latent_features,
        pixel_values=None,
        image_grid_thw=None,
        few_shot_examples=None,
        max_new_tokens=10,
    ):
        # True: keep latent; False: drop latent
        if latent_features is None or latent_features.numel() == 0:
            return True
        
        num_latent_tokens = latent_features.shape[1]
        has_image = pixel_values is not None and image_grid_thw is not None and image_grid_thw.numel() > 0


        judge_input_ids = self._build_judge_input_ids(
            tokenizer=tokenizer,
            question=question,
            num_latent_tokens=num_latent_tokens,
            few_shot_examples=few_shot_examples,
            image_grid_thw=image_grid_thw if has_image else None,
            device=latent_features.device,
        )

        seq_len = judge_input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=judge_input_ids.device)
        position_ids = cache_position.unsqueeze(0).expand(judge_input_ids.shape[0], -1)
        attention_mask = torch.ones_like(judge_input_ids, dtype=torch.long, device=judge_input_ids.device)

        latent_grid = torch.tensor([[1, self.target_h, self.target_w]], dtype=torch.long, device=latent_features.device)
        if has_image:
            combined_grid_thw = torch.cat([image_grid_thw, latent_grid], dim=0)
        else:
            combined_grid_thw = latent_grid

        # Judge should see original query and question image, and generated latent features
        model_inputs = self.prepare_inputs_for_generation(
            input_ids=judge_input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values if has_image else None,
            image_grid_thw=image_grid_thw if has_image else None,
            vision_features=latent_features,
            use_cache=True,
            is_first_iteration=True,
            cache_position=cache_position,
            full_grid_thw=combined_grid_thw,
        )
        outputs = self.forward(**model_inputs)

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        generated = []
        
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            current_seq_len = past_key_values[0][0].shape[-2]
            step_cache_position = torch.tensor([current_seq_len], device=judge_input_ids.device)
            step_position_ids = step_cache_position.unsqueeze(0).expand(1, -1)
            step_attention_mask = torch.ones(
                (1, current_seq_len + 1),
                dtype=torch.long,
                device=judge_input_ids.device,
            )

            model_inputs = self.prepare_inputs_for_generation(
                input_ids=next_token,
                past_key_values=past_key_values,
                attention_mask=step_attention_mask,
                position_ids=step_position_ids,
                use_cache=True,
                is_first_iteration=False,
                cache_position=step_cache_position,
            )
            outputs = self.forward(**model_inputs)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        judge_text = tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
        print(f"[Debug] Judge output: {judge_text}")
        if "KEEP_LATENT" in judge_text:
            return True
        
        return False

    @torch.no_grad()
    def _rebuild_cache_with_zero_latent(
        self,
        pre_latent_history_ids,
        zero_latent,
        device,
        pixel_values=None,
        image_grid_thw=None,
        running_grid_thw=None,
    ):
        # Rebuild cache with the same interleaved latent scaffold, but replace latent features with zeros.
        self._rope_deltas_cache = None

        batch_size = pre_latent_history_ids.shape[0]
        latent_len = int(zero_latent.shape[1])

        latent_start = torch.full(
            (batch_size, 1),
            self.latent_start_token_id,
            dtype=torch.long,
            device=device,
        )
        latent_pad = torch.full(
            (batch_size, latent_len),
            self.latent_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        latent_end = torch.full(
            (batch_size, 1),
            self.latent_end_token_id,
            dtype=torch.long,
            device=device,
        )

        rebuilt_input_ids = torch.cat(
            [pre_latent_history_ids, latent_start, latent_pad, latent_end],
            dim=1,
        )

        seq_len = rebuilt_input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones_like(rebuilt_input_ids, dtype=torch.long, device=device)

        effective_grid = running_grid_thw if running_grid_thw is not None and running_grid_thw.numel() > 0 else image_grid_thw

        model_inputs = self.prepare_inputs_for_generation(
            input_ids=rebuilt_input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=effective_grid,
            vision_features=zero_latent,
            use_cache=True,
            is_first_iteration=True,
            cache_position=cache_position,
            full_input_ids=rebuilt_input_ids,
        )
        outputs = self.forward(**model_inputs)

        return outputs, rebuilt_input_ids


    @torch.no_grad()
    def generate_interleaved(
        self,
        input_ids,
        tokenizer,
        pixel_values=None,
        image_grid_thw=None,
        max_new_tokens=512,
        max_latent_tokens=36,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0, 
        eos_token_ids=None,
        add_noise=False,
        noise_scale=1.0,
        return_latent_tensors=False,
        save_embeddings=False,
        save_embeddings_dir=None,
        enable_latent_judge=False,
        few_shot_judge_examples=None,
        max_judge_tokens=8,
    ):
        self.eval()
        batch_size = input_ids.shape[0]
        past_key_values = None

        if max_latent_tokens is None:
            max_latent_tokens = self.target_vision_tokens
        else:
            max_latent_tokens = max(0, min(max_latent_tokens, self.target_vision_tokens))

        if self.latent_pad_token_id is None:
            max_latent_tokens = 0

        # Clear the local rope_deltas cache.
        self._rope_deltas_cache = None 

        if batch_size > 1:
            if return_latent_tensors:
                result = {"response": [], "latent_tensors": []}
            else:
                results = []
            
            # Count images for each sample.
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)
            counts = [int(x.item()) for x in image_nums]
            
            # Split image_grid_thw.
            # grid shape: (Total_Images, 3) -> [T, H, W]
            ig = image_grid_thw
            ig_splits = []
            if isinstance(ig, torch.Tensor) and ig.shape[0] == sum(counts):
                ig_splits = list(torch.split(ig, counts, dim=0))
            else:
                ig_splits = [None] * batch_size

            # Dynamically split pixel_values.
            pv = pixel_values
            pv_splits = []
            
            if pv is not None and isinstance(pv, torch.Tensor):
                total_grid_pixels = ig.prod(dim=1).sum().item() # sum(t*h*w)
                
                # Compute the actual pixel_values length along dim=0.
                total_pv_len = pv.shape[0]
                
                # Compute a scale factor. Standard RGB data should have factor 3.0.
                # DDP or custom preprocessing may change the factor, but relative proportions remain stable.
                if total_grid_pixels > 0:
                    scale_factor = total_pv_len / total_grid_pixels
                else:
                    scale_factor = 0
                
                current_offset = 0
                for b_idx, n_imgs in enumerate(counts):
                    if n_imgs == 0 or ig_splits[b_idx] is None:
                        pv_splits.append(None)
                        continue
                    
                    # Current sample grids.
                    sample_grids = ig_splits[b_idx] # (n_imgs, 3)
                    
                    # Theoretical sum of t*h*w for this sample.
                    sample_grid_area = sample_grids.prod(dim=1).sum().item()
                    
                    # Use the global scale factor to estimate this sample's pixel_values length.
                    split_len = int(round(sample_grid_area * scale_factor))
                    
                    end_offset = min(current_offset + split_len, total_pv_len)
                    pv_slice = pv[current_offset : end_offset]
                    pv_splits.append(pv_slice)
                    
                    current_offset = end_offset
            else:
                pv_splits = [None] * batch_size

            for b in range(batch_size):
                single_pv = pv_splits[b]
                single_ig = ig_splits[b]

                out = self.generate_interleaved(
                    input_ids[b:b+1],
                    tokenizer,
                    pixel_values=single_pv,
                    image_grid_thw=single_ig,
                    max_new_tokens=max_new_tokens,
                    max_latent_tokens=max_latent_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_ids=eos_token_ids,
                    add_noise=add_noise,
                    noise_scale=noise_scale,
                    return_latent_tensors=return_latent_tensors,
                    save_embeddings=save_embeddings,
                    save_embeddings_dir=save_embeddings_dir,
                    enable_latent_judge=enable_latent_judge,
                    few_shot_judge_examples=few_shot_judge_examples,
                    max_judge_tokens=max_judge_tokens,
                )
                if return_latent_tensors:
                    result["response"].append(out["response"])
                    result["latent_tensors"].append(out["latent_tensors"])
                else:
                    results.append(out)

            if return_latent_tensors:
                return result
            return results

        generated_segments = []
        # per-sample buffers
        current_text_ids = [[] for _ in range(batch_size)]
        all_generated_ids = [[] for _ in range(batch_size)]

        latent_start_id = self.latent_start_token_id
        latent_end_id = self.latent_end_token_id
        
        im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_token_ids is None:
            eos_token_ids = [tokenizer.eos_token_id, im_end_token_id]
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids = {int(t) for t in eos_token_ids if t is not None and int(t) >= 0}
        
        latent_pad_token_id = self.latent_pad_token_id if self.latent_pad_token_id is not None else tokenizer.convert_tokens_to_ids("<|latent_pad|>")

        valid_latent_ids = (
            latent_start_id is not None
            and latent_end_id is not None
            and latent_pad_token_id is not None
            and int(latent_start_id) >= 0
            and int(latent_end_id) >= 0
            and int(latent_pad_token_id) >= 0
        )

        allow_latent = valid_latent_ids

        def _mask_control_tokens(logits):
            if latent_end_id is not None and int(latent_end_id) >= 0:
                logits[:, latent_end_id] = float("-inf")
            if latent_pad_token_id is not None and int(latent_pad_token_id) >= 0:
                logits[:, latent_pad_token_id] = float("-inf")
            if not allow_latent and latent_start_id is not None and int(latent_start_id) >= 0:
                logits[:, latent_start_id] = float("-inf")
            return logits

        if allow_latent:
            latent_input_id_placeholder = torch.tensor(
                [[latent_pad_token_id]],
                device=input_ids.device, dtype=torch.long
            ).repeat(batch_size, 1)
        else:
            latent_input_id_placeholder = None
        force_skip_future_latent = False

        full_history_ids = input_ids.clone()
        # Initialize global grid history.
        if image_grid_thw is not None:
            running_grid_thw = image_grid_thw.clone()
        else:
            running_grid_thw = torch.tensor([], device=input_ids.device, dtype=torch.long).reshape(0, 3)

        query_vision_norm = None
        if add_noise and pixel_values is not None and image_grid_thw is not None:
            with torch.no_grad():
                query_image_embeds = self._qwen_visual()(
                    pixel_values, grid_thw=image_grid_thw
                )
            if query_image_embeds.numel() > 0:
                query_vision_norm = torch.linalg.norm(
                    query_image_embeds.float(), dim=-1
                ).mean()

        # Prefill prompt and input image.
        seq_length = input_ids.shape[1]
        
        cache_position = torch.arange(seq_length, device=input_ids.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones_like(position_ids)

        model_inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            is_first_iteration=True,
            cache_position=cache_position
        )
        
        # Hidden states are not needed during prefill.
        outputs = self.forward(
            **model_inputs, 
            save_embeddings=save_embeddings, 
            save_embeddings_dir=save_embeddings_dir
        )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        _mask_control_tokens(logits)
        if allow_latent:
            logits[:, latent_start_id] = float("-inf")
        # initialize per-sample history from prompt (exclude vision/placeholders)
        try:
            special_filter = set([
                self.latent_pad_token_id,
                getattr(self.config, "image_token_id", -1),
                getattr(self.config, "vision_start_token_id", -1),
                getattr(self.config, "video_token_id", -1),
                tokenizer.eos_token_id,
            ])
            for b in range(batch_size):
                toks = input_ids[b].tolist()
                filtered = [t for t in toks if t not in special_filter]
                if len(filtered) > 200:
                    filtered = filtered[-200:]
                all_generated_ids[b].extend(filtered)
        except Exception:
            pass

        next_token_id = self._sample_token(
            logits,
            history_ids=all_generated_ids,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        # next_token_id shape: (batch,1)
        next_token_ints = next_token_id.squeeze(-1).tolist()
        # scalar for single-sample path
        next_token_int = next_token_ints[0]

        step = 0
        # for step in range(max_new_tokens):
        while step < max_new_tokens:
            if next_token_int in eos_token_ids:
                for b in range(batch_size):
                    if current_text_ids[b]:
                        text_str = tokenizer.decode(current_text_ids[b], skip_special_tokens=True)
                        generated_segments.append(f"{text_str}")
                        current_text_ids[b] = []
                if im_end_token_id is not None and int(im_end_token_id) >= 0:
                    generated_segments.append("<|im_end|>")
                break
            
            full_history_ids = torch.cat([full_history_ids, next_token_id], dim=1)

            # append per-sample tokens
            for b in range(batch_size):
                all_generated_ids[b].append(next_token_ints[b])
            # Current KV cache length.
            current_seq_len = past_key_values[0][0].shape[-2]
            cache_position = torch.tensor([current_seq_len], device=input_ids.device)
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.ones(
                (batch_size, current_seq_len + 1), 
                dtype=torch.long, device=input_ids.device
            )

            # Trigger latent vision generation.
            if next_token_int == latent_start_id:
                pre_latent_history_ids = full_history_ids[:, :-1].clone()
                pre_latent_generated_segments = list(generated_segments)
                pre_latent_current_text_ids = [list(x) for x in current_text_ids]
                pre_latent_all_generated_ids = [list(x) for x in all_generated_ids]
                pre_latent_running_grid_thw = running_grid_thw.clone()

                pre_latent_logits = logits.clone()
                pre_latent_kv_len = past_key_values[0][0].shape[-2] 

                # flush per-sample text segments
                for b in range(batch_size):
                    if current_text_ids[b]:
                        text_str = tokenizer.decode(current_text_ids[b], skip_special_tokens=True)
                        generated_segments.append(f"{text_str}")
                        current_text_ids[b] = []
                
                # generated_segments.append("<|latent_start|>")

                model_inputs = self.prepare_inputs_for_generation(
                    input_ids=next_token_id, 
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    is_first_iteration=False,
                    cache_position=cache_position
                )
                
                outputs = self.forward(**model_inputs, output_hidden_states=True)
                
                past_key_values = outputs.past_key_values
                curr_vision_hidden = outputs.hidden_states[-1][:, -1:, :]
                
                vision_features_list = []

                # Precompute 3D vision position IDs with the global context.
                latent_pad_block_tmp = torch.full((batch_size, max_latent_tokens), latent_pad_token_id, dtype=torch.long, device=input_ids.device)
                mock_input_ids = torch.cat([full_history_ids, latent_pad_block_tmp], dim=1)
                current_latent_grid_tmp = torch.tensor(
                    [[1, self.target_h, self.target_w]], 
                    dtype=torch.long, device=input_ids.device
                ).repeat(batch_size, 1)
                
                if running_grid_thw is not None and running_grid_thw.numel() > 0:
                    temp_run_grid = torch.cat([running_grid_thw, current_latent_grid_tmp], dim=0)
                else:
                    temp_run_grid = current_latent_grid_tmp
                
                mock_len = mock_input_ids.shape[1]
                mock_cache_position = torch.arange(mock_len, device=input_ids.device)
                mock_position_ids = mock_cache_position.unsqueeze(0).expand(batch_size, -1)
                mock_attention_mask = torch.ones_like(mock_input_ids)

                mock_inputs = self.prepare_inputs_for_generation(
                    input_ids=mock_input_ids,
                    past_key_values=None,
                    attention_mask=mock_attention_mask,
                    position_ids=mock_position_ids,
                    pixel_values=None,
                    image_grid_thw=temp_run_grid,
                    vision_features=torch.zeros((batch_size, max_latent_tokens, self.vision_output_dim), device=input_ids.device), # Force RoPE computation.
                    use_cache=True,
                    is_first_iteration=True,
                    cache_position=mock_cache_position,
                    full_input_ids=mock_input_ids
                )
                full_3d_position_ids = mock_inputs["position_ids"]
                latent_3d_position_ids = full_3d_position_ids[..., -max_latent_tokens:]

                # Vision Loop
                for v_step in range(max_latent_tokens):
                    step_position_ids_vis = latent_3d_position_ids[..., v_step : v_step + 1]
                    # Update position metadata inside the loop.
                    current_seq_len = past_key_values[0][0].shape[-2]
                    cache_position_vis = torch.tensor([current_seq_len], device=input_ids.device)
                    position_ids_vis = cache_position_vis.unsqueeze(0).expand(batch_size, -1)
                    attention_mask_vis = torch.ones((batch_size, current_seq_len + 1), dtype=torch.long, device=input_ids.device)

                    pred_latent = self._project_vision_prediction(curr_vision_hidden)
                    if add_noise:
                        # Replace the predicted latent/PCA coefficients, then pass
                        # the random coefficients through the same PCA decode path.
                        pred_latent = torch.randn_like(pred_latent)
                    recon_vision = self._decode_vision_latent_with_pca(pred_latent.view(-1, pred_latent.shape[-1]))
                    recon_vision = recon_vision.view(curr_vision_hidden.shape[0], curr_vision_hidden.shape[1], -1)
                    if np.random.rand() < 0.05:
                        recon_vision_norm = torch.linalg.norm(recon_vision, dim=-1).mean()
                        curr_vision_hidden_norm = torch.linalg.norm(curr_vision_hidden, dim=-1).mean()
                        # print(f"[Debug] curr_vision_hidden norm: {curr_vision_hidden_norm.item():.4f}, recon_vision norm: {recon_vision_norm.item():.4f}")

                    if add_noise:
                        recon_vision_norm = torch.linalg.norm(
                            recon_vision.float(), dim=-1, keepdim=True
                        ).clamp_min(1e-6).to(dtype=recon_vision.dtype)
                        if query_vision_norm is not None:
                            target_norm = query_vision_norm.to(
                                device=recon_vision.device,
                                dtype=recon_vision.dtype,
                            ).view(1, 1, 1)
                        else:
                            target_norm = torch.linalg.norm(
                                recon_vision.detach().float(),
                                dim=-1,
                                keepdim=True,
                            ).to(dtype=recon_vision.dtype)
                        recon_vision = (recon_vision / recon_vision_norm) * target_norm * float(noise_scale)
                    
                    vision_features_list.append(recon_vision)
                    
                    model_inputs_vis = self.prepare_inputs_for_generation(
                        input_ids=latent_input_id_placeholder, 
                        # inputs_embeds=curr_vision_hidden,
                        inputs_embeds=recon_vision,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask_vis,
                        position_ids=step_position_ids_vis,
                        use_cache=True,
                        is_first_iteration=False,
                        cache_position=cache_position_vis
                    )
                    # model_inputs_vis['input_ids'] = None
                    outputs = self.forward(**model_inputs_vis, output_hidden_states=True)
                    
                    past_key_values = outputs.past_key_values
                    curr_vision_hidden = outputs.hidden_states[-1][:, -1:, :]
                
                if len(vision_features_list) > 0:
                    full_vis = torch.cat(vision_features_list, dim=1)
                    generated_segments.append("<|latent_start|><|latent_tokens|><|latent_end|>")
                    # Update global grid history
                    current_latent_grid = torch.tensor(
                        [[1, self.target_h, self.target_w]], 
                        dtype=torch.long, device=input_ids.device
                    )
                    running_grid_thw = torch.cat([running_grid_thw, current_latent_grid], dim=0)
                    if save_embeddings:
                        save_dir = save_embeddings_dir if save_embeddings_dir is not None else "./saved_embeddings"
                        save_idx = getattr(self, "_save_latent_idx", 0)
                        self._save_latent_idx = save_idx + 1
                        self._save_tensors_dict(
                            {"latent_features": full_vis.cpu().detach()},
                            dir_path=save_dir,
                            file_name=f"latent_features_{self._save_latent_idx}.pt"
                        )
                else:
                    full_vis = curr_vision_hidden.new_empty(batch_size, 0, curr_vision_hidden.shape[-1])
                    generated_segments.append("<|latent_start|><|latent_end|>")

                num_generated_vis_tokens = full_vis.shape[1]
                latent_pad_block = torch.full(
                    (batch_size, num_generated_vis_tokens),
                    latent_pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device
                )
                full_history_ids = torch.cat([full_history_ids, latent_pad_block], dim=1)

                # Recompute RoPE.
                self._rope_deltas_cache = None

                num_vis_tokens = full_vis.shape[1]
                latent_pad = latent_pad_token_id
                
                current_seq_len = past_key_values[0][0].shape[-2]
                cache_position_end = torch.tensor([current_seq_len], device=input_ids.device)
                position_ids_end = cache_position_end.unsqueeze(0).expand(batch_size, -1)
                attention_mask_end = torch.ones((batch_size, current_seq_len + 1), dtype=torch.long, device=input_ids.device)
                input_ids_end = torch.tensor([[latent_end_id]], device=input_ids.device, dtype=torch.long).repeat(batch_size, 1)

                full_input_ids_rope = torch.cat([full_history_ids, input_ids_end], dim=1)

                for b in range(batch_size):
                    all_generated_ids[b].extend([latent_pad] * num_vis_tokens + [latent_end_id])

                model_inputs_end = self.prepare_inputs_for_generation(
                    input_ids=input_ids_end,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask_end,
                    position_ids=position_ids_end,
                    use_cache=True,
                    is_first_iteration=False,
                    cache_position=cache_position_end,
                    image_grid_thw=running_grid_thw,
                    full_input_ids=full_input_ids_rope,
                )
                outputs = self.forward(**model_inputs_end)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                _mask_control_tokens(logits)
                # logits[:, latent_start_id] = float("-inf")  # prevent immediate re-entry 
                full_history_ids = full_input_ids_rope

                should_keep_latent = True
                if enable_latent_judge:
                    try:
                        question_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                        question_text = re.sub(r"<image>|<\|vision_start\|>|<\|image_pad\|>|<\|vision_end\|>", "", question_text)
                    except Exception:
                        question_text = ""

                    backup_rope_deltas = getattr(self, "_rope_deltas_cache", None)

                    should_keep_latent = self._judge_latent_with_image(
                        tokenizer=tokenizer,
                        question=question_text,
                        latent_features=full_vis,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        few_shot_examples=few_shot_judge_examples,
                        max_new_tokens=max_judge_tokens,
                    )
                    self._rope_deltas_cache = backup_rope_deltas

                if not should_keep_latent:
                    if isinstance(past_key_values, tuple):
                        past_key_values = tuple(
                            (k[:, :, :pre_latent_kv_len, :], v[:, :, :pre_latent_kv_len, :])
                            for k, v in past_key_values
                        )
                    elif hasattr(past_key_values, "crop"):
                        # Support DynamicCache.
                        past_key_values.crop(pre_latent_kv_len)
                        if hasattr(past_key_values, "_seen_tokens"):
                            past_key_values._seen_tokens = pre_latent_kv_len
                    else:
                        # Other cache types are unsupported.
                        raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")
                    
                    self._rope_deltas_cache = None  # Force-reset the RoPE cache.

                    # Restore logits from before latent entry and resample.
                    logits = pre_latent_logits.clone()
                    # Mask the rejected latent branch.
                    _mask_control_tokens(logits)
                    logits[:, latent_start_id] = float("-inf")

                    full_history_ids = pre_latent_history_ids
                    running_grid_thw = pre_latent_running_grid_thw
                    full_vis = None

                    generated_segments = pre_latent_generated_segments
                    current_text_ids = [list(x) for x in pre_latent_current_text_ids]
                    all_generated_ids = [list(x) for x in pre_latent_all_generated_ids]

                    force_skip_future_latent = True

                    next_token_id = self._sample_token(
                        logits,
                        history_ids=all_generated_ids,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample
                    )
                    next_token_ints = next_token_id.squeeze(-1).tolist()
                    next_token_int = next_token_ints[0]

                    continue

                _mask_control_tokens(logits)
                logits[:, latent_start_id] = float("-inf")  # prevent immediate re-entry 
                # Sample Next Token
                next_token_id = self._sample_token(
                    logits,
                    history_ids=all_generated_ids,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample
                )
                next_token_ints = next_token_id.squeeze(-1).tolist()
                next_token_int = next_token_ints[0]

                step += (num_vis_tokens + 2)  # +1 for latent_start_id, +1 for latent_end_id
                continue

            # Normal text generation.
            # append to per-sample text buffers
            for b in range(batch_size):
                current_text_ids[b].append(next_token_ints[b])
            
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                is_first_iteration=False,
                cache_position=cache_position
            )
            outputs = self.forward(**model_inputs)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            _mask_control_tokens(logits)
            if allow_latent and force_skip_future_latent:
                logits[:, latent_start_id] = float("-inf")
            next_token_id = self._sample_token(
                logits,
                history_ids=all_generated_ids,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            )
            next_token_ints = next_token_id.squeeze(-1).tolist()
            next_token_int = next_token_ints[0]

            step += 1

        for b in range(batch_size):
            if current_text_ids[b]:
                text_str = tokenizer.decode(current_text_ids[b], skip_special_tokens=True)
                generated_segments.append(f"{text_str}")

        if return_latent_tensors:
            return {
                "response": "\n".join(generated_segments),
                "latent_tensors": full_vis if 'full_vis' in locals() else None
            }

        return "\n".join(generated_segments)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        vision_features=None,
        image_grid_thw=None,
        second_per_grid_ts=None,
        is_first_iteration=False,
        full_input_ids=None,
        full_grid_thw=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and position_ids is not None and position_ids.dim() == 3:
            model_inputs["position_ids"] = position_ids
            if use_cache:
                model_inputs["pixel_values"] = None
            if vision_features is not None:
                model_inputs["vision_features"] = vision_features
            return model_inputs

        rope_input_ids = full_input_ids if full_input_ids is not None else model_inputs.get("input_ids", input_ids)
        has_latent = (self.latent_pad_token_id is not None and (rope_input_ids == self.latent_pad_token_id).any())

        force_recalc = (vision_features is not None) or has_latent

        if (cache_position[0] == 0 or not use_cache) or self._rope_deltas_cache is None or force_recalc:
            if rope_input_ids is not None and self.latent_pad_token_id is not None:
                rope_input_ids = rope_input_ids.clone()
                mask = (rope_input_ids == self.latent_pad_token_id)
                rope_input_ids[mask] = self.config.image_token_id

            rope_attention_mask = attention_mask
            if rope_attention_mask is not None and rope_input_ids is not None:
                if rope_attention_mask.shape[-1] != rope_input_ids.shape[-1]:
                    rope_attention_mask = torch.ones_like(rope_input_ids, dtype=rope_attention_mask.dtype)

            # current_grid = model_inputs.get("image_grid_thw", image_grid_thw)
            current_grid = full_grid_thw if full_grid_thw is not None else model_inputs.get("image_grid_thw", image_grid_thw)

            if vision_features is not None and current_grid is None:
                batch_size = input_ids.shape[0]
                current_grid = torch.tensor(
                    [[1, self.target_h, self.target_w]], 
                    dtype=torch.long, device=input_ids.device
                ).repeat(batch_size, 1)
                model_inputs["image_grid_thw"] = current_grid

            vision_positions, rope_deltas = self._qwen_get_rope_index(
                rope_input_ids,
                image_grid_thw=current_grid,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=rope_attention_mask,
            )

            target_len = model_inputs["position_ids"].shape[-1]
            if vision_positions.shape[-1] != target_len:
                vision_positions = vision_positions[..., -target_len:]

            model_inputs["rope_deltas"] = rope_deltas
            try:
                self._rope_deltas_cache = rope_deltas.cpu().tolist()
            except Exception:
                self._rope_deltas_cache = None
            self._qwen_clear_rope_deltas()

            if model_inputs["position_ids"].dim() == 3:
                text_positions = model_inputs["position_ids"][0:1] # [1, B, Seq]
            else:
                text_positions = model_inputs["position_ids"].unsqueeze(0)
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        else:
            source_ids = model_inputs.get("input_ids", input_ids)
            batch_size, seq_len = source_ids.shape[0], source_ids.shape[1]
            device = source_ids.device
            current_pos = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1)
            
            cached = model_inputs.get("rope_deltas", None) or kwargs.get("rope_deltas", None) or getattr(self, "_rope_deltas_cache", None)
            if isinstance(cached, torch.Tensor):
                rope_deltas = cached.to(device=device, dtype=cache_position.dtype).detach()
            elif isinstance(cached, list):
                rope_deltas = torch.tensor(cached, device=device, dtype=cache_position.dtype)
            else:
                rope_deltas = torch.zeros((batch_size,), dtype=cache_position.dtype, device=device)

            delta = cache_position[0] + rope_deltas

            # Support 1D cache tensors.
            if delta.dim() == 1:
                delta = delta.unsqueeze(1) # Convert to [B, 1] for expansion.

            if delta.shape[1] != batch_size and delta.shape[0] != batch_size:
                delta = delta.expand(batch_size, 1)

            vision_positions = current_pos.unsqueeze(0).expand(3, -1, -1) + delta
            vision_positions = vision_positions.to(dtype=model_inputs["position_ids"].dtype)
            
            if model_inputs["position_ids"].dim() == 3:
                text_positions = model_inputs["position_ids"][0:1]
            else:
                text_positions = model_inputs["position_ids"].unsqueeze(0)
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if (not is_first_iteration and use_cache):
            model_inputs["pixel_values"] = None
        model_inputs["vision_features"] = vision_features

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]

            image_mask = (
                inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]

            video_mask = (
                inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id
        
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
    
__all__ = ["GapForConditionalGeneration"]
