import os
import copy
import re
import random
from PIL import Image
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Qwen2_5_VLModel, AutoProcessor

from gap import GapConfig

from .io_utils import *

class InterleaveDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        image_root_dirs: Dict[str, str],
        latent_seq_len: int = 36,         # Number of generated latent tokens, e.g. 6x6.
        max_seq_len: int = 4096,          # Maximum context length.
        # Native Qwen2.5-VL special tokens.
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        image_pad_token: str = "<|image_pad|>",
        # GAP-specific latent tokens.
        latent_start_token: str = "<|latent_start|>",
        latent_end_token: str = "<|latent_end|>",
        latent_pad_token: str = "<|latent_pad|>",
        # parser tokens
        parser_start_token: str = "<parser>",
        parser_end_token: str = "</parser>",
        parse_prompt: str = "\nPlease parse latent information from the above conversation and provide it in a structured format between <parser> and </parser> tags. The latent information should be concise and directly related to the visual content of the conversation.",
        is_train: bool = True,
        # If True, train on footer tokens such as <|im_end|> and newlines.
        # The default keeps the original behavior and masks footer tokens.
        learn_footer_tokens: bool = False,
        no_latent_loss: bool = False,
        no_split: bool = False,
        case_probs: Optional[List[float]] = None,
        pure_text_mode: bool = False,
        acc_threshold: float = 0.0,
    ):
        """
        Args:
            data_paths: List of JSONL metadata paths.
            image_root_dirs: Mapping from data source to image root, e.g. {'coco': '/path/to/coco'}.
        """
        super().__init__()
        self.no_split = no_split
        self.acc_threshold = acc_threshold
        if case_probs is None:
            case_probs = [1/3, 1/3, 1/3]
        if len(case_probs) != 3 or not all(0 <= p <= 1 for p in case_probs) or abs(sum(case_probs) - 1.0) > 1e-6:
            raise ValueError(f"case_probs must be a list of 3 probabilities that sum to 1. Got: {case_probs}")
        self.case_probs = case_probs
        
        # Load metadata.
        self.data = []
        for path in data_paths:
            print(f"[Dataset] Loading metadata from {path}...")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            item = self._strip_latent_from_successful_item(
                                item,
                                latent_start_token=latent_start_token,
                                latent_end_token=latent_end_token,
                                latent_pad_token=latent_pad_token,
                            )
                            msg_key = "messages" if "messages" in item else "message"
                            original_messages = item.get(msg_key, [])

                            if self.no_split:
                                item_single = copy.deepcopy(item)
                                item_single["parse_mode"] = False
                                self.data.append(item_single)
                                continue
                            
                            # Normalize the assistant message and defer parser handling to __getitem__.
                            item_processed = copy.deepcopy(item)
                            msgs_processed = []
                            has_parser = False
                            parser_text = ""
                            
                            for msg in original_messages:
                                msg_copy = copy.deepcopy(msg)
                                if msg_copy["role"].lower() == "assistant":
                                    msg_copy["dialog"] = repair_think_answer_tags(msg_copy["dialog"])
                                    pattern = rf'({re.escape(parser_start_token)}.*?{re.escape(parser_end_token)})'
                                    matches = re.findall(pattern, msg_copy["dialog"], flags=re.DOTALL)
                                    if matches:
                                        has_parser = True
                                        parser_text = "\n".join(matches)
                                        normal_text = re.sub(pattern, '', msg_copy["dialog"], flags=re.DOTALL).strip()
                                        msg_copy["dialog"] = normal_text
                                msgs_processed.append(msg_copy)
                            
                            item_processed["original_msgs"] = msgs_processed
                            item_processed["has_parser"] = has_parser
                            if has_parser:
                                item_processed["parser_text"] = parser_text

                            self.data.append(item_processed)

            except Exception as e:
                print(f"[Error] Failed to load {path}: {e}")
        
        print(f"[Dataset] Loaded {len(self.data)} samples. (Mode: {'Train' if is_train else 'Eval'})")

        # Store core components.
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_root_dirs = image_root_dirs
        self.latent_seq_len = latent_seq_len
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.placeholder_str = IMAGE_PLACEHOLDER
        self.parser_start_token = parser_start_token
        self.parser_end_token = parser_end_token
        self.parse_prompt = parse_prompt
        
        # Cache token IDs for speed.
        self.ignore_index = -100
        
        # Native vision tokens.
        self.vision_start_id = tokenizer.convert_tokens_to_ids(vision_start_token)
        self.vision_end_id = tokenizer.convert_tokens_to_ids(vision_end_token)
        self.image_pad_id = tokenizer.convert_tokens_to_ids(image_pad_token)
        
        # Latent generation tokens.
        self.pure_text_mode = pure_text_mode
        if pure_text_mode:
            self.latent_start_id = None
            self.latent_end_id = None
            self.latent_pad_id = None
        else:
            self.latent_start_id = tokenizer.convert_tokens_to_ids(latent_start_token)
            self.latent_end_id = tokenizer.convert_tokens_to_ids(latent_end_token)
            self.latent_pad_id = tokenizer.convert_tokens_to_ids(latent_pad_token)

        # ChatML special tokens.
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        # Newline token IDs. Keep newline and role encoded separately to avoid token merging.
        self.nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        # Pre-encode role IDs.
        self.role_ids = {
            "system": tokenizer.encode("system", add_special_tokens=False),
            "user": tokenizer.encode("user", add_special_tokens=False),
            "assistant": tokenizer.encode("assistant", add_special_tokens=False),
        }

        # Build the fixed system prompt block.
        # Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        sys_text_ids = tokenizer.encode("You are a helpful assistant.", add_special_tokens=False)
        self.system_block_ids = (
            [self.im_start_id] + self.role_ids["system"] + self.nl_ids + 
            sys_text_ids + 
            [self.im_end_id] + self.nl_ids
        )
        self.system_block_labels = [self.ignore_index] * len(self.system_block_ids)

        # Controls whether footer tokens are included in the LM loss.
        self.learn_footer_tokens = learn_footer_tokens
        self.no_latent_loss = no_latent_loss

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _strip_latent_text(dialog: str, latent_start_token: str, latent_end_token: str, latent_pad_token: str) -> Tuple[str, bool]:
        if not dialog:
            return dialog, False

        has_latent_tokens = all(
            token in dialog for token in (latent_start_token, latent_end_token, latent_pad_token)
        )
        if not has_latent_tokens:
            return dialog, False

        latent_span_pattern = re.compile(
            rf"{re.escape(latent_start_token)}.*?{re.escape(latent_end_token)}",
            flags=re.DOTALL,
        )
        dialog = latent_span_pattern.sub("", dialog)
        dialog = dialog.replace(latent_pad_token, "")
        dialog = re.sub(r"\n{3,}", "\n\n", dialog)
        return dialog.strip(), True

    @staticmethod
    def _remove_target_placeholders_from_message(msg: dict) -> dict:
        image_types = msg.get("image_types")
        image_filenames = msg.get("image_filenames")
        if not isinstance(image_types, list) or not isinstance(image_filenames, list):
            return msg

        dialog = msg.get("dialog", "")
        parts = dialog.split(IMAGE_PLACEHOLDER)
        filtered_paths = []
        filtered_types = []

        if len(parts) > 1:
            rebuilt = [parts[0]]
            for idx in range(len(parts) - 1):
                img_type = image_types[idx] if idx < len(image_types) else None
                img_path = image_filenames[idx] if idx < len(image_filenames) else None

                if img_type != "target":
                    rebuilt.append(IMAGE_PLACEHOLDER)
                    if img_path is not None:
                        filtered_paths.append(img_path)
                    if idx < len(image_types):
                        filtered_types.append(img_type)

                rebuilt.append(parts[idx + 1])

            for idx in range(len(parts) - 1, len(image_filenames)):
                img_type = image_types[idx] if idx < len(image_types) else None
                if img_type != "target":
                    filtered_paths.append(image_filenames[idx])
                    if idx < len(image_types):
                        filtered_types.append(img_type)

            msg["dialog"] = "".join(rebuilt)
        else:
            for idx, img_path in enumerate(image_filenames):
                img_type = image_types[idx] if idx < len(image_types) else None
                if img_type != "target":
                    filtered_paths.append(img_path)
                    if idx < len(image_types):
                        filtered_types.append(img_type)

        msg["image_filenames"] = filtered_paths
        msg["image_types"] = filtered_types
        return msg

    def _strip_latent_from_successful_item(
        self,
        item: dict,
        latent_start_token: str,
        latent_end_token: str,
        latent_pad_token: str,
    ) -> dict:
        try:
            accuracy = float(item.get("accuracy", 0.0))
        except (TypeError, ValueError):
            accuracy = 0.0

        if accuracy <= self.acc_threshold:
            return item

        msg_key = "messages" if "messages" in item else "message"
        original_messages = item.get(msg_key, [])
        if not isinstance(original_messages, list):
            return item

        item_copy = copy.deepcopy(item)
        cleaned_messages = []
        removed_any = False

        for msg in item_copy.get(msg_key, []):
            msg_copy = copy.deepcopy(msg)
            if msg_copy.get("role", "").lower() == "assistant":
                cleaned_dialog, stripped = self._strip_latent_text(
                    msg_copy.get("dialog", ""),
                    latent_start_token=latent_start_token,
                    latent_end_token=latent_end_token,
                    latent_pad_token=latent_pad_token,
                )
                if stripped:
                    msg_copy["dialog"] = cleaned_dialog
                    removed_any = True

                before_target_fields = (
                    msg_copy.get("dialog", ""),
                    tuple(msg_copy.get("image_filenames", []) or []),
                    tuple(msg_copy.get("image_types", []) or []),
                )
                msg_copy = self._remove_target_placeholders_from_message(msg_copy)
                after_target_fields = (
                    msg_copy.get("dialog", ""),
                    tuple(msg_copy.get("image_filenames", []) or []),
                    tuple(msg_copy.get("image_types", []) or []),
                )
                if before_target_fields != after_target_fields:
                    removed_any = True
            cleaned_messages.append(msg_copy)

        if removed_any:
            item_copy[msg_key] = cleaned_messages
            item_copy["need_latent"] = False
            item_copy["latent_removed_at_data_level"] = True

        return item_copy

    def _load_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Image load failed: {path}, using black image. Error: {e}")
            return Image.new("RGB", (336, 336), (0, 0, 0))

    def _process_image(self, image_path, is_target=False):
        image = self._load_image(image_path)
        MIN_EDGE = 28
        w, h = image.size
        if w < MIN_EDGE or h < MIN_EDGE:
            scale = MIN_EDGE / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        if is_target:
            limit_size = int((self.latent_seq_len ** 0.5) * 28)
            image = image.resize((limit_size, limit_size), resample=Image.BICUBIC)
            out = self.processor.image_processor(
                images=image, 
                return_tensors="pt",
                do_resize=False
                # min_pixels=self.latent_seq_len * 28 * 28, 
                # max_pixels=self.latent_seq_len * 28 * 28,
            )
            return out
        else:
            # keep API consistent: use keyword 'images' (AutoProcessor expects 'images' param)
            return self.processor.image_processor(
                images=image,
                return_tensors="pt",
            )
        
    def _encode_text_with_reflection_mask(
        self, text, is_assistant
    ):
        if not text:
            return [], []
        
        if not is_assistant:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return ids, [self.ignore_index] * len(ids)
        
        tag_open = "<reflection>"
        tag_close = "</reflection>"

        input_ids = []
        labels = []

        i = 0
        while i < len(text):
            next_open = text.find(tag_open, i)
            next_close = text.find(tag_close, i)
            candidates = [p for p in [next_open, next_close] if p != -1]
            if not candidates:
                chunk = text[i:]
                chunk_ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                input_ids.extend(chunk_ids)
                labels.extend(chunk_ids)
                break

            next_tag = min(candidates)

            if next_tag > i:
                chunk = text[i:next_tag]
                chunk_ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                input_ids.extend(chunk_ids)
                labels.extend(chunk_ids)

            tag = tag_open if next_tag == next_open else tag_close
            tag_ids = self.tokenizer.encode(tag, add_special_tokens=False)
            input_ids.extend(tag_ids)
            labels.extend([self.ignore_index] * len(tag_ids))

            i = next_tag + len(tag)

        return input_ids, labels


    def __getitem__(self, idx):
        item = self.data[idx]
        data_source = item.get("data_source", "default")
        image_root_dir = self.image_root_dirs.get(data_source, "")
        if self.pure_text_mode:
            need_latent = False
        elif item.get("latent_removed_at_data_level", False):
            need_latent = False
        elif "accuracy" in item:
            accuracy = float(item["accuracy"])
            need_latent = accuracy <= self.acc_threshold
        elif "need_latent" in item:
            need_latent = bool(item["need_latent"])
        else:
            need_latent = False
        
        # Data augmentation: randomly drop the auxiliary image and/or parser.
        has_parser = item.get("has_parser", False)
        original_msgs = copy.deepcopy(item.get("original_msgs", item.get("messages", item.get("message", []))))
        messages = []

        if has_parser and self.is_train:
            case = random.choices([1, 2, 3], weights=self.case_probs, k=1)[0]
            
            for msg in original_msgs:
                if msg["role"].lower() == "assistant":
                    parser_user_msg = {
                        "role": "user", 
                        "dialog": self.parse_prompt,
                        "image_filenames": [],
                        "image_types": []
                    }
                    parser_assistant_msg = {
                        "role": "assistant",
                        "dialog": item["parser_text"],
                        "image_filenames": [],
                        "image_types": []
                    }

                    if case == 1:
                        # 1. Keep auxiliary image and parser.
                        messages.append(msg)
                        messages.append(parser_user_msg)
                        messages.append(parser_assistant_msg)
                        
                    elif case == 2:
                        # 2. Drop auxiliary image, keep parser.
                        msg_no_img = copy.deepcopy(msg)
                        msg_no_img["dialog"] = msg_no_img["dialog"].replace(self.placeholder_str, "")
                        if "image_types" in msg_no_img:
                            filtered_paths = []
                            filtered_types = []
                            for i_idx, img_type in enumerate(msg_no_img["image_types"]):
                                if img_type != "target":
                                    filtered_paths.append(msg_no_img["image_filenames"][i_idx])
                                    filtered_types.append(img_type)
                            msg_no_img["image_filenames"] = filtered_paths
                            msg_no_img["image_types"] = filtered_types

                        messages.append(msg_no_img)
                        messages.append(parser_user_msg)
                        messages.append(parser_assistant_msg)
                        
                    elif case == 3:
                        # 3. Keep auxiliary image, drop parser.
                        messages.append(msg)

                else:
                    messages.append(msg)
        else:
            # In inference mode, or when parser text is absent, keep the full single-turn form.
            for msg in original_msgs:
                messages.append(msg)
                if has_parser and msg["role"].lower() == "assistant":
                    messages.append({
                        "role": "user", 
                        "dialog": self.parse_prompt,
                        "image_filenames": [],
                        "image_types": []
                    })
                    messages.append({
                        "role": "assistant",
                        "dialog": item["parser_text"],
                        "image_filenames": [],
                        "image_types": []
                    })

        # Core token containers.
        input_ids = []
        labels = []
        
        # Vision tensor containers.
        input_pixel_values_list = []
        input_image_grid_thw_list = []
        target_pixel_values_list = []
        target_image_grid_thw_list = []
        target_image_seq_lens = []
        
        # Ground truth fields used for validation.
        raw_gt_ids = None
        prompt_ids = None

        # System prompt
        input_ids.extend(self.system_block_ids)
        labels.extend(self.system_block_labels)

        for msg_idx, msg in enumerate(messages):
            role = msg["role"].lower()
            content = msg["dialog"]
            image_paths = msg.get("image_filenames", [])
            image_types = msg.get("image_types", None)
            
            is_assistant = (role == "assistant")

            # [Validation Special Logic]
            if (not self.is_train) and is_assistant and (msg_idx == len(messages) - 1):
                raw_gt_ids = self.tokenizer.encode(content, add_special_tokens=False)
                # if len(image_paths) > 0:
                #     latent_seq = [self.latent_start_id] + [self.latent_pad_id] * self.latent_seq_len + [self.latent_end_id]
                #     raw_gt_ids = raw_gt_ids

                prompt_ids = list(input_ids)       

            # Header
            role_token_ids = self.role_ids.get(role, self.role_ids["user"])
            header_ids = [self.im_start_id] + role_token_ids + self.nl_ids
            
            input_ids.extend(header_ids)
            labels.extend([self.ignore_index] * len(header_ids))

            # Process interleaved text and image placeholders.
            parts = content.split(self.placeholder_str)
            
            for i, part_text in enumerate(parts):
                # Text segment.
                if part_text:
                    text_ids, text_labels = self._encode_text_with_reflection_mask(part_text, is_assistant=is_assistant)
                    input_ids.extend(text_ids)
                    labels.extend(text_labels)

                # Image segment inserted between text chunks.
                if i < len(image_paths):
                    if os.path.isfile(image_paths[i]):
                        img_path = image_paths[i]  # Absolute path.
                    else:
                        img_path = os.path.join(image_root_dir, image_paths[i])

                    img_type = None
                    if image_types and i < len(image_types):
                        img_type = image_types[i]

                    if img_type == "input":
                        proc_out = self._process_image(img_path, is_target=False)
                        input_pixel_values_list.append(proc_out["pixel_values"])
                        input_image_grid_thw_list.append(proc_out["image_grid_thw"])
                        
                        grid = proc_out["image_grid_thw"][0]
                        seq_len = int((grid[1] // 2) * (grid[2] // 2))

                        img_seq = [self.vision_start_id] + [self.image_pad_id] * seq_len + [self.vision_end_id]
                        input_ids.extend(img_seq)
                        labels.extend([self.ignore_index] * len(img_seq))

                    elif is_assistant and need_latent:
                        proc_out = self._process_image(img_path, is_target=True)
                        target_pixel_values_list.append(proc_out["pixel_values"])
                        target_image_grid_thw_list.append(proc_out["image_grid_thw"])
                        grid = proc_out["image_grid_thw"][0]
                        seq_len = int((grid[1] // 2) * (grid[2] // 2))
                        if seq_len > self.latent_seq_len:
                            raise ValueError(
                                f"Target image feature length {seq_len} exceeds configured latent_seq_len {self.latent_seq_len} "
                                f"for image: {img_path}. Increase latent_seq_len or reduce target resolution."
                            )
                        target_image_seq_lens.append(seq_len)

                        # Build latent token sequence: <|latent_start|> ...pads... <|latent_end|>.
                        latent_seq = [self.latent_start_id] + [self.latent_pad_id] * self.latent_seq_len + [self.latent_end_id]
                        input_ids.extend(latent_seq)

                        if self.no_latent_loss:
                            labels.extend([self.ignore_index] * len(latent_seq))
                        else:
                            labels.append(self.latent_start_id)
                            labels.extend([self.ignore_index] * self.latent_seq_len)
                            labels.append(self.latent_end_id)

                    elif img_type == "target":
                        continue

                    else:
                        proc_out = self._process_image(img_path, is_target=False)
                        input_pixel_values_list.append(proc_out["pixel_values"])
                        input_image_grid_thw_list.append(proc_out["image_grid_thw"])
                        
                        # Compute Qwen's dynamic vision token length.
                        grid = proc_out["image_grid_thw"][0]
                        # Qwen2.5-VL logic: (h//2) * (w//2).
                        seq_len = int((grid[1] // 2) * (grid[2] // 2))
                        
                        # Build input vision token sequence.
                        img_seq = [self.vision_start_id] + [self.image_pad_id] * seq_len + [self.vision_end_id]
                        input_ids.extend(img_seq)
                        labels.extend([self.ignore_index] * len(img_seq))

            # Footer
            footer_ids = [self.im_end_id] + self.nl_ids
            input_ids.extend(footer_ids)
            if self.learn_footer_tokens and is_assistant:
                labels.extend(footer_ids)
            else:
                labels.extend([self.ignore_index] * len(footer_ids)) 

        # Eval trigger used only in inference mode.
        # Append <|im_start|>assistant\n to start generation.
        if not self.is_train:
            trigger_ids = [self.im_start_id] + self.role_ids["assistant"] + self.nl_ids
            input_ids.extend(trigger_ids)
            labels.extend([self.ignore_index] * len(trigger_ids))

        # Truncate: keep the tail (assistant response) so valid labels are preserved.
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]
        
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "need_latent": torch.tensor(need_latent, dtype=torch.bool),
        }

        # Attach validation ground truth if available.
        if raw_gt_ids is not None:
            result["ground_truth_ids"] = torch.tensor(raw_gt_ids, dtype=torch.long)

        if prompt_ids is not None:
            if len(prompt_ids) > self.max_seq_len:
                prompt_ids = prompt_ids[:self.max_seq_len]
            result["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.long)

        # Helper for concatenating tensor lists.
        def safe_concat(tensor_list):
            if len(tensor_list) > 0:
                return torch.cat(tensor_list, dim=0)
            return None

        # Fill image tensor fields.
        result["input_pixel_values"] = safe_concat(input_pixel_values_list)
        result["input_image_grid_thw"] = safe_concat(input_image_grid_thw_list)
        
        result["target_pixel_values"] = safe_concat(target_pixel_values_list)
        result["target_image_grid_thw"] = safe_concat(target_image_grid_thw_list)
        result["target_image_seq_lens"] = (
            torch.tensor(target_image_seq_lens, dtype=torch.long)
            if len(target_image_seq_lens) > 0
            else None
        )

        return result
    
@dataclass
class InterleaveDataCollator:
    tokenizer: AutoTokenizer
    ignore_index: int = -100

    def __call__(self, examples):
        filter_examples = []
        max_len = 0
        for instance in examples:
            if instance is not None and len(instance["input_ids"]) > 0:
                max_len = max(len(instance["input_ids"]), max_len)
                filter_examples.append(instance)

        if len(filter_examples) == 0:
            return {}
        
        batch_size = len(filter_examples)

        # Initialize tensors with the shape (B, L)
        input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        input_pixel_values_list = []
        input_image_grid_thw_list = []

        target_pixel_values_list = []
        target_image_grid_thw_list = []
        target_image_seq_lens_list = []
        target_image_counts = []
        need_latent_list = []

        for i, item in enumerate(filter_examples):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = item["input_ids"]
            labels[i, :seq_len] = item["labels"]
            attention_mask[i, :seq_len] = 1

            if item["input_pixel_values"] is not None:
                input_pixel_values_list.append(item["input_pixel_values"])
                input_image_grid_thw_list.append(item["input_image_grid_thw"])

            need_latent_list.append(item.get("need_latent", torch.tensor(True)).item())

            if item["target_pixel_values"] is not None:
                target_pixel_values_list.append(item["target_pixel_values"])
                target_image_grid_thw_list.append(item["target_image_grid_thw"])
                if item.get("target_image_seq_lens") is not None:
                    target_image_seq_lens_list.append(item["target_image_seq_lens"])
                target_image_counts.append(item["target_image_grid_thw"].shape[0])
            else:
                target_image_counts.append(0)

        if len(input_pixel_values_list) > 0:
            input_pixel_values = torch.cat(input_pixel_values_list, dim=0)
            input_image_grid_thw = torch.cat(input_image_grid_thw_list, dim=0)
        else:
            input_pixel_values = None
            input_image_grid_thw = None

        if len(target_pixel_values_list) > 0:
            target_pixel_values = torch.cat(target_pixel_values_list, dim=0)
            target_image_grid_thw = torch.cat(target_image_grid_thw_list, dim=0)
            target_image_seq_lens = (
                torch.cat(target_image_seq_lens_list, dim=0)
                if len(target_image_seq_lens_list) > 0
                else None
            )
        else:
            target_pixel_values = None
            target_image_grid_thw = None
            target_image_seq_lens = None

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": input_pixel_values,
            "image_grid_thw": input_image_grid_thw,
            "target_pixel_values": target_pixel_values,
            "target_image_grid_thw": target_image_grid_thw,
            "target_image_seq_lens": target_image_seq_lens,
            "target_image_counts": torch.tensor(target_image_counts, dtype=torch.long),
            "need_latent": torch.tensor(need_latent_list, dtype=torch.bool),
        }

        return batch


class QwenDataset(Dataset):
    """
    Standard SFT dataset for Qwen2.5-VL (or Qwen3-VL).

    Each sample follows the Qwen chat template:
        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        <|im_start|>user\n...<|im_end|>\n
        <|im_start|>assistant\n...<|im_end|>\n

    Only assistant turns are supervised (labels != -100).

    Metadata format (JSONL or list of dicts):
        {
            "messages": [                   # or "message"
                {
                    "role": "user",
                    "dialog": "Describe <|vision_start|><|image_pad|><|vision_end|>",
                    "image_filenames": ["path/to/img.jpg"]
                },
                {
                    "role": "assistant",
                    "dialog": "This is a cat."
                }
            ]
        }

    Parameters
    ----------
    meta_file : str | list[dict]
        Path to a JSONL file, or an already-loaded list of dicts.
    tokenizer : AutoTokenizer
        Qwen tokenizer (must contain <|im_start|>, <|im_end|>, vision tokens).
    image_processor : AutoProcessor
        Qwen processor whose ``image_processor`` will be called for each image.
    image_root : str
        Root directory that is prepended to relative image paths.
    max_seq_len : int
        Maximum token length; sequences longer than this are truncated.
    is_train : bool
        If False, the last assistant turn is stored as ``ground_truth_ids``
        and a trigger header is appended for generation.
    system_prompt : str
        System message content.
    """

    def __init__(
        self,
        meta_file: Union[str, List[dict]],
        tokenizer: AutoTokenizer,
        image_processor: Optional[AutoProcessor] = None,
        image_root: str = "",
        image_root_dirs: Optional[Dict[str, str]] = None,
        max_seq_len: int = 16384,
        is_train: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ):
        super().__init__()

        # ------ load data ------
        if isinstance(meta_file, list):
            self.data = meta_file
        else:
            self.data = []
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        print(f"[QwenDataset] Loaded {len(self.data)} samples (train={is_train})")

        self.tokenizer = tokenizer
        self.processor = image_processor
        self.image_root = image_root
        self.image_root_dirs = image_root_dirs or {}
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.ignore_index = -100

        # ------ special token ids ------
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_ids = tokenizer.encode("\n", add_special_tokens=False)

        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        self.role_ids = {
            "system": tokenizer.encode("system", add_special_tokens=False),
            "user": tokenizer.encode("user", add_special_tokens=False),
            "assistant": tokenizer.encode("assistant", add_special_tokens=False),
        }

        # pre-encode system block
        sys_text_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
        self.system_block_ids = (
            [self.im_start_id] + self.role_ids["system"] + self.nl_ids
            + sys_text_ids
            + [self.im_end_id] + self.nl_ids
        )
        self.system_block_labels = [self.ignore_index] * len(self.system_block_ids)

    # ---------- helpers ----------

    def _resolve_image_path(self, path: str) -> str:
        """Return an absolute path, trying ``image_root`` and ``image_root_dirs`` as prefixes."""
        # 1) already absolute and exists
        if os.path.isabs(path) and os.path.exists(path):
            return path
        # 2) try primary image_root
        if self.image_root:
            candidate = os.path.join(self.image_root, path)
            if os.path.exists(candidate):
                return candidate
        # 3) try each root in image_root_dirs dict
        for _, root in self.image_root_dirs.items():
            if not root:
                continue
            candidate = os.path.join(root, path)
            if os.path.exists(candidate):
                return candidate
        return path  # fallback; may be missing

    def _process_image(self, path: str):
        """Load one image and return (pixel_values, image_grid_thw, seq_len)."""
        actual = self._resolve_image_path(path)
        img = Image.open(actual).convert("RGB")
        vision_inputs = self.processor.image_processor(images=img, return_tensors="pt")

        pv = vision_inputs.get("pixel_values", None)
        gthw = vision_inputs.get("image_grid_thw", None)

        if pv is None or gthw is None:
            return None, None, 1

        if not isinstance(gthw, torch.Tensor):
            gthw = torch.tensor(gthw)
        if gthw.dim() == 1:
            gthw = gthw.unsqueeze(0)

        g = gthw[0]
        seq_len = max(int((g[1].item() // 2) * (g[2].item() // 2)), 1)
        return pv, gthw[:1], seq_len

    # ---------- __len__ / __getitem__ ----------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get("messages", item.get("message", []))

        input_ids: List[int] = []
        labels: List[int] = []
        pixel_values_list: List[torch.Tensor] = []
        image_grid_thw_list: List[torch.Tensor] = []
        image_pad_token_count = 0

        raw_gt_ids = None
        prompt_ids = None

        # system
        input_ids.extend(self.system_block_ids)
        labels.extend(self.system_block_labels)

        for msg_idx, msg in enumerate(messages):
            role = msg["role"].lower()
            content = msg.get("dialog", "")
            is_assistant = (role == "assistant")

            image_paths = msg.get("image_filenames", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            # assistant should not carry image placeholders
            if is_assistant:
                content = content.replace(IMAGE_PLACEHOLDER, "")
                content = content.replace("<image>", "")
                content = content.replace("<|image|>", "")

            # Eval: keep last assistant GT
            if (not self.is_train) and is_assistant and (msg_idx == len(messages) - 1):
                raw_gt_ids = self.tokenizer.encode(content, add_special_tokens=False)
                prompt_ids = list(input_ids)

            # --- header ---
            role_token_ids = self.role_ids.get(role, self.role_ids["user"])
            header_ids = [self.im_start_id] + role_token_ids + self.nl_ids
            input_ids.extend(header_ids)
            labels.extend([self.ignore_index] * len(header_ids))

            # --- content ---
            if is_assistant:
                # supervised: encode text, labels = token ids
                text_ids = self.tokenizer.encode(content, add_special_tokens=False)
                input_ids.extend(text_ids)
                labels.extend(text_ids)
            else:
                # non-assistant: text is masked; images are inserted at placeholders
                content = content.replace("<image>", IMAGE_PLACEHOLDER)
                content = content.replace("<|image|>", IMAGE_PLACEHOLDER)

                parts = content.split(IMAGE_PLACEHOLDER)
                for i, part_text in enumerate(parts):
                    if part_text:
                        txt_ids = self.tokenizer.encode(part_text, add_special_tokens=False)
                        input_ids.extend(txt_ids)
                        labels.extend([self.ignore_index] * len(txt_ids))

                    # insert image between parts
                    if i < len(parts) - 1:
                        if i < len(image_paths) and image_paths[i]:
                            try:
                                pv, gthw, seq_len = self._process_image(image_paths[i])
                                if pv is not None:
                                    pixel_values_list.append(pv)
                                    image_grid_thw_list.append(gthw)
                                    img_seq = ([self.vision_start_id]
                                               + [self.image_pad_id] * seq_len
                                               + [self.vision_end_id])
                                    input_ids.extend(img_seq)
                                    labels.extend([self.ignore_index] * len(img_seq))
                                    image_pad_token_count += seq_len
                                else:
                                    self._insert_placeholder(input_ids, labels)
                                    image_pad_token_count += 1
                            except Exception as e:
                                print(f"[QwenDataset][Warn] Image error {image_paths[i]}: {e}")
                                self._insert_placeholder(input_ids, labels)
                                image_pad_token_count += 1
                        else:
                            self._insert_placeholder(input_ids, labels)
                            image_pad_token_count += 1

            # --- footer ---
            footer_ids = [self.im_end_id] + self.nl_ids
            input_ids.extend(footer_ids)
            if is_assistant:
                # supervise <|im_end|> so model learns to stop
                labels.extend([self.im_end_id] + [self.ignore_index] * len(self.nl_ids))
            else:
                labels.extend([self.ignore_index] * len(footer_ids))

        # Eval trigger
        if not self.is_train:
            trigger_ids = [self.im_start_id] + self.role_ids["assistant"] + self.nl_ids
            input_ids.extend(trigger_ids)
            labels.extend([self.ignore_index] * len(trigger_ids))

        # concat vision tensors
        processed_pixel_values = None
        processed_image_grid_thw = None
        if pixel_values_list:
            try:
                processed_pixel_values = torch.cat(pixel_values_list, dim=0)
                processed_image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
            except Exception as e:
                print(f"[QwenDataset][Warn] concat vision fail idx={idx}: {e}")

        # alignment check
        if processed_image_grid_thw is not None:
            expected_feat = 0
            for g in processed_image_grid_thw:
                expected_feat += max(int((g[1].item() // 2) * (g[2].item() // 2)), 1)
            if image_pad_token_count != expected_feat:
                print(
                    f"[QwenDataset][Skip] token-feature mismatch idx={idx}: "
                    f"tokens={image_pad_token_count}, features={expected_feat}"
                )
                return None

        # truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": processed_pixel_values,
            "image_grid_thw": processed_image_grid_thw,
        }

        if raw_gt_ids is not None:
            result["ground_truth_ids"] = torch.tensor(raw_gt_ids, dtype=torch.long)
        if prompt_ids is not None:
            if len(prompt_ids) > self.max_seq_len:
                prompt_ids = prompt_ids[: self.max_seq_len]
            result["prompt_ids"] = torch.tensor(prompt_ids, dtype=torch.long)

        return result

    def _insert_placeholder(self, input_ids: List[int], labels: List[int]):
        """Insert a minimal vision placeholder (1 image_pad token)."""
        img_seq = [self.vision_start_id, self.image_pad_id, self.vision_end_id]
        input_ids.extend(img_seq)
        labels.extend([self.ignore_index] * 3)


@dataclass
class QwenDataCollator:
    """
    Data collator for standard Qwen2.5-VL SFT.

    Pads ``input_ids`` / ``labels`` to the batch max length and stacks
    ``pixel_values`` / ``image_grid_thw`` across samples.

    The output dict is directly consumable by
    ``Qwen2_5_VLForConditionalGeneration.forward()``.
    """

    tokenizer: AutoTokenizer
    ignore_index: int = -100

    def __call__(self, examples):
        # filter None (skipped samples)
        filtered = [ex for ex in examples if ex is not None and len(ex["input_ids"]) > 0]
        if not filtered:
            return {}

        bsz = len(filtered)
        max_len = max(len(ex["input_ids"]) for ex in filtered)

        input_ids = torch.full((bsz, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        all_pixel_values = []
        all_image_grid_thw = []

        for i, item in enumerate(filtered):
            l = len(item["input_ids"])
            input_ids[i, :l] = item["input_ids"]
            labels[i, :l] = item["labels"]
            attention_mask[i, :l] = 1

            pv = item.get("pixel_values", None)
            gthw = item.get("image_grid_thw", None)
            if pv is not None:
                all_pixel_values.append(pv)
            if gthw is not None:
                all_image_grid_thw.append(gthw)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Stack vision tensors across the batch (Qwen model expects concatenated tensors).
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        # optional eval fields
        if any("ground_truth_ids" in x for x in filtered):
            batch["ground_truth_ids"] = [
                x.get("ground_truth_ids", torch.empty((0,), dtype=torch.long))
                for x in filtered
            ]
        if any("prompt_ids" in x for x in filtered):
            batch["prompt_ids"] = [
                x.get("prompt_ids", torch.empty((0,), dtype=torch.long))
                for x in filtered
            ]

        return batch
