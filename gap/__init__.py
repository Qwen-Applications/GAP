from typing import TYPE_CHECKING
from transformers.utils import is_torch_available

from .configuration_gap import GapConfig


if is_torch_available():
    from .modeling_gap import GapForConditionalGeneration
