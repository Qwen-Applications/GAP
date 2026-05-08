from dataclasses import dataclass
# from modeling.qwen2 import Qwen2Config
from transformers import Qwen2_5_VLConfig
# from modeling.deepencoder import DeepSeekOCRVisionConfig

class GapConfig(Qwen2_5_VLConfig):
    model_type = "gap"
    def __init__(
        self,
        # Qwen2.5 config
        bos_token_id=151643,
        eos_token_id=151644,
        vocab_size=152064,
        vision_start_token_id=151652,
        image_token_id=151655,
        video_token_id=151656,
        vision_end_token_id=151653,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=28,
        attention_dropout=0.0,
        vision_config=None,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        # vision head config
        vision_feature_size=3584,
        projector_hidden_act="silu",
        num_projector_layers=2,
        ignore_index=-100,
        target_h=28,
        target_w=28,
        latent_start_token_id=None,
        latent_end_token_id=None,
        latent_pad_token_id=None,
        vision_loss_weight=0.01,
        use_sinkhorn_loss=False,
        is_transformer_block=False,
        num_transformer_layers=1,
        # vision pca path
        vision_pca_mu_path=None,
        vision_pca_W_path=None,
        vision_head_mode="linear",
        schedule_sampling_enabled=False,
        schedule_sampling_ratio=0.0,
        schedule_sampling_detach_features=True,
        # Flash-attention
        torch_dtype="bfloat16",
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            vision_config=vision_config,
            **kwargs,
        )
        
        self.vision_feature_size = vision_feature_size
        self.projector_hidden_act = projector_hidden_act
        self.num_projector_layers = num_projector_layers
        self.ignore_index = ignore_index
        self.target_h = target_h
        self.target_w = target_w
        self.latent_start_token_id = latent_start_token_id
        self.latent_end_token_id = latent_end_token_id
        self.latent_pad_token_id = latent_pad_token_id
        self.vision_loss_weight = vision_loss_weight
        self.torch_dtype = torch_dtype
        self.is_transformer_block = is_transformer_block
        self.num_transformer_layers = num_transformer_layers
        self.use_sinkhorn_loss = use_sinkhorn_loss
        self.vision_pca_mu_path = vision_pca_mu_path
        self.vision_pca_W_path = vision_pca_W_path
        self.vision_head_mode = vision_head_mode
        self.schedule_sampling_enabled = schedule_sampling_enabled
        self.schedule_sampling_ratio = schedule_sampling_ratio
        self.schedule_sampling_detach_features = schedule_sampling_detach_features
        self.model_type = "gap"
