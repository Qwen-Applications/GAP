# Option A — Custom vLLM Integration for GAP Interleaved Generation

Plan for bringing `generate_interleaved` (`gap/modeling_gap.py:1198`) onto vLLM. This is a design doc, not a kickoff — read end to end before committing engineering time.

## What we are integrating

`generate_interleaved` is a custom decoding loop with three modes:

1. **Text mode** — standard autoregressive token sampling with a logits mask that forbids `<|latent_end|>` / `<|latent_pad|>`.
2. **Latent mode** — entered when the sampler emits `<|latent_start|>`. For up to `max_latent_tokens` steps the model:
   - takes the last hidden state,
   - projects it via `_project_vision_prediction` → PCA-coded latent,
   - decodes via `_decode_vision_latent_with_pca` → a continuous embedding in vision-feature space,
   - feeds that **embedding** (not a token id) into the next forward via `inputs_embeds`, with a `<|latent_pad|>` placeholder id,
   - collects the new hidden state and repeats.
3. **Closeout** — emits `<|latent_end|>` and resumes text mode. Before doing so, runs a "mock prefill" over `full_history_ids` ⨁ latent block to recompute 3D RoPE position_ids over the accumulated `running_grid_thw`.

Optional knobs that complicate the integration:
- `enable_latent_judge` — runs a side `_judge_latent_with_image` call that may **reject** the latent block; on reject, the KV cache is **cropped back** to `pre_latent_kv_len`.
- `add_noise` — replaces `pred_latent` with random coefficients before PCA decode (intervention study).
- `return_latent_tensors` / `save_embeddings` — exfiltrates intermediate latents.

## Why it doesn't drop into vLLM

vLLM assumes (a) discrete token IDs throughout decode, (b) the engine owns the loop, (c) the paged KV cache only grows. GAP violates all three:
- Continuous embeddings as decode-step inputs (vLLM's `inputs_embeds` support is **prefill-only** today).
- A model-decided control-flow switch mid-decode.
- KV cache truncation on judge reject.

Plus 3D-RoPE with a `running_grid_thw` that grows every latent block, mirroring Qwen2-VL but with extra state.

## Target stack

- **vLLM 0.7.x, V1 engine** (`VLLM_USE_V1=1`). V1's executor/scheduler split is what makes this tractable; V0 is too monolithic.
- Pin vLLM exactly. Plan for a soft fork — upstream API churn between minor versions is real.
- Reuse vLLM's existing `Qwen2_5_VLForConditionalGeneration` model as the backbone (prefill, vision tower, 3D RoPE plumbing). Add GAP-specific behavior on top.

## Directory layout

```
Gap/inference/
├── PLAN.md                          # this file
├── README.md                        # how to run once implemented
├── pyproject.toml                   # local extra: pip install -e ".[vllm]"
├── vllm_gap/
│   ├── __init__.py                  # registers the GAP model with vLLM's ModelRegistry
│   ├── model.py                     # GapForVLLM, subclasses Qwen2_5_VL vLLM impl
│   ├── latent_step.py               # one-step latent decode (project → PCA decode → forward)
│   ├── latent_loop.py               # the max_latent_tokens loop, called from the runner
│   ├── rope_3d.py                   # running_grid_thw bookkeeping + position_ids recompute
│   ├── logits_processor.py          # masks <|latent_end|>/<|latent_pad|>; surfaces <|latent_start|>
│   └── kv_snapshot.py               # (Phase 3) snapshot/restore for latent-judge rollback
├── engine/
│   ├── runner_patch.py              # monkey-patches / subclass of ModelRunner.execute_model
│   ├── scheduler_hook.py            # detects sequences entering latent mode, isolates them
│   └── output_processor.py          # collects per-sequence latent tensors for return_latent_tensors
├── serve.py                         # OpenAI-compatible server entrypoint
├── client_example.py                # minimal HTTP request demo
├── tests/
│   ├── test_text_only_parity.py     # vLLM == HF when max_latent_tokens=0
│   ├── test_single_latent_parity.py # vLLM == HF on a fixed-seed greedy prompt with one latent block
│   ├── test_multi_latent_parity.py  # multiple latent blocks within one generation
│   └── test_concurrency.py          # N concurrent requests, no cross-talk in latent state
└── bench/
    ├── bench_latency.py             # single-request latency: vLLM vs. HF generate_interleaved
    └── bench_throughput.py          # batched throughput at concurrency 1/8/16/32
```

## Phased plan

### Phase 0 — Scoping spike (≈1 week)
- Pin vLLM version. Stand up vanilla Qwen2.5-VL inference via vLLM on the GAP base weights (no latent path). Confirm logits parity with HF `forward` on the same prompt to ε ≤ 1e-3 (bf16 kernels differ).
- Read vLLM V1 source: `vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/sample/sampler.py`, `vllm/v1/engine/core.py`. Identify exact override points.
- **Exit criterion**: a one-page "API contract" listing every vLLM symbol we will subclass or monkey-patch, pinned to a commit.

### Phase 1 — Text-mode trigger detection (≈1–2 weeks)
- Register `GapForVLLM` in vLLM's `ModelRegistry`. For prefill, identical to Qwen2.5-VL.
- Add `LatentLogitsProcessor` that:
  - hard-masks `<|latent_end|>` and `<|latent_pad|>` (mirrors `_mask_control_tokens`),
  - hard-masks `<|latent_start|>` if `max_latent_tokens == 0` (parity with `allow_latent` flag).
- Verify with `tests/test_text_only_parity.py` that greedy decode matches HF `generate_interleaved(max_latent_tokens=0)` token-for-token.
- **Milestone**: text-only parity test green; throughput within ~10% of plain Qwen2.5-VL on vLLM.

### Phase 2 — Latent loop, single sequence (≈2–3 weeks)
The hard phase. Three sub-tasks:

**2a. Mid-decode embed injection.** Extend `GpuModelRunner.execute_model` to accept an optional `latent_inputs_embeds: Tensor` per sequence. When present for a given sequence-id at a given step, that sequence's input is the embed (with `<|latent_pad|>` as the token-id slot, only used for position bookkeeping). All other sequences in the batch use normal token-id inputs. This is invasive — needs changes to:
- The input-tensor builder (`_prepare_inputs`) to route some sequences through `inputs_embeds`.
- The block manager — no change needed, KV grows the same way.
- The sampler — for latent-mode sequences, **don't sample**. Instead, write the new hidden state into a per-sequence buffer for the next latent step.

**2b. Latent control flow.** Implement `LatentLoopController` that lives in the engine layer:
- On `<|latent_start|>` sample, mark the sequence as `latent_mode=True`, `latent_steps_remaining=max_latent_tokens`.
- For each subsequent step, run `_project_vision_prediction` + `_decode_vision_latent_with_pca` on the last hidden state, feed the result back via 2a as `latent_inputs_embeds`, append `<|latent_pad|>` to the token log, decrement counter.
- When counter hits 0, append `<|latent_end|>` to the token log, run one closeout forward to populate KV for that token, mark `latent_mode=False`.

**2c. 3D RoPE bookkeeping.** Maintain `running_grid_thw` per sequence in engine state. Before each latent block, replicate the "mock prefill" position-id computation (`prepare_inputs_for_generation` with `vision_features=zeros`). Memoize: `position_ids` for a given `(running_grid_thw, current_seq_len)` are deterministic, so cache by hash to avoid recomputing every step.

- **Milestone**: `tests/test_single_latent_parity.py` and `test_multi_latent_parity.py` pass — vLLM output matches HF `generate_interleaved` on fixed-seed greedy decode for prompts with 1 and 3+ latent blocks. Tolerance: token-id exact match for greedy; logits ε ≤ 1e-2 for sampled.

### Phase 3 — Latent-judge rollback (deferred / optional)
**Recommendation: skip for v1.** vLLM's paged KV cache doesn't expose a "crop to N tokens" primitive. Options:
- **(a) Drop `enable_latent_judge` from the vLLM path.** Document the limitation. Use vLLM only for runs with judge disabled.
- **(b) Snapshot block tables before the latent block; restore on reject.** Implementable by copying allocated block IDs into a side list and reattaching to a forked sequence. Touches the block manager and roughly doubles peak memory during the latent window.

Pick (a) unless ablations show the judge is load-bearing for benchmark scores. If we end up needing (b), budget another 2 weeks.

### Phase 4 — Continuous batching with mixed modes (≈3–4 weeks)
- Naive: when **any** sequence enters latent mode, run latent steps **alone** (drop other sequences from the step). Simple, throughput hit only proportional to fraction of time in latent mode.
- Better: heterogeneous batches where some indices use `inputs_embeds` and others use `input_ids` in the same forward. Requires the input builder + attention metadata to handle both. ~2× the impl cost of the naive path.
- Recommendation: ship naive, profile. Only do the heterogeneous path if profiling shows latent-mode time is the bottleneck.

### Phase 5 — Serve, bench, ship (≈1 week)
- Wrap as OpenAI-compatible server via vLLM's `vllm serve` with our model registered.
- `bench_latency.py`: single-request median + p95 vs. HF `generate_interleaved`.
- `bench_throughput.py`: throughput at concurrency 8/16/32; expect the win to live here.
- Wire into VLMEvalKit: add a `Gap_vLLM` entry in `Gap_evaluation/vlmeval/config.py` that hits the vLLM server instead of loading the model in-process. Run MathVista_MINI on both, require score parity within ±0.5 absolute.

## Risks and open questions

| Risk | Severity | Mitigation |
|---|---|---|
| vLLM V1 API churn between 0.7.x point releases | High | Pin a commit; treat as soft fork. Re-evaluate per release. |
| Numerical drift (bf16 kernels) breaks parity tests | Medium | Use logits ε tolerance, not exact equality. Greedy parity on token ids is fine. |
| 3D-RoPE mock prefill is expensive and erodes vLLM's perf win | Medium | Memoize position_ids by `(grid_thw_hash, seq_len)`. Likely cheap after first hit. |
| Latent-judge KV crop is a real blocker if we need the judge | High | Decide upfront: is the judge required? If yes, Phase 3 option (b) and budget +2 weeks. |
| Heterogeneous batches (Phase 4) need kernel-level changes | Medium | Ship naive path first; defer heterogeneous unless profiling forces it. |
| `_decode_vision_latent_with_pca` runs CPU-heavy ops mid-loop | Low | Profile; port to GPU if it shows up. |

## Total budget (rough)

| Phase | Estimate |
|---|---|
| 0 — Scoping spike | 1 week |
| 1 — Text-mode + trigger | 1–2 weeks |
| 2 — Latent loop, single seq | 2–3 weeks |
| 3 — KV rollback (optional) | 0 or +2 weeks |
| 4 — Mixed-mode batching | 3–4 weeks (naive: 1 week) |
| 5 — Serve, bench, integrate | 1 week |
| **Total (v1, naive batching, no judge rollback)** | **~6–8 weeks** |
| **Total (with judge + heterogeneous batching)** | **~10–13 weeks** |

## Decision points before kickoff

1. **Is `enable_latent_judge` required at inference time, or train-time only?** This is the biggest single scoping question. If train-only, drop Phase 3 entirely.
2. **Are we OK with vLLM 0.7.x V1 (still maturing) or do we need 0.6.x V0?** V0 is more stable but V1 is much more amenable to this work. Recommend V1.
3. **Look at SGLang as an alternative target before kickoff.** SGLang's Python-DSL runtime (`sgl.fork`, `sgl.gen`) lets you write custom decoding control flow as ordinary Python that the runtime executes. The latent loop maps onto SGLang's primitives more naturally than vLLM's. A 2-day spike comparing the two would change the plan; do it before Phase 0.
4. **What's the actual deployment target?** If it's batch evaluation on VLMEvalKit, plain HF + flash-attn-2 is probably enough and Option A is overkill. If it's a serving product, the throughput-at-concurrency win justifies the work.

## Files to read first (for whoever picks this up)

- `gap/modeling_gap.py:1197-1850` — the full `generate_interleaved` definition.
- `gap/modeling_gap.py:_project_vision_prediction`, `_decode_vision_latent_with_pca`, `_mask_control_tokens` — the GAP-specific math.
- `vllm/v1/worker/gpu_model_runner.py` — where Phase 2a edits land.
- `vllm/v1/sample/sampler.py` — where the logits processor plugs in.
- `vllm/model_executor/models/qwen2_5_vl.py` — the backbone we subclass.
