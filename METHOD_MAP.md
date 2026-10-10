# Method-to-code mapping

This document connects the paper's method to the implementation without redistributing the manuscript or benchmark outputs.

## Online EDU segmentation

- `Streaming_EDU/src/final_edu/model.py`: `StreamingTransformerEDU.forward` combines token, shape, cue, character, and online-state features. `prefix_proj` and cumulative means implement prefix history.
- `lookahead_mask` limits attention to positions satisfying `j <= i + k`.
- `EDU_codex/src/stream_edu/train_variant.py`: end-boundary BCE, auxiliary start-boundary BCE, distance classification, boundary-density MSE, and weak-label supervision. The README command sets auxiliary weights to 0.22, 0.18, 0.32, and 0.02.
- `Streaming_EDU/src/final_edu/segmenter.py`: incremental commitment, threshold decisions, and end-of-input flushing. Checkpoint configuration determines the actual architecture.
- The deployment implementation also contains deterministic safeguards for numbers, brackets, measurement expressions, and phrases. Some safeguards wait for additional context. These are retained implementation details; deployment behavior is not identical to thresholding an isolated boundary head.

## SFT and arrival preemption

- `StreamingThinker/dataloader_hf.py`: constructs input and target segments, lengths, and supervision masks using markers such as `<EOQ>`, `<EOS>`, and `<EOT>`.
- `StreamingThinker/Qwen3/qwen_streaming.py`: streaming attention and position handling. `streaming_trainer.py` supplies the SFT interface.
- `StreamingThinker/online_edu_segmenter.py`: converts incrementally available text into committed EDUs and maintains their arrival timeline.
- `StreamingThinker/generation/generate.py`: `arrival_preemption` checks available input at generation-token boundaries and updates visible context; `finish_eot` waits for the current local thought to finish.

## EDU-HRPO

- `ST_RL/EDU_HRPO/counterfactual.py` and `fsdp_integration.py`: construct the full, stream-only, and no-stream answer views. The no-stream view is shared within a prompt group.
- `rewards.py:compute_edu_blhrpo_advantages`: quality components, helpful streaming contributions, persistent interference, correction burden, stage attribution, and group normalization.
- `latency.py:compute_latency_batch`: converts arrival intervals and generation throughput into overlap capacity, separates hidden and exposed tokens, and calculates EDU budgets, length rewards, trigger penalties, and trajectory costs.
- `trainer.py` calls `ST_RL/HRPO/hrpo.py:hrpo_loss`, preserving HRPO policy clipping.
- `enable_edu_controller=false` is the default. `controller.py` is retained because its data types also support fixed EDU grouping; the optional dynamic controller is not part of the paper's default entry point.
- `process_verifier.py` supplies an optional interface. The paper configuration does not require another verifier model. Pass a positive, pre-calibrated `reward_tau` explicitly.
- RL input timing uses tokenizer tokens per second. Evaluation can use English words per minute; these units require separate calibration.

## Scope

The release includes training and inference code with local dependencies. It excludes weights, datasets, prediction outputs, broad parameter searches, job queues, and personal server launch scripts. It does not claim unperformed end-to-end numerical reproduction.
