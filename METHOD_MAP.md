# Method-to-code mapping

This document connects the paper's method to the implementation without redistributing the manuscript or benchmark outputs.

## Online EDU segmentation

- `Streaming_EDU/src/final_edu/model.py`: `StreamingTransformerEDU.forward` combines token, shape, cue, character, and online-state features. `prefix_proj` and cumulative means implement prefix history.
- `lookahead_mask` limits attention to positions satisfying `j <= i + k`.
- `EDU_codex/src/stream_edu/train_variant.py`: end-boundary BCE, auxiliary start-boundary BCE, distance classification, boundary-density MSE, and weak-label supervision. The README command sets auxiliary weights to 0.22, 0.18, 0.32, and 0.02.
- `Streaming_EDU/src/final_edu/segmenter.py`: incremental commitment, threshold decisions, and end-of-input flushing. Checkpoint configuration determines the actual architecture.
- The deployment implementation also contains deterministic safeguards for numbers, brackets, measurement expressions, and phrases. Some safeguards wait for additional context. These are retained implementation details; deployment behavior is not identical to thresholding an isolated boundary head.

## EDU-aligned SFT and Arrival-Triggered Preemption (ATP)

- `StreamingThinker/dataloader_hf.py`: constructs input and target segments, lengths, and supervision masks using markers such as `<EOQ>`, `<EOS>`, and `<EOT>`.
- `StreamingThinker/Qwen3/qwen_streaming.py`: streaming attention and position handling. `streaming_trainer.py` supplies the SFT interface.
- `StreamingThinker/online_edu_segmenter.py`: converts incrementally available text into committed EDUs and maintains their arrival timeline.
- `StreamingThinker/generation/generate.py`: `arrival_preemption` (ATP) checks available input at generation-token boundaries, incrementally prefills new source input, and resumes with the updated visible prefix. Existing reasoning tokens and their caches are preserved. `finish_eot` (WTC) waits for the current local thought to finish before updating context.

## EDU-HRPO

- `ST_RL/EDU_HRPO/counterfactual.py` and `fsdp_integration.py`: construct the full, stream-only, and no-stream answer views. The no-stream view is shared within a prompt group.
- `rewards.py:compute_edu_blhrpo_advantages`: quality components, helpful streaming contributions, persistent interference, correction burden, stage attribution, and group normalization.
- `latency.py:compute_latency_batch`: converts arrival intervals and generation throughput into overlap capacity, separates hidden and exposed tokens, and calculates EDU budgets, length rewards, trigger penalties, and trajectory costs.
- `trainer.py` calls `ST_RL/HRPO/hrpo.py:hrpo_loss`, preserving HRPO policy clipping.
- `enable_edu_controller=false` is the default. `controller.py` is retained because its data types also support fixed EDU grouping; the optional dynamic controller is not part of the paper's default entry point.
- `process_verifier.py` supplies an optional interface. No separate process verifier is used in the paper. Pass a positive, pre-calibrated `reward_tau` explicitly.
- RL input timing uses tokenizer tokens per second. Evaluation can use English words per minute; these units require separate calibration.

## Reward definitions and paper configuration

All three answer views use the same complete source input and task scorer. The full view scores the sampled trajectory (`Q`); the stream-only probe retains its local thoughts and generates an answer with deep thinking disabled (`Q_S`); the no-stream probe replaces local thoughts with `<ignore>` while retaining boundary markers, then performs deep reasoning (`Q_0`). The no-stream probe is shared within a problem group. Probes use greedy decoding with limits of 16 and 512 new tokens, respectively, and supply reward measurements rather than additional policy-update trajectories.

With `[x]_+ = max(x, 0)`, attribution follows the paper:

- Helpful contribution: `[Q_S - Q_0]_+`.
- Persistent interference: `[Q_0 - max(Q_S, Q)]_+`.
- Correction burden: `[Q - Q_S]_+ * min(1, L_D / max(B_D, 1))`.
- Streaming attribution: helpful contribution minus persistent interference minus `0.5` times correction burden.
- Deep attribution: `Q - Q_S`.

Quality is `(exact + 0.2 * partial) / 1.2`. Exact correctness is binary; extractive QA uses answer-token F1 for partial credit, while mathematics and classification use zero partial credit. A valid probe requires a complete, nonempty boxed answer. Truncation is tracked separately. If only the no-stream probe is invalid, streaming attribution becomes `Q_S - 0.5 * correction_burden`; deep attribution is unchanged. An invalid stream-only probe masks both attribution advantages, leaving other rewards available.

For EDU `g`, overlap capacity is `floor((tau[g+1] - tau[g]) * v_gen)`. The final EDU has zero additional waiting allowance. Overlapped and exposed generated tokens incur weights `0.1` and `1.0`. The EDU budget is `clip(4 + 0.4 * input_token_count + 16 * relevance, 4, 64)`, where relevance is the proportion of EDU terms overlapping with the question. Fixed EDU grouping sets control uncertainty to zero, so the optional uncertainty term contributes nothing. Streaming length rewards penalize budget overruns and nonempty-thought frequency (`0.05`); deep length rewards penalize generation beyond the calibrated deep budget. Global efficiency is gated by exact correctness and both stage-format checks, and decays exponentially with weighted streaming cost plus deep generation length.

[`configs/edu_hrpo_paper.yaml`](configs/edu_hrpo_paper.yaml) specifies the paper's attribution weights (`0.4`, `0.2`), format weight (`2`), streaming/deep cost weights (`0.02`, `0.05`), and global efficiency weight (`0.05`). Reward components are normalized within each problem group. Attribution normalization uses valid stream-only probes; fewer than two valid samples or a standard deviation below `1e-8` yields zero normalized attribution advantage.

HRPO clips streaming-local, deep-local, and global terms separately, averages each over its applicable response tokens, and combines the global term with `0.05` times the two local terms. The clipping range is `0.2`; source and padding positions are excluded; the KL coefficient is zero. Calibrate generation throughput, the deep budget, and the fixed cost scale before training rather than treating them as universal constants.

## Scope

The release includes training and inference code with local dependencies. It excludes weights, datasets, prediction outputs, broad parameter searches, job queues, and personal server launch scripts. It does not claim unperformed end-to-end numerical reproduction.
