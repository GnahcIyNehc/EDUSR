# Validation scope

This document describes checks of the English release, not a replication of the paper's experimental results.

- Source syntax and manifest integrity are checked with `python tools/check_source.py`.
- Core unit tests cover rewards, arrival-aware costs, counterfactual views, training logic, and isolated FSDP integration.
- Entry-point smoke checks request help for segmentation, segmenter training/evaluation, input preparation, SFT, streaming evaluation, and RL.
- README links and figure paths are checked locally. The three supplied figures are visually inspected after rendering; identifying PDF metadata is removed from release copies.
- The release uses a fresh Git history with a repository-local anonymous author identity. No prior repository history is included. This reduces accidental attribution but does not guarantee unlinkability of publicly shared research material.

Test counts and entry-point outcomes are recorded in `VALIDATION_RESULTS.json` and `ENTRYPOINT_CHECKS.json` after verification. The source directories and original figures are not modified.

Validation uses Python 3.11 on Windows with existing local dependencies. No full GPU inference, SFT, or distributed vLLM/FSDP training is rerun. Model/data assets and calibrated cost parameters are still required. Installation in a clean Linux GPU environment has not been validated by these checks.

Developer comments retain their original language; documentation, CLI descriptions, and runtime diagnostics are localized into English. Algorithmic behavior and HRPO clipping are unchanged.
