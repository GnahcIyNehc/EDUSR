# Data and checkpoint formats

## Segmenter training

The loader expects `train/gum_train.online.jsonl` and `dev/gum_dev.online.jsonl` under the specified data root. Each line contains aligned sequences:

```json
{"tokens":["A","battery","charges","."],"end_labels":[0,0,0,1],"start_labels":[1,0,0,0]}
```

This is a synthetic format example. `start_labels` is optional in the loader. Convert other corpora using their official EDU annotations and keep training, validation, and test splits separate. The expected filenames are a compatibility convention.

## Streaming input and SFT

Evaluation data is a JSON array of records containing `Instruct_Context`, with `instruct` for the question and `context` for the supporting text. SFT also needs aligned local reasoning, deep reasoning, and the final answer:

```json
[
  {
    "Instruct_Context": {
      "instruct": "How many units remain?<EOQ>",
      "context": "There are ten units.<EOS>Three units are consumed.<EOS>"
    },
    "streaming_reasoning": "Need the remaining quantity.<EOQ>The initial quantity is ten.<EOT>Subtract three from ten.<EOT>",
    "deep_reasoning": "Ten minus three equals seven.",
    "answer": "\\boxed{7}"
  }
]
```

This is a synthetic illustration, not training data from the experiments. The number of local reasoning segments must match the total number of question/context input units. The final question reasoning segment ends with `<EOQ>`; context reasoning segments end with `<EOT>`.

`configs/sft.json` points to `data/sft_train.json` at the repository root. Inside task configurations, relative `file_path` values resolve against `StreamingThinker/`. Adjust prompts and answer formats to match the task. The complete parsing logic is in `StreamingThinker/dataloader_hf.py`.

`prepare-inputs` changes input granularity but does not create teacher reasoning targets. Task prompt templates are not datasets.

## RL

JSON arrays and JSONL are supported. A recommended record is:

```json
{"question":"How many units remain?","context":["There are ten units.","Three units are consumed."],"answer":"7"}
```

Each context-list element should be an already segmented EDU. Supported aliases are defined in `ST_RL/training_utils.py:records_to_samples`.

## External assets

- Supply Qwen3, SFT, and RL checkpoints separately and comply with their licenses.
- Supply or train the segmenter checkpoint, including its vocabulary and configuration. Load trusted weights only.
- GUM/eRST: https://github.com/amir-zeldes/gum
- SciDTB: https://github.com/PKU-TANGENT/SciDTB
- STAC through DISRPT: https://github.com/disrpt/sharedtask2025

No corpus data or experimental predictions are redistributed in this repository.
