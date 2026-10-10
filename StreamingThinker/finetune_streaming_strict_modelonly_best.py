from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, TrainerCallback, TrainingArguments

import utils
from dataloader_hf import StreamingDataCollator
from streaming_trainer import StreamingSFTTrainer


REPO_ROOT = Path(__file__).resolve().parent
BASE_SPECIAL_TOKENS = ["<SEP>", "<EOQ>", "<EOS>", "<EOT>", "<EOR>", "<EOA>", "<ignore>"]
HIERARCHICAL_EDU_TOKENS = ["<EOE>", "<EOP>"]
SUPPORTED_BACKBONES = ["Qwen3"]


def parse_args():
    parser = ArgumentParser(description="Fine-tune StreamingThinker with batch or streaming supervision.")

    parser.add_argument(
        "--model_name_or_path",
        "--LLM_path",
        dest="model_name_or_path",
        required=True,
        help="Base model or local checkpoint path passed to transformers.from_pretrained().",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        "--Tokenizer_path",
        dest="tokenizer_name_or_path",
        default=None,
        help="Tokenizer path or Hugging Face model ID. Defaults to --model_name_or_path.",
    )
    parser.add_argument(
        "--model_backbone",
        "--LLM_backbone",
        dest="model_backbone",
        default="Qwen3",
        choices=SUPPORTED_BACKBONES,
        help="Streaming model implementation to load.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention kernel used during training. SDPA reduces long-sequence activation memory.",
    )
    parser.add_argument(
        "--data_config",
        "--params",
        dest="data_config",
        default=str(REPO_ROOT / "configs" / "params_qwen_D2.json"),
        help="JSON config that defines the dataset path and prompt template.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "outputs" / "checkpoints"),
        help="Directory used by Hugging Face Trainer for checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        "--log_dir",
        dest="logging_dir",
        default=str(REPO_ROOT / "outputs" / "logs"),
        help="Directory for Trainer logs.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional experiment name shown by the selected reporter.",
    )

    parser.add_argument("--training_mode", default="streaming", choices=["batch", "streaming"])
    parser.add_argument("--split_mode", default="sentence", choices=["word", "sentence", "token"])
    parser.add_argument("--num_train_epochs", "--epochs", dest="num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--per_device_train_batch_size",
        "--per_bs",
        dest="per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device micro batch size.",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        "--acc_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument(
        "--lr_scheduler_type",
        "--warmup_type",
        dest="lr_scheduler_type",
        default="linear",
    )
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument(
        "--save_only_model",
        action=BooleanOptionalAction,
        default=True,
        help="Save model weights only, skipping optimizer/scheduler states to reduce checkpoint size.",
    )
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum training steps. -1 means use epochs.")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint path, or 'true' to resume from the latest checkpoint under output_dir.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--report_to", default="none", help="Trainer reporter, e.g. 'wandb' or 'none'.")
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.0,
        help="Deterministic fraction held out for eval_loss model selection. Use 0 to disable evaluation.",
    )
    parser.add_argument(
        "--select_best_by_train_loss",
        action=BooleanOptionalAction,
        default=False,
        help="Without validation, retain the checkpoint with the lowest epoch mean logged training loss.",
    )
    parser.add_argument(
        "--use_eoe_tokens",
        action=BooleanOptionalAction,
        default=True,
        help="Register <EOE>/<EOP>. Disable this for the no-EOE Full-EDU dataset.",
    )

    parser.add_argument(
        "--use_lora",
        "--lora",
        dest="use_lora",
        action=BooleanOptionalAction,
        default=False,
        help="Enable LoRA fine-tuning.",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--gradient_checkpointing",
        "--gradient_checkpoint",
        dest="gradient_checkpointing",
        action=BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--batch_from_position_zero",
        "--is_batch_from_zero",
        dest="batch_from_position_zero",
        action=BooleanOptionalAction,
        default=False,
        help="Reset target position indices for batch decoding.",
    )
    parser.add_argument(
        "--deepspeed",
        "--deepspeed_path",
        dest="deepspeed_config",
        default=str(REPO_ROOT / "configs" / "deepspeed_config.json"),
        help="DeepSpeed config file used by Trainer. Use 'none' to disable DeepSpeed.",
    )

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class TrainLossBestCallback(TrainerCallback):
    """Track the mean of logged training losses separately for each epoch."""

    def __init__(self):
        self.epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.epoch_losses.append(float(logs["loss"]))

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.epoch_losses:
            print(f"WARNING: no logged training losses were collected for epoch {state.epoch}.")
            return control

        epoch_mean_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        self.epoch_losses.clear()
        is_better = state.best_metric is None or epoch_mean_loss < state.best_metric
        if is_better:
            state.best_metric = epoch_mean_loss
            state.best_global_step = state.global_step
        print(
            f"Epoch train-loss selection: epoch={state.epoch}, "
            f"mean_logged_loss={epoch_mean_loss:.8f}, is_best={is_better}, "
            f"best_loss={state.best_metric:.8f}, best_step={state.best_global_step}"
        )
        return control


def build_tokenizer(tokenizer_name_or_path, config, use_eoe_tokens=True):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side="right", config=config)
    special_tokens = list(BASE_SPECIAL_TOKENS)
    if use_eoe_tokens:
        special_tokens.extend(HIERARCHICAL_EDU_TOKENS)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_class(model_backbone):
    if model_backbone == "Qwen3":
        from Qwen3.qwen_streaming import Qwen3ForCausalLM_stream

        return Qwen3ForCausalLM_stream

    raise ValueError(f"Unsupported model_backbone: {model_backbone}")


def main():
    args = parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in [0, 1).")
    if args.validation_fraction > 0 and args.select_best_by_train_loss:
        raise ValueError(
            "Choose one checkpoint-selection policy: validation eval_loss or training loss, not both."
        )
    setup_seed(args.seed)

    output_dir = Path(args.output_dir)
    logging_dir = Path(args.logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    params = utils.Params(args.data_config)
    params.file_path = utils.resolve_path(params.file_path, REPO_ROOT)

    tokenizer_name_or_path = args.tokenizer_name_or_path or args.model_name_or_path

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config._attn_implementation = args.attn_implementation
    config.use_cache = False

    tokenizer = build_tokenizer(tokenizer_name_or_path, config, use_eoe_tokens=args.use_eoe_tokens)
    model_cls = get_model_class(args.model_backbone)

    data_collator = StreamingDataCollator(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        end_Instruct=params.end_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        reasoning_connect_1=params.reasoning_connect_1,
        reasoning_connect_2=params.reasoning_connect_2,
        reasoning_connect_3=params.reasoning_connect_3,
        training_mode=args.training_mode,
        split_mode=args.split_mode,
        pe_cache_start=0,
        is_batch_from_pe_zero=args.batch_from_position_zero,
    )

    report_to = [] if args.report_to == "none" else [args.report_to]
    deepspeed_config = None if str(args.deepspeed_config).lower() in {"", "none", "false", "no"} else args.deepspeed_config
    full_dataset = data_collator.dataset_loader()
    if args.validation_fraction > 0:
        dataset_split = full_dataset.train_test_split(
            test_size=args.validation_fraction,
            seed=args.seed,
            shuffle=True,
        )
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        save_strategy = "epoch"
        eval_strategy = "epoch"
        track_best_checkpoint = True
        print(
            f"Dataset split: train={len(train_dataset)}, eval={len(eval_dataset)}, "
            f"validation_fraction={args.validation_fraction:.4f}"
        )
    else:
        train_dataset = full_dataset
        eval_dataset = None
        save_strategy = "epoch" if args.select_best_by_train_loss else "steps"
        eval_strategy = "no"
        track_best_checkpoint = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        remove_unused_columns=False,
        logging_dir=str(logging_dir),
        logging_steps=args.logging_steps,
        report_to=report_to,
        run_name=args.run_name or output_dir.name,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        save_safetensors=True,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=True,
        deepspeed=deepspeed_config,
        seed=args.seed,
        eval_strategy=eval_strategy,
        # ZeRO-3 model-only checkpoints are Hugging Face weights, not native
        # DeepSpeed checkpoints. Track and retain the best one without asking
        # Trainer to reload it through deepspeed_load_checkpoint at shutdown.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss" if track_best_checkpoint else None,
        greater_is_better=False if track_best_checkpoint else None,
        prediction_loss_only=True,
    )

    model = model_cls.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "up_proj", "down_proj","lm_head"],
            lora_dropout=args.lora_dropout,
            bias="all",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")
    print(f"Grad size:  {sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e9:.2f} GB")

    callbacks = [TrainLossBestCallback()] if args.select_best_by_train_loss else None
    trainer = StreamingSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator.collate_fn,
        training_mode=args.training_mode,
        callbacks=callbacks,
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        if str(args.resume_from_checkpoint).lower() == "true":
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if trainer.is_world_process_zero():
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        selection = {
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "final_checkpoint": str(checkpoints[-1]) if checkpoints else None,
            "selection_metric": (
                "eval_loss"
                if eval_dataset is not None
                else "epoch_mean_logged_train_loss"
                if args.select_best_by_train_loss
                else None
            ),
        }
        with (output_dir / "checkpoint_selection.json").open("w", encoding="utf-8") as file:
            json.dump(selection, file, ensure_ascii=False, indent=2)
        print("Checkpoint selection:", selection)


if __name__ == "__main__":
    main()
