import torch
from transformers import Trainer


class StreamingSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        training_mode="batch",
        skip_oom_batches=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.training_mode = training_mode
        self.skip_oom_batches = skip_oom_batches

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        model_inputs = dict(inputs)

        try:
            labels = model_inputs.pop("labels")
            position_ids = model_inputs.pop("position_ids", None)
            training_mode = model_inputs.pop("training_mode", self.training_mode)
            split_mode = model_inputs.pop("split_mode", None)
            model_inputs.pop("source_seg_len", None)
            model_inputs.pop("target_seg_len", None)
            lengths = model_inputs.pop("_lengths", None)

            outputs = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=position_ids,
                training_mode=training_mode,
                split_mode=split_mode,
                is_training=True,
                labels=labels,
                _lengths=lengths,
                # 训练损失不需要返回逐层注意力矩阵，长序列下保留它们会显著增加显存。
                output_attentions=False,
                num_items_in_batch=num_items_in_batch,
            )
            loss = outputs.loss
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise

            print(f"[Rank {self.args.local_rank}] OOM detected at step {self.state.global_step}")
            input_ids = model_inputs.get("input_ids")
            if input_ids is not None:
                print(f"OOM batch shape: {tuple(input_ids.shape)}")

            torch.cuda.empty_cache()
            # DeepSpeedEngine 的 zero_grad 在部分版本中不接受 set_to_none。
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()
            if not self.skip_oom_batches:
                raise RuntimeError(
                    "CUDA out of memory during training. Aborting instead of "
                    "silently skipping the batch. Reduce the micro batch size, "
                    "increase gradient accumulation, or validate the longest "
                    "sample before restarting."
                ) from error

            print(
                "WARNING: skipping this OOM batch because "
                "--skip_oom_batches was explicitly enabled."
            )
            loss = torch.zeros((), device=self.args.device, requires_grad=True)
            outputs = None

        return (loss, outputs) if return_outputs else loss
