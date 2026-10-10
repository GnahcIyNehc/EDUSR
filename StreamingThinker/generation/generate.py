from transformers.generation.utils import *
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import DynamicCache
from .Stopping_criteria import StopTokenCriteria
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch


def _extract_complete_boxed_answer(text):
    """Return the first complete boxed answer body, supporting nested braces."""
    if not text:
        return ""

    match = re.search(r"\\box(?:ed)?\s*\{", text)
    if match is None:
        match = re.search(r"(?<![A-Za-z])box(?:ed)?\s*\{", text)
    if match is None:
        return ""

    open_brace = text.find("{", match.end() - 1)
    depth = 0
    body_start = open_brace + 1
    for index in range(open_brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = text[body_start:index].strip()
                while answer.startswith("{") and answer.endswith("}"):
                    answer = answer[1:-1].strip()
                return answer
    return ""

# #...
# from .GPU_monitor import PerformanceMonitor, setup_logger
# logger = setup_logger("./logs/experiment_results.log")

# ...
# add pop method to DynamicCache
class DynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()

    def pop(self):
        self._seen_tokens -= 1

        # Update the cache
        target_key_cache = []
        target_value_cache = []
        for key_cache,value_cache in zip(self.key_cache, self.value_cache):
            target_key_cache.append(key_cache[...,:-1,:])
            target_value_cache.append(value_cache[...,:-1,:])
        self.key_cache = target_key_cache
        self.value_cache = target_value_cache

class unified_PreTrainedModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


    # The generate method inherited from transformers.generation.utils.GenerationMixin, where GenerationMixin is first inherited by PreTrainedModel
    @torch.no_grad()
    def generate(
        self,
        # generate_mode: str = "batch", # must be "batch", "streaming"
        # split_mode: Optional[str] = None, # must be one of ["token", "word", "sentence"] if generate_mode == "streaming"
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        generate_mode = kwargs.get("generate_mode", "batch") # must be "batch", "streaming"
        split_mode = kwargs.get("split_mode", None) # must be one of ["token", "word", "sentence"] if generate_mode == "streaming"
        # pe_cache_length = kwargs.get("pe_cache_length", 0) 
        is_batch_from_pe_zero = kwargs.get("is_batch_from_pe_zero", False) 

        if generate_mode == "batch": # same as transformers.generation.utils.GenerationMixin.generate
            wait_lagging = None
            if is_batch_from_pe_zero is False:
                return super().generate(
                    inputs,
                    generation_config,
                    logits_processor,
                    stopping_criteria,
                    prefix_allowed_tokens_fn,
                    synced_gpus,
                    assistant_model,
                    streamer,
                    negative_prompt_ids,
                    negative_prompt_attention_mask,
                    use_model_defaults,
                    **kwargs,
                ), wait_lagging
            elif is_batch_from_pe_zero is True:
                kwargs['prefill_length'] = kwargs['attention_mask'].sum(axis=1) # input length for each sample
                kwargs['is_prefill'] = True # input length for each sample
                return super().generate(
                    inputs,
                    generation_config,
                    logits_processor,
                    stopping_criteria,
                    prefix_allowed_tokens_fn,
                    synced_gpus,
                    assistant_model,
                    streamer,
                    negative_prompt_ids,
                    negative_prompt_attention_mask,
                    use_model_defaults,
                    **kwargs,
                ), wait_lagging

        
        elif generate_mode == "streaming":
            assert split_mode in ["token", "word", "sentence"], f"streaming_split must be one of ['token', 'word', 'sentence'], but got {split_mode}."
            result, wait_lagging = self.streaming_generate( 
                split_mode,              
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                assistant_model,
                streamer,
                negative_prompt_ids,
                negative_prompt_attention_mask,
                use_model_defaults,
                **kwargs,
            )
            return result, wait_lagging
        
        else:
            raise ValueError(f"generate_mode must be one of ['batch', 'streaming'], but got {generate_mode}.")

    # ✅ STEP 1: Add the overridden method directly into this class.
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,

        prefill_length = None,
        is_prefill = False,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """
        # prefill_length = kwargs.pop("prefill_length", None)
        # is_prefill = kwargs.pop("is_prefill", False)

        if prefill_length is None:
            return super().prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                inputs_embeds,
                cache_position,
                **kwargs,
            )
        else: # for group position encoding
            # 1. Handle BC:
            model_inputs = {}
            # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
            if self._supports_cache_class:
                model_inputs["cache_position"] = cache_position
            # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
            #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
            #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
            elif cache_position is None:
                past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

            # 2. Generic cache-dependent input preparation
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
                inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                    input_ids, inputs_embeds, cache_position
                )

            # 3. Prepare base model inputs
            input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
            if not self.config.is_encoder_decoder:
                if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                    model_inputs[input_ids_key] = None
                    model_inputs["inputs_embeds"] = inputs_embeds
                else:
                    # `clone` calls in this function ensure a consistent stride. See #32227
                    model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                    model_inputs["inputs_embeds"] = None
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

            # 4. Create missing `position_ids` on the fly
            encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
            attention_mask = (
                kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
            )
            attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
            position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
            if (
                attention_mask is not None
                and kwargs.get(position_ids_key) is None
                and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
            ):
                # ...
                if is_prefill is True: # prefill
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    # # set to False
                    # model_inputs['is_prefill'] = False
                    # model_inputs['prefill_length'] = prefill_length
                else:
                    position_ids = attention_mask.long().cumsum(-1) - 1 - prefill_length.unsqueeze(1)
                    # model_inputs['is_prefill'] = False
                    # model_inputs['prefill_length'] = prefill_length


                position_ids.masked_fill_(attention_mask == 0, 1)
                kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

            # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
            for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
                model_input = kwargs.get(model_input_name)
                if model_input is not None:
                    if past_key_values is not None:
                        current_input_length = (
                            model_inputs["inputs_embeds"].shape[1]
                            if model_inputs.get("inputs_embeds") is not None
                            else model_inputs[input_ids_key].shape[1]
                        )
                        model_input = model_input[:, -current_input_length:]
                        model_input = model_input.clone(memory_format=torch.contiguous_format)
                    model_inputs[model_input_name] = model_input

            # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                if model_inputs["inputs_embeds"] is not None:
                    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                    device = model_inputs["inputs_embeds"].device
                else:
                    batch_size, sequence_length = model_inputs[input_ids_key].shape
                    device = model_inputs[input_ids_key].device

                # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
                # the 4D causal mask exists, it should be present in the base model (XXXModel class).
                base_model = getattr(self, self.base_model_prefix, None)
                if base_model is None:
                    causal_mask_creation_function = getattr(
                        self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                    )
                else:
                    causal_mask_creation_function = getattr(
                        base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                    )
                if causal_mask_creation_function is None:
                    logger.warning_once(
                        f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                        "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                        "writing code, see Llama for an example implementation. If you're a user, please report this "
                        "issue on GitHub."
                    )
                else:
                    attention_mask = causal_mask_creation_function(
                        attention_mask,
                        sequence_length=sequence_length,
                        target_length=past_key_values.get_max_cache_shape(),
                        dtype=self.dtype,
                        device=device,
                        cache_position=cache_position,
                        batch_size=batch_size,
                        config=self.config,
                        past_key_values=past_key_values,
                    )
            if attention_mask is not None:
                model_inputs[attention_mask_key] = attention_mask

            if encoder_attention_mask is not None:
                model_inputs["attention_mask"] = encoder_attention_mask

            # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
            for key, value in kwargs.items():
                if key not in model_inputs:
                    model_inputs[key] = value

            # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
            model_inputs.pop("labels", None)
            return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        #...
        model_kwargs['is_prefill'] = False
        return model_kwargs

      
    def streaming_generate(
        self,
        streaming_split: str = "word", # must be one of ["token", "word", "sentence"] 
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            use_model_defaults (`bool`, *optional*):
                When it is `True`, unset parameters in `generation_config` will be set to the model-specific default
                generation configuration (`model.generation_config`), as opposed to the global defaults
                (`GenerationConfig()`). If unset, models saved starting from `v4.50` will consider this flag to be
                `True`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        
        '''prepare'''
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # Transformers 4.55 已移除该私有校验接口；旧版本存在时仍执行。
        if hasattr(self, "_validate_model_class"):
            self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation
        latency_recorder = kwargs.pop("latency_recorder", None)
        online_edu_boundary_stream = kwargs.pop("online_edu_boundary_stream", None)
        force_complete_source = kwargs.pop("force_complete_source", False)
        max_write_tokens_before_read = kwargs.pop("max_write_tokens_before_read", None)
        defer_thinking_until_context = kwargs.pop("defer_thinking_until_context", False)
        finish_thought_before_read = kwargs.pop(
            "finish_thought_before_read", False
        )
        read_policy = kwargs.pop("read_policy", None)
        initial_source_segments_before_thinking = kwargs.pop(
            "initial_source_segments_before_thinking", None
        )
        force_reflection_after_draft = kwargs.pop(
            "force_reflection_after_draft", False
        )
        forced_reflection_prompt = kwargs.pop(
            "forced_reflection_prompt", None
        )
        reflection_trigger_token = kwargs.pop(
            "reflection_trigger_token", "<EOA>"
        )
        forced_reflection_min_new_tokens = kwargs.pop(
            "forced_reflection_min_new_tokens", 0
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs_streaming(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )


        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)


        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache



        
        # 10. go into different generation modes
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result, wait_lagging = self._sample_streaming(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                tokenizer=tokenizer, # must provide
                latency_recorder=latency_recorder,
                online_edu_boundary_stream=online_edu_boundary_stream,
                force_complete_source=force_complete_source,
                max_write_tokens_before_read=max_write_tokens_before_read,
                defer_thinking_until_context=defer_thinking_until_context,
                finish_thought_before_read=finish_thought_before_read,
                read_policy=read_policy,
                initial_source_segments_before_thinking=initial_source_segments_before_thinking,
                force_reflection_after_draft=force_reflection_after_draft,
                forced_reflection_prompt=forced_reflection_prompt,
                reflection_trigger_token=reflection_trigger_token,
                forced_reflection_min_new_tokens=forced_reflection_min_new_tokens,
                **model_kwargs,
            )
        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam sample
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


        return result, wait_lagging

    def _validate_model_kwargs_streaming(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation_streaming).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
    ''' Separating the KV cache allows for loading source and target caches independently. 
        Merging the KV cache enables the target to attend to the entire KV cache content during the autoregressive decoding stage. 
        In the attention mechanism, the read phase uses the source cache, while the write phase uses the past cache.  
    '''
    def merge_source_target(self):
        # reset
        assert self.past_key_values is not None 
        assert self.source_key_values is not None
        assert self.target_key_values is not None
        self.past_key_values.key_cache = []
        self.past_key_values.value_cache = []
        if self.target_key_values.get_seq_length()==0:
            self.past_key_values.key_cache = self.source_key_values.key_cache.copy()
            self.past_key_values.value_cache = self.source_key_values.value_cache.copy()
        else:
            for source_key_cache, source_value_cache, target_key_cache, target_value_cache in zip(self.source_key_values.key_cache, self.source_key_values.value_cache, self.target_key_values.key_cache, self.target_key_values.value_cache):
                self.past_key_values.key_cache.append(torch.cat((source_key_cache, target_key_cache), dim=2))
                self.past_key_values.value_cache.append(torch.cat((source_value_cache, target_value_cache), dim=2))
                        
    def separate_source_target(self):
        assert self.past_key_values is not None 
        assert self.source_key_values is not None
        assert self.target_key_values is not None
        source_length = self.source_key_values.get_seq_length()
        if self.past_key_values.get_seq_length()> source_length:
            # reset
            self.source_key_values.key_cache = []
            self.source_key_values.value_cache = []
            self.target_key_values.key_cache = []
            self.target_key_values.value_cache = []
            for key_cache, value_cache in zip(self.past_key_values.key_cache, self.past_key_values.value_cache):
                self.source_key_values.key_cache.append(key_cache[...,:source_length,:])
                self.source_key_values.value_cache.append(value_cache[...,:source_length,:])
                self.target_key_values.key_cache.append(key_cache[...,source_length:,:])
                self.target_key_values.value_cache.append(value_cache[...,source_length:,:])

    # Copied and revised from GenerationMixin._get_initial_cache_position
    def _get_initial_cache_position_for_streaming(self, input_length, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        assert self.source_key_values is not None
        cache_position = torch.arange(
                self.source_key_values.get_seq_length(), input_length[0], dtype=torch.int64, device=model_kwargs.get('assistant_token').device
            )

        # past_length = 0
        # if model_kwargs.get("past_key_values") is not None:
        #     cache = model_kwargs["past_key_values"]
        #     past_length = 0
        #     if not isinstance(cache, Cache):
        #         past_length = cache[0][0].shape[2]
        #     elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
        #         past_length = cache.get_seq_length()

        #     cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    # Copied from transformers.models.persimmon.modeling_persimmon.Phi3ForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation_streaming(
        self,
        input_ids,
        input_length,
        attention_mask=None,
        use_cache=True,
        streaming =False,
        ReadAction = True,

        pe_cache_length = 0,
        assistant_token = None,
        source_words=None,
        end_Instruct = None,
        target_txt = None,
        **kwargs,
    ):
        # ...
        model_inputs = kwargs.copy()

        if streaming:

            if self.source_key_values is not None:
                past_source_length = self.source_key_values.get_seq_length()
                # past_target_length = self.target_key_values.get_seq_length()
                past_target_length = self.past_key_values.get_seq_length() - past_source_length

            if ReadAction:
                position_ids_source = torch.arange(past_source_length,input_length[0]).to(assistant_token.device).unsqueeze(0)
                position_ids = position_ids_source.clone().detach()
                past_length = past_source_length 
                input_ids = input_ids[:,past_source_length:input_length[0]]
            elif not ReadAction: 
                num_tokens = input_ids.shape[-1]
                if past_target_length==0:
                    # position_ids_target = range(pe_cache_length, pe_cache_length+num_tokens)
                    # position_ids = torch.tensor([position_ids_target]).to(assistant_token.device).squeeze(0)
                    position_ids = torch.arange(pe_cache_length, pe_cache_length+num_tokens).to(assistant_token.device).unsqueeze(0)
                else:
                    position_ids_target = past_target_length + pe_cache_length
                    position_ids = torch.tensor([position_ids_target]).to(assistant_token.device).unsqueeze(0)
                past_length = past_source_length
                input_ids = input_ids

            # model_inputs = {"input_ids": input_ids}

            if ReadAction:
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "source_key_values":self.source_key_values,
                    }
                )
            else:            
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "past_key_values":self.past_key_values,
                    }
                )
        return model_inputs
    

    def _sample_streaming(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        tokenizer: Optional["PreTrainedTokenizerBase"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        
        # ...
        # init cache
        self.source_key_values = DynamicCache()
        self.target_key_values = DynamicCache()
        self.past_key_values = DynamicCache()
        latency_recorder = model_kwargs.pop("latency_recorder", None)
        online_edu_boundary_stream = model_kwargs.pop("online_edu_boundary_stream", None)
        online_mode = online_edu_boundary_stream is not None
        force_complete_source = bool(model_kwargs.pop("force_complete_source", False))
        max_write_tokens_before_read = model_kwargs.pop(
            "max_write_tokens_before_read", None
        )
        if max_write_tokens_before_read is not None:
            max_write_tokens_before_read = int(max_write_tokens_before_read)
            if max_write_tokens_before_read <= 0:
                max_write_tokens_before_read = None
        defer_thinking_until_context = bool(
            model_kwargs.pop("defer_thinking_until_context", False)
        )
        finish_thought_before_read = bool(
            model_kwargs.pop("finish_thought_before_read", False)
        )
        read_policy = model_kwargs.pop("read_policy", None)
        if read_policy is None:
            read_policy = (
                "finish_eot"
                if finish_thought_before_read
                else "arrival_preemption"
            )
        if read_policy not in {
            "arrival_preemption",
            "finish_eot",
            "sentence_finish_eot",
        }:
            raise ValueError(f"Unsupported read_policy: {read_policy}")
        finish_thought_before_read = read_policy == "finish_eot"
        sentence_finish_eot = read_policy == "sentence_finish_eot"
        initial_source_segments_before_thinking = model_kwargs.pop(
            "initial_source_segments_before_thinking", None
        )
        if initial_source_segments_before_thinking is not None:
            initial_source_segments_before_thinking = int(
                initial_source_segments_before_thinking
            )
        force_reflection_after_draft = bool(
            model_kwargs.pop("force_reflection_after_draft", False)
        )
        forced_reflection_prompt = model_kwargs.pop(
            "forced_reflection_prompt", None
        )
        reflection_trigger_token = str(
            model_kwargs.pop("reflection_trigger_token", "<EOA>")
        )
        forced_reflection_min_new_tokens = int(
            model_kwargs.pop("forced_reflection_min_new_tokens", 0)
        )
        if forced_reflection_min_new_tokens < 0:
            raise ValueError("forced_reflection_min_new_tokens must be >= 0")
        if force_reflection_after_draft:
            if not isinstance(forced_reflection_prompt, str) or not forced_reflection_prompt.strip():
                raise ValueError(
                    "force_reflection_after_draft requires a non-empty "
                    "forced_reflection_prompt"
                )
        timing_enabled = isinstance(latency_recorder, dict)
        sync_cuda_for_latency = bool(latency_recorder.get("synchronize_cuda", True)) if timing_enabled else False

        def latency_now():
            if sync_cuda_for_latency and torch.cuda.is_available():
                torch.cuda.synchronize()
            if online_mode and hasattr(online_edu_boundary_stream, "latency_now"):
                return online_edu_boundary_stream.latency_now()
            return time.perf_counter()

        if timing_enabled:
            latency_recorder["start_time"] = latency_now()
            latency_recorder["forced_read_count"] = 0
            latency_recorder["finish_thought_before_read"] = (
                finish_thought_before_read
            )
            latency_recorder["read_policy"] = read_policy
            latency_recorder["fast"] = bool(
                online_mode and getattr(online_edu_boundary_stream, "fast", False)
            )
            latency_recorder["sentence_transition_finish_eot"] = (
                sentence_finish_eot
            )
            latency_recorder["force_reflection_after_draft"] = (
                force_reflection_after_draft
            )
            latency_recorder["reflection_trigger_token"] = (
                reflection_trigger_token
                if force_reflection_after_draft
                else None
            )
            latency_recorder["forced_reflection_min_new_tokens"] = (
                forced_reflection_min_new_tokens
                if force_reflection_after_draft
                else 0
            )

        # monitor = model_kwargs.get("monitor", None)

        # ...
        # tokenizer = model_kwargs.get("tokenizer", None)
        end_Instruct = model_kwargs.get("end_Instruct", '<|end|>')
        ReadAction_criteria = StopTokenCriteria(tokenizer,max_new_tokens=50, end_Instruct = end_Instruct)

        split_mode = model_kwargs.get("split_mode", True)

        _lengths = model_kwargs.get("_lengths", None)
        source_seg_len = _lengths[0]['source_seg_len']
        prefix_source_segments = int(_lengths[0].get('prefix_source_segments', 0))
        prefix_source_segments = max(0, min(prefix_source_segments, len(source_seg_len)))
        prompt_prefill_mode = _lengths[0].get(
            "prompt_prefill_mode",
            "immediate" if prefix_source_segments else None,
        )
        prefix_source_tokens = int(sum(source_seg_len[:prefix_source_segments]))
        total_source_tokens_raw = int(sum(source_seg_len))
        total_streaming_source_tokens = max(0, total_source_tokens_raw - prefix_source_tokens)
        first_real_read_segments = 1
        if prefix_source_segments:
            # The prompt prefix is read in its own forward pass. The first real
            # input segment is read next, before any target/thinking token is generated.
            first_real_read_segments = min(len(source_seg_len), prefix_source_segments + 1)
            initial_read_segments = prefix_source_segments
        else:
            initial_read_segments = 1
        initial_read_tokens_raw = int(sum(source_seg_len[:initial_read_segments])) if source_seg_len else 0
        first_real_read_tokens_raw = int(sum(source_seg_len[:first_real_read_segments])) if source_seg_len else 0
        if timing_enabled:
            latency_recorder["source_segments"] = len(source_seg_len)
            latency_recorder["prefix_source_segments"] = prefix_source_segments
            latency_recorder["prompt_prefix_tokens"] = prefix_source_tokens
            latency_recorder["prompt_prefill_mode"] = prompt_prefill_mode
            latency_recorder["prompt_rate_limited"] = False
            latency_recorder["streaming_source_segments"] = max(0, len(source_seg_len) - prefix_source_segments)
            latency_recorder["raw_first_read_tokens"] = first_real_read_tokens_raw
            latency_recorder["raw_total_source_tokens"] = total_source_tokens_raw
            latency_recorder["first_read_tokens"] = max(0, first_real_read_tokens_raw - prefix_source_tokens)
            latency_recorder["total_source_tokens"] = total_streaming_source_tokens
            if prefix_source_segments == 0:
                latency_recorder["prefix_prefill_done_time"] = latency_recorder["start_time"]
        # source_seg_len = [_length['source_seg_len'] for _length in _lengths]
        # _lengths_index = model_kwargs.get("_lengths_index", None)                  
        ReadAction = True

        wait_k = model_kwargs.get("wait_k", None)
        # source_words = model_kwargs['wait_k'] #source_words
        target_words = 0  #target_words
        max_distance = 10
        
        next_tokens = model_kwargs['assistant_token'].unsqueeze(0)
        target_tokens = [next_tokens[:,:-1], next_tokens[:,-1:]]
        target_start_len = sum(tokens.shape[-1] for tokens in target_tokens)
        target_ids = torch.cat(target_tokens, dim=-1)
        max_new_target_tokens = generation_config.max_new_tokens
        reflection_triggered = False
        reflection_completed = False
        forced_reflection_token_ids = []
        forced_reflection_token_cursor = 0
        reflection_prompt_finishing_this_token = False
        reflection_content_start_output_token_index = None
        reflection_trigger_token_id = None
        reflection_early_stop_token_ids = set()
        if force_reflection_after_draft:
            reflection_trigger_token_id = tokenizer.convert_tokens_to_ids(
                reflection_trigger_token
            )
            if (
                reflection_trigger_token_id is None
                or reflection_trigger_token_id == tokenizer.unk_token_id
                or not (0 <= int(reflection_trigger_token_id) < int(self.config.vocab_size))
            ):
                raise ValueError(
                    "Invalid reflection_trigger_token for this tokenizer/model: "
                    f"{reflection_trigger_token!r}"
                )
            forced_reflection_token_ids = tokenizer(
                forced_reflection_prompt,
                add_special_tokens=False,
            )["input_ids"]
            if not forced_reflection_token_ids:
                raise ValueError("forced_reflection_prompt tokenized to an empty sequence")
            for token_text in (
                reflection_trigger_token,
                "</think>",
                "<|im_end|>",
            ):
                token_id = tokenizer.convert_tokens_to_ids(token_text)
                if (
                    token_id is not None
                    and token_id != tokenizer.unk_token_id
                    and 0 <= int(token_id) < int(self.config.vocab_size)
                ):
                    reflection_early_stop_token_ids.add(int(token_id))
            if timing_enabled:
                latency_recorder["forced_reflection_prompt"] = (
                    forced_reflection_prompt
                )
                latency_recorder["forced_reflection_prompt_tokens"] = len(
                    forced_reflection_token_ids
                )
        target_tokens_this_write = []
        wait_lagging = [] # record the lagging of each target token
        # attentions_source = []
        # attentions_target = []
        # if wait_k is not None:
        #     input_length = (sum(source_seg_len[:model_kwargs['wait_k']+1]), 1)
        #     source_input_length = sum(source_seg_len[:model_kwargs['wait_k']+1])
        # else:
        #     input_length = (sum(source_seg_len[:2]), 1)
        input_length = (initial_read_tokens_raw, 1)
        pending_read_segments = initial_read_segments
        pending_online_event_index = None
        pending_online_final = False
        pending_online_source = None
        online_source_complete = False
        write_tokens_since_read = 0
        last_online_arrival_token_index = 0
        last_online_arrival_word_index = 0
        last_read_online_utterance_index = None
        last_read_online_source = None
        last_read_online_is_final_instruct = False

        def next_online_segment_matches_current_utterance() -> bool:
            if online_edu_boundary_stream is None:
                return False
            segment = online_edu_boundary_stream.peek_segment(block=False)
            if segment is None:
                return False
            if last_read_online_utterance_index is None:
                return True
            return (
                segment.source == last_read_online_source
                and segment.utterance_index == last_read_online_utterance_index
            )

        def queue_next_online_segment(block=True, read_reason=None):
            nonlocal input_ids
            nonlocal input_length
            nonlocal pending_read_segments
            nonlocal pending_online_event_index
            nonlocal pending_online_final
            nonlocal pending_online_source
            nonlocal online_source_complete
            nonlocal last_online_arrival_token_index
            nonlocal last_online_arrival_word_index
            nonlocal last_read_online_utterance_index
            nonlocal last_read_online_source
            nonlocal last_read_online_is_final_instruct

            segment = online_edu_boundary_stream.next_segment(block=block)
            if segment is None:
                if online_edu_boundary_stream.exhausted:
                    online_source_complete = True
                return False

            segment_tokens = tokenizer(
                segment.text,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(input_ids.device)
            segment_token_count = int(segment_tokens.shape[-1])
            if segment_token_count == 0:
                raise ValueError("Online EDU emitted an empty token segment.")

            input_ids = torch.cat([input_ids, segment_tokens], dim=-1)
            attention_mask = model_kwargs.get("attention_mask")
            if attention_mask is not None:
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(
                            (attention_mask.shape[0], segment_token_count)
                        ),
                    ],
                    dim=-1,
                )

            source_seg_len.append(segment_token_count)
            _lengths[0]["source_seg_len"] = source_seg_len
            _lengths[0]["source_token_len"] = int(sum(source_seg_len))
            pending_read_segments = len(source_seg_len)
            input_length = (int(sum(source_seg_len)), 1)
            pending_online_event_index = len(online_edu_boundary_stream.events) - 1
            pending_online_final = bool(segment.final_source_segment)
            pending_online_source = segment.source
            last_read_online_utterance_index = segment.utterance_index
            last_read_online_source = segment.source
            # 只有实际带有 <EOQ> 的最终问题片段才能结束问题阶段。
            last_read_online_is_final_instruct = (
                segment.source == "instruct" and "<EOQ>" in segment.text
            )
            online_edu_boundary_stream.mark_llm_read_started(
                pending_online_event_index
            )
            online_edu_boundary_stream.events[pending_online_event_index][
                "target_token_index_at_read"
            ] = int(target_ids.shape[-1] - target_start_len)
            online_edu_boundary_stream.events[pending_online_event_index][
                "read_reason"
            ] = read_reason
            last_online_arrival_token_index = int(
                online_edu_boundary_stream.events[pending_online_event_index].get(
                    "arrival_token_index", 0
                )
            )
            last_online_arrival_word_index = int(
                online_edu_boundary_stream.events[pending_online_event_index].get(
                    "arrival_word_index", 0
                )
            )

            if timing_enabled and len(source_seg_len) == prefix_source_segments + 1:
                first_read_tokens_raw = int(sum(source_seg_len))
                latency_recorder["raw_first_read_tokens"] = first_read_tokens_raw
                latency_recorder["first_read_tokens"] = max(
                    0, first_read_tokens_raw - prefix_source_tokens
                )
            return True
        # input_length = [(_source_seg_len[0], 1) for _source_seg_len in source_seg_len]


        

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True


        #...
        # sentence-by-sentence split
        if split_mode in ['sentence']:
            vocab_size = getattr(self.config, "vocab_size", None)

            def is_valid_token_id(token_id):
                if token_id is None or token_id == tokenizer.unk_token_id:
                    return False
                if vocab_size is not None and not (0 <= int(token_id) < int(vocab_size)):
                    return False
                return True

            sperate_tokens = [
                "<SEP>",
                "<EOQ>",
                "<EOE>",
                "<EOS>",
                "<EOP>",
                "<EOT>",
                "<EOR>",
                "<EOA>",
                "<ignore>",
            ]
            sperate_tokens_ids = {
                token_id
                for token_id in tokenizer.convert_tokens_to_ids(sperate_tokens)
                if is_valid_token_id(token_id)
            }
            eot_token_id = tokenizer.convert_tokens_to_ids("<EOT>")
            eot_token_id = eot_token_id if is_valid_token_id(eot_token_id) else None
            eop_token_id = tokenizer.convert_tokens_to_ids("<EOP>")
            eop_token_id = eop_token_id if is_valid_token_id(eop_token_id) else None
            eoq_token_id = tokenizer.convert_tokens_to_ids("<EOQ>")
            eoq_token_id = eoq_token_id if is_valid_token_id(eoq_token_id) else None
            # Already_complete = False
            # Already_complete = len(source_seg_len)-1
            Already_complete = None if online_mode else len(source_seg_len)
            Read_count= 0

            def is_source_complete():
                return online_source_complete if online_mode else Already_complete == 0

            terminal_token_ids = set()
            eos_token_tensor = getattr(generation_config, "_eos_token_tensor", None)
            if eos_token_tensor is not None:
                terminal_token_ids.update(
                    int(token_id)
                    for token_id in eos_token_tensor.reshape(-1).tolist()
                    if is_valid_token_id(int(token_id))
                )
            terminal_token_ids.update(
                token_id
                for token_id in tokenizer.convert_tokens_to_ids(
                    ["<EOA>", end_Instruct]
                )
                if is_valid_token_id(token_id)
            )

            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                if ReadAction:
                    # prepare model inputs
                    self.separate_source_target()
                    model_inputs = self.prepare_inputs_for_generation_streaming(input_ids, input_length, ReadAction=ReadAction, streaming =True,
                                                                    **model_kwargs)
                    # prepare variable output controls (note: some models won't accept all output controls)
                    model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                    model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                    _outputs = self(  
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        ReadAction = ReadAction,
                    )
                    ReadAction = False
                    self.merge_source_target()
                    newly_read_segments = max(0, pending_read_segments - Read_count)
                    if not online_mode:
                        Already_complete -= newly_read_segments
                    Read_count = pending_read_segments
                    if online_mode and pending_online_event_index is not None:
                        just_read_online_source = pending_online_source
                        online_edu_boundary_stream.mark_llm_read_finished(
                            pending_online_event_index
                        )
                        if pending_online_final:
                            online_source_complete = True
                        pending_online_event_index = None
                        pending_online_final = False
                        pending_online_source = None
                    else:
                        just_read_online_source = None
                    write_tokens_since_read = 0
                    if (
                        timing_enabled
                        and prefix_source_segments
                        and Read_count >= prefix_source_segments
                        and "prefix_prefill_done_time" not in latency_recorder
                    ):
                        latency_recorder["prefix_prefill_done_time"] = latency_now()
                    if (
                        online_mode
                        and prefix_source_segments
                        and Read_count <= prefix_source_segments
                        and not online_source_complete
                    ):
                        if queue_next_online_segment(
                            read_reason="prefix_prefill_chain"
                        ):
                            ReadAction = True
                            continue
                    elif (
                        not online_mode
                        and prefix_source_segments
                        and Read_count <= prefix_source_segments
                        and Read_count < len(source_seg_len)
                    ):
                        pending_read_segments = min(Read_count + 1, len(source_seg_len))
                        input_length = (sum(source_seg_len[:pending_read_segments]), 1)
                        ReadAction = True
                        continue
                    if (
                        online_mode
                        and defer_thinking_until_context
                        and just_read_online_source == "instruct"
                        and not online_source_complete
                    ):
                        if queue_next_online_segment(
                            read_reason="defer_until_context"
                        ):
                            ReadAction = True
                            continue
                    if (
                        online_mode
                        and read_policy in {
                            "arrival_preemption",
                            "sentence_finish_eot",
                        }
                        and not online_source_complete
                    ):
                        boundary_ready_to_read = (
                            queue_next_online_segment(
                                block=False,
                                read_reason="post_read_arrival_preemption",
                            )
                            if read_policy == "arrival_preemption"
                            else (
                                queue_next_online_segment(
                                    block=False,
                                    read_reason=(
                                        "post_read_same_sentence_preemption"
                                    ),
                                )
                                if next_online_segment_matches_current_utterance()
                                else False
                            )
                        )
                        if boundary_ready_to_read:
                            ReadAction = True
                            continue
                    if (
                        not online_mode
                        and initial_source_segments_before_thinking is not None
                        and Read_count < initial_source_segments_before_thinking
                        and Read_count < len(source_seg_len)
                    ):
                        if (
                            timing_enabled
                            and Read_count == initial_source_segments_before_thinking - 1
                            and "content_stream_start_time" not in latency_recorder
                        ):
                            latency_recorder["content_stream_start_time"] = latency_now()
                            latency_recorder["content_stream_start_source_tokens"] = int(
                                input_length[0]
                            )
                        pending_read_segments = min(Read_count + 1, len(source_seg_len))
                        input_length = (
                            sum(source_seg_len[:pending_read_segments]),
                            1,
                        )
                        ReadAction = True
                        continue
                    if (
                        online_mode
                        and timing_enabled
                        and "content_stream_start_time" not in latency_recorder
                    ):
                        context_started_at = getattr(
                            online_edu_boundary_stream,
                            "context_started_at",
                            None,
                        )
                        if context_started_at is not None:
                            latency_recorder["content_stream_start_time"] = (
                                context_started_at
                            )
                            latency_recorder["content_stream_start_source_tokens"] = int(
                                sum(source_seg_len[: max(0, Read_count - 1)])
                            )
                    source_complete_now = (
                        online_source_complete if online_mode else Already_complete == 0
                    )
                    if (
                        timing_enabled
                        and source_complete_now
                        and "input_complete_time" not in latency_recorder
                    ):
                        latency_recorder["input_complete_time"] = latency_now()
                        latency_recorder["input_complete_target_token_index"] = int(
                            target_ids.shape[-1] - target_start_len
                        )
                        latency_recorder["input_complete_output_token_index"] = int(
                            target_ids.shape[-1]
                        )
                elif not ReadAction:
                    # self.separate_source_target()
                    # prepare model inputs
                    model_inputs = self.prepare_inputs_for_generation_streaming(next_tokens, input_length, ReadAction=ReadAction, streaming =True,
                                                                    **model_kwargs)
                    # prepare variable output controls (note: some models won't accept all output controls)
                    model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                    model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                    outputs = self(  
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        seq_length = input_length,
                        ReadAction=ReadAction,
                    )

                    # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
                    model_kwargs = self._update_model_kwargs_for_generation(
                        outputs,
                        model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder,
                    )
                    if synced_gpus and this_peer_finished:
                        continue

                    # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                    # (the clone itself is always small)
                    next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                    # pre-process distribution
                    next_token_scores = logits_processor(input_ids, next_token_logits)
                    if force_complete_source and not is_source_complete():
                        for terminal_token_id in terminal_token_ids:
                            if 0 <= int(terminal_token_id) < next_token_scores.shape[-1]:
                                next_token_scores[:, terminal_token_id] = -float("inf")
                    reflection_tokens_generated = (
                        int(target_ids.shape[-1])
                        - int(reflection_content_start_output_token_index)
                        if reflection_content_start_output_token_index
                        is not None
                        else 0
                    )
                    if (
                        force_reflection_after_draft
                        and reflection_triggered
                        and forced_reflection_token_cursor
                        >= len(forced_reflection_token_ids)
                        and reflection_content_start_output_token_index
                        is not None
                        and reflection_tokens_generated
                        < forced_reflection_min_new_tokens
                    ):
                        for token_id in reflection_early_stop_token_ids:
                            if 0 <= token_id < next_token_scores.shape[-1]:
                                next_token_scores[:, token_id] = -float("inf")

                    # Store scores, attentions and hidden_states when required
                    if return_dict_in_generate:
                        if output_scores:
                            scores += (next_token_scores,)
                        if output_logits:
                            raw_logits += (next_token_logits,)
                        if output_attentions:
                            decoder_attentions += (
                                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                            )
                            if self.config.is_encoder_decoder:
                                cross_attentions += (outputs.cross_attentions,)

                        if output_hidden_states:
                            decoder_hidden_states += (
                                (outputs.decoder_hidden_states,)
                                if self.config.is_encoder_decoder
                                else (outputs.hidden_states,)
                            )

                    # Token selection. Once a draft <EOA> is observed, the
                    # configured reflection intervention is written token by
                    # token into the same target stream. Every forced token is
                    # consumed by the normal next iteration, so target KV cache
                    # continuity is preserved without a second generation call.
                    forcing_reflection_prompt = (
                        reflection_triggered
                        and forced_reflection_token_cursor
                        < len(forced_reflection_token_ids)
                    )
                    reflection_prompt_finishing_this_token = False
                    if forcing_reflection_prompt:
                        forced_token_id = forced_reflection_token_ids[
                            forced_reflection_token_cursor
                        ]
                        forced_reflection_token_cursor += 1
                        reflection_prompt_finishing_this_token = (
                            forced_reflection_token_cursor
                            == len(forced_reflection_token_ids)
                        )
                        next_tokens = torch.full(
                            (batch_size,),
                            int(forced_token_id),
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                    elif do_sample:
                        probs = nn.functional.softmax(next_token_scores, dim=-1)
                        # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_tokens = torch.argmax(next_token_scores, dim=-1)

                    # finished sentences should have their next token be a padding token
                    if has_eos_stopping_criteria:
                        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                    # # update generated ids, model inputs, and length for next step
                    # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                    if streamer is not None:
                        streamer.put(next_tokens.cpu())


                    next_tokens = next_tokens.unsqueeze(0)
                    target_tokens.append(next_tokens) # for output
                    target_ids = torch.cat(target_tokens,dim=-1)
                    write_tokens_since_read += 1
                    if reflection_prompt_finishing_this_token:
                        reflection_content_start_output_token_index = int(
                            target_ids.shape[-1]
                        )
                        if timing_enabled:
                            latency_recorder[
                                "reflection_content_start_time"
                            ] = latency_now()
                            latency_recorder[
                                "reflection_content_start_output_token_index"
                            ] = reflection_content_start_output_token_index
                            latency_recorder[
                                "reflection_content_start_target_token_index"
                            ] = int(
                                target_ids.shape[-1] - target_start_len
                            )
                    if timing_enabled and "first_thought_time" not in latency_recorder:
                        latency_recorder["first_thought_time"] = latency_now()
                        latency_recorder["first_thought_target_token_index"] = int(target_ids.shape[-1] - target_start_len)
                        latency_recorder["raw_source_tokens_read_at_first_thought"] = int(input_length[0])
                        latency_recorder["source_tokens_read_at_first_thought"] = max(0, int(input_length[0]) - prefix_source_tokens)
                        if "content_stream_start_time" in latency_recorder:
                            latency_recorder["first_thought_after_content_start_s"] = (
                                latency_recorder["first_thought_time"]
                                - latency_recorder["content_stream_start_time"]
                            )
                        if "content_stream_start_source_tokens" in latency_recorder:
                            latency_recorder["content_tokens_until_first_thought"] = max(
                                0,
                                int(input_length[0])
                                - int(
                                    latency_recorder[
                                        "content_stream_start_source_tokens"
                                    ]
                                ),
                            )
                        if online_mode:
                            content_arrival_start = (
                                online_edu_boundary_stream.context_start_arrival_token_index
                                or 0
                            )
                            latency_recorder[
                                "content_arrival_tokens_until_first_thought"
                            ] = max(
                                0,
                                last_online_arrival_token_index
                                - content_arrival_start,
                            )
                            content_word_start = (
                                online_edu_boundary_stream.context_start_arrival_word_index
                                or 0
                            )
                            latency_recorder[
                                "content_arrival_words_until_first_thought"
                            ] = max(
                                0,
                                last_online_arrival_word_index
                                - content_word_start,
                            )
                    if timing_enabled and "answer_time" not in latency_recorder:
                        answer_start_index = 0
                        reflection_answer_scope_ready = True
                        if force_reflection_after_draft:
                            reflection_answer_scope_ready = (
                                reflection_triggered
                                and forced_reflection_token_cursor
                                >= len(forced_reflection_token_ids)
                                and "reflection_content_start_output_token_index"
                                in latency_recorder
                            )
                            if reflection_answer_scope_ready:
                                answer_start_index = int(
                                    latency_recorder[
                                        "reflection_content_start_output_token_index"
                                    ]
                                )
                        elif force_complete_source and "input_complete_target_token_index" in latency_recorder:
                            answer_start_index = (
                                target_start_len
                                + latency_recorder["input_complete_target_token_index"]
                            )
                        boxed_answer = None
                        if reflection_answer_scope_ready:
                            generated_text = tokenizer.decode(
                                target_ids[0, answer_start_index:],
                                skip_special_tokens=False,
                            )
                            boxed_answer = _extract_complete_boxed_answer(
                                generated_text
                            )
                        if boxed_answer:
                            if force_complete_source and not is_source_complete():
                                if "early_answer_time" not in latency_recorder:
                                    latency_recorder["early_answer_time"] = latency_now()
                                    latency_recorder["early_answer_text"] = boxed_answer
                                    latency_recorder["early_answer_target_token_index"] = int(
                                        target_ids.shape[-1] - target_start_len
                                    )
                            else:
                                latency_recorder["answer_time"] = latency_now()
                                latency_recorder["answer_target_token_index"] = int(target_ids.shape[-1] - target_start_len)
                                latency_recorder["answer_text_at_detection"] = boxed_answer
                                latency_recorder["source_segments_read_at_answer"] = int(Read_count)
                                latency_recorder["source_segments_remaining_at_answer"] = (
                                    0
                                    if online_mode and online_source_complete
                                    else None
                                    if online_mode
                                    else int(Already_complete)
                                )
                                latency_recorder["raw_source_tokens_read_at_answer"] = int(input_length[0])
                                latency_recorder["source_tokens_read_at_answer"] = max(0, int(input_length[0]) - prefix_source_tokens)

                    if (
                        timing_enabled
                        and force_reflection_after_draft
                        and not reflection_triggered
                        and is_source_complete()
                        and "draft_answer_time" not in latency_recorder
                    ):
                        draft_text = tokenizer.decode(
                            target_ids[0],
                            skip_special_tokens=False,
                        )
                        draft_answer = _extract_complete_boxed_answer(
                            draft_text
                        )
                        if draft_answer:
                            latency_recorder["draft_answer_time"] = latency_now()
                            latency_recorder["draft_answer_text"] = draft_answer
                            latency_recorder[
                                "draft_answer_output_token_index"
                            ] = int(target_ids.shape[-1])

                    generated_reflection_trigger = (
                        force_reflection_after_draft
                        and not reflection_triggered
                        and is_source_complete()
                        and reflection_trigger_token_id is not None
                        and next_tokens[0].item()
                        == int(reflection_trigger_token_id)
                    )
                    if generated_reflection_trigger:
                        reflection_triggered = True
                        forced_reflection_token_cursor = 0
                        if timing_enabled:
                            latency_recorder["reflection_triggered"] = True
                            latency_recorder[
                                "reflection_trigger_time"
                            ] = latency_now()
                            latency_recorder[
                                "reflection_prompt_start_output_token_index"
                            ] = int(target_ids.shape[-1])
                            latency_recorder[
                                "reflection_prompt_start_target_token_index"
                            ] = int(
                                target_ids.shape[-1] - target_start_len
                            )
                    elif (
                        force_reflection_after_draft
                        and reflection_triggered
                        and not reflection_completed
                        and not forcing_reflection_prompt
                        and reflection_trigger_token_id is not None
                        and next_tokens[0].item()
                        == int(reflection_trigger_token_id)
                    ):
                        reflection_completed = True
                        if timing_enabled:
                            latency_recorder["reflection_completed"] = True
                            latency_recorder[
                                "reflection_complete_time"
                            ] = latency_now()
                            latency_recorder[
                                "reflection_complete_output_token_index"
                            ] = int(target_ids.shape[-1])

                    stop_requested = stopping_criteria(target_ids, scores)
                    target_budget_exhausted = (
                        max_new_target_tokens is not None
                        and target_ids.shape[-1] - target_start_len
                        >= max_new_target_tokens
                    )
                    if force_complete_source and not is_source_complete():
                        unfinished_sequences = torch.ones_like(unfinished_sequences)
                    else:
                        unfinished_sequences = unfinished_sequences & ~stop_requested
                        if target_budget_exhausted:
                            unfinished_sequences = torch.zeros_like(unfinished_sequences)
                    this_peer_finished = unfinished_sequences.max() == 0
                    cur_len += 1

                    # This is needed to properly delete outputs.logits which may be very large for first iteration
                    # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                    del outputs

                    # next_tokens = next_tokens.unsqueeze(0)
                    # target_tokens.append(next_tokens) # for output
                    # target_ids = torch.cat(target_tokens,dim=-1)


                    generated_separator = next_tokens[0].item() in sperate_tokens_ids
                    generated_eot = next_tokens[0].item() == eot_token_id
                    generated_eop = next_tokens[0].item() == eop_token_id
                    generated_eoq = next_tokens[0].item() == eoq_token_id
                    # <EOQ> 仅在模型刚处理完最终问题片段时才允许推进输入。
                    generated_final_question_end = (
                        generated_eoq and last_read_online_is_final_instruct
                    )
                    generated_boundary_thought_end = (
                        generated_eot
                        or generated_eop
                        or generated_final_question_end
                    )
                    read_budget_reached = (
                        max_write_tokens_before_read is not None
                        and write_tokens_since_read >= max_write_tokens_before_read
                    )
                    must_continue_source = (
                        force_complete_source
                        and not is_source_complete()
                        and (
                            bool(stop_requested.max().item())
                            or target_budget_exhausted
                        )
                    )
                    should_read = (
                        generated_separator
                        or read_budget_reached
                        or must_continue_source
                    )
                    if online_mode:
                        ReadAction = False
                        if finish_thought_before_read:
                            strict_should_read = (
                                generated_boundary_thought_end or must_continue_source
                            )
                            if strict_should_read and not online_source_complete:
                                if (
                                    timing_enabled
                                    and must_continue_source
                                    and not generated_boundary_thought_end
                                ):
                                    latency_recorder[
                                        "forced_read_count"
                                    ] = (
                                        latency_recorder.get(
                                            "forced_read_count", 0
                                        )
                                        + 1
                                    )
                                ReadAction = queue_next_online_segment(
                                    block=True,
                                    read_reason=(
                                        "forced_continue_source"
                                        if must_continue_source
                                        and not generated_boundary_thought_end
                                        else (
                                            "finish_eop"
                                            if generated_eop
                                            else (
                                                "finish_eoq"
                                                if generated_final_question_end
                                                else "finish_eot"
                                            )
                                        )
                                    ),
                                )
                                if (
                                    not ReadAction
                                    and timing_enabled
                                    and "input_complete_time"
                                    not in latency_recorder
                                ):
                                    latency_recorder[
                                        "input_complete_time"
                                    ] = latency_now()
                                    latency_recorder[
                                        "input_complete_target_token_index"
                                    ] = int(
                                        target_ids.shape[-1]
                                        - target_start_len
                                    )
                                    latency_recorder[
                                        "input_complete_output_token_index"
                                    ] = int(target_ids.shape[-1])
                        elif sentence_finish_eot:
                            boundary_arrived_same_utterance = False
                            if (
                                not online_source_complete
                                and next_online_segment_matches_current_utterance()
                            ):
                                boundary_arrived_same_utterance = (
                                    queue_next_online_segment(
                                        block=False,
                                        read_reason=(
                                            "same_sentence_arrival_preemption"
                                        ),
                                    )
                                )
                            if boundary_arrived_same_utterance:
                                ReadAction = True
                            else:
                                sentence_boundary_should_read = (
                                    generated_eot
                                    or generated_final_question_end
                                    or must_continue_source
                                )
                                if (
                                    sentence_boundary_should_read
                                    and not online_source_complete
                                ):
                                    if (
                                        timing_enabled
                                        and must_continue_source
                                        and not (
                                            generated_eot
                                            or generated_final_question_end
                                        )
                                    ):
                                        latency_recorder[
                                            "forced_read_count"
                                        ] = (
                                            latency_recorder.get(
                                                "forced_read_count", 0
                                            )
                                            + 1
                                        )
                                    ReadAction = queue_next_online_segment(
                                        block=True,
                                        read_reason=(
                                            "forced_continue_source"
                                            if must_continue_source
                                            and not (
                                                generated_eot
                                                or generated_final_question_end
                                            )
                                            else (
                                                "question_transition_after_eoq"
                                                if generated_final_question_end
                                                else "sentence_transition_after_eot"
                                            )
                                        ),
                                    )
                                    if (
                                        not ReadAction
                                        and timing_enabled
                                        and "input_complete_time"
                                        not in latency_recorder
                                    ):
                                        latency_recorder[
                                            "input_complete_time"
                                        ] = latency_now()
                                        latency_recorder[
                                            "input_complete_target_token_index"
                                        ] = int(
                                            target_ids.shape[-1]
                                            - target_start_len
                                        )
                                        latency_recorder[
                                            "input_complete_output_token_index"
                                        ] = int(target_ids.shape[-1])
                        else:
                            boundary_arrived = False
                            if not online_source_complete:
                                boundary_arrived = queue_next_online_segment(
                                    block=False,
                                    read_reason="arrival_preemption",
                                )
                            if boundary_arrived:
                                ReadAction = True
                            elif should_read and not online_source_complete:
                                if timing_enabled and read_budget_reached:
                                    latency_recorder["forced_read_count"] = (
                                        latency_recorder.get(
                                            "forced_read_count", 0
                                        )
                                        + 1
                                    )
                                ReadAction = queue_next_online_segment(
                                    block=True,
                                    read_reason=(
                                        "forced_continue_source"
                                        if must_continue_source
                                        else "separator_or_budget_read"
                                    ),
                                )
                                if (
                                    not ReadAction
                                    and timing_enabled
                                    and "input_complete_time"
                                    not in latency_recorder
                                ):
                                    latency_recorder[
                                        "input_complete_time"
                                    ] = latency_now()
                                    latency_recorder[
                                        "input_complete_target_token_index"
                                    ] = int(
                                        target_ids.shape[-1]
                                        - target_start_len
                                    )
                                    latency_recorder[
                                        "input_complete_output_token_index"
                                    ] = int(target_ids.shape[-1])
                    else:
                        if should_read and Already_complete != 0:
                            ReadAction = True
                            if timing_enabled and read_budget_reached:
                                latency_recorder["forced_read_count"] = (
                                    latency_recorder.get("forced_read_count", 0) + 1
                                )
                        if Already_complete == 0:
                            ReadAction = False

                        if ReadAction:
                            # source_input_length = input_length[0] + source_seg_len[-Already_complete+1] # or + source_seg_len[Read_count+1]
                            pending_read_segments = min(Read_count + 1, len(source_seg_len))
                            source_input_length = sum(source_seg_len[:pending_read_segments])
                            target_input_length = 1
                            input_length = (source_input_length, target_input_length) 
    
        
        # monitor.report_and_reset(1)


        if timing_enabled:
            latency_recorder["end_time"] = latency_now()
            if online_mode:
                source_start_time = getattr(
                    online_edu_boundary_stream, "source_start_time", None
                )
                source_arrival_complete_time = getattr(
                    online_edu_boundary_stream,
                    "source_arrival_complete_time",
                    None,
                )
                source_arrival_duration_s = getattr(
                    online_edu_boundary_stream,
                    "source_arrival_duration_s",
                    None,
                )
                if source_start_time is not None:
                    latency_recorder["source_start_time"] = float(
                        source_start_time
                    )
                if source_arrival_complete_time is not None:
                    latency_recorder["source_arrival_complete_time"] = float(
                        source_arrival_complete_time
                    )
                if source_arrival_duration_s is not None:
                    latency_recorder["source_arrival_duration_s"] = float(
                        source_arrival_duration_s
                    )
            if "input_complete_time" in latency_recorder:
                # 旧字段表示模型消费完最后一个输入段，而不是源输入物理到达。
                latency_recorder["model_read_complete_time"] = latency_recorder[
                    "input_complete_time"
                ]
            total_source_tokens_raw = int(sum(source_seg_len))
            latency_recorder["source_segments"] = len(source_seg_len)
            latency_recorder["streaming_source_segments"] = max(
                0, len(source_seg_len) - prefix_source_segments
            )
            latency_recorder["raw_total_source_tokens"] = total_source_tokens_raw
            latency_recorder["total_source_tokens"] = max(
                0, total_source_tokens_raw - prefix_source_tokens
            )
            latency_recorder["generated_target_tokens"] = int(target_ids.shape[-1] - target_start_len)
            if force_reflection_after_draft:
                latency_recorder.setdefault(
                    "reflection_triggered", reflection_triggered
                )
                latency_recorder.setdefault(
                    "reflection_completed", reflection_completed
                )
                reflection_content_start = latency_recorder.get(
                    "reflection_content_start_output_token_index"
                )
                latency_recorder["reflection_stage_target_tokens"] = (
                    max(0, int(target_ids.shape[-1]) - int(reflection_content_start))
                    if reflection_content_start is not None
                    else None
                )
                latency_recorder["reflection_generation_s"] = (
                    latency_recorder["end_time"]
                    - latency_recorder["reflection_content_start_time"]
                    if "reflection_content_start_time" in latency_recorder
                    else None
                )
            input_complete_target_tokens = latency_recorder.get(
                "input_complete_target_token_index"
            )
            if input_complete_target_tokens is not None:
                streaming_stage_target_tokens = max(
                    0,
                    min(
                        int(input_complete_target_tokens),
                        latency_recorder["generated_target_tokens"],
                    ),
                )
                latency_recorder["streaming_stage_target_tokens"] = (
                    streaming_stage_target_tokens
                )
                latency_recorder["post_input_complete_target_tokens"] = max(
                    0,
                    latency_recorder["generated_target_tokens"]
                    - streaming_stage_target_tokens,
                )
            latency_recorder["total_generation_s"] = latency_recorder["end_time"] - latency_recorder["start_time"]
            latency_recorder["first_thought_latency_s"] = (
                latency_recorder["first_thought_time"] - latency_recorder["start_time"]
                if "first_thought_time" in latency_recorder
                else None
            )
            latency_recorder["first_thought_after_prefix_s"] = (
                latency_recorder["first_thought_time"] - latency_recorder["prefix_prefill_done_time"]
                if "first_thought_time" in latency_recorder and "prefix_prefill_done_time" in latency_recorder
                else None
            )
            latency_recorder["input_complete_to_end_s"] = (
                latency_recorder["end_time"] - latency_recorder["input_complete_time"]
                if "input_complete_time" in latency_recorder
                else None
            )
            latency_recorder["total_after_prefix_s"] = (
                latency_recorder["end_time"] - latency_recorder["prefix_prefill_done_time"]
                if "prefix_prefill_done_time" in latency_recorder
                else None
            )
            latency_recorder["answer_after_input_complete_s"] = (
                latency_recorder["answer_time"] - latency_recorder["input_complete_time"]
                if "answer_time" in latency_recorder and "input_complete_time" in latency_recorder
                else None
            )
            latency_recorder["source_start_to_answer_s"] = (
                latency_recorder["answer_time"]
                - latency_recorder["source_start_time"]
                if "answer_time" in latency_recorder
                and "source_start_time" in latency_recorder
                else None
            )
            latency_recorder["source_arrival_complete_to_answer_s"] = (
                latency_recorder["answer_time"]
                - latency_recorder["source_arrival_complete_time"]
                if "answer_time" in latency_recorder
                and "source_arrival_complete_time" in latency_recorder
                else None
            )
            latency_recorder["model_read_complete_to_answer_s"] = (
                latency_recorder["answer_time"]
                - latency_recorder["model_read_complete_time"]
                if "answer_time" in latency_recorder
                and "model_read_complete_time" in latency_recorder
                else None
            )
            latency_recorder["answer_after_prefix_s"] = (
                latency_recorder["answer_time"] - latency_recorder["prefix_prefill_done_time"]
                if "answer_time" in latency_recorder and "prefix_prefill_done_time" in latency_recorder
                else None
            )
            latency_recorder["answer_latency_s"] = (
                latency_recorder["answer_time"] - latency_recorder["start_time"]
                if "answer_time" in latency_recorder
                else None
            )
            latency_recorder["input_completed"] = "input_complete_time" in latency_recorder
            latency_recorder["answer_before_input_complete"] = (
                "answer_time" in latency_recorder
                and (
                    "input_complete_time" not in latency_recorder
                    or latency_recorder["answer_time"] < latency_recorder["input_complete_time"]
                )
            )
            if online_mode:
                latency_recorder["skipped_source_arrival_wait_time_s"] = float(
                    getattr(
                        online_edu_boundary_stream.stats,
                        "skipped_arrival_wait_time_s",
                        0.0,
                    )
                )


        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=target_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=target_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            if not wait_lagging:
                wait_lagging = [target_ids.shape[-1] - target_start_len]
            return target_ids, wait_lagging
        
