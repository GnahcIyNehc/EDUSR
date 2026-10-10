"""Direct-training helpers for GRPO and HRPO scripts.

The scripts in ``ST_RL/GRPO`` and ``ST_RL/HRPO`` intentionally avoid a heavy
rollout stack. They use HuggingFace ``generate`` for rollouts and then compute
sampled-response log probabilities from the causal-LM logits.
"""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from ST_RL.prompt_protocol import PromptProtocol

# 与句子模型 SFT 配置完全一致；所有 HRPO rollout 入口必须复用该常量。
SFT_STREAMING_INSTRUCTION = (
    "<|im_start|>user\n"
    "You are a streaming reasoner to perform a streaming reasoning with deep thinking task. "
    "1. In the streaming reasoning, you should perform a sentence-by-sentence streaming reasoning, "
    "i.e. when you encounter an '<EOS>' in the Input context, perform reasoning about the sentence "
    "that before this '<EOS>', focus on: (1) Summarize or extract key information, (2) Reformulate "
    "the meaning in clearer or more explicit terms, (3) Minimal, natural logic-based extensions. "
    "Importantly, end the reasoning for each Original Input sentence with '<EOT>'. 2. In the deeper "
    "reasoning, you should perform a global and deep reasoning with input and streaming reasoning: "
    "(1) Your goal is to condense and refine the streaming shallow reasoning into a focused, "
    "high-level reasoning trace that directly leads to the final answer. (2) Focus on identifying "
    "the most relevant insights that directly support answering the question. Keep your reasoning "
    "concise, focused, and deeper. 3.Your number or equation answer must be enclosed in "
    "\\boxed{{}}. Start your reasoning based on following content:"
)

# 保留旧名称，确保旧版 HRPO 直训入口也使用同一套 SFT 提示词。
HRPO_SFT_INSTRUCTION = SFT_STREAMING_INSTRUCTION


@dataclass
class RLSample:
    question: str
    context_segments: List[str]
    answer: str
    raw: Dict[str, Any]


@dataclass
class GeneratedBatch:
    prompts: List[str]
    prompt_ids: List[List[int]]
    response_ids: List[List[int]]
    response_texts: List[str]
    group_ids: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_records(path: str) -> List[Dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))
    if data_path.suffix.lower() in {".jsonl", ".jl"}:
        records = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "train", "samples", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("expected a JSON list, JSONL file, or dict containing data/train/samples/examples")


def _first_present(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def split_stream_context(context: Any) -> List[str]:
    if context is None:
        return []
    if isinstance(context, list):
        return [str(item).strip() for item in context if str(item).strip()]
    text = str(context).strip()
    if not text:
        return []
    if any(token in text for token in ("<EOS>", "<EOE>", "<EOQ>", "<EOT>")):
        pattern = re.compile(r"(.*?(?:<EOS>|<EOE>|<EOQ>|<EOT>))", re.DOTALL)
        pieces = pattern.findall(text)
        consumed = sum(len(piece) for piece in pieces)
        tail = text[consumed:]
        if tail.strip():
            pieces.append(tail)
        cleaned = [re.sub(r"<(?:EOS|EOE|EOQ|EOT)>", "", piece).strip() for piece in pieces]
        return [piece for piece in cleaned if piece]
    sentence_pieces = re.split(r"(?<=[.!?。！？])\s+", text)
    return [piece.strip() for piece in sentence_pieces if piece.strip()]


def records_to_samples(records: Iterable[Dict[str, Any]]) -> List[RLSample]:
    samples: List[RLSample] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        instruct_context = record.get("Instruct_Context") or {}
        if not isinstance(instruct_context, dict):
            instruct_context = {}

        question = _first_present(
            record,
            ("question", "prompt", "query", "instruction", "instruct", "Q", "input"),
        )
        if question is None:
            question = _first_present(instruct_context, ("instruct", "question", "prompt"))

        context = _first_present(record, ("context", "contexts", "passage", "story"))
        if context is None:
            context = _first_present(instruct_context, ("context", "contexts", "passage"))

        answer = _first_present(record, ("answer", "target", "result", "label", "A"))
        if answer is None:
            answer = _first_present(instruct_context, ("answer", "target", "result"))

        if question is None or answer is None:
            continue
        samples.append(
            RLSample(
                question=str(question).strip(),
                context_segments=split_stream_context(context),
                answer=str(answer).strip(),
                raw=record,
            )
        )
    if not samples:
        raise ValueError("no usable samples found; each record needs a question/prompt and answer/target")
    return samples


def load_samples(path: str, max_samples: Optional[int] = None) -> List[RLSample]:
    samples = records_to_samples(load_json_records(path))
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    return samples


def format_context(context_segments: Sequence[str]) -> str:
    if not context_segments:
        return ""
    return "\n".join("%d. %s" % (idx + 1, segment) for idx, segment in enumerate(context_segments))


def build_grpo_prompt(sample: RLSample, protocol: Optional["PromptProtocol"] = None) -> str:
    if protocol is not None:
        return build_hrpo_prompt(sample, protocol) + protocol.target_prefix
    context = format_context(sample.context_segments)
    context_block = "\nContext:\n%s\n" % context if context else ""
    return (
        "You are a careful reasoning assistant. Solve the problem and put the final answer inside \\boxed{}.\n"
        "Question:\n%s\n%s"
        "Answer:\n" % (sample.question, context_block)
    )


def build_hrpo_prompt(sample: RLSample, protocol: Optional["PromptProtocol"] = None) -> str:
    question_stream = format_question_stream(sample.question)
    context_stream = format_context_stream(sample)
    instruction = protocol.instruction if protocol is not None else HRPO_SFT_INSTRUCTION
    end_instruction = protocol.end_instruction if protocol is not None else "<|im_end|>\n"
    return "%s<SEP>%s%s%s" % (instruction, question_stream, context_stream, end_instruction)


def _split_with_end_tokens(text: str, end_token_pattern: str) -> List[str]:
    pattern = re.compile(r"(.*?(?:%s))" % end_token_pattern, re.DOTALL)
    chunks = pattern.findall(text or "")
    consumed_len = sum(len(chunk) for chunk in chunks)
    tail = (text or "")[consumed_len:]
    if tail:
        chunks.append(tail)
    return [chunk for chunk in chunks if chunk]


def split_question_stream(question: str) -> List[str]:
    chunks = _split_with_end_tokens(str(question or ""), r"<EOE>|<EOS>|<EOQ>")
    if not chunks:
        chunks = [str(question or "")]
    if not chunks[-1].endswith("<EOQ>"):
        chunks[-1] = chunks[-1] + "<EOQ>"
    return chunks


def count_question_stream_segments(question: str) -> int:
    return len(split_question_stream(question))


def format_question_stream(question: str) -> str:
    return "".join(split_question_stream(question))


def format_context_stream(sample: RLSample) -> str:
    instruct_context = sample.raw.get("Instruct_Context") or {}
    if isinstance(instruct_context, dict):
        raw_context = instruct_context.get("context")
        if raw_context:
            return str(raw_context)
    context = "".join("%s<EOS>" % segment for segment in sample.context_segments)
    return context


def strip_pad(ids: Sequence[int], pad_token_id: int) -> List[int]:
    output = list(ids)
    while output and output[-1] == pad_token_id:
        output.pop()
    return output


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
) -> GeneratedBatch:
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(float(temperature), 1e-5),
        "top_p": top_p,
        "num_return_sequences": num_generations,
        "pad_token_id": pad_token_id,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if top_k and top_k > 0:
        generation_kwargs["top_k"] = top_k
    with torch.no_grad():
        outputs = model.generate(**encoded, **generation_kwargs)
    tokenizer.padding_side = old_padding_side

    prompt_width = encoded["input_ids"].shape[1]
    flat_prompts: List[str] = []
    prompt_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    response_texts: List[str] = []
    group_ids: List[int] = []

    for sample_idx, prompt in enumerate(prompts):
        raw_prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for gen_idx in range(num_generations):
            row_idx = sample_idx * num_generations + gen_idx
            completion = strip_pad(outputs[row_idx, prompt_width:].detach().cpu().tolist(), pad_token_id)
            flat_prompts.append(prompt)
            prompt_ids.append(list(raw_prompt_ids))
            response_ids.append(completion)
            response_texts.append(tokenizer.decode(completion, skip_special_tokens=True))
            group_ids.append(sample_idx)
    return GeneratedBatch(
        prompts=flat_prompts,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_texts=response_texts,
        group_ids=torch.tensor(group_ids, dtype=torch.long),
    )


def _pad_full_sequences(
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    pad_token_id = tokenizer.pad_token_id
    prompt_lens = [len(ids) for ids in prompt_ids]
    response_lens = [len(ids) for ids in response_ids]
    max_len = max([p + r for p, r in zip(prompt_lens, response_lens)] + [1])
    rows = []
    masks = []
    for p_ids, r_ids in zip(prompt_ids, response_ids):
        full = list(p_ids) + list(r_ids)
        pad = max_len - len(full)
        rows.append(full + [pad_token_id] * pad)
        masks.append([1] * len(full) + [0] * pad)
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        prompt_lens,
        response_lens,
    )


def compute_response_log_probs(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs, response_mask, _ = _compute_response_statistics(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        device=device,
        require_grad=require_grad,
        compute_entropy=False,
    )
    return log_probs, response_mask


def categorical_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """根据分类分布 logits 计算精确熵。"""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def _compute_response_statistics(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
    compute_entropy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, prompt_lens, response_lens = _pad_full_sequences(
        tokenizer, prompt_ids, response_ids, device
    )
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        max_response_len = max(response_lens + [1])
        log_probs_out = torch.zeros(
            (len(response_ids), max_response_len),
            dtype=torch.float32,
            device=logits.device,
        )
        response_mask = torch.zeros((len(response_ids), max_response_len), dtype=torch.bool, device=device)
        token_entropy_out = torch.zeros_like(log_probs_out)
        for row_idx, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens)):
            if response_len <= 0:
                continue
            start = prompt_len - 1
            end = start + response_len
            token_logits = logits[row_idx, start:end, :].float()
            target_ids = torch.tensor(response_ids[row_idx], dtype=torch.long, device=token_logits.device)
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            gathered = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            log_probs_out[row_idx, :response_len] = gathered
            response_mask[row_idx, :response_len] = True
            if compute_entropy:
                detached_log_probs = token_log_probs.detach()
                token_entropy_out[row_idx, :response_len] = -(
                    detached_log_probs.exp() * detached_log_probs
                ).sum(dim=-1)
    return log_probs_out, response_mask, token_entropy_out


def compute_response_log_probs_and_entropy(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算回复 log-prob、有效 token 掩码和完整词表策略熵。"""
    return _compute_response_statistics(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        device=device,
        require_grad=require_grad,
        compute_entropy=True,
    )


def compute_response_log_probs_in_batches(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按微批计算采样回复的 log-prob，降低 4B 模型训练时的显存峰值。"""
    total = len(response_ids)
    if micro_batch_size <= 0 or micro_batch_size >= total:
        return compute_response_log_probs(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            device=device,
            require_grad=require_grad,
        )

    max_response_len = max([len(ids) for ids in response_ids] + [1])
    log_prob_chunks = []
    mask_chunks = []
    for start in range(0, total, micro_batch_size):
        end = min(start + micro_batch_size, total)
        chunk_log_probs, chunk_mask = compute_response_log_probs(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids[start:end],
            response_ids=response_ids[start:end],
            device=device,
            require_grad=require_grad,
        )
        pad_width = max_response_len - chunk_log_probs.shape[1]
        if pad_width > 0:
            chunk_log_probs = torch.nn.functional.pad(chunk_log_probs, (0, pad_width))
            chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_width), value=False)
        log_prob_chunks.append(chunk_log_probs)
        mask_chunks.append(chunk_mask)
    return torch.cat(log_prob_chunks, dim=0), torch.cat(mask_chunks, dim=0)


def compute_response_log_probs_and_entropy_in_batches(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    device: torch.device,
    require_grad: bool,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按微批计算回复 log-prob、有效 token 掩码和精确策略熵。"""
    total = len(response_ids)
    if micro_batch_size <= 0 or micro_batch_size >= total:
        return compute_response_log_probs_and_entropy(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            device=device,
            require_grad=require_grad,
        )

    max_response_len = max([len(ids) for ids in response_ids] + [1])
    log_prob_chunks = []
    mask_chunks = []
    entropy_chunks = []
    for start in range(0, total, micro_batch_size):
        end = min(start + micro_batch_size, total)
        chunk_log_probs, chunk_mask, chunk_entropy = compute_response_log_probs_and_entropy(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids[start:end],
            response_ids=response_ids[start:end],
            device=device,
            require_grad=require_grad,
        )
        pad_width = max_response_len - chunk_log_probs.shape[1]
        if pad_width > 0:
            chunk_log_probs = torch.nn.functional.pad(chunk_log_probs, (0, pad_width))
            chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_width), value=False)
            chunk_entropy = torch.nn.functional.pad(chunk_entropy, (0, pad_width))
        log_prob_chunks.append(chunk_log_probs)
        mask_chunks.append(chunk_mask)
        entropy_chunks.append(chunk_entropy)
    return (
        torch.cat(log_prob_chunks, dim=0),
        torch.cat(mask_chunks, dim=0),
        torch.cat(entropy_chunks, dim=0),
    )


def compute_token_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    return [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def boxed_format_reward(text: str) -> float:
    return float("\\boxed{" in (text or ""))


def parse_hrpo_response(text: str, expected_segments: int) -> Tuple[List[str], str, str]:
    """Return streaming segments, deep text, and answer-bearing full text."""
    raw = text or ""
    stream_segments: List[str] = []
    cursor = 0
    while True:
        marker_hits = []
        for token in ("<EOT>", "<EOQ>"):
            marker = raw.find(token, cursor)
            if marker >= 0:
                marker_hits.append((marker, token))
        if not marker_hits:
            break
        eor_marker = raw.find("<EOR>", cursor)
        marker, token = min(marker_hits, key=lambda item: item[0])
        if eor_marker >= 0 and eor_marker < marker:
            break
        segment = raw[cursor : marker + len(token)]
        stream_segments.append(segment)
        cursor = marker + len(token)
        if expected_segments > 0 and len(stream_segments) >= expected_segments:
            break
    remainder = raw[cursor:].strip()
    eor_idx = remainder.find("<EOR>")
    deep_text = remainder[: eor_idx + len("<EOR>")] if eor_idx >= 0 else remainder
    return stream_segments, deep_text.strip(), raw


def save_checkpoint(model: Any, tokenizer: Any, output_dir: str, step: int) -> str:
    checkpoint_dir = Path(output_dir) / ("checkpoint-%06d" % step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    return str(checkpoint_dir)


def sample_minibatch(samples: Sequence[RLSample], batch_size: int) -> List[RLSample]:
    if batch_size >= len(samples):
        return random.sample(list(samples), len(samples))
    return random.sample(list(samples), batch_size)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0
