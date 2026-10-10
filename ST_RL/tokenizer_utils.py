"""恢复 StreamingThinker 训练时使用的特殊 token 词表。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence


STREAM_SPECIAL_TOKENS_NO_EOE = (
    "<SEP>",
    "<EOQ>",
    "<EOS>",
    "<EOT>",
    "<EOR>",
    "<EOA>",
    "<ignore>",
)
STREAM_SPECIAL_TOKENS_WITH_EOE = (
    "<SEP>",
    "<EOQ>",
    "<EOE>",
    "<EOS>",
    "<EOP>",
    "<EOT>",
    "<EOR>",
    "<EOA>",
    "<ignore>",
)


@dataclass(frozen=True)
class StreamingTokenizerInfo:
    """记录词表恢复结果，供训练日志和启动前检查使用。"""

    vocab_size: int
    token_set: str
    added_tokens: int
    token_ids: dict[str, int]


def _tokens_are_atomic(tokenizer: Any, tokens: Sequence[str]) -> bool:
    """检查每个流式边界是否都编码为唯一的单 token。"""
    for token in tokens:
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            return False
        if int(encoded[0]) != int(tokenizer.convert_tokens_to_ids(token)):
            return False
    return True


def load_streaming_tokenizer(
    model_name_or_path: str,
    *,
    config: Optional[Any] = None,
    tokenizer_name_or_path: Optional[str] = None,
    padding_side: Optional[str] = None,
    prefer_eoe: bool = False,
) -> tuple[Any, StreamingTokenizerInfo]:
    """按模型 embedding 大小恢复 SFT 时的特殊 token，并拒绝静默错配。"""
    from transformers import AutoConfig, AutoTokenizer

    model_config = config or AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    expected_vocab_size = int(getattr(model_config, "vocab_size", 0) or 0)
    if expected_vocab_size <= 0:
        raise ValueError("Model config is missing a valid vocab_size.")

    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    tokenizer_kwargs = {
        "config": model_config,
        "trust_remote_code": True,
    }
    if padding_side:
        tokenizer_kwargs["padding_side"] = padding_side

    candidates = (
        ("with_eoe", STREAM_SPECIAL_TOKENS_WITH_EOE),
        ("no_eoe", STREAM_SPECIAL_TOKENS_NO_EOE),
    )
    if not prefer_eoe:
        candidates = tuple(reversed(candidates))

    initial = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    for token_set_name, tokens in (("checkpoint", ()), *candidates):
        tokenizer = (
            initial
            if token_set_name == "checkpoint"
            else AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        )
        before_size = len(tokenizer)
        if tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": list(tokens)}
            )
        if len(tokenizer) != expected_vocab_size:
            continue
        required_tokens = (
            STREAM_SPECIAL_TOKENS_NO_EOE
            if token_set_name == "checkpoint"
            else tokens
        )
        if not _tokens_are_atomic(tokenizer, required_tokens):
            continue
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        token_ids = {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in required_tokens
        }
        return tokenizer, StreamingTokenizerInfo(
            vocab_size=len(tokenizer),
            token_set=token_set_name,
            added_tokens=len(tokenizer) - before_size,
            token_ids=token_ids,
        )

    raise ValueError(
        "Cannot restore a streaming tokenizer aligned with model embeddings: "
        f"tokenizer={tokenizer_source} model_vocab_size={expected_vocab_size} "
        f"checkpoint_tokenizer_size={len(initial)}。"
    )
