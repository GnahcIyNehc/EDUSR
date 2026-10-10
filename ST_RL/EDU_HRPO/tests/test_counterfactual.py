from types import SimpleNamespace

import pytest

from ST_RL.EDU_HRPO.counterfactual import (
    FORBIDDEN_PROBE_MARKERS,
    CounterfactualProbeRunner,
    ProbeInput,
    assert_no_stream_prompt,
    build_no_stream_probe_prompt,
    build_stream_probe_prompt,
)
from ST_RL.EDU_HRPO.task_scorers import ClassificationScorer, MathTaskScorer


def make_protocol():
    # 构造与 SFT 连接顺序一致的最小测试协议。
    return SimpleNamespace(
        target_prefix="<|im_start|>assistant\n<think>\n<SEP>",
        reasoning_connect_2="Let's think about the question deeply.\n",
        reasoning_connect_3="</think>\n\n",
    )


def test_counterfactual_prompts_only_use_sft_protocol_tokens():
    # 验证两类探针不再插入训练时未见过的伪标记或临时指令。
    protocol = make_protocol()
    stream_prompt = build_stream_probe_prompt(
        "Question<EOQ><|im_end|>\n",
        ("first thought<EOT>", "second thought<EOT>"),
        protocol,
        answer_prefix="\\boxed{",
    )
    no_stream_prompt = build_no_stream_probe_prompt(
        "Question<EOQ><|im_end|>\n",
        ("<EOQ>", "<EOT>"),
        protocol,
    )

    assert stream_prompt.endswith("second thought<EOT></think>\n\n\\boxed{")
    assert no_stream_prompt.endswith(
        "<ignore><EOQ><ignore><EOT>Let's think about the question deeply.\n"
    )
    assert not any(marker in stream_prompt for marker in FORBIDDEN_PROBE_MARKERS)
    assert not any(marker in no_stream_prompt for marker in FORBIDDEN_PROBE_MARKERS)
    assert_no_stream_prompt(no_stream_prompt, ["normal hidden thought"])
    with pytest.raises(ValueError, match="leaked"):
        assert_no_stream_prompt(
            no_stream_prompt + "normal hidden thought",
            ["normal hidden thought"],
        )


def test_probe_runner_shares_one_no_stream_probe_per_prompt():
    # 验证同一问题的多条轨迹只共享一次 no-stream 深度探针。
    calls = []

    def generate(prompts, temperature, max_tokens):
        calls.append((tuple(prompts), temperature, max_tokens))
        return ["\\boxed{2}" for _ in prompts]

    runner = CounterfactualProbeRunner(
        generate_batch=generate,
        scorer=MathTaskScorer(require_boxed=True),
        protocol=make_protocol(),
        stream_max_tokens=16,
        no_stream_max_tokens=128,
    )
    inputs = [
        ProbeInput("p0", "1+1?<EOQ>", ("add one<EOT>",), ("<EOQ>",), "2"),
        ProbeInput("p0", "1+1?<EOQ>", ("use arithmetic<EOT>",), ("<EOQ>",), "2"),
        ProbeInput("p1", "2-1?<EOQ>", ("subtract<EOT>",), ("<EOQ>",), "1"),
    ]
    result = runner.run(inputs)

    assert len(result.stream_outputs) == 3
    assert len(result.no_stream_by_prompt) == 2
    assert len(calls[0][0]) == 3
    assert len(calls[1][0]) == 2
    assert calls[0][1:] == (0.0, 16)
    assert calls[1][1:] == (0.0, 128)


def test_stream_probe_reconstructs_sft_boxed_prefix_before_scoring():
    # 模型只续写标签与右括号，评分文本必须恢复为完整 SFT boxed 答案。
    def generate(prompts, temperature, max_tokens):
        del temperature, max_tokens
        return [
            "entailment}" if prompt.endswith("\\boxed{") else "\\boxed{entailment}"
            for prompt in prompts
        ]

    runner = CounterfactualProbeRunner(
        generate_batch=generate,
        scorer=ClassificationScorer(),
        protocol=make_protocol(),
        stream_answer_prefix="\\boxed{",
    )
    result = runner.run(
        [
            ProbeInput(
                "p0",
                "Claim<EOQ>",
                ("state<EOT>",),
                ("<EOQ>",),
                "\\boxed{entailment}",
            )
        ]
    )

    assert result.stream_outputs[0].answer == "\\boxed{entailment}"
    assert result.stream_outputs[0].exact == 1.0
    assert result.stream_outputs[0].valid is True
