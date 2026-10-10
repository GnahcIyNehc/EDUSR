"""EDU vLLM/FSDP 接入层的轻量回归测试。"""

from unittest.mock import patch

import torch

from ST_RL.EDU_HRPO.config import EDUBLHRPOConfig
from ST_RL.EDU_HRPO.controller import ControlAction
from ST_RL.EDU_HRPO.fsdp_integration import (
    _build_dynamic_groups,
    _build_fixed_groups,
    _reduce_edu_diagnostics,
    build_advantages_adapter,
    compose_reporting_rewards,
    prepare_edu_group,
)
from ST_RL.EDU_HRPO.controller import EDUBoundaryEvent
from ST_RL.EDU_HRPO.task_scorers import ClassificationScorer, ExtractiveQAScorer
from ST_RL.EDU_HRPO.trainer import EDUBLHRPOTrainerCore
from ST_RL.HRPO import train_hrpo_fsdp as hrpo_runner
from ST_RL.training_utils import RLSample


class FakeTokenizer:
    """只实现接入层测试所需的编码和解码接口。"""

    pieces = {
        10: "<ignore><EOQ>",
        11: "Relevant fact<EOT>",
        12: "Therefore \\boxed{yes}<EOA>",
    }

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(max(len(str(text).split()), 1)))}

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


def test_boxed_nonmath_scorers_are_format_compatible():
    classification = ClassificationScorer().score("\\boxed{yes}", "\\boxed{yes}")
    extractive = ExtractiveQAScorer().score(
        "\\boxed{New York}",
        "\\boxed{New York}",
    )
    assert classification.exact == 1.0
    assert extractive.exact == 1.0
    assert extractive.part == 1.0
    assert ClassificationScorer().is_valid_probe("\\boxed{yes}") is True
    assert ClassificationScorer().is_valid_probe("yes") is False
    assert ClassificationScorer().is_valid_probe("\\boxed{yes") is False


def test_prepare_group_builds_records_and_edu_advantages():
    tokenizer = FakeTokenizer()
    sample = RLSample(
        question="Classify this claim.<EOQ>",
        context_segments=["Relevant fact"],
        answer="\\boxed{yes}",
        raw={"Instruct_Context": {"instruct": "Classify this claim.<EOQ>", "context": "Relevant fact<EOS>"}},
    )
    completion = {
        "token_ids": [10, 11, 12],
        "text": "<ignore><EOQ>Relevant fact<EOT>Therefore \\boxed{yes}<EOA>",
        "finish_reason": "stop",
        "response_segment_token_lengths": [1, 1, 1],
        "stream_probe_text": "\\boxed{yes}",
    }
    item = {
        "sample_index": 0,
        "source_segments": ["prefix", "question<EOQ>", "context<EOS>"],
        "source_segment_token_ids": [[1], [2, 3], [4, 5]],
        "target_prefix_token_ids": [6],
        "expected_boundaries": ["<EOQ>", "<EOT>"],
        "question_segments": 1,
        "no_stream_probe_text": "\\boxed{no}",
        "completions": [dict(completion), dict(completion)],
    }
    prepare_edu_group._base_prepare = hrpo_runner.prepare_group
    group = prepare_edu_group(
        item=item,
        samples=[sample],
        tokenizer=tokenizer,
        expected_generations=2,
        accuracy_reward_mode="boxed",
        task_type="classification",
        input_tokens_per_second=2.0,
        max_merge=3,
    )
    assert len(group.records) == 2
    assert group.records[0].stream_groups[0].action == ControlAction.SKIP
    assert group.records[0].stream_groups[1].action == ControlAction.THINK
    assert group.records[0].stream_probe.exact == 1.0
    assert group.records[0].no_stream_probe.exact == 0.0
    assert group.records[0].stream_probe.valid is True
    assert group.records[0].no_stream_probe.valid is True

    config = EDUBLHRPOConfig(
        rollouts_per_prompt=2,
        generation_tokens_per_second=20.0,
        deep_budget=8.0,
        reward_tau=20.0,
    )
    adapter = build_advantages_adapter(
        group,
        config,
        torch.device("cpu"),
        EDUBLHRPOTrainerCore,
    )
    assert adapter.streaming.shape == (2,)
    assert adapter.deep.shape == (2,)
    assert adapter.global_.shape == (2,)
    assert torch.isfinite(adapter.reward_raw).all()
    reporting = compose_reporting_rewards(adapter, config)
    assert torch.equal(reporting["reward"], adapter.reward_raw)
    assert torch.equal(reporting["reward_raw"], adapter.reward_raw)
    assert torch.equal(reporting["advantage"], adapter.advantage)
    with patch("ST_RL.EDU_HRPO.fsdp_integration.dist.all_reduce"):
        diagnostics = _reduce_edu_diagnostics([adapter], torch.device("cpu"))
    assert diagnostics["stream_probe_valid_rate"] == 1.0
    assert diagnostics["no_stream_probe_valid_rate"] == 1.0
    assert diagnostics["stream_probe_accuracy"] == 1.0
    assert diagnostics["no_stream_accuracy"] == 0.0


def test_controller_is_opt_in_and_fixed_mode_never_merges_edus():
    tokenizer = FakeTokenizer()
    events = [
        EDUBoundaryEvent(0, "first", 2, 2, 1.0, 1.0),
        EDUBoundaryEvent(1, "second", 3, 5, 2.0, 2.0),
    ]
    stage_texts = ["<defer><EOQ>", "reason<EOT>"]

    fixed = _build_fixed_groups(events, stage_texts, tokenizer, "question")
    dynamic = _build_dynamic_groups(
        events,
        stage_texts,
        [0.5, 0.25],
        tokenizer,
        "question",
        max_merge=3,
    )

    assert len(fixed) == 2
    assert [group.edu_count for group in fixed] == [1, 1]
    assert fixed[0].action == ControlAction.THINK
    assert fixed[0].thought == "<defer>"
    assert all(group.control_entropy == 0.0 for group in fixed)
    assert len(dynamic) == 1
    assert dynamic[0].edu_count == 2


def test_edu_controller_is_disabled_by_default():
    assert EDUBLHRPOConfig().enable_edu_controller is False
