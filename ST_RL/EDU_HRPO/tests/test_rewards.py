import torch

from ST_RL.common import group_normalize
from ST_RL.EDU_HRPO.config import EDUBLHRPOConfig
from ST_RL.EDU_HRPO.rewards import EDURewardBatch, compute_edu_blhrpo_advantages


def make_config(**overrides):
    values = dict(
        generation_tokens_per_second=10.0,
        deep_budget=4.0,
        reward_tau=20.0,
        budget_min=0.0,
        budget_max=64.0,
        budget_base=0.0,
        budget_input_scale=0.0,
        budget_uncertainty_scale=0.0,
        budget_relevance_scale=0.0,
    )
    values.update(overrides)
    return EDUBLHRPOConfig(**values)


def make_batch(all_equal=False):
    exact = torch.tensor([1.0, 0.0, 1.0, 0.0])
    part = torch.tensor([0.2, 0.8, 0.4, 0.6])
    if all_equal:
        exact = torch.ones(4)
        part = torch.full((4,), 0.5)
    return EDURewardBatch(
        group_ids=torch.tensor([0, 0, 1, 1]),
        exact=exact,
        part=part,
        stream_probe_exact=torch.tensor([1.0, 1.0, 0.0, 0.0]) if not all_equal else exact,
        stream_probe_part=part.clone(),
        no_stream_exact=torch.tensor([0.0, 0.0, 1.0, 1.0]) if not all_equal else exact,
        no_stream_part=part.clone(),
        stream_probe_valid=torch.ones(4),
        no_stream_probe_valid=torch.ones(4),
        stream_probe_truncated=torch.zeros(4),
        no_stream_probe_truncated=torch.zeros(4),
        stream_verifier=torch.zeros(4),
        stream_format=torch.ones(4),
        deep_format=torch.ones(4),
        deep_lengths=torch.tensor([4.0, 8.0, 4.0, 8.0]) if not all_equal else torch.full((4,), 4.0),
        thought_lengths=torch.tensor([[4.0], [8.0], [4.0], [8.0]]) if not all_equal else torch.full((4, 1), 4.0),
        input_lengths=torch.full((4, 1), 4.0),
        group_edu_counts=torch.ones(4, 1),
        group_mask=torch.ones(4, 1),
        think_group_mask=torch.ones(4, 1),
        control_uncertainty=torch.zeros(4, 1),
        relevance=torch.zeros(4, 1),
        slack_tokens=torch.zeros(4, 1),
    )


def test_each_reward_component_is_group_normalized_before_weighting():
    config = make_config(gamma_part=0.2, beta_eff=0.05)
    batch = make_batch()
    result = compute_edu_blhrpo_advantages(batch, config, reward_tau=20.0)

    expected_exact = group_normalize(batch.exact, batch.group_ids)
    expected_part = group_normalize(batch.part, batch.group_ids)
    expected_global = (
        expected_exact
        + config.gamma_part * expected_part
        + config.beta_eff * result.normalized_components["efficiency"]
    )
    assert torch.allclose(result.normalized_components["exact"], expected_exact)
    assert torch.allclose(result.normalized_components["part"], expected_part)
    assert torch.allclose(result.global_, expected_global)


def test_zero_variance_groups_return_zero_without_nan():
    result = compute_edu_blhrpo_advantages(
        make_batch(all_equal=True),
        make_config(),
        reward_tau=20.0,
    )
    assert torch.isfinite(result.streaming).all()
    assert torch.isfinite(result.deep).all()
    assert torch.isfinite(result.global_).all()
    assert torch.equal(result.streaming, torch.zeros_like(result.streaming))
    assert torch.equal(result.deep, torch.zeros_like(result.deep))
    assert torch.equal(result.global_, torch.zeros_like(result.global_))


def test_wrong_trajectories_still_receive_stream_and_deep_length_rewards():
    batch = make_batch()
    batch.exact.zero_()
    result = compute_edu_blhrpo_advantages(batch, make_config(), reward_tau=20.0)

    assert result.components.stream_length_reward[1] < 0
    assert result.components.deep_length_reward[1] < 0
    assert torch.equal(result.components.efficiency_reward, torch.zeros(4))


def test_attribution_distinguishes_stream_harm_and_deep_regression():
    batch = make_batch()
    result = compute_edu_blhrpo_advantages(
        batch,
        make_config(rho_fix=0.5),
        reward_tau=20.0,
    )

    # 第一条最终和 stream-only 都正确，而 no-stream 错，流式贡献为正。
    assert result.components.stream_attribution[0] > 0
    assert result.components.deep_attribution[0] == 0
    # 第二条 stream-only 正确但最终错误，深度阶段应收到负归因。
    assert result.components.deep_attribution[1] < 0
    # 第四条 no-stream 正确但正常轨迹错误，流式归因为负。
    assert result.components.stream_attribution[3] < 0


def test_shared_no_stream_view_survives_group_normalization():
    # Q0 通过帮助/伤害门控改变组内排序，不能再退化成被中心化消掉的常数。
    low_q0 = make_batch()
    low_q0.group_ids.zero_()
    low_q0.no_stream_exact.zero_()
    low_q0.no_stream_part.zero_()
    high_q0 = make_batch()
    high_q0.group_ids.zero_()
    high_q0.no_stream_exact.fill_(1.0)
    high_q0.no_stream_part.zero_()

    low_result = compute_edu_blhrpo_advantages(
        low_q0,
        make_config(),
        reward_tau=20.0,
    )
    high_result = compute_edu_blhrpo_advantages(
        high_q0,
        make_config(),
        reward_tau=20.0,
    )

    assert not torch.allclose(
        low_result.normalized_components["stream_attribution"],
        high_result.normalized_components["stream_attribution"],
    )


def test_invalid_stream_probe_receives_no_counterfactual_advantage():
    # 生成失败不是错误答案；对应轨迹的两级反事实优势必须保持为零。
    batch = make_batch()
    batch.group_ids.zero_()
    batch.stream_probe_valid[1] = 0.0
    result = compute_edu_blhrpo_advantages(
        batch,
        make_config(),
        reward_tau=20.0,
    )

    assert result.normalized_components["stream_attribution"][1] == 0.0
    assert result.normalized_components["deep_attribution"][1] == 0.0
