"""EDU-BLHRPO 的集中配置与校验。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class EDUBLHRPOConfig:
    """实验计划第 5--8 节使用的核心超参数。"""

    clip_range: float = 0.2
    kl_coef: float = 0.0
    local_weight: float = 0.05
    normalize_eps: float = 1e-8
    format_weight: float = 2.0
    rollouts_per_prompt: int = 12

    gamma_attr_s: float = 0.20
    gamma_attr_d: float = 0.10
    rho_v: float = 0.25
    rho_fix: float = 0.50

    exact_weight: float = 1.0
    gamma_part: float = 0.20
    beta_eff: float = 0.05
    beta_s: float = 0.02
    beta_d: float = 0.05

    alpha_in: float = 0.10
    alpha_over: float = 1.00
    stream_cost_mode: str = "overlap_aware"
    budget_min: float = 4.0
    budget_max: float = 64.0
    budget_base: float = 4.0
    budget_input_scale: float = 0.4
    budget_uncertainty_scale: float = 8.0
    budget_relevance_scale: float = 16.0
    trigger_penalty: float = 0.05

    # 这两个值必须由训练硬件和句子基线派生，不能在测试集上调节。
    generation_tokens_per_second: Optional[float] = None
    deep_budget: Optional[float] = None
    reward_tau: Optional[float] = None
    tau_warmup_correct_rollouts: int = 512

    # 控制器是独立消融项；默认只启用第 5--8 节奖励，不改变原 SFT 动作协议。
    enable_edu_controller: bool = False
    max_merge: int = 3
    stream_probe_max_tokens: int = 16
    no_stream_probe_max_tokens: int = 512
    probe_temperature: float = 0.0

    def validate(self, require_derived: bool = False) -> None:
        """尽早拒绝会改变奖励语义或产生非法数值的配置。"""

        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must be in (0, 1).")
        if self.kl_coef < 0 or self.local_weight < 0:
            raise ValueError("KL and local advantage weights must be nonnegative.")
        for name in (
            "gamma_attr_s",
            "gamma_attr_d",
            "rho_v",
            "rho_fix",
            "exact_weight",
            "gamma_part",
            "beta_eff",
            "beta_s",
            "beta_d",
            "alpha_in",
            "alpha_over",
            "trigger_penalty",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError("%s must be nonnegative." % name)
        if self.exact_weight != 1.0:
            raise ValueError("exact_weight must remain 1.0 so auxiliary rewards cannot replace the task objective.")
        if self.format_weight != 2.0:
            raise ValueError("format_weight must remain 2.0 as in the AdaSR baseline.")
        if self.rollouts_per_prompt < 2:
            raise ValueError("rollouts_per_prompt must be at least 2 to compute group-relative advantages.")
        if self.stream_cost_mode not in {"overlap_aware", "uniform_small"}:
            raise ValueError("stream_cost_mode must be overlap_aware or uniform_small.")
        if self.budget_min < 0 or self.budget_max < self.budget_min:
            raise ValueError("Invalid EDU budget bounds.")
        if not isinstance(self.enable_edu_controller, bool):
            raise ValueError("enable_edu_controller must be a boolean.")
        if self.max_merge < 1:
            raise ValueError("max_merge must be at least 1.")
        if self.stream_probe_max_tokens < 1 or self.no_stream_probe_max_tokens < 1:
            raise ValueError("Counterfactual probe lengths must be positive.")
        if self.probe_temperature != 0.0:
            raise ValueError("Counterfactual probes must use temperature=0.")
        if self.tau_warmup_correct_rollouts < 1:
            raise ValueError("tau_warmup_correct_rollouts must be positive.")
        if self.generation_tokens_per_second is not None:
            if self.generation_tokens_per_second <= 0:
                raise ValueError("generation_tokens_per_second must be positive.")
        if self.deep_budget is not None and self.deep_budget < 0:
            raise ValueError("deep_budget must be nonnegative.")
        if self.reward_tau is not None and self.reward_tau <= 0:
            raise ValueError("reward_tau must be positive.")
        if require_derived:
            if self.generation_tokens_per_second is None:
                raise ValueError("Measure generation_tokens_per_second before training.")
            if self.deep_budget is None:
                raise ValueError("Derive deep_budget from the sentence baseline before training.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EDUBLHRPOConfig":
        """从映射加载配置，同时拒绝拼写错误的字段。"""

        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("Unknown configuration field: %s" % ", ".join(unknown))
        config = cls(**dict(values))
        config.validate(require_derived=False)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str) -> EDUBLHRPOConfig:
    """读取 JSON 或 YAML 配置；YAML 支持由 PyYAML 提供。"""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Install PyYAML to read YAML configurations.") from error
        values = yaml.safe_load(text)
    if not isinstance(values, Mapping):
        raise ValueError("The top-level configuration must be an object.")
    return EDUBLHRPOConfig.from_mapping(values)
