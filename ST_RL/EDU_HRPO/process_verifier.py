"""冻结过程验证器的最小适配接口。"""

from __future__ import annotations

from typing import Any, Callable, Sequence


class NullProcessVerifier:
    """没有可靠验证器时必须显式返回零，不能用 actor 自评替代。"""

    frozen = True

    def score(self, prompt: str, stream_steps: Sequence[str], metadata: Any = None) -> float:
        return 0.0


class CallableProcessVerifier:
    """包装外部冻结 PAV/PRM，并强制输出落在 [0,1]。"""

    frozen = True

    def __init__(self, score_fn: Callable[[str, Sequence[str], Any], float]) -> None:
        self.score_fn = score_fn

    def score(self, prompt: str, stream_steps: Sequence[str], metadata: Any = None) -> float:
        value = float(self.score_fn(prompt, stream_steps, metadata))
        if not 0.0 <= value <= 1.0:
            raise ValueError("Process verifier scores must be in [0, 1].")
        return value

