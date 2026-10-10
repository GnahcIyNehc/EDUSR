"""stream-only 与 no-stream 三视图反事实探针。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Sequence, Tuple

from .task_scorers import TaskScore

if TYPE_CHECKING:
    from ST_RL.prompt_protocol import PromptProtocol


FORBIDDEN_PROBE_MARKERS = (
    "<STREAM_STATE>",
    "</STREAM_STATE>",
    "<ANSWER_PROBE>",
    "<NO_STREAM>",
)
VALID_NO_STREAM_TOKENS = ("<ignore>", "<skip>")


@dataclass(frozen=True)
class ProbeInput:
    prompt_id: str
    full_input: str
    stream_state: Tuple[str, ...]
    expected_boundaries: Tuple[str, ...]
    answer: str

    @property
    def edu_count(self) -> int:
        """返回真实输入 EDU 数量，避免用数量猜测末尾边界类型。"""
        return len(self.expected_boundaries)


@dataclass(frozen=True)
class ProbeOutput:
    answer: str
    exact: float
    part: float
    valid: bool = True

    @property
    def quality(self) -> TaskScore:
        return TaskScore(exact=self.exact, part=self.part)


@dataclass(frozen=True)
class CounterfactualResult:
    stream_outputs: Tuple[ProbeOutput, ...]
    no_stream_by_prompt: Dict[str, ProbeOutput]
    stream_prompts: Tuple[str, ...]
    no_stream_prompts: Dict[str, str]


def build_stream_probe_target_segments(
    stream_state: Sequence[str],
    reasoning_connect_3: str,
    answer_prefix: str = "",
) -> Tuple[str, ...]:
    """沿用真实流式状态，关闭 think 后直接进入答案前缀。"""
    connector = str(reasoning_connect_3 or "")
    if "</think>" not in connector:
        raise ValueError("A stream-only probe requires an SFT connector containing </think>.")
    segments = tuple(str(segment) for segment in stream_state)
    if any(not segment for segment in segments):
        raise ValueError("A stream-only probe must not contain empty streaming-state segments.")
    return segments + (connector + str(answer_prefix or ""),)


def build_no_stream_probe_target_segments(
    expected_boundaries: Sequence[str],
    reasoning_connect_2: str,
    skip_token: str = "<ignore>",
) -> Tuple[str, ...]:
    """以 SFT 已见过的忽略状态替换浅层 thought，再进入深度推理。"""
    if skip_token not in VALID_NO_STREAM_TOKENS:
        raise ValueError("A no-stream probe may use only <ignore> or <skip>.")
    boundaries = tuple(str(boundary) for boundary in expected_boundaries)
    if not boundaries or any(boundary not in ("<EOT>", "<EOQ>", "<EOP>") for boundary in boundaries):
        raise ValueError("A no-stream probe requires a valid response boundary for every real input EDU.")
    connector = str(reasoning_connect_2 or "")
    if not connector:
        raise ValueError("The no-stream probe is missing the SFT deep-reasoning connector.")
    return tuple(f"{skip_token}{boundary}" for boundary in boundaries) + (connector,)


def render_protocol_probe_prompt(
    full_input: str,
    target_prefix: str,
    target_segments: Sequence[str],
) -> str:
    """按原 SFT 文本顺序渲染探针，不插入训练时未出现的提示标记。"""
    prompt = str(full_input) + str(target_prefix) + "".join(str(value) for value in target_segments)
    unexpected = [marker for marker in FORBIDDEN_PROBE_MARKERS if marker in prompt]
    if unexpected:
        raise ValueError("The counterfactual probe contains a non-SFT marker: %s" % ", ".join(unexpected))
    return prompt


def build_stream_probe_prompt(
    full_input: str,
    stream_state: Sequence[str],
    protocol: "PromptProtocol",
    answer_prefix: str = "",
) -> str:
    """使用注册的 SFT 协议构造 stream-only 探针。"""
    segments = build_stream_probe_target_segments(
        stream_state,
        protocol.reasoning_connect_3,
        answer_prefix=answer_prefix,
    )
    return render_protocol_probe_prompt(full_input, protocol.target_prefix, segments)


def build_no_stream_probe_prompt(
    full_input: str,
    expected_boundaries: Sequence[str],
    protocol: "PromptProtocol",
    skip_token: str = "<ignore>",
) -> str:
    """使用注册的 SFT 协议构造 no-stream 深度推理探针。"""
    segments = build_no_stream_probe_target_segments(
        expected_boundaries,
        protocol.reasoning_connect_2,
        skip_token=skip_token,
    )
    return render_protocol_probe_prompt(full_input, protocol.target_prefix, segments)


def assert_no_stream_prompt(prompt: str, normal_stream_thoughts: Iterable[str]) -> None:
    """确保 no-stream 输入没有伪标记，也没有泄漏正常轨迹思考文本。"""
    unexpected = [marker for marker in FORBIDDEN_PROBE_MARKERS if marker in prompt]
    if unexpected:
        raise ValueError("The no-stream probe contains a non-SFT marker: %s" % ", ".join(unexpected))
    if "<ignore>" not in prompt and "<skip>" not in prompt:
        raise ValueError("The no-stream probe is missing the SFT streaming ignore state.")
    for thought in normal_stream_thoughts:
        normalized = str(thought).strip()
        if normalized and normalized in prompt:
            raise ValueError("The no-stream probe leaked normal streaming thoughts.")


class CounterfactualProbeRunner:
    """批量执行每轨迹短探针和每 prompt 共享的 no-stream 探针。"""

    def __init__(
        self,
        generate_batch: Callable[[Sequence[str], float, int], Sequence[str]],
        scorer: Any,
        protocol: "PromptProtocol",
        stream_max_tokens: int = 16,
        no_stream_max_tokens: int = 512,
        no_stream_token: str = "<ignore>",
        stream_answer_prefix: str = "",
    ) -> None:
        if stream_max_tokens < 1 or no_stream_max_tokens < 1:
            raise ValueError("Probe generation lengths must be positive.")
        if no_stream_token not in VALID_NO_STREAM_TOKENS:
            raise ValueError("no_stream_token must be <ignore> or <skip>.")
        self.generate_batch = generate_batch
        self.scorer = scorer
        self.protocol = protocol
        self.stream_max_tokens = int(stream_max_tokens)
        self.no_stream_max_tokens = int(no_stream_max_tokens)
        self.no_stream_token = no_stream_token
        self.stream_answer_prefix = str(stream_answer_prefix or "")

    def _score(self, prediction: str, answer: str) -> ProbeOutput:
        score = self.scorer.score(prediction, answer)
        validity_function = getattr(self.scorer, "is_valid_probe", None)
        valid = bool(validity_function(prediction)) if validity_function is not None else bool(str(prediction).strip())
        return ProbeOutput(
            answer=str(prediction),
            exact=float(score.exact),
            part=float(score.part),
            valid=valid,
        )

    def run(self, inputs: Sequence[ProbeInput]) -> CounterfactualResult:
        if not inputs:
            raise ValueError("Counterfactual probe inputs must not be empty.")
        stream_prompts = tuple(
            build_stream_probe_prompt(
                item.full_input,
                item.stream_state,
                self.protocol,
                answer_prefix=self.stream_answer_prefix,
            )
            for item in inputs
        )
        stream_answers = tuple(
            self.generate_batch(stream_prompts, 0.0, self.stream_max_tokens)
        )
        if len(stream_answers) != len(inputs):
            raise RuntimeError("Stream-only probe output and rollout counts differ.")
        stream_outputs = tuple(
            self._score(self.stream_answer_prefix + prediction, item.answer)
            for prediction, item in zip(stream_answers, inputs)
        )

        unique = {}  # type: Dict[str, ProbeInput]
        thoughts = {}  # type: Dict[str, List[str]]
        for item in inputs:
            previous = unique.get(item.prompt_id)
            if previous is not None:
                if (
                    previous.full_input != item.full_input
                    or previous.answer != item.answer
                    or previous.expected_boundaries != item.expected_boundaries
                ):
                    raise ValueError("Counterfactual base inputs differ for the same prompt_id.")
            else:
                unique[item.prompt_id] = item
            thoughts.setdefault(item.prompt_id, []).extend(item.stream_state)

        prompt_ids = tuple(unique)
        no_stream_prompts = {
            prompt_id: build_no_stream_probe_prompt(
                unique[prompt_id].full_input,
                unique[prompt_id].expected_boundaries,
                self.protocol,
                skip_token=self.no_stream_token,
            )
            for prompt_id in prompt_ids
        }
        for prompt_id, prompt in no_stream_prompts.items():
            assert_no_stream_prompt(prompt, thoughts[prompt_id])
        no_stream_answers = tuple(
            self.generate_batch(
                tuple(no_stream_prompts[prompt_id] for prompt_id in prompt_ids),
                0.0,
                self.no_stream_max_tokens,
            )
        )
        if len(no_stream_answers) != len(prompt_ids):
            raise RuntimeError("No-stream probe output and prompt counts differ.")
        no_stream_by_prompt = {
            prompt_id: self._score(prediction, unique[prompt_id].answer)
            for prompt_id, prediction in zip(prompt_ids, no_stream_answers)
        }
        return CounterfactualResult(
            stream_outputs=stream_outputs,
            no_stream_by_prompt=no_stream_by_prompt,
            stream_prompts=stream_prompts,
            no_stream_prompts=no_stream_prompts,
        )
