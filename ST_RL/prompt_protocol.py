"""加载并校验 SFT、RL 与评测共用的流式提示协议。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "prompt_registry.json"
BOUNDARY_MODE_ALIASES = {
    "sentence": "sentence",
    "sent": "sentence",
    "edu": "edu",
    "full_edu": "edu",
    "full-edu": "edu",
}


def _first_text(value: Any, field_name: str) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value or "")
    if not text:
        raise ValueError(f"Prompt configuration field {field_name} must not be empty.")
    return text


def _normalize_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


@dataclass(frozen=True)
class PromptProtocol:
    """一套与具体 SFT 模型绑定的完整流式提示协议。"""

    name: str
    dataset: str
    boundary_mode: str
    instruction: str
    end_instruction: str
    assistant_instruction: str
    reasoning_connect_1: str
    reasoning_connect_2: str
    reasoning_connect_3: str
    config_path: str
    rollout_bootstrap_target_segments: Tuple[str, ...] = ()

    @property
    def source_prefix(self) -> str:
        return self.instruction + "<SEP>"

    @property
    def target_prefix(self) -> str:
        return self.reasoning_connect_1 + "<SEP>"

    @property
    def protocol_hash(self) -> str:
        payload = {
            "instruction": self.instruction,
            "end_instruction": self.end_instruction,
            "assistant_instruction": self.assistant_instruction,
            "reasoning_connect_1": self.reasoning_connect_1,
            "reasoning_connect_2": self.reasoning_connect_2,
            "reasoning_connect_3": self.reasoning_connect_3,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "boundary_mode": self.boundary_mode,
            "config_path": self.config_path,
            "protocol_hash": self.protocol_hash,
            "rollout_bootstrap_target_segments": list(
                self.rollout_bootstrap_target_segments
            ),
        }


def load_prompt_protocol(
    config_path: str | Path,
    *,
    name: str = "",
    dataset: str = "",
    boundary_mode: str = "",
) -> PromptProtocol:
    """从 StreamingThinker 风格 JSON 配置加载提示协议。"""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt configuration does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        if not raw or not isinstance(raw[0], Mapping):
            raise ValueError(f"Prompt configuration must contain one JSON object: {path}")
        raw = raw[0]
    if not isinstance(raw, Mapping):
        raise ValueError(f"Prompt configuration must be a JSON object or a one-object list: {path}")

    instruction = _first_text(raw.get("Instruct", raw.get("instruction")), "Instruct")
    end_instruction = _first_text(raw.get("end_Instruct"), "end_Instruct")
    assistant_instruction = _first_text(raw.get("assistant_Instruct"), "assistant_Instruct")
    reasoning_connect_1 = _first_text(raw.get("reasoning_connect_1"), "reasoning_connect_1")
    reasoning_connect_2 = _first_text(raw.get("reasoning_connect_2"), "reasoning_connect_2")
    reasoning_connect_3 = _first_text(raw.get("reasoning_connect_3"), "reasoning_connect_3")
    bootstrap_value = raw.get("rollout_bootstrap_target_segments") or []
    if isinstance(bootstrap_value, str):
        bootstrap_value = [bootstrap_value]
    if not isinstance(bootstrap_value, list) or any(
        not isinstance(value, str) or not value for value in bootstrap_value
    ):
        raise ValueError(
            f"rollout_bootstrap_target_segments must be a list of nonempty strings: {path}"
        )

    if not instruction.startswith("<|im_start|>user\n"):
        raise ValueError(f"Instruct must start with the Qwen user marker: {path}")
    if assistant_instruction != reasoning_connect_1:
        raise ValueError(f"assistant_Instruct must match reasoning_connect_1: {path}")
    if "<think>" not in reasoning_connect_1:
        raise ValueError(f"reasoning_connect_1 must enter the think stage: {path}")
    if "</think>" not in reasoning_connect_3:
        raise ValueError(f"reasoning_connect_3 must close the think stage: {path}")

    configured_mode = boundary_mode or str(raw.get("boundary_mode") or "")
    normalized_mode = BOUNDARY_MODE_ALIASES.get(str(configured_mode).lower(), configured_mode)
    return PromptProtocol(
        name=name or str(raw.get("protocol_name") or path.stem),
        dataset=_normalize_dataset_name(dataset),
        boundary_mode=str(normalized_mode or ""),
        instruction=instruction,
        end_instruction=end_instruction,
        assistant_instruction=assistant_instruction,
        reasoning_connect_1=reasoning_connect_1,
        reasoning_connect_2=reasoning_connect_2,
        reasoning_connect_3=reasoning_connect_3,
        config_path=str(path),
        rollout_bootstrap_target_segments=tuple(bootstrap_value),
    )


def resolve_prompt_config(
    dataset: str,
    boundary_mode: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """按数据集和边界模式从注册表解析提示配置。"""
    registry_file = Path(registry_path).expanduser().resolve()
    with registry_file.open("r", encoding="utf-8-sig") as handle:
        registry = json.load(handle)
    datasets = registry.get("datasets", {})
    requested = _normalize_dataset_name(dataset)
    canonical = ""
    for name, entry in datasets.items():
        aliases = {_normalize_dataset_name(name)}
        aliases.update(_normalize_dataset_name(alias) for alias in entry.get("aliases", []))
        if requested in aliases:
            canonical = name
            break
    if not canonical:
        available = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset {dataset!r}; available datasets: {available}")

    mode = BOUNDARY_MODE_ALIASES.get(str(boundary_mode or "").lower())
    if not mode:
        raise KeyError("boundary_mode must be sentence or edu.")
    relative_path = datasets[canonical].get("protocols", {}).get(mode)
    if not relative_path:
        raise KeyError(f"Dataset {canonical} has no registered {mode} prompt configuration.")
    return (registry_file.parent / relative_path).resolve()


def load_registered_prompt_protocol(
    dataset: str,
    boundary_mode: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> PromptProtocol:
    """解析注册表并返回带数据集元信息的提示协议。"""
    config_path = resolve_prompt_config(dataset, boundary_mode, registry_path)
    return load_prompt_protocol(
        config_path,
        name=config_path.stem,
        dataset=dataset,
        boundary_mode=boundary_mode,
    )


def load_training_prompt_protocol(
    *,
    prompt_config: str = "",
    dataset: str = "",
    boundary_mode: str = "",
    registry_path: Optional[str] = None,
) -> PromptProtocol:
    """训练入口统一加载函数；显式配置优先，注册表作为便利入口。"""
    if prompt_config:
        explicit = load_prompt_protocol(
            prompt_config,
            dataset=dataset,
        )
        requested_mode = BOUNDARY_MODE_ALIASES.get(str(boundary_mode or "").lower(), boundary_mode)
        if explicit.boundary_mode and requested_mode and explicit.boundary_mode != requested_mode:
            raise ValueError(
                f"Prompt configuration boundary mode is {explicit.boundary_mode}, but training requests {requested_mode}。"
            )
        if dataset and requested_mode:
            registered = load_registered_prompt_protocol(
                dataset,
                requested_mode,
                registry_path or DEFAULT_REGISTRY_PATH,
            )
            if explicit.protocol_hash != registered.protocol_hash:
                raise ValueError(
                    "Explicit PROMPT_CONFIG differs from the registered dataset/boundary template: "
                    f"dataset={dataset} boundary_mode={requested_mode}。"
                )
        return explicit
    if dataset and boundary_mode:
        return load_registered_prompt_protocol(
            dataset,
            boundary_mode,
            registry_path or DEFAULT_REGISTRY_PATH,
        )
    raise ValueError("Provide --prompt-config or both --dataset and --boundary-mode.")
