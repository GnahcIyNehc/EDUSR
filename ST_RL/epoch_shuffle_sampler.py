"""提供可精确恢复的按 Epoch 确定性随机采样器。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpochShuffleSampler:
    """每个 Epoch 只生成一次排列，并从持久化游标继续读取。"""

    dataset_size: int
    batch_size: int
    shuffle_seed: int
    epoch: int = 0
    data_cursor: int = 0
    _permutation_epoch: int | None = field(default=None, init=False, repr=False)
    _permutation: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dataset_size = int(self.dataset_size)
        self.batch_size = int(self.batch_size)
        self.shuffle_seed = int(self.shuffle_seed)
        self.epoch = int(self.epoch)
        self.data_cursor = int(self.data_cursor)
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        if not 0 <= self.data_cursor < self.dataset_size:
            raise ValueError("data_cursor must be within the current epoch data range.")

    def _get_permutation(self) -> list[int]:
        """按基础种子和 Epoch 编号生成唯一且可重建的排列。"""
        if self._permutation_epoch != self.epoch:
            permutation = list(range(self.dataset_size))
            random.Random(self.shuffle_seed + self.epoch).shuffle(permutation)
            self._permutation = permutation
            self._permutation_epoch = self.epoch
        return self._permutation

    def next_batch(self) -> list[int]:
        """读取下一批索引；批次跨 Epoch 时自动切换到下一排列。"""
        selected: list[int] = []
        while len(selected) < self.batch_size:
            permutation = self._get_permutation()
            take = min(self.batch_size - len(selected), self.dataset_size - self.data_cursor)
            selected.extend(permutation[self.data_cursor : self.data_cursor + take])
            self.data_cursor += take
            if self.data_cursor == self.dataset_size:
                self.epoch += 1
                self.data_cursor = 0
                self._permutation_epoch = None
                self._permutation = []
        return selected

    def state_dict(self) -> dict[str, int]:
        """返回足以重建下一批题目顺序的最小持久化状态。"""
        return {
            "format_version": 1,
            "epoch": self.epoch,
            "shuffle_seed": self.shuffle_seed,
            "data_cursor": self.data_cursor,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        dataset_size: int,
        batch_size: int,
        shuffle_seed: int,
    ) -> "EpochShuffleSampler":
        """校验 checkpoint 契约后恢复采样器。"""
        required = {
            "epoch",
            "shuffle_seed",
            "data_cursor",
            "dataset_size",
            "batch_size",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Sampler resume state is missing fields: {missing}")
        expected = {
            "shuffle_seed": int(shuffle_seed),
            "dataset_size": int(dataset_size),
            "batch_size": int(batch_size),
        }
        mismatches = {
            key: {"checkpoint": int(state[key]), "current": value}
            for key, value in expected.items()
            if int(state[key]) != value
        }
        if mismatches:
            raise ValueError(f"Sampler resume configuration differs from the current run: {mismatches}")
        return cls(
            dataset_size=dataset_size,
            batch_size=batch_size,
            shuffle_seed=shuffle_seed,
            epoch=int(state["epoch"]),
            data_cursor=int(state["data_cursor"]),
        )

    @classmethod
    def from_consumed_samples(
        cls,
        *,
        dataset_size: int,
        batch_size: int,
        shuffle_seed: int,
        consumed_samples: int,
    ) -> "EpochShuffleSampler":
        """为仅给出 step_offset 的新任务推导确定性的读取位置。"""
        consumed_samples = int(consumed_samples)
        if consumed_samples < 0:
            raise ValueError("consumed_samples must be nonnegative.")
        epoch, data_cursor = divmod(consumed_samples, int(dataset_size))
        return cls(
            dataset_size=dataset_size,
            batch_size=batch_size,
            shuffle_seed=shuffle_seed,
            epoch=epoch,
            data_cursor=data_cursor,
        )
