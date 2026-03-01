from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

PAD_TOKEN_ID = 0


class CharDataset(ABC, Dataset[tuple[torch.Tensor, torch.Tensor]]):
    vocab_size: int
    ctoi: dict[str, int]
    itoc: dict[int, str]

    @abstractmethod
    def __init__(self, max_len: int, split: str = "train") -> None: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...
