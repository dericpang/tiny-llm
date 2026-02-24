import torch
from torch.utils.data import Dataset

DATA = [
    "My name is Deric.",
    "Jenny is my wife.",
    "I work at Figma",
    "I was born in Las Vegas.",
    "I live in Queens.",
    "I take the W train to work",
    "I grew up playing tennis.",
    "These days, I really enjoy playing poker!"
]


class SimpleCharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, max_len: int):
        text = "".join(DATA)
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ctoi = {ch: i for i, ch in enumerate(chars)}
        self.itoc = {i: ch for i, ch in enumerate(chars)}

        self.examples: list[torch.Tensor] = []
        for example in DATA:
            indexes = [self.ctoi[c] for c in example] + [0] * (max_len - len(example))
            self.examples.append(torch.tensor(indexes, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        return tokens[:-1], tokens[1:]

    def encode(self, text: str) -> list[int]:
        return [self.ctoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itoc[i] for i in tokens)
