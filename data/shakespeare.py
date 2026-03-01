from pathlib import Path

import requests
import tiktoken
import torch
from torch.utils.data import Dataset


OUT_DIR = Path(__file__).parent.parent / "out" / "shakespeare"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_SPLIT = 0.9


def _get_text() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUT_DIR / "input.txt"
    if not input_path.exists():
        print("Downloading Shakespeare dataset...")
        input_path.write_text(requests.get(DATA_URL).text, encoding="utf-8")
    return input_path.read_text(encoding="utf-8")


class ShakespeareDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, max_len: int, split: str = "train"):
        text = _get_text()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab
        encoded = self.tokenizer.encode(text)
        n = int(len(encoded) * TRAIN_SPLIT)
        split_data = encoded[:n] if split == "train" else encoded[n:]
        self.examples: list[torch.Tensor] = []
        for i in range(0, len(split_data) - max_len, max_len):
            chunk = split_data[i : i + max_len + 1]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        return tokens[:-1], tokens[1:]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class ShakespeareCharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, max_len: int, split: str = "train"):
        text = _get_text()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ctoi = {ch: i for i, ch in enumerate(chars)}
        self.itoc = {i: ch for i, ch in enumerate(chars)}

        n = int(len(text) * TRAIN_SPLIT)
        split_text = text[:n] if split == "train" else text[n:]

        encoded = [self.ctoi[c] for c in split_text]
        self.examples: list[torch.Tensor] = []
        for i in range(0, len(encoded) - max_len, max_len):
            chunk = encoded[i : i + max_len + 1]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        return tokens[:-1], tokens[1:]

    def encode(self, text: str) -> list[int]:
        return [self.ctoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itoc[i] for i in tokens)
