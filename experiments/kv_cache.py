import argparse
import time
from itertools import islice
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.shakespeare import ShakespeareDataset, ShakespeareCharDataset
from data.tiny_char import TinyCharDataset
from train import TrainDataset
from model import LLM

MAX_BATCHES = 50
DATASETS: dict[str, Callable[[int, str], TrainDataset]] = {
    "simple_char": TinyCharDataset,
    "shakespeare_char": ShakespeareCharDataset,
    "shakespeare": ShakespeareDataset,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset", type=str, choices=list(DATASETS.keys()), default="simple"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device", device)

    model = LLM.from_pretrained(args.checkpoint, map_location=device).to(device)
    model.eval()

    dataset = DATASETS[args.dataset](model.max_len, "dev")
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=args.batch_size
    )

    num_batches = min(MAX_BATCHES, len(loader))

    print("Generating without KV caching...")
    start = time.time()
    for x, _ in tqdm(
        islice(loader, num_batches), total=num_batches, desc="Without KV cache"
    ):
        x = x.to(device)
        model.generate(x[:, :5], use_kv_cache=False)
    end = time.time()

    print("Generating with KV caching...")
    kv_start = time.time()
    for x, _ in tqdm(
        islice(loader, num_batches), total=num_batches, desc="With KV cache"
    ):
        x = x.to(device)
        model.generate(x[:, :5])
    kv_end = time.time()

    print(
        f"Generating without KV caching took {end - start:.2f}s ({(end - start) / num_batches:.2f}s per batch)"
    )
    print(
        f"Generating with KV caching took {kv_end - kv_start:.2f}s ({(kv_end - kv_start) / num_batches:.2f}s per batch)"
    )


if __name__ == "__main__":
    main()
