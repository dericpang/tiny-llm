import argparse
from typing import Callable
import os
import time
from itertools import cycle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.char_dataset import CharDataset
from data.shakespeare import ShakespeareCharDataset
from data.simple import SimpleCharDataset
from model import LLM, LLMConfig

DATASETS: dict[str, Callable[[int, str], CharDataset]] = {
    "simple": SimpleCharDataset,
    "shakespeare": ShakespeareCharDataset,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)

    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), default="simple")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Optimized for Mac M1/M2/M3 chips
    else:
        device = torch.device("cpu")
    print("Using device", device)

    train_dataset: CharDataset = DATASETS[args.dataset](args.max_len, "train")
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset: CharDataset = DATASETS[args.dataset](args.max_len, "dev")
    dev_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    config = LLMConfig(vocab_size=train_dataset.vocab_size, d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, max_len=args.max_len, rope_theta=args.rope_theta)
    model = LLM(config).to(device)
    print("Vocab size:", train_dataset.vocab_size)
    print("Model has", f"{sum(p.numel() for p in model.parameters()):,}", "parameters")
    for name, param in model.named_parameters():
        print(name, f"{param.numel():,}")
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.time()
    for i, (x, y) in enumerate(cycle(train_loader)):
        model.train()
        if i >= args.steps:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()  # type: ignore[no-untyped-call]

        if i % 10 == 0:
            print(f"Step {i} | Loss: {loss.item():.4f}")

        is_last = (i + 1) == args.steps
        is_save_step = args.save_steps is not None and (i + 1) % args.save_steps == 0
        if args.output_dir is not None and (is_last or is_save_step):
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"step_{i + 1}.pt")
            model.eval()
            total_dev_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for dev_x, dev_y in dev_loader:
                    dev_x, dev_y = dev_x.to(device), dev_y.to(device)
                    _, batch_loss, _ = model(dev_x, dev_y)
                    assert batch_loss is not None
                    total_dev_loss += batch_loss.item()
                    num_batches += 1
            avg_dev_loss = total_dev_loss / num_batches
            print(f"Step {i + 1} | Dev loss: {avg_dev_loss:.4f}")
            model.train()
            torch.save({"config": vars(config), "state_dict": model.state_dict()}, out_path)
            print(f"Saved checkpoint to {out_path}")
    end = time.time()
    print(f"Training took {end - start:.2f}s ({args.steps / (end - start):.2f} steps/second)")


if __name__ == "__main__":
    main()
