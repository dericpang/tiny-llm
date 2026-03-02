import argparse
import os
import time
from itertools import cycle
from typing import Callable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.shakespeare import ShakespeareCharDataset, ShakespeareDataset
from data.simple import SimpleCharDataset
from model import LLM, LLMConfig


TrainDataset = SimpleCharDataset | ShakespeareCharDataset | ShakespeareDataset

DATASETS: dict[str, Callable[[int, str], TrainDataset]] = {
    "simple_char": SimpleCharDataset,
    "shakespeare_char": ShakespeareCharDataset,
    "shakespeare": ShakespeareDataset,
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--flash_attention", type=bool, default=True)

    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), default="simple_char")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], default=None)

    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Optimized for Mac M1/M2/M3 chips
    else:
        device = torch.device("cpu")
    print("Using device", device)

    train_dataset = DATASETS[args.dataset](args.max_len, "train")
    assert hasattr(train_dataset, "vocab_size")
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = DATASETS[args.dataset](args.max_len, "dev")
    dev_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    config = LLMConfig(vocab_size=train_dataset.vocab_size, hidden_size=args.d_model, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, num_layers=args.num_layers, max_len=args.max_len, rope_theta=args.rope_theta, flash_attention=args.flash_attention)
    print("Model config:", config)
    model = LLM(config).to(device)
    print("Vocab size:", train_dataset.vocab_size)
    print("Model has", f"{sum(p.numel() for p in model.parameters()):,}", "parameters")
    if args.debug:
        for name, param in model.named_parameters():
            print(name, f"{param.numel():,}")
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.time()
    pbar = tqdm(enumerate(cycle(train_loader)), total=args.steps, desc="Training")
    for i, (x, y) in pbar:
        model.train()
        if i >= args.steps:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()  # type: ignore[no-untyped-call]

        pbar.set_postfix(loss=f"{loss.item():.4f}")  # type: ignore[no-untyped-call]

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
