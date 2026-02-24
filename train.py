import argparse
import os
import time
from itertools import cycle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.simple import SimpleCharDataset
from model import LLM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--iterations", type=int, default=200)
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

    dataset = SimpleCharDataset(args.max_len)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LLM(dataset.vocab_size, args.d_model, args.num_heads, args.num_layers, args.max_len).to(device)
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start = time.time()
    model.train()
    for i, (x, y) in enumerate(cycle(loader)):
        if i >= args.iterations:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()  # type: ignore[no-untyped-call]

        if i % 10 == 0:
            print(f"Iteration {i} | Loss: {loss.item():.4f}")

        is_last = (i + 1) == args.iterations
        is_save_step = args.save_steps is not None and (i + 1) % args.save_steps == 0
        if args.output_dir is not None and (is_last or is_save_step):
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"model_step_{i + 1}.pt")
            torch.save({"config": {"vocab_size": dataset.vocab_size, "d_model": args.d_model, "num_heads": args.num_heads, "num_layers": args.num_layers, "max_len": args.max_len}, "state_dict": model.state_dict()}, out_path)
            print(f"Saved checkpoint to {out_path}")
    end = time.time()
    print(f"Training took {end - start:.2f}s ({args.iterations / (end - start):.2f} iterations/second)")


if __name__ == "__main__":
    main()
