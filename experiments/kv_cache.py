import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from data.simple import SimpleCharDataset
from model import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
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

    dataset = SimpleCharDataset(model.max_len)
    x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)

    print("Generating without KV caching...")
    start = time.time()
    generated = model.generate(x[:, :5], use_kv_cache=False)
    for example in generated:
        print(dataset.decode([int(idx) for idx in example]))
    end = time.time()
    print(f"Generating took {end - start:.2f}s ({(end - start) / len(generated):.2f}s per example)")

    print("Generating with KV caching...")
    start = time.time()
    generated = model.generate(x[:, :5])
    for example in generated:
        print(dataset.decode([int(idx) for idx in example]))
    end = time.time()
    print(f"Generating took {end - start:.2f}s ({(end - start) / len(generated):.2f}s per example)")


if __name__ == "__main__":
    main()
