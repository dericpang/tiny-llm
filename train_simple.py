import torch
import torch.optim as optim

from model import LLM


MAX_LEN = 64
LEARNING_RATE = 1e-4
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


def main():
    # character to index
    text = "".join(DATA)
    chars: list[str] = sorted(list(set(text)))
    vocab_size: int = len(chars)
    ctoi = {ch: i for i, ch in enumerate(chars)}
    itoc = {i: ch for i, ch in enumerate(chars)}

    batch = []
    for example in DATA:
        indexes = [ctoi[c] for c in example] + [0] * (MAX_LEN - len(example))
        batch.append(indexes)

    data = torch.tensor(batch, dtype=torch.long)
    x = data[:, :-1].contiguous()
    y = data[:, 1:].contiguous()

    model = LLM(vocab_size, 512, 8, 6, MAX_LEN)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for i in range(200):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        print(f"Iteration {i:3d} | Loss: {loss.item():.4f}")

    print("Generating...")

    model.eval()
    generated = model.generate(x[:, :5], max_new_tokens=64, top_p=0.9)
    for example in generated:
        print("".join([itoc[int(idx)] for idx in example]))


if __name__ == "__main__":
    main()
