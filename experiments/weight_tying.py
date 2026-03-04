import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.shakespeare import ShakespeareDataset
from model import LLM, LLMConfig
from train import train


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Optimized for Mac M1/M2/M3 chips
    else:
        device = torch.device("cpu")
    print("Using device", device)

    train_dataset = ShakespeareDataset(64, "train")
    assert hasattr(train_dataset, "vocab_size")
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=8, shuffle=True
    )
    dev_dataset = ShakespeareDataset(64, "dev")
    dev_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dev_dataset, batch_size=8, shuffle=True
    )

    config = LLMConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
        num_layers=6,
        max_len=64,
        tie_weights=False,
        rope_theta=10_000.0,
        flash_attention=True,
    )
    config_tie_weights = LLMConfig(
        vocab_size=train_dataset.vocab_size,
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
        num_layers=6,
        max_len=64,
        tie_weights=True,
        rope_theta=10_000.0,
        flash_attention=True,
    )

    model = LLM(config)
    model_tie_weights = LLM(config_tie_weights)

    print("Model WITHOUT weight tying has", f"{sum(p.numel() for p in model.parameters()):,}", "parameters")
    print("Model WITH weight tying has", f"{sum(p.numel() for p in model_tie_weights.parameters()):,}", "parameters")

    print("Training model WITHOUT weight tying...")
    train_out = train(
        model=model,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optim.Adam(model.parameters(), lr=1e-4),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        steps=500,
        save_steps=None,
        output_dir=None,
    )

    print("Training model WITH weight tying...")
    train_out_tie_weights = train(
        model=model_tie_weights,
        config=config_tie_weights,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optim.Adam(model_tie_weights.parameters(), lr=1e-4),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        steps=500,
        save_steps=None,
        output_dir=None,
    )
    print(
        f"Training without weight tying took {train_out['time']:.2f}s ({500 / train_out['time']:.2f} steps/second) and ended with loss {train_out['loss']:.4f}"
    )
    print(
        f"Training with weight tying took {train_out_tie_weights['time']:.2f}s ({500 / train_out_tie_weights['time']:.2f} steps/second) and ended with loss {train_out_tie_weights['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
