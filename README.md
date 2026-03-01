# Tiny LLM

Features:

- Pre-layer normalization
- Nucleus sampling
- Embeddings and final projection tied together

TODOs:

- SwiGLU activation
- Rotary position embeddings
- Flash attention

## Commands

Train a small character LLM on Shakespeare data:

```
uv run -m train --output_dir out/shakespeare --save_steps 100 --dataset shakespeare --steps=500
```

Run the KV-cache experiment:

```
uv run -m experiments.kv_cache --checkpoint out/shakespeare/step_500.pt --dataset shakespeare
```
