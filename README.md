# Tiny LLM

Features:

- Pre-layer normalization
- Nucleus sampling
- Weight tying between embedding and output layers
- SwiGLU
- Rotary position embeddings
- Flash attention
- Grouped query attention

## Train

Train a small LLM on Shakespeare data:

```
uv run -m train --dataset=shakespeare --output_dir=out/shakespeare
```

## Experiments

These tiny experiments demonstrate known research results.

### Weight-tying parameter efficiency

```
uv run -m experiments.weight_tying
```

### KV-cache decoding experiment


```
uv run -m train --dataset=shakespeare --output_dir=out/shakespeare && \
uv run -m experiments.kv_cache --dataset=shakespeare --checkpoint=out/shakespeare/step_500.pt
```
