import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self._d_model = d_model
        self._num_heads = num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)

        self.mask: torch.Tensor
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))
    
    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        past_T = kv_cache[0].shape[1] if kv_cache else 0

        new_kv_cache = None
        q, k, v = self.qkv(x).split(self._d_model, dim=-1)  # (B, T, d_model)
        if kv_cache:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)  # (B, T + T_{i-1}, d_model)
            v = torch.cat([past_v, v], dim=1)  # (B, T + T_{i-1}, d_model)
        new_kv_cache = (k, v) 

        q = q.view(B, T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, T, d_head)
        k = k.view(B, past_T + T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, past_T + T, d_head)
        v = v.view(B, past_T + T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, past_T + T, d_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))  # (B, num_heads, T, past_T + T)
        mask = self.mask[past_T:past_T + T, :past_T + T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, num_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o(y)
        return y, new_kv_cache


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model*4)
        self.l2 = nn.Linear(d_model*4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.gelu(x)
        x = self.l2(x) 
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_len:int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadMaskedAttention(d_model, num_heads, max_len)
        self.ff = FeedForwardNetwork(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_out, kv_cache = self.attention(self.ln1(x), kv_cache)
        x = x + attention_out
        x = x + self.ff(self.ln2(x))
        return x, kv_cache


class LLM(nn.Module):
    @classmethod
    def from_pretrained(cls, path: str, map_location: str | torch.device | None = None) -> "LLM":
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_len: int) -> None:
        super().__init__()
        self.max_len: int = max_len

        self.tok_embed: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed: nn.Embedding = nn.Embedding(max_len, d_model)
        self.blocks: nn.ModuleList = nn.ModuleList([DecoderBlock(d_model, num_heads, max_len) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head: nn.Linear = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.tok_embed.weight


    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        _, T = x.shape
        assert T <= self.max_len
        past_T = kv_cache[0][0].shape[1] if kv_cache else 0
        if kv_cache:
            assert len(kv_cache) == len(self.blocks)
            assert past_T + T <= self.max_len

        token_embeddings = self.tok_embed(x)  # (B, seq_len, d_model)
        positions = torch.arange(past_T, past_T + T, device=x.device, dtype=torch.long)
        positional_encodings = self.pos_embed(positions)  # (seq_len, d_model)
        x = token_embeddings + positional_encodings

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, block in enumerate(self.blocks):
            x, new_block_kv_cache = block(x, kv_cache[i] if kv_cache else None)
            new_kv_cache.append(new_block_kv_cache)

        x = self.ln(x)
        logits = self.lm_head(x)  # B, seq_len, vocab_size

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

        return logits, loss, new_kv_cache

    def generate(self, x: torch.Tensor, top_p: float = 0.9, use_kv_cache: bool = True) -> torch.Tensor:
        _, T = x.shape
        kv_cache = None
        for _ in range(self.max_len - T):
            logits = None
            if use_kv_cache and kv_cache:
                logits, _, kv_cache = self(x[:, -1:], None, kv_cache)
            else:
                logits, _, kv_cache = self(x)
            logits = logits[:, -1, :]

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, x_next], dim=1)

        return x
