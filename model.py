import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN_ID = 0


@dataclass
class LLMConfig:
    vocab_size: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    max_len: int
    rope_theta: float
    flash_attention: bool


class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        head_size = config.hidden_size // config.num_heads

        self.config = config

        self.qkv = nn.Linear(config.hidden_size, (config.num_heads + 2 * config.num_kv_heads) * head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.mask: torch.Tensor  # (max_len, max_len)
        self.register_buffer("mask", torch.tril(torch.ones(config.max_len, config.max_len)) == 0)

        self.rope_cos: torch.Tensor  # (1, 1, max_len, d_head // 2)
        self.register_buffer("rope_cos", torch.cos(torch.arange(0, config.hidden_size // config.num_heads, 2) * (1.0 / config.rope_theta)).unsqueeze(0).repeat(config.max_len, 1)[None, None, :, :])
        self.rope_sin: torch.Tensor  # (1, 1, max_len, d_head // 2)
        self.register_buffer("rope_sin", torch.sin(torch.arange(0, config.hidden_size // config.num_heads, 2) * (1.0 / config.rope_theta)).unsqueeze(0).repeat(config.max_len, 1)[None, None, :, :])

    def _apply_rope(self, x: torch.Tensor, T: int, past_T: int) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)  # each (B, num_heads, T, d_head // 2)
        return torch.cat([x1 * self.rope_cos[:, :, past_T:past_T + T] - x2 * self.rope_sin[:, :, past_T:past_T + T], x2 * self.rope_cos[:, :, past_T:past_T + T] + x1 * self.rope_sin[:, :, past_T:past_T + T]], dim=-1)
    
    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        head_size = C // self.config.num_heads
        past_T = kv_cache[0].shape[2] if kv_cache else 0

        q, k, v = self.qkv(x).split([self.config.num_heads * head_size, self.config.num_kv_heads * head_size, self.config.num_kv_heads * head_size], dim=-1)  # (B, T, num_heads * head_size | num_kv_heads * head_size)

        q = q.view(B, T, self.config.num_heads, head_size).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = self._apply_rope(q, T, past_T)

        k = k.view(B, T, self.config.num_kv_heads, head_size).transpose(1, 2)  # (B, num_kv_heads, T, head_size)
        k = self._apply_rope(k, T, past_T)

        v = v.view(B, T, self.config.num_kv_heads, head_size).transpose(1, 2)  # (B, num_kv_heads, T, head_size)

        if kv_cache:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # (B, num_kv_heads, past_T + T, d_head)
            v = torch.cat([past_v, v], dim=2)  # (B, num_kv_heads, past_T + T, d_head)
        new_kv_cache = (k, v)

        k = torch.repeat_interleave(k, self.config.num_heads // self.config.num_kv_heads, dim=1)  # (B, num_heads, T, head_size)
        v = torch.repeat_interleave(v, self.config.num_heads // self.config.num_kv_heads, dim=1)  # (B, num_heads, T, head_size)

        mask = self.mask[past_T:past_T + T, :past_T + T]
        if self.config.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0).view(B, T, C)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))  # (B, num_heads, T, past_T + T)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, num_heads, T, d_head)
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.o(y)
        return y, new_kv_cache


class SwiGLU(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gate = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.down_proj = nn.Linear(4 * config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.rn1 = nn.RMSNorm(config.hidden_size)
        self.attention = MultiHeadMaskedAttention(config)
        self.swiglu = SwiGLU(config)
        self.rn2 = nn.RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_out, kv_cache = self.attention(self.rn1(x), kv_cache)
        x = x + attention_out
        x = x + self.swiglu(self.rn2(x))
        return x, kv_cache


class LLM(nn.Module):
    @classmethod
    def from_pretrained(cls, path: str, map_location: str | torch.device | None = None) -> "LLM":
        checkpoint = torch.load(path, map_location=map_location)
        config = LLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.max_len: int = config.max_len

        self.tok_embed: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks: nn.ModuleList = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.RMSNorm(config.hidden_size)
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weight = self.tok_embed.weight


    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        _, T = x.shape
        assert T <= self.max_len
        past_T = kv_cache[0][0].shape[1] if kv_cache else 0
        if kv_cache:
            assert len(kv_cache) == len(self.blocks)
            assert past_T + T <= self.max_len

        x = self.tok_embed(x)  # (B, seq_len, d_model)

        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for i, block in enumerate(self.blocks):
            x, new_block_kv_cache = block(x, kv_cache[i] if kv_cache else None)
            new_kv_cache.append(new_block_kv_cache)

        x = self.ln(x)
        logits = self.lm_head(x)  # B, seq_len, vocab_size

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=PAD_TOKEN_ID)

        return logits, loss, new_kv_cache

    def generate(self, x: torch.Tensor, top_p: float = 0.9, use_kv_cache: bool = True) -> torch.Tensor:
        _, T = x.shape
        kv_cache = None
        for _ in range(self.max_len - T):
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
