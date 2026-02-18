import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self._num_heads = num_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q(x)  # (B, d_model)
        k = self.k(x)  # (B, d_model)
        v = self.v(x)  # (B, d_model)

        q = q.view(B, T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, T, d_head)
        k = k.view(B, T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, T, d_head)
        v = v.view(B, T, self._num_heads, C // self._num_heads).transpose(1, 2)  # (B, num_heads, T, d_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))  # (B, num_heads, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, num_heads, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o(y)
        return y


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
    def __init__(self, d_model:int, num_heads:int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadMaskedAttention(d_model, num_heads)
        self.ff = FeedForwardNetwork(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        attention_out = self.attention(x)
        x = x + attention_out
        x = self.ln2(x)
        x = x + self.ff(x)
        return x


class LLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_len: int) -> None:
        super().__init__()
        self._max_len = max_len

        self.tok_embed: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed: nn.Embedding = nn.Embedding(max_len, d_model)
        self.blocks: nn.ModuleList[DecoderBlock] = nn.ModuleList([DecoderBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        print(vocab_size)
        self.lm_head: nn.Linear = nn.Linear(d_model, vocab_size)


    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B, T = x.shape  # x has shape (batch, seq_len)
        assert T <= self._max_len

        token_embeddings = self.tok_embed(x)  # (B, seq_len, d_model)
        positions = torch.arange(0, T, device=x.device, dtype=torch.long)
        positional_encodings = self.pos_embed(positions)
        x = token_embeddings + positional_encodings

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.lm_head(x)  # B, seq_len, vocab_size

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

        return logits, loss

    def generate(self, x: torch.Tensor, max_new_tokens: int, top_p: float = 0.9) -> torch.Tensor:
        for _ in range(max_new_tokens):
            x_cond = x[:, -self._max_len:]
            logits, _ = self(x_cond)
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
