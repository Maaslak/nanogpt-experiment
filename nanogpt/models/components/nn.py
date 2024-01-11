import torch
from torch import nn


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.value_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.key_proj = nn.Linear(embedding_dim, head_size, bias=False)
        self.query_proj = nn.Linear(embedding_dim, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, E = x.shape

        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        weights = (query @ key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.embedding_dim))

        weights = weights.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
        weights = torch.nn.functional.softmax(weights, dim=-1)
        # print(weights)
        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, block_size):
        super().__init__()
        self.head_size = embedding_dim // n_heads
        if self.head_size * n_heads != embedding_dim:
            raise ValueError("n_heads has to be a factor of embedding_dim")
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embedding_dim, self.head_size, block_size) for _ in range(n_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], axis=-1)


class FeadForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.LayerNorm(2 * embedding_dim),
            nn.ReLU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, embedding_dim, n_heads, block_size):
        super().__init__()
        self.multiatt = MultiHeadAttention(embedding_dim, n_heads, block_size=block_size)
        self.ff = FeadForward(embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.multiatt(self.ln(x))
        x = x + self.ff(x)
        return x
