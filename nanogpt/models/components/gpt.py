import torch
from torch import nn

from .nn import Block, FeadForward


class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, ids, targets=None):
        logits = self.embedding(ids)
        B, T, C = logits.shape
        loss = None
        if targets is not None:
            # print(targets)
            loss = nn.functional.cross_entropy(logits.view(B * T, C), targets.reshape(B * T))
        return logits, loss


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, n_heads, n_blocks):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # C, E
        self.position_embedding = nn.Embedding(block_size, embedding_dim)  # T, E
        self.attention_blocks = nn.Sequential(
            *[Block(embedding_dim, n_heads, block_size=block_size) for _ in range(n_blocks)]
        )
        self.ff = FeadForward(embedding_dim)
        self.proj = nn.Linear(embedding_dim, vocab_size)

        self.register_buffer("pos", torch.arange(block_size))

    def forward(self, ids):
        x = self.embedding(ids)
        B, T, E = x.shape
        x = x + self.position_embedding(self.pos)

        x = self.attention_blocks(x)
        x = self.ff(x)
        out = self.proj(x)
        return out
