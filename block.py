import torch
import torch.nn as nn
from head import Head
from feedforward import FeedForward


class Block(nn.Module):
    """One transformer block: communication (attention) + computation (feed-forward)."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.attention = Head(n_embd, head_size, block_size)
        self.proj = nn.Linear(head_size, n_embd)   # match attention output back to n_embd
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)            # normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)            # normalize before feed-forward

    def forward(self, x):
        # residual: x + (what the sub-layer computes)
        x = x + self.proj(self.attention(self.ln1(x)))
        x = x + self.feed_forward(self.ln2(x))
        return x


# --- quick test ---
if __name__ == "__main__":
    torch.manual_seed(1337)
    x = torch.randn(1, 8, 32)
    block = Block(n_embd=32, head_size=32, block_size=8)
    out = block(x)
    print("input shape: ", x.shape)
    print("output shape:", out.shape)