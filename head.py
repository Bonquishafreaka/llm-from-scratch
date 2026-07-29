import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        # Same three linear layers as before -- set up once
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # The mask, stored so we don't rebuild it every time.
        # register_buffer = "part of the module, but not a learnable weight"
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: [batch, time, n_embd]  (we'll add the batch dimension soon)
        B, T, C = x.shape

        k = self.key(x)      # what each token offers
        q = self.query(x)    # what each token seeks
        v = self.value(x)    # what each token hands over

        # relevance scores, scaled
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # no peeking: mask the future (only up to T, the current length)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        out = weights @ v    # blend the values
        return out

                # --- quick test ---
if __name__ == "__main__":
    torch.manual_seed(1337)
    B, T, C = 1, 8, 32        # batch=1, time=8 tokens, channels=32
    x = torch.randn(B, T, C)  # fake input in the right shape
    head = Head(n_embd=32, head_size=16, block_size=8)
    out = head(x)
    print("input shape: ", x.shape)
    print("output shape:", out.shape)