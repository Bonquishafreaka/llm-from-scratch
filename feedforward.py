import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """A simple per-token processing layer: expand, nonlinearity, shrink."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand to 4x width
            nn.ReLU(),                        # nonlinearity
            nn.Linear(4 * n_embd, n_embd),   # shrink back to original size
        )

    def forward(self, x):
        return self.net(x)


# --- quick test ---
if __name__ == "__main__":
    torch.manual_seed(1337)
    x = torch.randn(1, 8, 32)     # [batch, time, channels]
    ff = FeedForward(n_embd=32)
    out = ff(x)
    print("input shape: ", x.shape)
    print("output shape:", out.shape)