import torch
import numpy as np
from model import GPT, GPTConfig
from data import TokenShardDataloader

GPT2_EOT = 50256

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

config = GPTConfig()
model = GPT(config)
model.to(device)

steps = 1000

data_loader = TokenShardDataloader(
    shards_dir="./tiny_shakespeare",
    B=8,
    T=1024,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
)

x, y = data_loader.get_batch()
x, y = x.to(device), y.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(steps):
    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optim.step()
    print(loss.item())
