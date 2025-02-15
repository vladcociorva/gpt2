import torch
import numpy as np
import random
from model import GPT, GPTConfig
from data import TokenShardDataloader

GPT2_EOT = 50256

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cpu"
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda"

config = GPTConfig()
model = GPT(config)
model.to(device)

steps = 5

data_loader = TokenShardDataloader(
    B=8,
    T=1024,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
    shards_dir="./tiny_shakespeare",
)

x, y = data_loader.get_batch()
x, y = x.to(device), y.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(steps):
    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optim.step()
    print(f'{loss.item():.6f}')
