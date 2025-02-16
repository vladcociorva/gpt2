import time
import torch
import tiktoken
import numpy as np
import random
from model import GPT, GPTConfig
from data import TokenShardDataloader

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")
GPT2_EOT = GPT2_TOKENIZER.eot_token

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

config = GPTConfig()
model = GPT(config)
model.to(device)

steps = 1000
gen_interval = 10

data_loader = TokenShardDataloader(
    B=8,  # max "nice" number that fits on a single 3090 (i.e. 24GB vram)
    T=1024,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
    shards_dir="./tiny_shakespeare",
)

x, y = data_loader.get_batch()
x, y = x.to(device), y.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(steps):
    start = time.perf_counter()

    optim.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # this returns the initial norm (before clipping)
    optim.step()

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_elapsed = end - start  # seconds
    tokens_processed = data_loader.B * data_loader.T

    print(
        f"step {step:<5} | loss {loss.item():.6f} | time {time_elapsed*1e3:.2f}ms | tokens/s {tokens_processed / time_elapsed:.2f} | grad norm {norm:.4f}"
    )
    if step % gen_interval == 0:
        xg = torch.tensor([GPT2_EOT], dtype=torch.long, device=device).view(1, -1)
        yg = model.generate(xg, max_new_tokens=32)
        print(GPT2_TOKENIZER.decode(yg[0].tolist()))
