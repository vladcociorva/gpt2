import time
import math
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

data_loader = TokenShardDataloader(
    B=8,  # max "nice" number that fits on a single 3090 (i.e. 24GB vram)
    T=1024,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
    shards_dir="./fineweb_10b",
)

# GPT3 paper mentions warm up over the first 375m tokens. And doing decay over 260b tokens then training with min lr.
# All GPT3 variants were trained on 300b tokens.
# A rough heuristic to translate it on a per step level: 
# - linear warm-up over 0.12% tokens (375m/300b)  
# - cosine decay until 86% tokens (260b/300b) 
# - train with min lr for the remaining 14% of tokens
initial_lr = 6e-4
steps_per_epoch = data_loader.total_tokens // (data_loader.B * data_loader.T)
warmup_steps = round(0.0012 * steps_per_epoch)
decay_steps = round(0.86 * steps_per_epoch)
def cosine_decay_w_linear_warmup(step):
    # Should return a coefficient that the current lr will be multiplied with
    if step < warmup_steps:
        return (step+1) / max(1, warmup_steps)  # linear increase
    if step < warmup_steps + decay_steps:
        progress = (step - warmup_steps) / max(1, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * 0.9 + 0.1
    return 0.1

validation_cadence = 100

optim = torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95), eps=1e-8)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cosine_decay_w_linear_warmup)
for step in range(steps_per_epoch):
    start = time.perf_counter()

    optim.zero_grad()

    x, y = data_loader.get_batch()
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # this returns the initial norm (before clipping)
    optim.step()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_elapsed = end - start  # seconds
    tokens_processed = data_loader.B * data_loader.T

    print(
        f"step {step:>5} | lr {lr:.6f} | loss {loss.item():.6f} | grad norm {norm:.3f} | time {time_elapsed*1e3:.2f}ms | tokens/s {tokens_processed / time_elapsed:.2f} "
    )
    if step % validation_cadence == 0:
        xg = torch.tensor([GPT2_EOT], dtype=torch.long, device=device).view(1, -1)
        yg = model.generate(xg, max_new_tokens=32)
        print(GPT2_TOKENIZER.decode(yg[0].tolist()))
