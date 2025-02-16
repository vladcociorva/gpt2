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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig()
model = GPT(config)
model.to(device)
model = torch.compile(model)

B = 16   # micro-batch size
T = 1024 # sequence length
global_batch_size = 524_288 # 2**19 tokens; gpt3 paper mentions "0.5M" for the 125M params model
grad_accum_steps = global_batch_size // (B * T) 

data_loader = TokenShardDataloader(
    B=B,
    T=T,
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
steps_per_epoch = data_loader.total_tokens // global_batch_size
print(f'steps/epoch: {steps_per_epoch}')
warmup_steps = round(0.001 * steps_per_epoch)
print(f'lr warmup steps: {warmup_steps}')
decay_steps = round(0.866 * steps_per_epoch)
print(f'lr decay steps: {decay_steps}')
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

optim = torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cosine_decay_w_linear_warmup)
for step in range(steps_per_epoch):
    start = time.perf_counter()

    optim.zero_grad()

    batch_loss = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = data_loader.get_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(x.device.type, torch.bfloat16):
            logits, loss = model(x, y)
            loss /= grad_accum_steps
            batch_loss += loss.detach()
            loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # this returns the initial norm (before clipping)
    optim.step()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    torch.cuda.synchronize()
    end = time.perf_counter()

    time_elapsed = end - start  # seconds

    print(
        f"step {step:>5} | lr {lr:.6f} | loss {batch_loss.item():.6f} | grad norm {norm:.3f} | time {time_elapsed*1e3:.2f}ms | tokens/s {B * T * grad_accum_steps / time_elapsed:.2f} "
    )
    if step % validation_cadence == 0:
        xg = torch.tensor([GPT2_EOT], dtype=torch.long, device=device).view(1, -1)
        yg = model.generate(xg, max_new_tokens=32)
        print(GPT2_TOKENIZER.decode(yg[0].tolist()))
