import time
import math
import torch
import tiktoken
import numpy as np
import random
from model import GPT, GPTConfig
from data import TokenShardDataloader

DATASET_DIR = "./fineweb_10b"
GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")
GPT2_EOT = GPT2_TOKENIZER.eot_token
EVAL_SAMPLE_PREFIX = [GPT2_EOT] + GPT2_TOKENIZER.encode_ordinary("The meaning of life is")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
train_data_loader = TokenShardDataloader(
    B=B,
    T=T,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
    shards_dir=f"{DATASET_DIR}/train",
)
steps_per_epoch = train_data_loader.total_tokens // global_batch_size
print(f'steps/epoch: {steps_per_epoch}')

val_data_loader = TokenShardDataloader(
    B=B,
    T=T,
    token_dtype=np.uint16,
    pad_value=GPT2_EOT,
    shards_dir=f"{DATASET_DIR}/val",
)

# GPT3 paper mentions warm up over the first 375m tokens. And doing decay over 260b tokens, then training with min lr (10% of initial).
# All GPT3 variants were trained on 300b tokens, so:
# - linear warm-up over 0.12% tokens (375m/300b)  
# - cosine decay over 86.6% tokens (260b/300b) 
# - train with min lr for the remaining ~13% of tokens
# Roughly translating the same proportions to a per step level
lr_warmup_steps = round(0.001 * steps_per_epoch)
lr_decay_steps = round(0.866 * steps_per_epoch)
def cosine_decay_w_linear_warmup(step):
    # Should return a coefficient that the current lr will be multiplied with
    if step < lr_warmup_steps:
        return (step+1) / max(1, lr_warmup_steps)  # linear increase
    if step < lr_warmup_steps + lr_decay_steps:
        progress = (step - lr_warmup_steps) / max(1, lr_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * 0.9 + 0.1
    return 0.1

val_cadence = 100
val_loss_eval_steps = 20

optim = model.config_optimizer(weight_decay=0.1, lr=6e-4)  # matching gpt3-125M setup 
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cosine_decay_w_linear_warmup)
for step in range(steps_per_epoch):
    start = time.perf_counter()

    optim.zero_grad()

    batch_loss = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_data_loader.get_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(x.device.type, torch.bfloat16):
            logits, loss = model(x, y)
            # averaging and backpropping every micro step (instead of accumulating and calling backpropping once on the average) to
            # be more efficient with torch's computation graph (i.e. to not retain the computation for all micro-steps)
            # also good for gradient stability
            loss /= grad_accum_steps 
            loss.backward()
            batch_loss += loss.detach()

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
    if step % val_cadence == 0:
        # calculate loss on the validation split
        model.eval()
        val_data_loader.reset() # to evaluate the same tokens every time
        with torch.no_grad():
            val_loss = 0.0
            for val_step in range(val_loss_eval_steps):
                x, y = val_data_loader.get_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(x.device.type, torch.bfloat16):
                    logits, loss = model(x, y)
                    val_loss += loss.detach()

            val_loss /= val_loss_eval_steps
            print(f'val loss {val_loss.item():.6f}')

        # sample some random tokens to see where the model's at
        xg = torch.tensor(EVAL_SAMPLE_PREFIX, dtype=torch.long, device=device).view(1, -1)
        yg = model.generate(xg, max_new_tokens=256)
        print(GPT2_TOKENIZER.decode(yg[0].tolist()))

        model.train()
