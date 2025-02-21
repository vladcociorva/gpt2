import os
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken
import numpy as np
import random
import wandb
import argparse
from model import GPT, GPTConfig
from data import TokenShardDataloader
from utils import load_checkpoint_dir, save_checkpoint_dir
from evals import HellaSwag


# --------------------- Argument Parsing ---------------------
parser = argparse.ArgumentParser(description="GPT training script with DDP")
# micro-batch: 16 for local runs on 3090 (24GB); 64 for cloud runs on A100s/H100s (80GB)
# for 0.5M batch size: micro batch 64, seq length 1024, world size 8 -> no grad accumulation (i.e., on 8xH100)
parser.add_argument("--micro_batch_size", type=int, default=16,
                    help="Micro batch size (default: 16)")
parser.add_argument("--dataset_dir", type=str, default="./fineweb_10b",
                    help="Dataset directory (default: ./fineweb_10b)")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                    help="Directory for saving checkpoints (default: ./checkpoints)")
parser.add_argument("--from_checkpoint", type=str, default=None,
                    help="Path of a checkpoint directory to resume training from (default: None)")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of epochs to train for (default: 1)")
parser.add_argument("--wandb_project", type=str, default=None,
                    help="Name of the wandb project; (default: None; i.e. no wandb logging)")
args = parser.parse_args()


# --------------------- Global Configurations ---------------------
GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")
GPT2_EOT = GPT2_TOKENIZER.eot_token
EVAL_SAMPLE_PREFIX = [GPT2_EOT] + GPT2_TOKENIZER.encode_ordinary("The meaning of life is")


# --------------------- Distributed Setup ---------------------
ddp = "WORLD_SIZE" in os.environ
if ddp: 
    dist.init_process_group(backend='nccl')

world_size = int(os.getenv('WORLD_SIZE', 1))
local_rank = int(os.getenv('LOCAL_RANK', 0))
master_process = (local_rank == 0) # only one process should do logging, checkpointing, etc.
print_master = lambda s: print(s) if master_process else ...


# --------------------- Torch config ---------------------
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------- Reproducibility ---------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


# --------------------- Model init ---------------------
config = GPTConfig()
model = GPT(config)
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if ddp else model

# --------------------- Training setup ---------------------
GLOBAL_BATCH_SIZE = 524_288 # 2**19 tokens; gpt3 paper mentions "0.5M" for the 125M params model
SEQ_LENGTH = 1024 # GPT2 paper

B = args.micro_batch_size
T = SEQ_LENGTH
assert GLOBAL_BATCH_SIZE % (B * T * world_size) == 0
grad_accum_steps = GLOBAL_BATCH_SIZE // (B * T * world_size)

train_data_loader = TokenShardDataloader(
    B=B,
    T=T,
    token_dtype=np.dtype(np.uint16),
    pad_value=GPT2_EOT,
    shards_dir=f"{args.dataset_dir}/train",
    local_rank=local_rank,
    world_size=world_size
)
steps_per_epoch = train_data_loader.total_tokens // GLOBAL_BATCH_SIZE
print_master(f'epochs: {args.epochs}')
print_master(f'steps/epoch: {steps_per_epoch}')
print_master(f"global grad accumulation steps: {grad_accum_steps}")

val_data_loader = TokenShardDataloader(
    B=B,
    T=T,
    token_dtype=np.dtype(np.uint16),
    pad_value=GPT2_EOT,
    shards_dir=f"{args.dataset_dir}/val",
    local_rank=local_rank,
    world_size=world_size
)
hellaswag_eval = HellaSwag()

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

optim = raw_model.config_optimizer(weight_decay=0.1, lr=6e-4)  # matching gpt3-125M setup 
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cosine_decay_w_linear_warmup)

start_step = 0
val_cadence = 100
val_loss_eval_steps = 20
checkpoint_cadence = 300

if args.from_checkpoint:
    print_master(f'resuming training from checkpoint: {args.from_checkpoint}')
    start_step = load_checkpoint_dir(args.from_checkpoint, raw_model, optim, scheduler, train_data_loader)


# --------------------- Weights and Biases setup ---------------------
log_to_wandb = args.wandb_project is not None
print_master(f'logging to wandb - {log_to_wandb}')
if log_to_wandb and master_process:
    wandb.init(
        project=args.wandb_project,
        config={
            'dataset': args.dataset_dir,
            'training_tokens': train_data_loader.total_tokens,
            'global_batch_size': GLOBAL_BATCH_SIZE,
            'micro_batch': B,
            'sequence_length': T,
            'grad_accum_steps': grad_accum_steps,
            'epochs': args.epochs,
            'steps_per_epoch': steps_per_epoch
        }
    )


# --------------------- Training loop ---------------------
for step in range(start_step, args.epochs * steps_per_epoch):
    if master_process and step % val_cadence == 0:
        print_master('--------------------- Validation ---------------------')
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
            print_master(f'val loss {val_loss.item():.6f}')

        # eval is under @torch.no_grad
        accuracy = hellaswag_eval.eval(raw_model, GPT2_TOKENIZER, device)
        print_master(f'hellaswag accuracy: {accuracy:.4f}')

        if log_to_wandb:
            wandb.log({
                'val_loss': val_loss.item(),
                'hellaswag_acc': accuracy
            })

        # sample some random tokens to see where the model's at
        print_master('\nmodel sampling:')
        xg = torch.tensor(EVAL_SAMPLE_PREFIX, dtype=torch.long, device=device).view(1, -1)
        yg = raw_model.generate(xg, max_new_tokens=256)
        print_master(GPT2_TOKENIZER.decode(yg[0].tolist()))
        model.train()
        print_master('------------------------------------------------------')

    if master_process and step % checkpoint_cadence == 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{step}/')
        print_master(f'saving training checkpoint at {checkpoint_path}')
        save_checkpoint_dir(checkpoint_path, step, raw_model, optim, scheduler, train_data_loader)

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

            # ddp syncs grads across each process on every backward call; 
            # for efficiency, accumulate grads locally and only sync on the final update
            if ddp and micro_step < grad_accum_steps - 1:
                with model.no_sync():
                    loss.backward()
            else: 
                loss.backward()
            batch_loss += loss.detach()

    # the master process only sees its own batch_loss; sync across all processes for logging
    if ddp: 
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # this returns the initial norm (before clipping)
    optim.step()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    torch.cuda.synchronize()
    end = time.perf_counter()

    loss = batch_loss.item()
    time_elapsed = end - start  # seconds
    tokens_procesed = B * T * grad_accum_steps * world_size / time_elapsed # tokens/s

    if log_to_wandb and master_process:
        wandb.log({
            'lr': lr,
            'loss': loss,
            'grad_norm': norm,
            'step_time': time_elapsed*1e3,
            'tokens_processed': tokens_procesed
        })

    print_master(
        f"step {step:>5} | lr {lr:.6f} | loss {loss:.6f} | grad norm {norm:.3f} | time {time_elapsed*1e3:.2f}ms | tokens/s {tokens_procesed}"
    )

if master_process: 
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{step}/')
    save_checkpoint_dir(checkpoint_path, step, raw_model, optim, scheduler, train_data_loader)

if ddp:
    dist.destroy_process_group()
