import torch
from model import GPT, GPTConfig

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

config = GPTConfig()
model = GPT(config)
model.to(device)

B = 4
T = 32

idx = torch.randint(0, config.vocab_size, (B, T), device=device)
targets = torch.randint(0, config.vocab_size, (B, T), device=device)

logits, loss = model(idx, targets)
print(loss.item())
