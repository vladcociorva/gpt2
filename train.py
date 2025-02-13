import torch
from model import GPT, GPTConfig

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

config = GPTConfig()
model = GPT(config)
model.to(device)

print(model)

idx = torch.randint(0, config.vocab_size, (4, 8), device=device)
print(model(idx).shape)
