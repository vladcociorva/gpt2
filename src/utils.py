import os
import torch
import random
import numpy as np


def save_checkpoint_dir(dir, step, raw_model, optimizer, scheduler, dataloader):
    os.makedirs(dir, exist_ok=True)
    weights_path = os.path.join(dir, "weights.pt")
    torch.save(raw_model.state_dict(), weights_path)

    optim_path = os.path.join(dir, "optim.pt")
    optim_dict = {
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "dataloader_state": dataloader.state_dict()
    }
    torch.save(optim_dict, optim_path)

    rng_path = os.path.join(dir, "rng.pt")
    rng_dict = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "np_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate()
    }
    torch.save(rng_dict, rng_path)

def load_checkpoint_dir(dir, raw_model, optimizer, scheduler, dataloader):
    weights_path = os.path.join(dir, "weights.pt")
    optim_path = os.path.join(dir, "optim.pt")
    rng_path = os.path.join(dir, "rng.pt")

    weights = torch.load(weights_path, map_location='cpu')
    raw_model.load_state_dict(weights)

    optim_dict = torch.load(optim_path, map_location='cpu')
    step = optim_dict["step"]
    optimizer.load_state_dict(optim_dict["optimizer_state"])
    scheduler.load_state_dict(optim_dict["scheduler_state"])
    dataloader.load_state_dict(optim_dict["dataloader_state"])

    rng_dict = torch.load(rng_path, map_location='cpu', weights_only=False)
    torch.set_rng_state(rng_dict["torch_rng_state"])
    torch.cuda.set_rng_state_all(rng_dict["cuda_rng_state"])
    np.random.set_state(rng_dict["np_rng_state"])
    random.setstate(rng_dict["py_rng_state"])

    return step
