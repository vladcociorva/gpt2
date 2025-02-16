import os
import glob
import torch
import numpy as np


class TokenShardDataloader():
    def __init__(self, shards_dir, B, T, token_dtype, pad_value):
        self.B, self.T = B, T

        self.pad_value = pad_value
        self.token_dtype = token_dtype
        self.shard_paths = sorted(glob.glob(f'{shards_dir}/*.bin'))

        token_size = np.dtype(token_dtype).itemsize

        self.total_tokens = 0
        for shard_path in self.shard_paths:
            self.total_tokens += os.path.getsize(shard_path) // token_size
        
        print(f"Total tokens: {self.total_tokens}")

        self.current_shard = -1
        self._load_next_shard()

    def _load_next_shard(self):
        # TODO: Shuffling the samples around on an epoch end
        self.current_shard = (self.current_shard + 1) % len(self.shard_paths)
        self.data_ptr = 0
        self.shard_mmap = np.memmap(self.shard_paths[self.current_shard], dtype=self.token_dtype, mode='r')

    def get_batch(self):
        # Always return B sequences of T tokens, meaning that multiple distinct documents from the
        # initial dataset might be packed in a single sequence (if they are shorter than T).
        # This conforms with the GPT3 paper.
        B, T = self.B, self.T

        remaining_tokens = len(self.shard_mmap) - self.data_ptr
        required_tokens = B * T + 1

        if remaining_tokens >= required_tokens:
            tokens = self.shard_mmap[self.data_ptr : self.data_ptr + required_tokens]
            self.data_ptr += required_tokens
        else:
            # pad the rest of this last sample
            tokens = self.shard_mmap[self.data_ptr:]
            pad_tokens = np.full(required_tokens - remaining_tokens, self.pad_value, dtype=self.token_dtype)
            tokens = np.concatenate([tokens, pad_tokens])
            self.data_ptr = len(self.shard_mmap)
        
        if self.data_ptr >= len(self.shard_mmap):
            self._load_next_shard()

        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1].view(B, T)
        y = tokens[1:].view(B, T)
        return x, y
