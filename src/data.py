import os
import glob
import torch
import numpy as np


class TokenShardDataloader:
    def __init__(
        self,
        B: int,
        T: int,
        shards_dir: str,
        token_dtype: np.dtype,
        pad_value: int,
        world_size: int,
        local_rank: int
    ):
        self.B, self.T = B, T
        self.world_size = world_size
        self.local_rank = local_rank
        self.pad_value = pad_value
        self.token_dtype = token_dtype

        self._shard_paths = sorted(glob.glob(f"{shards_dir}/*.bin"))

        self.total_tokens = 0
        for shard_path in self._shard_paths:
            self.total_tokens += os.path.getsize(shard_path) // token_dtype.itemsize
        if local_rank == 0:
            print(f"Total tokens in {shards_dir}: {self.total_tokens}")

        self._token_slice_size = B * T + 1 # +1 for target of the last token in the batch
        self.reset()

    def reset(self):
        self._current_shard = -1
        self._load_next_shard()

    def _load_next_shard(self):
        # TODO: Shuffling the samples around on an epoch end
        self._current_shard = (self._current_shard + 1) % len(self._shard_paths)
        self._local_data_ptr = self.local_rank * self._token_slice_size
        self._shard_mmap = np.memmap(
            self._shard_paths[self._current_shard], dtype=self.token_dtype, mode="r"
        )

    # Always return B sequences of T tokens, meaning that multiple distinct documents from the
    # initial dataset might be packed in a single sequence (if they are shorter than T).
    # This conforms with the GPT3 paper.
    def get_batch(self):
        if self._local_data_ptr + self._token_slice_size <= len(self._shard_mmap):
            tokens = self._shard_mmap[self._local_data_ptr : self._local_data_ptr + self._token_slice_size]
            self._local_data_ptr += self.world_size * self._token_slice_size
        else:
            # pad the rest of this last sample
            tokens = self._shard_mmap[self._local_data_ptr:]
            pad_tokens = np.full(
                self._token_slice_size - len(tokens),
                self.pad_value,
                dtype=self.token_dtype,
            )
            tokens = np.concatenate([tokens, pad_tokens])
            self._local_data_ptr = len(self._shard_mmap)

        if self._local_data_ptr >= len(self._shard_mmap):
            self._load_next_shard()

        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1].view(self.B, self.T)
        y = tokens[1:].view(self.B, self.T)
        return x, y
