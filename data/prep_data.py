import os
import argparse
import tiktoken
import multiprocessing
import numpy as np
from tqdm import tqdm


SHARDS_MAX_SIZE = int(1e8)
BATCH_SIZE = 50
NUM_PROC = multiprocessing.cpu_count() - 1

def _load_fineweb_10b():
    from datasets import load_dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )
    return (sample["text"] for sample in dataset)

def _load_tiny_shakespeare():
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')

    # treating every section as a separate document
    import re
    sections = re.split(r"(\n\n)", content)
    # add back the separators to each section
    return [sections[i] + sections[i+1] for i in range(0, len(sections)-1, 2)]

_CONFIGS = {
    "fineweb-10B": {
        "shards_dir": "./fineweb_10b",
        "shard_file_name": "fineweb_10b_gpt2_toks_shard_{idx}.bin",
        "load_fn": _load_fineweb_10b
    }, 
    "tiny_shakespeare": {
        "shards_dir": "./tiny_shakespeare",
        "shard_file_name": "tiny_shakespeare_gpt2_toks_shard_{idx}.bin",
        "load_fn": _load_tiny_shakespeare
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["fineweb-10B", "tiny_shakespeare"], default="tiny_shakespeare")
args = parser.parse_args()

config = _CONFIGS[args.dataset]

encoding = tiktoken.get_encoding("gpt2")
def tokenize_fn(text):
    # Delimit each doc using the special EOT token, then encode
    tokens = [encoding._special_tokens["<|endoftext|>"]]
    tokens.extend(encoding.encode_ordinary(text))
    return tokens

def flush_shard(idx, buffer, buffer_fill_size):
    shard_path = os.path.join(config["shards_dir"], config["shard_file_name"].format(idx=idx))
    with open(shard_path, "wb") as f:
        # Only write out the actually filled portion
        buffer[:buffer_fill_size].tofile(f)

os.makedirs(config["shards_dir"], exist_ok=True)

with multiprocessing.Pool(NUM_PROC) as pool:
    print(f"Parallel tokenization using {NUM_PROC} processes.")

    samples = config["load_fn"]()

    shard_buffer, shard_fill_size, shard_idx = np.zeros(SHARDS_MAX_SIZE, dtype=np.uint16), 0, 0
    shard_pbar = tqdm(total=SHARDS_MAX_SIZE, desc=f"Shard {shard_idx}", unit='tokens')

    for token_batch in pool.imap(tokenize_fn, samples, chunksize=BATCH_SIZE):
        batch_i = 0
        while batch_i < len(token_batch):
            # How many tokens can we still fit in the current shard?
            needed = SHARDS_MAX_SIZE - shard_fill_size
            take = min(needed, len(token_batch) - batch_i)

            shard_buffer[shard_fill_size : shard_fill_size + take] = token_batch[batch_i : batch_i + take]
            shard_fill_size += take
            batch_i += take
            shard_pbar.update(take)

            if shard_fill_size == SHARDS_MAX_SIZE:
                flush_shard(shard_idx, shard_buffer, shard_fill_size)
                
                shard_pbar.close()

                shard_idx += 1
                shard_fill_size = 0
                
                shard_pbar = tqdm(total=SHARDS_MAX_SIZE, desc=f"Shard {shard_idx}", unit='tokens')

    # If leftover tokens in the final shard, write them out if not empty
    if shard_fill_size > 0:
        shard_pbar.close()
        flush_shard(shard_idx, shard_buffer, shard_fill_size)

print("Done")