# GPT2

GPT2 124M pre-training for educational purposes.

Weights available at https://huggingface.co/skyehigh/gpt2-124M-fineweb.

## Model architecture

GPT2 decoder-only transformer design (some details from the GPT3 paper which has a lot more information):

1. Token and positional embeddings
2. Stack of decoder layers
3. Final layer norm
4. Final linear projection (language modeling head)

### Token and positional embeddings
Turns the input token indices into dense vector representations:

**Token Embedding**: Maps token indices to n-dimensional (`d_model`) dense vectors. 

**Positional Embedding**: Adds a learned position-dependent vector to each token's embedding to encode order information.

These embeddings are summed together and fed into the decoder layers.

### Stack of decoder layers
Each decoder layer has:
- **LayerNorm** before each sub-layer (deviation from original Transformer paper which had it after each sub-layer, but consistent with GPT2 paper).
- **Multi-head self-attention** with a causal mask to ensure autoregressive behaviour. Each head independently processes a subspace of the hidden representation (`d_model` / `n_heads`).
- **Feed-Forward layer (MLP)** that expands the hidden dimension to 4x (i.e., `4 * d_model`), uses the **GELU** non-linearity in between and projects back to `d_model`.
- **Residual connections** after both the attention and MLP blocks.

### Final layer norm
After passing through all decoder layers, a final **layer normalization** is applied. 

### Final linear projection
Language modeling head -- this layer projects the `d_model` dimensional hidden state to vocabulary space (i.e. `d_model` -> `vocab_size`).

### Weight tying
The **final linear layer's** weight matrix is shared with the **token embedding layer**.

This reduces the number of trainable parameters and helps improve generalization by enforcing consistency between encoding and decoding representations.

## Model params
**Vocab size**: Set to **50304** (instead of GPT2's **50257**) to better mach CUDA kernel block sizes, which are typically powers of 2 (e.g., 16, 32, 64). `50304` divides 16, 32, 64. This provides a decent speed-up each training step.

**Context window (n_ctx)**:
Maximum block size of **1024** tokens, matching the original GPT2. 

**Number of layers (n_layers)**:
**12** decoder layers, matching the original GPT2.

**Hidden dimension (d_model)**:
Each token is represented in **768**-dimensional space, consistent with the original GPT2-124M and GPT3-125M.

**Attention heads (n_heads)**: **12** attention heads, matching GPT2-124M and GPT3-125M.

## Data
Used the `sample-10B` (**~10.3b** tokens) variant of the [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb). 

The dataset was tokenized with the gpt2 tokenizer and split into 100M token binary shards.

A shard is a flat sequence of `uint16`s (gpt2 vocabulary has 50257 tokens), which are the samples from the dataset, stored contiguously. Each sample was prefixed with the `<|endoftext|>` token before adding it to a shard.

This can be replicated by running `python data/prep_data --dataset fineweb-10B`. 

The shards can also be downloaded directly from [hugging face](https://huggingface.co/datasets/skyehigh/fineweb-10b-gpt2). This is also split into 103 training shards (**~10.2B** tokens, under `train/`) and 1 shard for validation (**100M** tokens, under `val/`).

## Training
### Weight init
Following the GPT2 (and 3) paper and code.

**Linear layers and token embeddings**
Initialized from a **normal distribution** with a mean of **0** and standard deviation of **0.02**.

**Positional embeddings**
Initialized from a normal distribution with a mean of **0** and standard deviation of **0.01**.

**Biases**
Initialized to **0**.

**Residual layers**
Weights of the residual layer projections are futher scaled at initialization by a factor of **1/sqrt(N)**, where **N** is the number of residual layers (i.e., 2 per decoder block).


### Performance optimizations
#### Flash attention 2
Accelerate the self-attention computation with `Flash Attention 2`. 
This is a memory-efficient approach with tiling and kernel fusion that significantly speeds up the computation by reducing GPU memory I/O. 

**~20%** time improvement per step (on a 3090)

#### Tensor Float 32 (TF32)
Runs calculations at the speed of fp16 but maintains fp32-like dynamic range.
Also enables tensor cores to kick in for faster computations.

**~25%** time improvement per step (on a 3090)

#### Mixed precision (bfloat16)
The training loop uses `automatic mixed precision`, allowing some operations to be performed in `bfloat16` instead of full-precision (fp32).

This makes the computations much faster (i.e., tensor cores kick in for matmuls), memory usage is lower allowing for larger micro batch sizes and the precision loss is minimal as `bfloat` maintains a large enough dynamic range, due to having a 8-bit exponent (unlike fp16, which has 5).

**~55%** time improvement per step (on a 3090)

#### Model compile
Model is compiled with `torch.compile` which from what I understand **dynamically optimizes execution** by fusing operations to minimize kernel launch overhead, reorders operations for better memory locality etc.

### Distributed setup
I use torch's distributed data parallel (ddp) to scale training across multiple GPUs.

### Optimization and hyper-params
#### Loss function
Trained using **cross-entropy loss** -- computs the difference between the predicted token (log) probabilities and the actual tokens.

#### Optimizer
`AdamW` with: 
- max learning rate **6e-4** (gpt3 paper) (using scheduling)
- betas **(0.99, 0.95)** (gpt3 paper)
- eps **1e-8** (gpt3 paper)
- weight decay **0.1** (gpt3 paper)
> weighy decay is applied only to linear layers (biases, layer norms and embeddings are exempt)

#### Learning Rate Scheduling
The learning rate is scheduled using a **cosine decay** with warm-up. Adapted from GPT3 paper.
- **warms up linearly** over the first 0.1% steps
- **cosine decay** over the next 86.6% steps
- **min learing rate** (10% of the initial value, i.e. **6e-5**) over the remaining steps.

The curve of the learning rate across the full one epoch training run: 
![](assets/lr.svg)


#### Global batch size
Follows GPT3 paper, which mentions using "0.5M" batch size for the 125M model.
I use **524_288** (2**19).

#### Sequence length
Fixed to `1024` tokens during training. Always pack `1024` tokens in each sequence (even if from different samples).
This is consistent with GPT3 paper.

#### Gradient accumulation
Gradients are accumulated over multiple micro-batches to effectively achieve the large global batch size without exceeding memory constraints.

`grad_accumulation_steps = 524_288 // (micro_batch_size * 1024 * world_size)`

When using ddp, gradients are synced only during the final micro-step to optimize communication overhead.

When running locally on a 3090 (24gb), the max `micro batch size` that can fit is **16** resulting in **32** grad accumulation steps.

However, on the full training run on a 8xH100 node, a `micro batch size` of 64 can fit in a single H100 (80gb) making the `grad_accumulation_steps` equal to 1, 
meaning we don't actually do any grad accumulation, resulting in much faster stepping.

### Gradient norm clipping
I clip the global norm of the gradient at **1.0** -- GPT3 paper.

The gradient norms were pretty stable (~0.3 on avg) across the whole training run, but there have been some big spikes during the first ~2000 steps.
![](assets/grad_norm.svg)

### Validation
Validation is done periodically -- once every **500** training steps.

#### Val loss
Evaluated on a validation split, averaged across **50** steps.

#### HellaSwag eval
The model is also evaluated on the **HellaSwag** benchmark.
HellaSwag is evaluated completion-style by computing the log-probability of each candidate answer's tokens, given the context, 
marking the one with the highest as the most likely answer of the model.

I report the average accuracy across the whole dataset. I use the `validation` split of the HellaSwag dataset.

### Run
The full training was run on a `8xH100` node for about 1 hour and 30 minutes.
Trained on **10,251,184,259** tokens, for one full epoch-- i.e., **19500** steps considering the **524_288** batch size.

Each training step took **~150ms** on average. In other words, we processed **3.4M** tokens per second on average.

The validation runs took considerably longer.

## Results
> TODO: compare with actual gpt2-124m on these metrics 


### Loss / val loss
Both the training loss and the validation loss are pretty consistent throughout the run (probably because it's impossbile for a model of this size to actually overfit the training data in one epoch). 

After one epoch, they eventually reached about `~3.31`.
![](assets/losses.svg)

### Hellswag
The `HellaSwag` accuracy consistenly increased across the training from random (1/4 = **0.25**) to about **0.285**.
![](assets/hellaswag.svg)

### Sampling: Meaning of life
I also randomly sampled from the model across the training run starting from the sequence `The meaning of life is`.

#### Step 0 (random weights):
```
<|endoftext|>The meaning of life is pitching SophiaPros Mé protagonist receiveboarding Chief nostalgiate RidertymologykefRegarding Jenny Instructor561ischer AgreementCONonen
```

#### Step 10000 (a bit over halfway):
```
<|endoftext|>The meaning of life is not to abandon ourselves, but to foster inner peace and understanding.
Do we truly need this precious gift and love?
The Bible phrase
To open someone’s heart, for they love the Lord and so do another.
```

#### Step 19000 (close to the end of the run):
```
<|endoftext|>The meaning of life is eternal
It can never be given to you.
It can never be moved or revised
And you can have it.
I am born without happiness
In Jesus’ Name
I am His true faith
And God Who gives
```

## Replicate
1. Install dependencies
    - If using uv: `uv sync && source .venv/bin/activate`
    - If using pip: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
2. Get dataset
    - Tokenize locally: e.g., run the `data/prep_data` script
    - Fetch archive from Hugging Face: https://huggingface.co/datasets/skyehigh/fineweb-10b-gpt2 
3. Run
    - [Optional] Set `WANDB_API_KEY` env variable for Weights and Biases logging
    - `./train.sh [3090 | 8xH100]`
        > This runs with pre-created configs, which automatically set some hyper-params like the **micro batch size** to good values. Created configs for a single 3090 (used locally during development for experimentation) and a 8xH100 node (used for the actual full run).
    - Or just run `src/train.py` 

## References
- [GPT2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT2 code](https://github.com/openai/gpt-2)
- [GPT3 paper](https://arxiv.org/pdf/2005.14165)  
- Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT) 🐐