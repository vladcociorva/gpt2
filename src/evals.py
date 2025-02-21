import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from tiktoken import Encoding

class HellaSwag():
    def __init__(self, split: str = 'validation'):
        self.dataset = load_dataset('Rowan/hellaswag', split=split, trust_remote_code=True)

    # Evaluates HellaSwag completion-style by computing the log-probability of each candidate answer's tokens, given the context, 
    # marking the one with the highest as the most likely answer of the model.
    # Returns the average accuracy across the whole dataset.
    @torch.no_grad
    def eval(self, model: torch.nn.Module, tokenizer: Encoding, device: torch.device) -> float:
        def _eval(sample):
            ctx = sample['ctx']
            candidates = sample['endings']
            label = int(sample['label'])

            ctx_ids = tokenizer.encode_ordinary(ctx)
            ctx_len = len(ctx_ids)
            candidates_ids = tokenizer.encode_ordinary_batch(candidates)
            candidates_len = [len(ids) for ids in candidates_ids]

            sequences = [ctx_ids + candidate_ids for candidate_ids in candidates_ids]
            max_len = max(len(seq) for seq in sequences)

            padded_sequences = []
            candidate_masks = []
            for seq, candidate_len in zip(sequences, candidates_len):
                pad_len = max_len - len(seq)
                padded_seq = seq + [tokenizer.eot_token] * pad_len
                padded_sequences.append(padded_seq)

                mask = [0] * ctx_len + [1] * candidate_len + [0] * pad_len
                candidate_masks.append(mask)

            x = torch.tensor(padded_sequences, device=device) # (4, max_len)
            mask = torch.tensor(candidate_masks, device=device) # (4, max_len)

            logits, _ = model(x) # (4, max_len, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1) # (4, max_len, vocab_size)
            token_log_probs = torch.gather(log_probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1) # (4, max_len)
            candidate_scores = (token_log_probs * mask).sum(dim=1) # (4, )
            
            most_likely = torch.argmax(candidate_scores).item()
            return most_likely == label
            

        acc = 0.0
        for sample in self.dataset:
            acc += _eval(sample)

        return acc / len(self.dataset)
