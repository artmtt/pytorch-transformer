import random
import torch
from torch import Tensor

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_auto_device(device: str = None) -> str:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def get_subsequent_mask(seq_size: int) -> Tensor:
    """
    Get triangular mask (for autoregresive tasks).
    """
    subsequent_mask = torch.triu(torch.ones((seq_size, seq_size)), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def greedy_decode(model, src: Tensor, src_mask: Tensor, max_seq_len: int, start_token_idx: int = None, end_token_idx: int = None, pad_token_idx: int = None, device = None) -> Tensor:
    """
    Greedy decoding for generating sequences during inference.
    - Sequences that reach `end_token_idx` (if provided) are padded with `pad_token_idx`.
    - If `start_token_idx` is None, `pad_token_idx` is used if provided, if not it will default to 0 for the initial token index to start generating.
    """
    if end_token_idx is not None and pad_token_idx is None:
        raise ValueError('If end_token_idx is provided, pad_token_idx must also be provided.')
    
    valid_end_token = end_token_idx is not None
    device = get_auto_device(device)
    src, src_mask = src.to(device), src_mask.to(device)
    batch_size = src.size(0)

    if start_token_idx is None:
        start_token_idx = 0 if pad_token_idx is None else pad_token_idx

    tgt = torch.full((batch_size, 1), start_token_idx, device=device, dtype=src.dtype)
    finished_gen = torch.zeros(batch_size, dtype=bool, device=device)

    with torch.no_grad():
        encoder_out = model.encode(src, src_mask)

        for _ in range(max_seq_len-1):
            tgt_mask = get_subsequent_mask(tgt.size(1)).type_as(src).unsqueeze(0).to(device)
            decoder_out = model.decode(encoder_out, tgt, src_mask, tgt_mask)
            
            # Pass the decoder logits corresponding to the last token in the sequences
            out_probs = model.compute_predictions(decoder_out[:, -1, :])
            
            _, next_tokens = torch.max(out_probs, dim=-1)

            if valid_end_token:
                next_tokens = torch.where(finished_gen, pad_token_idx, next_tokens)

            tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1).type_as(src)

            if valid_end_token:
                finished_gen |= next_tokens == end_token_idx
                if finished_gen.all():
                    break

    # Pad the tgt sequences after end_token_idx if needed
    tgt_seq_len = tgt.size(1)
    if pad_token_idx is not None and tgt_seq_len < max_seq_len:
        remaining_len = max_seq_len - tgt_seq_len
        padding_tensor = torch.empty((batch_size, remaining_len), dtype=tgt.dtype, device=device).fill_(pad_token_idx)
        tgt = torch.cat([tgt, padding_tensor], dim=-1)

    return tgt
