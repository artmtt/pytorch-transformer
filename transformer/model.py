"""
PyTorch implementation of Attention Is All You Need transformer.
Paper: https://arxiv.org/abs/1706.03762

References:
- The Annotated Transformer:
    https://github.com/harvardnlp/annotated-transformer

- Tensor2Tensor:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
"""


import math
from copy import deepcopy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import log_softmax
from .utils import greedy_decode, get_auto_device
from typing import Callable

class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_len: int, pad_token_idx: int = None) -> None:
        """
        Creates an embedding layer.
        """
        super().__init__()
        self.d_model = d_model
        self.d_model_sqrt = math.sqrt(d_model)
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, d_model, pad_token_idx)

    def forward(self, input: Tensor) -> Tensor:
        # Multiply by sqrt to prevent gradient vanishing and gradient exploding
        out_embedding = self.embedding(input) * self.d_model_sqrt
        return out_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, p_dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        # Positional Encoding table
        pos_enc = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # According to the original paper's formula
        # divisor = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        # pos_scaled = position / divisor
        # Equivalent alternative way
        div_mult = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_scaled = position * div_mult

        pos_enc[:, 0::2] = torch.sin(pos_scaled)
        pos_enc[:, 1::2] = torch.cos(pos_scaled)
        # Add dimension for batch: (batch, max_seq_len, d_model)
        pos_enc.unsqueeze_(0)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, input: Tensor) -> Tensor:
        curr_seq_len = input.size(1)
        curr_pos_enc = self.pos_enc[:, :curr_seq_len, :].detach()
        input = input + curr_pos_enc
        return self.dropout(input)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, p_dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError('Embedding dim needs to be divisible by the quantity of heads')

        self.d_model = d_model
        self.nhead = nhead
        # Dimensionality of key vectors (dk)
        self.head_dim = d_model // nhead
        # Weight matrices
        self.w_queries = nn.Linear(d_model, d_model, bias=False)
        self.w_keys = nn.Linear(d_model, d_model, bias=False)
        self.w_values = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p_dropout)

    @staticmethod
    def scaled_dot_product_attention(query: Tensor, key: Tensor, value, mask: Tensor = None, dropout: nn.Dropout = None, mask_filler: float = -1e20):
        head_dim = query.size(-1) # d_k
        # Q * transpose(K): (batch_size, nhead, max_seq_len, max_seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, mask_filler)
        
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # softmax(Q * transpose(K)) * V: (batch_size, nhead, max_seq_len, head_dim)
        attention_output = attention_scores @ value
        return attention_output, attention_scores

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        curr_query = self.w_queries(query)
        curr_key = self.w_keys(key)
        curr_value = self.w_values(value)

        batch_size = curr_query.size(0)
        # (batch_size, {q, k, v}_max_seq_len, d_model) -> (batch_size, {q, k, v}_max_seq_len, nhead, head_dim) -> (batch_size, nhead, {q, k, v}_max_seq_len, head_dim)
        curr_query = curr_query.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        curr_key = curr_key.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        curr_value = curr_value.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attention_output, self.attention_scores = MultiHeadAttention.scaled_dot_product_attention(curr_query, curr_key, curr_value, mask, self.dropout)
        
        # Head concatenation
        # (batch_size, nhead, max_seq_len, head_dim) -> (batch_size, max_seq_len, nhead, head_dim) -> (batch_size, max_seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(attention_output.size(0), -1, self.d_model)
        mha_out = self.w_out(attention_output)

        return mha_out
    

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(dim = -1, keepdim = True)
        std = input.std(dim = -1, keepdim = True)
        norm_out = self.alpha * (input - mean) / (std + self.eps) + self.bias
        return norm_out


class ResidualAndNorm(nn.Module):
    """
    Residual connection and normalization layer.
    """
    def __init__(self, features: int, p_dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.norm = LayerNorm(features, layer_norm_eps)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, input: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        residual_conn_out = input + self.dropout(sublayer(self.norm(input)))
        return residual_conn_out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, p_dropout: float) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p_dropout)
        self.linear_out = nn.Linear(dim_feedforward, d_model)

    def forward(self, input: Tensor) -> Tensor:
        feed_forward_act = torch.relu(self.linear_in(input))
        feed_forward_out = self.linear_out(self.dropout(feed_forward_act))
        return feed_forward_out
    

class EncoderLayer(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttention, feed_forward: FeedForward, p_dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.features = features
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_and_norm = nn.ModuleList([ResidualAndNorm(features, p_dropout, layer_norm_eps) for _ in range(2)])
        self.layer_norm_eps = layer_norm_eps

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        input = self.residual_and_norm[0](input, lambda x: self.self_attention(x, x, x, mask))
        input = self.residual_and_norm[1](input, self.feed_forward)
        return input


class Encoder(nn.Module):
    def __init__(self, base_layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(base_layer) for _ in range(num_layers)])
        self.norm = LayerNorm(base_layer.features, base_layer.layer_norm_eps)

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input, mask)
        return self.norm(input)


class DecoderLayer(nn.Module):
    def __init__(self, features: int, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, p_dropout: float, layer_norm_eps: float) -> None:
        super().__init__()
        self.features = features
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_and_norm = nn.ModuleList([ResidualAndNorm(features, p_dropout, layer_norm_eps) for _ in range(3)])
        self.layer_norm_eps = layer_norm_eps

    def forward(self, input: Tensor, encoder_out: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        input = self.residual_and_norm[0](input, lambda x: self.self_attention(x, x, x, tgt_mask))
        input = self.residual_and_norm[1](input, lambda x: self.cross_attention(x, encoder_out, encoder_out, src_mask))
        input = self.residual_and_norm[2](input, self.feed_forward)
        return input


class Decoder(nn.Module):
    def __init__(self, base_layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(base_layer) for _ in range(num_layers)])
        self.norm = LayerNorm(base_layer.features, base_layer.layer_norm_eps)

    def forward(self, input: Tensor, encoder_out: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input, encoder_out, src_mask, tgt_mask)
        return self.norm(input)


class OutputProjection(nn.Module):
    """
    Linear + Softmax block.
    """
    def __init__(self, d_model: int, vocab_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_len)

    def forward(self, input: Tensor) -> Tensor:
        # (batch_size, tgt_max_seq_len, d_model) -> (batch_size, tgt_max_seq_len, vocab_len)
        return log_softmax(self.proj(input), dim=-1)


class Transformer(nn.Module):
    """Transformer Model."""
    def __init__(
            self,
            src_vocab_len: int,
            tgt_vocab_len: int,
            start_token_idx: int = None,
            end_token_idx: int = None,
            pad_token_idx: int = None,
            src_max_seq_len: int = 5000,
            tgt_max_seq_len: int = 5000,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            dim_feedforward: int = 2048,
            p_dropout: float = 0.1,
            layer_norm_eps: float = 1e-5,
            device = None
        ) -> None:
        """
        Creates a new transformer model.

        Args:
            - d_model: The size of the embedding vectors.
            - nhead: Quantity of attention heads.

            d_model dim needs to be divisible by nhead.
        """
        super().__init__()

        self.src_vocab_len = src_vocab_len
        self.tgt_vocab_len = tgt_vocab_len
        
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.pad_token_idx = pad_token_idx

        self.src_max_seq_len = src_max_seq_len
        self.tgt_max_seq_len = tgt_max_seq_len
        self.d_model = d_model
        self.nhead = nhead

        attention = MultiHeadAttention(d_model, nhead, p_dropout)
        feed_forward = FeedForward(d_model, dim_feedforward, p_dropout)
        
        self.src_embedding = Embedding(d_model, src_vocab_len, pad_token_idx)
        self.tgt_embedding = Embedding(d_model, tgt_vocab_len, pad_token_idx)
        self.src_pos = PositionalEncoding(d_model, src_max_seq_len, p_dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_max_seq_len, p_dropout)

        self.encoder = Encoder(EncoderLayer(d_model, deepcopy(attention), deepcopy(feed_forward), p_dropout, layer_norm_eps), num_encoder_layers)
        self.decoder = Decoder(DecoderLayer(d_model, deepcopy(attention), deepcopy(attention), deepcopy(feed_forward), p_dropout, layer_norm_eps), num_decoder_layers)
        self.prediction_head = OutputProjection(d_model, tgt_vocab_len)
        
        self.device = get_auto_device(device)
        self.to(self.device)

        self._reset_parameters()
    
    def encode(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Encodes the soruce sequences inside `src`, masking them with `src_mask` if it's provided.

        Returns:
            A Tensor with dimensions: `(batch_size, src_max_seq_len, d_model)`.
        """
        src, src_mask = self._to_device(src, src_mask)
        src = self.src_pos(self.src_embedding(src))
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_out: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        """
        Decodes using the embedding representations inside `encoder_out` and providing the target sequences `tgt` to the decoder while masking with `src_mask` and `tgt_mask` respectively.

        Returns:
            A Tensor with dimensions: `(batch_size, tgt_max_seq_len, d_model)`.
        """
        encoder_out, tgt, src_mask, tgt_mask = self._to_device(encoder_out, tgt, src_mask, tgt_mask)
        tgt = self.tgt_pos(self.tgt_embedding(tgt))
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)

    def compute_predictions(self, decoder_out: Tensor, apply_softmax: bool = True) -> Tensor:
        """
        Computes the output of the model's prediction head (probabilities if `apply_softmax` is True, otherwise logits).
        
        Returns:
            A Tensor with dimensions: `(batch_size, tgt_max_seq_len, vocab_len)`.
        """
        decoder_out = decoder_out.to(self.device)
        if apply_softmax:
            return self.prediction_head(decoder_out)
        return self.prediction_head.proj(decoder_out)
    
    def generate_sequences(self, src: Tensor, src_mask: Tensor, max_seq_len: int = None) -> Tensor:
        """
        Returns the predicted sequence (in token IDs) based on the provided `src` input. Uses greedy decoding to generate the sequence.
        If `max_seq_len` is not provided, `self.tgt_max_seq_len` will be used by default.
        """
        src, src_mask = self._to_device(src, src_mask)
        if max_seq_len is None:
            max_seq_len = self.tgt_max_seq_len
        return greedy_decode(self, src, src_mask, max_seq_len, self.start_token_idx, self.end_token_idx, self.pad_token_idx, device=self.device)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None, tgt_mask: Tensor = None) -> Tensor:
        src, tgt, src_mask, tgt_mask = self._to_device(src, tgt, src_mask, tgt_mask)
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)

    def _reset_parameters(self) -> None:
        """
        Glorot/Xavier/fan_avg initialization of parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _to_device(self, *tensors):
        return [curr_tensor.to(self.device) if curr_tensor is not None else None for curr_tensor in tensors]
