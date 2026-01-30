"""
Analog LSTM for Language Modeling.

Standard LSTM architecture with AnalogLinear layers.
Used for WikiText-2 / Penn Treebank experiments.

The LSTM is particularly interesting for analog because:
1. Recurrent weights see many time steps → accumulates noise
2. Gates (sigmoid, tanh) have saturating regions → some noise tolerance
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from ..layers import AnalogLinear
from ..physics import PCMPhysicsEngine


class AnalogLSTMCell(nn.Module):
    """
    Single LSTM cell with analog linear projections.
    
    Standard LSTM equations:
        i = σ(W_xi @ x + W_hi @ h + b_i)  # input gate
        f = σ(W_xf @ x + W_hf @ h + b_f)  # forget gate
        g = tanh(W_xg @ x + W_hg @ h + b_g)  # cell candidate
        o = σ(W_xo @ x + W_ho @ h + b_o)  # output gate
        c' = f ⊙ c + i ⊙ g
        h' = o ⊙ tanh(c')
    
    We use two AnalogLinear layers: input→hidden and hidden→hidden,
    each projecting to 4×hidden (for i,f,g,o concatenated).
    """
    
    def __init__(self, input_size: int, hidden_size: int,
                 physics: Optional[PCMPhysicsEngine] = None):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input projection: [x] → [i,f,g,o]
        self.ih = AnalogLinear(input_size, 4 * hidden_size, physics=physics)
        # Hidden projection: [h] → [i,f,g,o]
        self.hh = AnalogLinear(hidden_size, 4 * hidden_size, physics=physics)
    
    def forward(self, x: torch.Tensor, 
                state: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor [batch, input_size]
            state: (h, c) each [batch, hidden_size]
            
        Returns:
            (h', c') new hidden and cell states
        """
        h_prev, c_prev = state
        
        # Combined projection for all gates
        gates = self.ih(x) + self.hh(h_prev)  # [batch, 4*hidden]
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Gate activations
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # cell candidate
        
        # Cell and hidden state updates
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
    
    def set_inference_time(self, t: float) -> None:
        self.ih.set_inference_time(t)
        self.hh.set_inference_time(t)


class AnalogLSTM(nn.Module):
    """
    Multi-layer LSTM language model with analog weights.
    
    Architecture:
        Embedding → LSTM layers → Linear decoder
        
    Hyperparameters match AWD-LSTM baseline:
        - 2 layers
        - 200/200 embedding/hidden size (small) or 400/1150 (large)
        - Tied embedding/decoder weights (optional)
        - Dropout between layers
    """
    
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 physics: Optional[PCMPhysicsEngine] = None,
                 tie_weights: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        # Embedding (digital, not quantized)
        # Rationale: embedding lookup is sparse, not suited for crossbar
        self.encoder = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embed_size if i == 0 else hidden_size
            self.layers.append(
                AnalogLSTMCell(input_dim, hidden_size, physics=physics)
            )
        
        # Output projection (analog)
        self.decoder = AnalogLinear(hidden_size, vocab_size, physics=physics)
        
        # Dropout between layers
        self.drop = nn.Dropout(dropout)
        
        # Weight initialization
        self._init_weights()
        
        # Optional weight tying
        if tie_weights and embed_size == hidden_size:
            self.decoder.weight = self.encoder.weight
    
    def _init_weights(self) -> None:
        """Standard init: uniform for embedding, zero bias for decoder."""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.decoder.bias is not None:
            self.decoder.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, 
                hidden: Tuple[List[torch.Tensor], List[torch.Tensor]]
                ) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: input tokens [seq_len, batch]
            hidden: ([h_0, h_1, ...], [c_0, c_1, ...])
            
        Returns:
            output: logits [seq_len, batch, vocab]
            hidden: updated hidden states
        """
        # Embed input tokens
        emb = self.drop(self.encoder(x))  # [seq_len, batch, embed]
        
        # Unpack hidden states
        h_states, c_states = hidden
        
        # Process sequence step by step
        # (Could be parallelized with cuDNN LSTM, but we need custom cell)
        outputs = []
        for t in range(x.size(0)):
            inp = emb[t]  # [batch, embed]
            new_h, new_c = [], []
            
            for i, layer in enumerate(self.layers):
                h_i, c_i = layer(inp, (h_states[i], c_states[i]))
                
                # Apply dropout between layers (not after last)
                if i < self.num_layers - 1:
                    inp = self.drop(h_i)
                else:
                    inp = h_i
                
                new_h.append(h_i)
                new_c.append(c_i)
            
            h_states, c_states = new_h, new_c
            outputs.append(inp)
        
        # Stack outputs and apply decoder
        output = torch.stack(outputs)  # [seq_len, batch, hidden]
        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        
        return decoded.view(output.size(0), output.size(1), -1), (h_states, c_states)
    
    def init_hidden(self, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initialize hidden states to zeros."""
        weight = next(self.parameters())
        h_states = [weight.new_zeros(batch_size, self.hidden_size) 
                    for _ in range(self.num_layers)]
        c_states = [weight.new_zeros(batch_size, self.hidden_size) 
                    for _ in range(self.num_layers)]
        return h_states, c_states
    
    def set_inference_time(self, t: float) -> None:
        """Set drift time for all analog layers."""
        for layer in self.layers:
            layer.set_inference_time(t)
        self.decoder.set_inference_time(t)


def create_lstm_lm(vocab_size: int, physics: Optional[PCMPhysicsEngine] = None,
                   size: str = 'small') -> AnalogLSTM:
    """
    Factory for LSTM language models.
    
    Args:
        vocab_size: vocabulary size
        physics: PCM physics engine (None for digital baseline)
        size: 'small' (200/200) or 'large' (400/1150)
    """
    if size == 'small':
        return AnalogLSTM(vocab_size, embed_size=200, hidden_size=200,
                          num_layers=2, dropout=0.5, physics=physics)
    elif size == 'large':
        return AnalogLSTM(vocab_size, embed_size=400, hidden_size=1150,
                          num_layers=3, dropout=0.65, physics=physics)
    else:
        raise ValueError(f"Unknown size: {size}")
