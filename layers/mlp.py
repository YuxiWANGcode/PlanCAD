import torch.nn as nn
import torch

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='tanh'): 
        super(MLP, self).__init__()
        
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class TransformerBlock(nn.Module):
    '''
    Transformer block to encode/decode high-dimension representation of sequential data
    '''
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward=256, 
                 num_layers=2, 
                 dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 10000, d_model))  # Large enough for dynamic input

    def forward(self, x):
        # x: (batch_size, num_tokens, seq_len, embed_dim)
        batch_size, num_tokens, seq_len, embed_dim = x.shape

        if embed_dim != self.d_model:
            raise ValueError(f"Input embed_dim ({embed_dim}) must match d_model ({self.d_model}).")

        # Flatten num_tokens and seq_len into a single sequence dimension for Transformer
        x = x.reshape(batch_size, num_tokens * seq_len, embed_dim)

        print(f"Transformer input shape: {x.shape}")

        # Add positional encoding
        seq_len_total = x.size(1)
        if seq_len_total > self.positional_encoding.size(1):
            raise ValueError(f"Sequence length ({seq_len_total}) exceeds positional encoding size ({self.positional_encoding.size(1)}).")

        x = x + self.positional_encoding[:, :seq_len_total, :]

        # Permute for Transformer input: (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)

        # Pass through Transformer Encoder
        y = self.transformer(x)  # Shape: (seq_len, batch_size, embed_dim)

        print(f"Transformer output shape: {y.shape}")

        # Permute back: (batch_size, seq_len, embed_dim)
        y = y.permute(1, 0, 2)
        return y
