import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerCouplet(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_pingze = nn.Linear(d_model, 2)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None, tgt_padding_mask: torch.Tensor = None):
        src_emb = self.embedding(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        transformer_out = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        word_logits = self.fc_out(transformer_out)
        pingze_logits = self.fc_pingze(transformer_out)

        return word_logits, pingze_logits
