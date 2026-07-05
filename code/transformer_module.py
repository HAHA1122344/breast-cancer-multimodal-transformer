import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".
    Adds positional encodings to input embeddings.

    Input shape: (seq_len, batch_size, d_model) OR (batch_size, seq_len, d_model)
    We'll accept (batch_size, seq_len, d_model) and return same shape.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)     # even indices
        pe[:, 1::2] = torch.cos(position * div_term)     # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encodings (batch_size, seq_len, d_model)
        """
        if x.dim() != 3:
            raise ValueError("PositionalEncoding expects input of shape (batch_size, seq_len, d_model)")
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return x


# -------------------------
# Modality Attention
# -------------------------
class ModalityAttention(nn.Module):
    """
    Computes modality-level attention weights for each sample.

    Input:
        modality_embeddings: (batch_size, n_modalities, embed_dim)
    Output:
        weighted_embeddings: (batch_size, n_modalities, embed_dim)
        attention_weights: (batch_size, n_modalities)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, modality_embeddings: torch.Tensor) -> tuple:
        """
        Args:
            modality_embeddings: (B, M, D)
        Returns:
            weighted_modalities: (B, M, D)
            attn_weights: (B, M)  (softmax across M)
        """
        B, M, D = modality_embeddings.shape
        # compute attention scores per modality
        h = self.activation(self.fc1(modality_embeddings))   # (B, M, H)
        h = self.dropout(h)
        scores = self.fc2(h).squeeze(-1)                      # (B, M)
        attn_weights = F.softmax(scores, dim=1)              # (B, M)
        attn_weights_expanded = attn_weights.unsqueeze(-1)   # (B, M, 1)
        weighted = modality_embeddings * attn_weights_expanded
        return weighted, attn_weights


# -------------------------
# Pathway-Guided Attention
# -------------------------
class PathwayGuidedAttention(nn.Module):
    """
    Pathway-guided attention for gene expression features.

    Two modes:
    1. Hard mask: binary pathway_mask (n_pathways, n_genes), each gene belongs to exactly one pathway
    2. Soft membership: soft_membership (n_genes, n_pathways), each gene has a weight for each pathway

    Pipeline:
        1. Aggregate genes within each pathway -> pathway representations (B, n_pathways)
        2. Learn attention weights over pathways (B, n_pathways)
        3. Weight pathways by attention, then broadcast back to genes (B, n_genes)

    Args:
        n_pathways: number of pathways/modules
        n_genes: number of genes
        pathway_mask: (n_pathways, n_genes) binary mask OR (n_genes, n_pathways) soft membership
        hidden_dim: hidden dimension for attention MLP
        dropout: dropout probability
        use_soft_membership: if True, pathway_mask is (n_genes, n_pathways) soft weights
    """

    def __init__(
        self,
        n_pathways: int,
        n_genes: int,
        pathway_mask: np.ndarray,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_soft_membership: bool = False
    ):
        super().__init__()
        self.n_pathways = n_pathways
        self.n_genes = n_genes
        self.use_soft_membership = use_soft_membership

        if use_soft_membership:
            # pathway_mask: (n_genes, n_pathways) soft weights
            self.register_buffer("soft_membership", torch.tensor(pathway_mask, dtype=torch.float32))
            self.register_buffer("pathway_mask", None)
        else:
            # pathway_mask: (n_pathways, n_genes) binary
            self.register_buffer("pathway_mask", torch.tensor(pathway_mask, dtype=torch.float32))
            self.register_buffer("soft_membership", None)

        # Pathway-level attention MLP
        self.attention = nn.Sequential(
            nn.Linear(n_pathways, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_pathways)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, n_genes) gene expression features
        Returns:
            attended_x: (B, n_genes) pathway-attended gene features
            pathway_weights: (B, n_pathways) attention weights over pathways
            pathway_scores: (B, n_pathways) raw pathway activations
        """
        B = x.size(0)
        device = x.device

        if self.use_soft_membership:
            # Soft membership: (n_genes, n_pathways)
            membership = self.soft_membership.to(device)  # (G, P)
            # Pathway representations: weighted sum of genes by membership
            pathway_repr = torch.matmul(x, membership)  # (B, P)
        else:
            # Hard mask: (n_pathways, n_genes)
            pathway_mask = self.pathway_mask.to(device)  # (P, G)
            gene_counts = pathway_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (P, 1)
            pathway_repr = torch.matmul(x, pathway_mask.t()) / gene_counts.t()  # (B, P)

        # Pathway-level attention
        pathway_scores = self.attention(pathway_repr)  # (B, P)
        pathway_weights = F.softmax(pathway_scores, dim=1)  # (B, P)

        # Weight pathways and broadcast back to genes
        if self.use_soft_membership:
            # Use soft membership to broadcast
            membership = self.soft_membership.to(device)  # (G, P)
            weighted_pathways = pathway_weights.unsqueeze(1) * membership.unsqueeze(0)  # (B, 1, P) * (1, G, P)
            attended_x = weighted_pathways.sum(dim=2)  # (B, G)
        else:
            pathway_mask = self.pathway_mask.to(device)  # (P, G)
            weighted_pathways = pathway_weights.unsqueeze(2) * pathway_mask.unsqueeze(0)  # (B, P, G)
            attended_x = weighted_pathways.sum(dim=1)  # (B, G)

        # Residual connection
        attended_x = x + attended_x

        return attended_x, pathway_weights, pathway_scores


# -------------------------
# Transformer Backbone
# -------------------------
class TransformerBackbone(nn.Module):
    """
    Transformer backbone to fuse modality latent embeddings and produce a fused patient representation.

    Typical usage:
        - Input modality_latents: (batch_size, n_modalities, latent_dim)
        - Backbone projects each modality latent to d_model
        - Optionally prepend a learnable CLS token
        - Add positional encodings and pass through TransformerEncoder
        - Return pooled embedding (CLS token or mean pool)

    Args:
        latent_dim: dimension of input latent vectors (from autoencoders)
        d_model: Transformer model dimension (embedding dimension)
        nhead: number of attention heads
        num_layers: number of stacked TransformerEncoderLayer
        dim_feedforward: FFN inner dimension
        dropout: dropout probability
        use_cls_token: if True, prepend a learnable CLS token and return its final representation
        max_seq_len: maximum supported sequence length for positional encoding
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        use_cls_token: bool = True,
        max_seq_len: int = 50
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # projection from latent space to Transformer d_model
        self.project = nn.Linear(latent_dim, d_model)

        # learned modality tokens (one per modality) are optional, but we'll add a learned "modality token"
        # instead of per-modality tokens we append nothing here; alternative: learn per-modality embeddings externally
        # We DO add a learnable CLS token if requested
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)
        else:
            self.cls_token = None

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + (1 if use_cls_token else 0))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True  # makes input (B, S, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_latents: torch.Tensor, modality_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            modality_latents: (B, M, latent_dim)
            modality_mask: Optional boolean mask (B, M) where True=valid modality, False=missing
                           If provided, we will mask positions corresponding to missing modalities by setting their embeddings to zero
                           and/or using attention_mask to prevent attending to those.
        Returns:
            fused_repr: (B, d_model)  pooled representation (CLS or mean)
            transformer_output: (B, S, d_model)  full sequence output
        """
        B, M, L = modality_latents.shape
        device = modality_latents.device

        # Project modalities to d_model
        x = self.project(modality_latents)  # (B, M, d_model)

        # Optionally mask missing modalities by zeroing their projected embeddings
        if modality_mask is not None:
            # modality_mask expected shape (B, M) with 1 for valid, 0 for missing (or True/False)
            mask = modality_mask.view(B, M, 1).to(device)
            x = x * mask.float()

        seq = x  # (B, M, d_model)

        # Prepend CLS token if requested
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            seq = torch.cat([cls, seq], dim=1)      # (B, 1+M, d_model)

        # Add positional encodings
        seq = self.pos_encoder(seq)  # (B, S, d_model)
        seq = self.dropout(seq)
        seq = self.norm(seq)

        # Build transformer attention mask if modality_mask provided
        # PyTorch TransformerEncoder (batch_first=True) accepts src_key_padding_mask of shape (B, S)
        src_key_padding_mask = None
        if modality_mask is not None:
            if self.use_cls_token:
                # prepend True for CLS token (not masked)
                cls_mask = torch.ones(B, 1, device=device, dtype=modality_mask.dtype)
                padd_mask = torch.cat([cls_mask, modality_mask.to(device)], dim=1)  # 1=valid
            else:
                padd_mask = modality_mask
            # Transformer expects mask with True at positions that should be masked (i.e., padding)
            # Here padd_mask: 1 valid, 0 missing -> need inverted boolean mask
            src_key_padding_mask = (~padd_mask.bool())  # (B, S) True for masked positions

        # Transformer forward; returns (B, S, d_model)
        transformer_out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        # Pooling: return CLS token representation if used, else mean pool across modalities (excluding masked)
        if self.use_cls_token:
            fused = transformer_out[:, 0, :]  # (B, d_model)
        else:
            if modality_mask is not None:
                mask = modality_mask.unsqueeze(-1).float()  # (B, M, 1)
                # mean of valid modality positions
                summed = (transformer_out * mask).sum(dim=1)  # (B, d_model)
                denom = mask.sum(dim=1).clamp(min=1.0)         # avoid division by zero
                fused = summed / denom
            else:
                fused = transformer_out.mean(dim=1)  # (B, d_model)

        return fused, transformer_out


# -------------------------
# Example usage / quick test
# -------------------------
if __name__ == "__main__":
    # Quick smoke test
    B = 8           # batch size
    M = 3           # number of modalities (e.g., clinical, genomic, lifestyle)
    latent_dim = 128
    d_model = 256

    # fake modality latent vectors
    x = torch.randn(B, M, latent_dim)

    # simulate modality mask (1 valid, 0 missing)
    modality_mask = torch.ones(B, M)
    modality_mask[0, 2] = 0  # sample 0 has modality 2 missing

    model = TransformerBackbone(
        latent_dim=latent_dim,
        d_model=d_model,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        use_cls_token=True,
        max_seq_len=10
    )

    fused, seq_out = model(x, modality_mask=modality_mask)
    print("Fused shape:", fused.shape)         # (B, d_model)
    print("Transformer seq_out shape:", seq_out.shape)  # (B, S, d_model)
