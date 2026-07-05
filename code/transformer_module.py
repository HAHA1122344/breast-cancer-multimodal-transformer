import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.dtype)


class ModalityAttention(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, modality_embeddings: torch.Tensor) -> tuple:
        B, M, D = modality_embeddings.shape
        h = self.activation(self.fc1(modality_embeddings))
        h = self.dropout(h)
        scores = self.fc2(h).squeeze(-1)
        attn_weights = F.softmax(scores, dim=1)
        weighted = modality_embeddings * attn_weights.unsqueeze(-1)
        return weighted, attn_weights


class PathwayGuidedAttention(nn.Module):
    def __init__(self, n_pathways: int, n_genes: int, pathway_mask: np.ndarray,
                 hidden_dim: int = 64, dropout: float = 0.1, use_soft_membership: bool = False):
        super().__init__()
        self.n_pathways = n_pathways
        self.n_genes = n_genes
        self.use_soft_membership = use_soft_membership
        if use_soft_membership:
            self.register_buffer("soft_membership", torch.tensor(pathway_mask, dtype=torch.float32))
            self.register_buffer("pathway_mask", None)
        else:
            self.register_buffer("pathway_mask", torch.tensor(pathway_mask, dtype=torch.float32))
            self.register_buffer("soft_membership", None)
        self.attention = nn.Sequential(
            nn.Linear(n_pathways, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_pathways)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        B = x.size(0)
        device = x.device
        if self.use_soft_membership:
            membership = self.soft_membership.to(device)
            pathway_repr = torch.matmul(x, membership)
        else:
            pathway_mask = self.pathway_mask.to(device)
            gene_counts = pathway_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pathway_repr = torch.matmul(x, pathway_mask.t()) / gene_counts.t()
        pathway_scores = self.attention(pathway_repr)
        pathway_weights = F.softmax(pathway_scores, dim=1)
        if self.use_soft_membership:
            attended_x = torch.matmul(pathway_weights.unsqueeze(1), self.soft_membership.to(device)).squeeze(1)
        else:
            attended_x = torch.matmul(pathway_weights.unsqueeze(1), self.pathway_mask.to(device)).squeeze(1)
        attended_x = x + attended_x
        return attended_x, pathway_weights, pathway_scores


class TransformerBackbone(nn.Module):
    def __init__(self, latent_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.3,
                 use_cls_token: bool = True, max_seq_len: int = 50):
        super().__init__()
        self.project = nn.Linear(latent_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if use_cls_token else None
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + (1 if use_cls_token else 0))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.use_cls_token = use_cls_token

    def forward(self, modality_latents: torch.Tensor, modality_mask: Optional[torch.Tensor] = None) -> tuple:
        B, M, L = modality_latents.shape
        x = self.project(modality_latents)
        if modality_mask is not None:
            x = x * modality_mask.view(B, M, 1).float()
        seq = x
        if self.use_cls_token:
            seq = torch.cat([self.cls_token.expand(B, -1, -1), seq], dim=1)
        seq = self.pos_encoder(seq)
        seq = self.dropout(seq)
        seq = self.norm(seq)
        src_key_padding_mask = None
        if modality_mask is not None:
            if self.use_cls_token:
                cls_mask = torch.ones(B, 1, device=seq.device, dtype=modality_mask.dtype)
                padd_mask = torch.cat([cls_mask, modality_mask.to(seq.device)], dim=1)
            else:
                padd_mask = modality_mask
            src_key_padding_mask = (~padd_mask.bool())
        transformer_out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)
        if self.use_cls_token:
            fused = transformer_out[:, 0, :]
        else:
            fused = transformer_out.mean(dim=1)
        return fused, transformer_out


class CrossModalPathwayAttention(nn.Module):
    def __init__(self, n_pathways: int, n_modalities: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pathway_to_modality = nn.Sequential(
            nn.Linear(n_pathways, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_modalities)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pathway_weights: torch.Tensor) -> tuple:
        modality_scores = self.pathway_to_modality(pathway_weights)
        modality_gates = F.softmax(modality_scores, dim=1)
        return modality_gates, modality_scores


class ModalityGatingNetwork(nn.Module):
    def __init__(self, n_modalities: int, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_modalities)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_embeddings: torch.Tensor) -> tuple:
        sample_repr = modality_embeddings.mean(dim=1)
        gate_scores = self.gate_net(sample_repr)
        gates = F.softmax(gate_scores, dim=1)
        gated_embeddings = modality_embeddings * gates.unsqueeze(-1)
        return gated_embeddings, gates


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = features.size(0)
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float().to(features.device)
        mask = torch.eye(B, device=features.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * labels_eq).sum(dim=1)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log((pos_sim + 1e-7) / (denom + 1e-7))
        return loss.mean()
