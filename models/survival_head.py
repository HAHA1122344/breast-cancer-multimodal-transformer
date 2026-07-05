import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧩 1. Cox Proportional Hazards Linear Head
# ============================================================

class SurvivalHead(nn.Module):
    """
    Cox-based linear survival prediction head.
    Maps the fused Transformer embedding (d_model) to a scalar log-risk score.
    The hazard ratio for a patient i is exp(risk_score_i).
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super(SurvivalHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — fused feature embedding for each patient
        Returns:
            log_risk: (B, 1) — log-risk score for Cox model
        """
        x = self.dropout(x)
        log_risk = self.linear(x)
        return log_risk


# ============================================================
# 🧠 2. DeepSurv Non-linear Head (Optional Variant)
# ============================================================

class DeepSurvHead(nn.Module):
    """
    DeepSurv variant: A non-linear feedforward neural network for survival analysis.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super(DeepSurvHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            log_risk: (B, 1)
        """
        return self.net(x)


# ============================================================
# ⚙️ 3. Cox Partial Log-Likelihood Loss
# ============================================================

def cox_partial_log_likelihood(log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Computes the Cox proportional hazards partial log-likelihood loss.

    Args:
        log_risk: (B, 1) model output (log hazard score for each patient)
        durations: (B,) survival times
        events: (B,) event indicators (1 = event occurred, 0 = censored)
    Returns:
        loss: scalar tensor (negative partial log-likelihood)
    """

    # Flatten tensors
    log_risk = log_risk.view(-1)
    durations = durations.view(-1)
    events = events.view(-1)

    # Sort by descending survival time (longest to shortest)
    order = torch.argsort(durations, descending=True)
    log_risk = log_risk[order]
    events = events[order]

    # Compute cumulative log-sum-exp of risks (numerically stable)
    log_cumsum_hazard = torch.logcumsumexp(log_risk, dim=0)

    # Partial log-likelihood: (risk_i - log(sum_j exp(risk_j))) over uncensored (events==1)
    log_likelihood = log_risk - log_cumsum_hazard
    neg_loss = -torch.sum(log_likelihood * events) / events.sum().clamp(min=1.0)

    return neg_loss


# ============================================================
# 📈 4. Risk Score Utilities
# ============================================================

def compute_risk_scores(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute risk scores for new patient embeddings.

    Args:
        model: trained SurvivalHead or DeepSurvHead
        x: (B, input_dim) patient embeddings
    Returns:
        risk_scores: (B,) risk values (higher = higher risk)
    """
    model.eval()
    with torch.no_grad():
        log_risk = model(x)
        risk_scores = torch.exp(log_risk).view(-1)
    return risk_scores


# ============================================================
# 🚀 Example Usage (Standalone Test)
# ============================================================
if __name__ == "__main__":
    import numpy as np

    # Create dummy dataset
    batch_size = 10
    input_dim = 256
    embeddings = torch.randn(batch_size, input_dim)

    # Simulate survival times and event indicators
    durations = torch.tensor(np.random.randint(1, 1000, size=batch_size), dtype=torch.float32)
    events = torch.tensor(np.random.randint(0, 2, size=batch_size), dtype=torch.float32)

    # Initialize survival head
    survival_head = SurvivalHead(input_dim)
    log_risk = survival_head(embeddings)

    # Compute Cox loss
    loss = cox_partial_log_likelihood(log_risk, durations, events)
    print(f"Cox partial log-likelihood loss: {loss.item():.6f}")

    # Compute risk scores
    risk_scores = compute_risk_scores(survival_head, embeddings)
    print("Sample risk scores:", risk_scores[:5])
