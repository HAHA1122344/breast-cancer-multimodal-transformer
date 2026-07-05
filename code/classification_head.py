import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 🧩 1. Classification Head Definition
# ============================================================

class ClassificationHead(nn.Module):
    """
    Multi-class classification head.

    Args:
        input_dim (int): Dimension of input embedding (e.g., 256 from Transformer)
        num_classes (int): Number of recurrence types or cancer risk categories
        hidden_dim (int): Hidden layer dimension (for optional MLP)
        dropout (float): Dropout probability for regularization
        use_mlp (bool): If True, adds one hidden layer between input and output
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_mlp: bool = True
    ):
        super(ClassificationHead, self).__init__()

        self.use_mlp = use_mlp
        self.dropout = nn.Dropout(dropout)

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — fused patient embeddings
        Returns:
            logits: (B, num_classes)
        """
        x = self.dropout(x)
        if self.use_mlp:
            logits = self.mlp(x)
        else:
            logits = self.fc(x)
        return logits


# ============================================================
# ⚙️ 2. Classification Loss
# ============================================================

def classification_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights=None) -> torch.Tensor:
    """
    Computes the weighted cross-entropy loss for multi-class classification.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion(logits, labels)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha=None, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (B, num_classes)
        labels: (B,) true class indices
        alpha: optional (num_classes,) weight per class
        gamma: focusing parameter (default 2.0)
        reduction: 'mean' or 'sum'
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    ce_loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    # Label smoothing
    if label_smoothing > 0:
        uniform_ce = -log_probs.mean(dim=1)
        ce_loss = (1 - label_smoothing) * ce_loss + label_smoothing * uniform_ce

    # Label smoothing
    if label_smoothing > 0:
        uniform_ce = -log_probs.mean(dim=1)
        ce_loss = (1 - label_smoothing) * ce_loss + label_smoothing * uniform_ce

    if alpha is not None:
        alpha = alpha.to(logits.device)
        at = alpha.gather(0, labels)
        ce_loss = at * ce_loss

    focal_weight = (1 - pt).pow(gamma)
    loss = focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def hierarchical_loss(logits: torch.Tensor, labels: torch.Tensor,
                       luminal_mask: torch.Tensor, nonluminal_mask: torch.Tensor,
                       alpha: float = 0.5) -> torch.Tensor:
    """
    Hierarchical classification loss:
    1. First stage: Luminal (0/1) vs Non-Luminal (2/3/4/5)
    2. Second stage: fine-grained within each super-class
    """
    B = logits.shape[0]

    # Super-class labels: 0=Luminal, 1=Non-Luminal
    super_labels = torch.where(
        (labels == 0) | (labels == 1),
        torch.zeros_like(labels),
        torch.ones_like(labels)
    )

    # Stage 1: Binary cross-entropy for super-class
    # Use logits for classes 0,1 (LumA=0, LumB=1) vs others
    stage1_logits = torch.stack([
        logits[:, 0] + logits[:, 1],  # Luminal score
        logits[:, 2:].sum(dim=1)       # Non-Luminal score
    ], dim=1)

    loss_stage1 = F.cross_entropy(stage1_logits, super_labels)

    # Stage 2: Cross-entropy for fine classes within each super-class
    loss_stage2 = F.cross_entropy(logits, labels)

    return alpha * loss_stage1 + (1 - alpha) * loss_stage2


# ============================================================
# 📊 3. Prediction & Evaluation Utilities
# ============================================================

def predict_class(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns predicted class indices.
    """
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    return preds


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes classification accuracy.
    """
    preds = predict_class(logits)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def predict_proba(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts raw logits to class probabilities using softmax.
    """
    return F.softmax(logits, dim=1)


# ============================================================
# 🚀 4. Example Usage (Standalone Test)
# ============================================================

if __name__ == "__main__":
    import numpy as np

    # Dummy inputs
    batch_size = 8
    input_dim = 256
    num_classes = 4

    # Random input embeddings (from Transformer)
    embeddings = torch.randn(batch_size, input_dim)

    # Random labels
    labels = torch.tensor(np.random.randint(0, num_classes, size=batch_size))

    # Initialize model
    model = ClassificationHead(input_dim=input_dim, num_classes=num_classes, use_mlp=True)
    logits = model(embeddings)

    # Compute loss
    loss = classification_loss(logits, labels)
    print(f"Cross-Entropy Loss: {loss.item():.6f}")

    # Predictions
    preds = predict_class(logits)
    acc = compute_accuracy(logits, labels)
    probs = predict_proba(logits)

    print(f"Predictions: {preds.tolist()}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Probabilities (first sample): {probs[0].detach().numpy()}")
