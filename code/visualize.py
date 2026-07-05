import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import torch
import shap

from models.unified_model import UnifiedMultimodalModel
from train import MultimodalNPZDataset, multimodal_collate


# ============================================================
# 🧩 1. Confusion Matrix & Metrics
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(out_path, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix to {out_path}/confusion_matrix.png")

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ============================================================
# 📈 2. Kaplan–Meier Survival Curves
# ============================================================
def plot_kaplan_meier(durations, events, risk_scores, out_path, num_groups=2):
    """
    Plots KM curves by splitting patients into risk groups (e.g., high vs low).
    """
    os.makedirs(out_path, exist_ok=True)
    kmf = KaplanMeierFitter()

    # Divide into quantile-based groups
    risk_scores = np.array(risk_scores)
    cutoff = np.quantile(risk_scores, 0.5) if num_groups == 2 else np.quantile(risk_scores, [0.33, 0.66])
    if num_groups == 2:
        groups = (risk_scores > cutoff).astype(int)
    else:
        groups = np.digitize(risk_scores, bins=cutoff)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for g in np.unique(groups):
        mask = groups == g
        label = f"Group {g+1}" if num_groups > 2 else ("High Risk" if g == 1 else "Low Risk")
        kmf.fit(durations[mask], event_observed=events[mask], label=label)
        kmf.plot(ax=ax, ci_show=False, color=colors[g % len(colors)])

    plt.title("Kaplan–Meier Survival Curves")
    plt.xlabel("Time (Days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_path, "kaplan_meier_curve.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved Kaplan–Meier curve to {out_path}/kaplan_meier_curve.png")


# ============================================================
# 🧠 3. SHAP Feature Importance
# ============================================================
def plot_shap_importance(model, data_loader, out_path, modality_names):
    """
    Computes SHAP feature importances on the model’s fused representation
    (interprets ClassificationHead).
    """
    os.makedirs(out_path, exist_ok=True)
    model.eval()

    # collect embeddings and labels
    all_fused = []
    all_labels = []
    with torch.no_grad():
        for modalities, labels, durations, events in data_loader:
            modalities = [m.cuda() if torch.cuda.is_available() else m for m in modalities]
            logits, log_risk, attn_weights, fused = model(modalities)
            all_fused.append(fused.cpu())
            all_labels.append(labels)
    X = torch.cat(all_fused, dim=0)
    y = torch.cat(all_labels, dim=0)

    # Explain using SHAP
    explainer = shap.Explainer(model.class_head, X)
    shap_values = explainer(X)

    shap.summary_plot(
        shap_values, X.numpy(), plot_type="bar", show=False, max_display=20
    )
    plt.title("Top SHAP Feature Importances (Fused Representation)")
    plt.savefig(os.path.join(out_path, "shap_feature_importance.png"), bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP summary plot to {out_path}/shap_feature_importance.png")


# ============================================================
# 💠 4. Modality Attention Heatmap
# ============================================================
def plot_modality_attention(model, data_loader, modality_names, out_path):
    os.makedirs(out_path, exist_ok=True)
    model.eval()

    attn_matrix = []
    with torch.no_grad():
        for modalities, labels, durations, events in data_loader:
            modalities = [m.cuda() if torch.cuda.is_available() else m for m in modalities]
            _, _, attn_weights, _ = model(modalities)
            attn_matrix.append(attn_weights.cpu().numpy())
    attn_matrix = np.concatenate(attn_matrix, axis=0)

    mean_attn = np.mean(attn_matrix, axis=0)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        mean_attn[np.newaxis, :],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=modality_names,
        yticklabels=["Attention"],
        cbar=False,
        ax=ax
    )
    plt.title("Average Modality Attention Weights")
    plt.savefig(os.path.join(out_path, "modality_attention_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved attention heatmap to {out_path}/modality_attention_heatmap.png")


# ============================================================
# 🧭 Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize Unified Model Results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_npz", type=str, required=True, help="Path to dataset (.npz)")
    parser.add_argument("--out_dir", type=str, default="results/plots", help="Output directory for plots")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--modality_names", type=str, nargs="+", default=["Clinical", "mRNA", "CNA", "Methylation"])
    parser.add_argument("--skip_shap", action="store_true", help="Skip SHAP analysis (slow on large models)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset
    dataset = MultimodalNPZDataset(args.data_npz)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=multimodal_collate)

    modality_dims = [m.shape[1] for m in dataset.modalities]
    num_modalities = len(modality_dims)

    # Use explicit latent dims matching checkpoint [64, 256, 256, 256, 128]
    ae_latent_dims = [64, 256, 256, 256, 128][:num_modalities]
    if len(ae_latent_dims) < num_modalities:
        ae_latent_dims.extend([128] * (num_modalities - len(ae_latent_dims)))

    # Load model
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_dims,
        ae_latent_dim=ae_latent_dims,
        transformer_cfg={"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512, "dropout": 0.3},
        num_classes=args.num_classes
    ).to(device)

    # Load checkpoint and conditionally init pathway attention
    state_dict = torch.load(args.model_path, map_location="cpu")
    if any(k.startswith("pathway_attention.") for k in state_dict.keys()):
        soft_key = "pathway_attention.soft_membership"
        use_soft = soft_key in state_dict
        import os
        soft_path = "data/soft_module_mask.npz"
        hard_path = "data/pathway_mask.npz"
        mask_path = soft_path if use_soft and os.path.exists(soft_path) else hard_path
        model.setup_pathway_attention(mask_path, use_soft_membership=use_soft)
    model.load_state_dict(state_dict)
    model.eval()

    # Collect predictions & risks
    all_labels, all_preds, all_durations, all_events, all_risks = [], [], [], [], []

    with torch.no_grad():
        for modalities, labels, durations, events in loader:
            modalities = [m.to(device) for m in modalities]
            logits, log_risk, attn_weights, fused = model(modalities)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_durations.extend(durations.numpy().tolist())
            all_events.extend(events.numpy().tolist())
            all_risks.extend(torch.exp(log_risk).cpu().numpy().reshape(-1).tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    durations = np.array(all_durations)
    events = np.array(all_events)
    risk_scores = np.array(all_risks)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1️⃣ Confusion Matrix
    class_names = [f"Class {i}" for i in range(args.num_classes)]
    plot_confusion_matrix(y_true, y_pred, class_names, args.out_dir)

    # 2️⃣ Kaplan–Meier Curves
    plot_kaplan_meier(durations, events, risk_scores, args.out_dir)

    # 3️⃣ SHAP Feature Importance (optional, slow)
    if not args.skip_shap:
        try:
            plot_shap_importance(model, loader, args.out_dir, args.modality_names)
        except Exception as e:
            print(f"[WARN] SHAP analysis failed: {e}")
    else:
        print("[INFO] Skipped SHAP analysis (--skip_shap)")

    # 4️⃣ Modality Attention Heatmap
    plot_modality_attention(model, loader, args.modality_names, args.out_dir)

    print(f"[INFO] Visualization complete. All plots saved to {args.out_dir}")


# ============================================================
# 🚀 Entry Point
# ============================================================
if __name__ == "__main__":
    main()
