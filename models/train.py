import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import argparse
import time
import copy
import numpy as np
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from lifelines.utils import concordance_index

# Import your model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.unified_model import UnifiedMultimodalModel
from models.survival_head import cox_partial_log_likelihood
from models.classification_head import classification_loss, focal_loss
import copy

# Class weights for imbalance: [490, 330, 171, 129, 142, 107]
# Higher weight for minority classes (Class 5 is most under-represented)
CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 2.9, 3.8, 3.4, 4.6])

# -----------------------------

# Dataset
# -----------------------------
class MultimodalNPZDataset(Dataset):
    """
    Loads data saved as a .npz file with arrays:
      mod_0, mod_1, ..., labels, durations, events
    """

    def __init__(self, npz_path: str, modality_keys: List[str] = None):
        data = np.load(npz_path, allow_pickle=True)
        # If modality_keys provided, use them; else auto-detect mod_ prefix
        if modality_keys:
            self.modalities = [data[k].astype(np.float32) for k in modality_keys]
        else:
            # detect all keys starting with 'mod_'
            mod_keys = sorted([k for k in data.keys() if k.startswith("mod_")])
            if len(mod_keys) == 0:
                raise ValueError(f"No modality arrays found in {npz_path}. Expected keys mod_0, mod_1, ...")
            self.modalities = [data[k].astype(np.float32) for k in mod_keys]

        # Required arrays
        if "labels" not in data or "durations" not in data or "events" not in data:
            raise ValueError(f"{npz_path} must contain 'labels', 'durations' and 'events' arrays.")
        self.labels = data["labels"].astype(np.int64)
        self.durations = data["durations"].astype(np.float32)
        self.events = data["events"].astype(np.float32)

        # number of samples
        self.N = self.labels.shape[0]
        # sanity checks: all modality arrays must have same N
        for arr in self.modalities:
            if arr.shape[0] != self.N:
                raise ValueError("All modality arrays must have same number of samples as labels.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return list of modality arrays for index idx, plus label, duration, event
        mods = [torch.from_numpy(arr[idx]) for arr in self.modalities]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        duration = torch.tensor(self.durations[idx], dtype=torch.float32)
        event = torch.tensor(self.events[idx], dtype=torch.float32)
        return mods, label, duration, event


# -----------------------------
# Collate fn for DataLoader
# -----------------------------
def multimodal_collate(batch):
    """
    Batch is list of tuples (mods_list, label, duration, event)
    We need to stack each modality separately.
    """
    batch_size = len(batch)
    num_modalities = len(batch[0][0])
    # prepare lists for each modality
    mods_per_modality = [[] for _ in range(num_modalities)]
    labels = []
    durations = []
    events = []

    for mods, lbl, dur, ev in batch:
        for i, m in enumerate(mods):
            mods_per_modality[i].append(m)
        labels.append(lbl)
        durations.append(dur)
        events.append(ev)

    # stack modalities
    mods_stacked = [torch.stack(lst, dim=0) for lst in mods_per_modality]
    labels = torch.stack(labels, dim=0)
    durations = torch.stack(durations, dim=0)
    events = torch.stack(events, dim=0)

    return mods_stacked, labels, durations, events


# -----------------------------
# Utility: compute metrics
# -----------------------------
def compute_classification_metrics(logits, labels):
    """
    Returns accuracy, precision, recall, f1 (macro)
    """
    probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    y_true = labels.detach().cpu().numpy()
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    return acc, prec, rec, f1, preds


# -----------------------------
# Training / Validation loop
# -----------------------------
def train(
    train_loader,
    val_loader,
    modality_input_dims,
    device,
    output_dir,
    num_classes=6,
    epochs=100,
    lr=1e-4,
    cls_weight=1.0,
    surv_weight=1.0,
    patience=15,
    save_every=1,
    ae_latent_dims=None,
    pathway_mask_path=None,
):
    print("[DEBUG] train() started, output_dir=", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Per-modality latent dims: large dims for high-dim modalities, smaller for low-dim
    if ae_latent_dims is None:
        ae_latent_dims = []
        for d in modality_input_dims:
            if d >= 500:
                ae_latent_dims.append(256)
            elif d >= 100:
                ae_latent_dims.append(192)
            elif d >= 50:
                ae_latent_dims.append(128)
            else:
                ae_latent_dims.append(64)

    print("[DEBUG] Creating model...")
    model = UnifiedMultimodalModel(
        modality_input_dims=modality_input_dims,
        ae_latent_dim=ae_latent_dims,
        transformer_cfg={"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512, "dropout": 0.3, "use_cls_token": True},
        num_classes=num_classes
    ).to(device)

    # Pathway-guided attention for mRNA
    if pathway_mask_path is not None:
        model.setup_pathway_attention(pathway_mask_path)

    # EMA for model weights
    ema = ModelEMA(model, decay=0.999, device=device)

# Cosine LR with linear warmup
    warmup_epochs = 5
    base_lr = lr
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    # Contrastive loss helper
    def _cosine_contrastive_loss(fused, labels, temperature=0.07):
        normed = F.normalize(fused, dim=1)
        sim_matrix = torch.matmul(normed, normed.T) / temperature
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float().to(fused.device)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        exp_sim = torch.exp(sim_matrix)
        mask = torch.eye(fused.size(0), device=fused.device).bool()
        exp_sim = exp_sim.masked_fill(mask, 0)
        pos = (exp_sim * labels_eq).sum(dim=1)
        denom = exp_sim.sum(dim=1)
        loss = -torch.log((pos + 1e-7) / (denom + 1e-7))
        return loss.mean()

    # Pathway consistency loss helper
    def _pathway_consistency_loss(pathway_weights, labels):
        """
        Encourage patients with the same cancer subtype to have similar pathway attention patterns.
        """
        if pathway_weights is None:
            return torch.tensor(0.0, device=device)
        normed = F.normalize(pathway_weights, dim=1)
        sim_matrix = torch.matmul(normed, normed.T)
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float().to(pathway_weights.device)
        mask = torch.eye(pathway_weights.size(0), device=pathway_weights.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0)
        pos_sim = (sim_matrix * labels_eq).sum(dim=1)
        denom = labels_eq.sum(dim=1).clamp(min=1.0)
        loss = -torch.log((pos_sim + 1e-7) / (denom + 1e-7))
        return loss.mean()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print("[DEBUG] Epoch loop started")
    for epoch in range(1, epochs + 1):
        # Warmup learning rate for first warmup_epochs
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for g in optimizer.param_groups:
                g['lr'] = base_lr * lr_scale
        else:
            scheduler.step()

        tic = time.time()
        model.train()
        train_losses = []
        train_cls_losses = []
        train_surv_losses = []

        for batch in train_loader:
            modalities, labels, durations, events = batch
            modalities = [m.to(device) for m in modalities]
            labels = labels.to(device)
            durations = durations.to(device)
            events = events.to(device)

            # Mixup augmentation with class-aware sampling
            if model.training and np.random.rand() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                batch_size = labels.size(0)
                # Mix with similar classes only
                index = torch.randperm(batch_size).to(device)
                for i in range(batch_size):
                    if np.random.rand() < 0.3:  # 30% chance to enforce same-class mixup
                        same_class_idx = (labels == labels[i]).nonzero().squeeze(1)
                        if len(same_class_idx) > 1:
                            index[i] = same_class_idx[torch.randint(0, len(same_class_idx), (1,)).to(device)]
                modalities_mixed = [lam * m + (1 - lam) * m[index] for m in modalities]
                labels_mixed = lam * F.one_hot(labels, num_classes).float() + (1 - lam) * F.one_hot(labels[index], num_classes).float()
                use_mixup = True
            else:
                modalities_mixed = modalities
                labels_mixed = labels
                use_mixup = False

            # Gaussian noise augmentation
            if model.training:
                modalities_mixed = [m + torch.randn_like(m) * 0.05 * (1.0 if model.training else 0.0) for m in modalities_mixed]

            optimizer.zero_grad()
            logits, log_risk, attn_weights, fused, reconstructions, pathway_weights = model(
                modalities_mixed, modality_mask=None, add_noise_to_ae=True, return_reconstructions=True
            )

            # Classification loss
            if use_mixup:
                loss_cls = -(labels_mixed * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss_cls = focal_loss(logits, labels, alpha=CLASS_WEIGHTS.to(device), gamma=2.0, label_smoothing=0.1)

            # Survival loss
            loss_surv = cox_partial_log_likelihood(log_risk, durations, events)
            # Reconstruction loss
            loss_recon = sum(F.mse_loss(rec, mod.detach()) for rec, mod in zip(reconstructions, modalities))
            # Contrastive loss (auxiliary)
            loss_contrast = _cosine_contrastive_loss(fused, labels)
            # Pathway consistency loss
            loss_pathway = _pathway_consistency_loss(pathway_weights, labels)

            loss = (cls_weight * loss_cls + surv_weight * loss_surv +
                    0.3 * loss_recon + 0.1 * loss_contrast + 0.1 * loss_pathway)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_cls_losses.append(loss_cls.item())
            train_surv_losses.append(loss_surv.item())

        # Validation
        model.eval()
        val_losses = []
        val_cls_losses = []
        val_surv_losses = []
        all_val_labels = []
        all_val_preds = []
        val_durations_list = []
        val_events_list = []
        val_risks = []

        with torch.no_grad():
            for batch in val_loader:
                modalities, labels, durations, events = batch
                modalities = [m.to(device) for m in modalities]
                labels = labels.to(device)
                durations = durations.to(device)
                events = events.to(device)

                outputs = model(
                    modalities, modality_mask=None, add_noise_to_ae=False, return_reconstructions=True
                )
                if len(outputs) == 6:
                    logits, log_risk, attn_weights, fused, _, pathway_weights = outputs
                else:
                    logits, log_risk, attn_weights, fused, _ = outputs
                    pathway_weights = None

                loss_cls = focal_loss(logits, labels, alpha=CLASS_WEIGHTS.to(device), gamma=2.0, label_smoothing=0.1)
                loss_surv = cox_partial_log_likelihood(log_risk, durations, events)
                loss = cls_weight * loss_cls + surv_weight * loss_surv

                val_losses.append(loss.item())
                val_cls_losses.append(loss_cls.item())
                val_surv_losses.append(loss_surv.item())

                # metrics
                acc, prec, rec, f1, preds = compute_classification_metrics(logits, labels)
                all_val_labels.extend(labels.detach().cpu().numpy().tolist())
                all_val_preds.extend(preds.tolist())

                # collect survival arrays for c-index
                val_durations_list.extend(durations.detach().cpu().numpy().tolist())
                val_events_list.extend(events.detach().cpu().numpy().tolist())
                val_risks.extend(torch.exp(log_risk).detach().cpu().numpy().reshape(-1).tolist())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses) if val_losses else float("nan")
        avg_train_cls = np.mean(train_cls_losses) if train_cls_losses else 0.0
        avg_train_surv = np.mean(train_surv_losses) if train_surv_losses else 0.0
        avg_val_cls = np.mean(val_cls_losses) if val_cls_losses else 0.0
        avg_val_surv = np.mean(val_surv_losses) if val_surv_losses else 0.0

        # Compute final metrics for validation set
        val_acc, val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0, 0.0
        if len(all_val_preds) > 0:
            val_acc = accuracy_score(all_val_labels, all_val_preds)
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average="macro", zero_division=0)

        # compute concordance index
        try:
            c_index = concordance_index(val_durations_list, -np.array(val_risks), val_events_list)
            # Note: many definitions expect higher risk -> shorter times, so we pass negative risk
        except Exception:
            c_index = float("nan")

        toc = time.time()
        epoch_time = toc - tic

        print(f"[Epoch {epoch:03d}] time: {epoch_time:.1f}s  train_loss: {avg_train_loss:.4f} (cls:{avg_train_cls:.4f}, surv:{avg_train_surv:.4f})  val_loss: {avg_val_loss:.4f} (cls:{avg_val_cls:.4f}, surv:{avg_val_surv:.4f})")
        print(f"             val_acc: {val_acc:.4f}  val_prec: {val_prec:.4f}  val_rec: {val_rec:.4f}  val_f1: {val_f1:.4f}  val_c-index: {c_index:.4f}")

        # scheduler step after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print("  [INFO] Best model updated and saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {epoch} epochs (no improvement in {patience} epochs).")
            break

    # load best weights before returning
    model.load_state_dict(best_model_wts)
    # final save
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    print(f"[INFO] Training completed. Best val loss: {best_val_loss:.4f}. Models saved in {output_dir}")

    return model


# -----------------------------
# CLI and main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train unified multi-modal Transformer model.")
    parser.add_argument("--train_npz", type=str, required=True, help="Path to train .npz (mod_*, labels, durations, events)")
    parser.add_argument("--val_npz", type=str, required=True, help="Path to validation .npz")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--surv_weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classification classes")
    parser.add_argument("--binarize_mutations", action="store_true", default=True, help="Binarize mutation modality (count -> presence/absence)")
    parser.add_argument("--pathway_mask_path", type=str, default=None, help="Path to pathway mask .npz file for pathway-guided attention")
    return parser.parse_args()



class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def state_dict(self):
        return self.ema.state_dict()

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load datasets
    train_ds = MultimodalNPZDataset(args.train_npz)
    val_ds = MultimodalNPZDataset(args.val_npz)

    # Optionally binarize mutation modality (last modality) for train/val
    if args.binarize_mutations:
        for ds in [train_ds, val_ds]:
            if len(ds.modalities) > 0:
                mut_mod = ds.modalities[-1]
                mut_mod = (mut_mod > 0).astype(np.float32)
                ds.modalities[-1] = mut_mod
        print("[INFO] Mutation modality binarized (count -> presence/absence)")

    # modality input dims
    modality_input_dims = [arr.shape[1] for arr in train_ds.modalities]
    print(f"[INFO] Detected {len(modality_input_dims)} modalities with dims: {modality_input_dims}")

    print("[DEBUG] Creating dataloaders...")
    os.environ["TORCH_SHARED_MEMORY_MANAGER"] = "1"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=multimodal_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=multimodal_collate, num_workers=0)

    # Train with per-modality latent dims matching checkpoint [64, 256, 256, 256, 128]
    _ = train(
        train_loader=train_loader,
        val_loader=val_loader,
        modality_input_dims=modality_input_dims,
        device=device,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        cls_weight=args.cls_weight,
        surv_weight=args.surv_weight,
        patience=args.patience,
        ae_latent_dims=[64, 256, 256, 256, 128],
        pathway_mask_path=args.pathway_mask_path
    )


if __name__ == "__main__":
    main()
